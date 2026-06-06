"""
Robust simultaneous smoother for non-Gaussian disturbances.

Generalizes the Gaussian smoother to heavy-tailed disturbances by
interpreting them as Gaussian scale mixtures and running iteratively
reweighted least squares (IRLS) on the same sparse block-tridiagonal
system that the Gaussian smoother uses.

Families:
  * Student-t  : Gaussian scale mixture, tau ~ Gamma(nu/2, nu/2);
                 weight (nu+m)/(nu+delta^2) with delta^2 the squared
                 Mahalanobis distance of the residual.
  * Laplace    : Gaussian scale mixture; weight 1 / sqrt(delta^2).
  * Huber      : NOT a proper scale mixture --- the weight
                 min(1, c/delta) is the IRLS working weight implied by
                 the Huber loss (Holland-Welsch), applied to the same
                 sparse system.

By default the weights are evaluated by plugging in residuals at the
current smoothed mean.  The resulting fixed point is the MAP / posterior
mode of the heavy-tailed model (IRLS = majorize-minimize on the
non-Gaussian objective).  For Student-t, exact_estep=True replaces the
plug-in squared distance by its posterior expectation under the current
Gaussian approximation,
    E[delta_t^2 | Y] = delta_hat_t^2 + tr(R^{-1} H Sigma_tt H')
(and the state-side analogue with Sigma_tt, Sigma_{t-1,t-1}, and the
lag-one block Sigma_{t,t-1}), which is the E-step quantity required by
latent-scale EM.  This costs one selected inversion per iteration.
"""

# simksmoother --- a simultaneous Kalman smoother in sparse linear algebra form.
# Copyright (C) 2026 Mico Mrkaic.
#
# Produced under the guidance and direction of Mico Mrkaic, with the
# assistance of AI (Claude, Anthropic).
#
# This file is part of the simksmoother package.
#
# simksmoother is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the LICENSE file in the package root,
# or <https://www.gnu.org/licenses/>.
#

import numpy as np

from .smooth import smooth_const, smooth_tv


def _robust_weight(family, delta2, m, nu, c):
    f = family.lower()
    if f == "student-t":
        return (nu + m) / (nu + delta2)
    if f == "laplace":
        return 1.0 / max(np.sqrt(delta2), 1e-10)
    if f == "huber":
        d = np.sqrt(delta2)
        return 1.0 if d <= c else c / d
    raise ValueError(f"Unknown family '{family}'")


def smooth_robust(Y, F, H, Q, R, a0, P0, family, *,
                  nu=4.0, c=1.345,
                  max_iter=50, tol=1e-6,
                  robust_state=False, return_cov=False,
                  exact_estep=False,
                  verbose=False,
                  Z=None, B=None, D=None):
    """
    IRLS smoother for heavy-tailed measurement (and optionally state) noise.

    Parameters
    ----------
    Y, F, H, Q, R, a0, P0 : as in smooth_const.
    family : {'gaussian', 'student-t', 'laplace', 'huber'}
        Distribution of v_t (and optionally w_t).
    nu : float
        Student-t degrees of freedom (default 4).
    c : float
        Huber tuning constant (default 1.345, the Holland-Welsch value).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on max |Delta X| between iterations.
    robust_state : bool
        If True, also apply heavy tails to the state innovations
        (recommended for structural-break detection).
    exact_estep : bool
        Student-t only.  If True, weight updates use the posterior
        expectation of the squared Mahalanobis distance (plug-in value
        plus the trace correction from the posterior covariance blocks)
        rather than the plug-in value alone.  This is the latent-scale
        EM E-step under the current Gaussian approximation; the default
        (False) is plug-in IRLS, whose fixed point is the posterior mode.
    Z, B, D : optional exogenous inputs.

    Returns
    -------
    Xhat : (n, T+1) array
    info : dict
        Keys: tau_v, tau_w (if robust_state), iters, converged,
        delta_history, family, (Ptt, Ptt1 if return_cov).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape

    F = np.atleast_2d(F).astype(float)
    H = np.atleast_2d(H).astype(float)
    Q = np.atleast_2d(Q).astype(float)
    R = np.atleast_2d(R).astype(float)
    n = F.shape[0]

    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z).astype(float)
        k = Z.shape[0]
        if B is None: B = np.zeros((n, k))
        if D is None: D = np.zeros((m, k))
        B = np.atleast_2d(B); D = np.atleast_2d(D)
        BZ = B @ Z
        DZ = D @ Z

    # Gaussian pass-through.
    if family.lower() == "gaussian":
        kwargs = dict(return_cov=return_cov)
        if have_exog:
            kwargs.update(Z=Z, B=B, D=D)
        Xhat, J, h, info0 = smooth_const(Y, F, H, Q, R, a0, P0, **kwargs)
        info = dict(info0)
        info.update(tau_v=np.ones(T), iters=1, converged=True,
                    delta_history=[], family="gaussian")
        return Xhat, info

    if exact_estep and family.lower() != "student-t":
        raise ValueError("exact_estep=True is only available for family='student-t'")

    Qi_full = np.linalg.inv(Q)
    tau_v = np.ones(T)
    tau_w = np.ones(T) if robust_state else None

    Xhat_prev = None
    delta_history = []
    converged = False
    iters_done = 0

    for it in range(1, max_iter + 1):
        iters_done = it

        Rcells = [R / tau_v[t] for t in range(T)]
        if robust_state:
            Qcells = [Q / tau_w[t] for t in range(T)]
        else:
            Qcells = Q

        kwargs = {}
        if have_exog:
            kwargs.update(Z=Z, Bseq=B, Dseq=D)
        if exact_estep:
            kwargs.update(return_cov=True)
        Xhat, _, _, info_it = smooth_tv(Y, F, H, Qcells, Rcells, a0, P0, **kwargs)
        if exact_estep:
            Ptt_it = info_it["Ptt"]
            Ptt1_it = info_it["Ptt1"]

        if Xhat_prev is not None:
            delta = float(np.max(np.abs(Xhat - Xhat_prev)))
            delta_history.append(delta)
            if verbose:
                print(f"  iter {it:2d}   |delta X|_inf = {delta:.3e}")
            if delta < tol:
                converged = True
                break
        Xhat_prev = Xhat.copy()

        # E-step: measurement weights.
        for t in range(T):
            yt = Y[:, t]
            mask = ~np.isnan(yt)
            if mask.any():
                Hm = H[mask, :]
                Rm_i = np.linalg.inv(R[np.ix_(mask, mask)])
                r = yt[mask] - Hm @ Xhat[:, t + 1]
                if have_exog:
                    r = r - DZ[mask, t]
                delta2 = float(r @ Rm_i @ r)
                if exact_estep:
                    # E[delta^2 | Y] = plug-in + tr(R_m^{-1} H_m Sigma_tt H_m').
                    delta2 += float(np.trace(Rm_i @ Hm @ Ptt_it[t + 1] @ Hm.T))
                m_eff = int(mask.sum())
                tau_v[t] = _robust_weight(family, delta2, m_eff, nu, c)
            else:
                tau_v[t] = 1.0

        if robust_state:
            for t in range(T):
                w = Xhat[:, t + 1] - F @ Xhat[:, t]
                if have_exog:
                    w = w - BZ[:, t]
                eta2 = float(w @ Qi_full @ w)
                if exact_estep:
                    # Var(x_t - F x_{t-1} | Y) trace correction.
                    S_tt = Ptt_it[t + 1]
                    S_mm = Ptt_it[t]
                    S_tm = Ptt1_it[t]
                    Ctr = (S_tt - F @ S_tm.T - S_tm @ F.T
                           + F @ S_mm @ F.T)
                    eta2 += float(np.trace(Qi_full @ Ctr))
                tau_w[t] = _robust_weight(family, eta2, n, nu, c)

    info = dict(
        tau_v=tau_v.copy(),
        iters=iters_done,
        converged=converged,
        delta_history=delta_history,
        family=family,
    )
    if robust_state:
        info["tau_w"] = tau_w.copy()

    if return_cov:
        Rcells_final = [R / tau_v[t] for t in range(T)]
        if robust_state:
            Qcells_final = [Q / tau_w[t] for t in range(T)]
        else:
            Qcells_final = Q
        kwargs = dict(return_cov=True)
        if have_exog:
            kwargs.update(Z=Z, Bseq=B, Dseq=D)
        Xhat, _, _, info_f = smooth_tv(Y, F, H, Qcells_final, Rcells_final,
                                        a0, P0, **kwargs)
        info["Ptt"] = info_f["Ptt"]
        info["Ptt1"] = info_f["Ptt1"]

    return Xhat, info
