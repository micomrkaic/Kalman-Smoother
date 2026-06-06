"""
Gaussian EM for constant linear state-space systems.

Iterates between an E-step (simultaneous smoother + selected inversion)
and an M-step using the full smoothed sufficient statistics, with
posterior-covariance corrections so that Q-hat and R-hat are unbiased.

With exogenous inputs Z, the M-step jointly estimates [F B] and [H D]
via augmented OLS-style regression on [x_{t-1}; z_t] and [x_t; z_t].
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

from .smooth import smooth_const
from .marginal_loglik import marginal_loglik


def _nearest_spd(M, jitter=1e-10):
    """Symmetrize and add jitter to keep M positive definite."""
    M = 0.5 * (M + M.T)
    eigs = np.linalg.eigvalsh(M)
    if eigs.min() < jitter:
        M = M + (jitter - eigs.min()) * np.eye(M.shape[0])
    return 0.5 * (M + M.T)


def em_const(Y, n, *,
             max_iter=200, tol=1e-6, jitter=1e-8, verbose=False,
             F=None, H=None, Q=None, R=None, a0=None, P0=None,
             Z=None, B=None, D=None):
    """
    Gaussian EM for x_t = F x_{t-1} + B z_t + w_t, y_t = H x_t + D z_t + v_t.

    Parameters
    ----------
    Y : (m, T) array.
    n : int, latent state dimension.
    max_iter, tol, jitter, verbose : EM controls.
    F, H, Q, R, a0, P0 : optional initializations.
    Z, B, D : optional exogenous (B and D are then jointly estimated).

    Returns
    -------
    params : dict with keys F, H, Q, R, a0, P0, and B, D if exogenous.
    Xhat : final smoothed state path.
    hist : list of marginal log-likelihood values, monotone non-decreasing.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape

    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z).astype(float)
        k = Z.shape[0]
        if B is None: B = np.zeros((n, k))
        if D is None: D = np.zeros((m, k))
        B = np.atleast_2d(B); D = np.atleast_2d(D)
        Szz = Z @ Z.T

    # Defaults.
    if F is None:
        F = 0.8 * np.eye(n)
    if H is None:
        # Initialize H from leading eigenvectors of the observed covariance.
        mask_cols = np.all(~np.isnan(Y), axis=0)
        if mask_cols.any():
            C = np.cov(Y[:, mask_cols])
            if np.ndim(C) == 0:
                C = np.array([[float(C)]])
        else:
            C = np.eye(m)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        V = eigvecs[:, order]
        H = np.zeros((m, n))
        H[:, :min(m, n)] = V[:, :min(m, n)]
    if Q is None: Q = np.eye(n)
    if R is None: R = 0.2 * np.eye(m)
    if a0 is None: a0 = np.zeros(n)
    if P0 is None: P0 = 10.0 * np.eye(n)

    F = np.atleast_2d(F).astype(float)
    H = np.atleast_2d(H).astype(float)
    Q = np.atleast_2d(Q).astype(float)
    R = np.atleast_2d(R).astype(float)
    a0 = np.atleast_1d(a0).astype(float).ravel()
    P0 = np.atleast_2d(P0).astype(float)

    fully_obs = np.where(np.all(~np.isnan(Y), axis=0))[0]
    hist = []

    for it in range(1, max_iter + 1):
        # ---- E step ----
        ekwargs = dict(return_cov=True)
        if have_exog:
            ekwargs.update(Z=Z, B=B, D=D)
        Xhat, J, h, info = smooth_const(Y, F, H, Q, R, a0, P0, **ekwargs)
        Ptt = info["Ptt"]
        Ptt1 = info["Ptt1"]

        S00 = np.zeros((n, n))
        S11 = np.zeros((n, n))
        S10 = np.zeros((n, n))
        if have_exog:
            S0z = np.zeros((n, k))
            S1z = np.zeros((n, k))
        for t in range(1, T + 1):
            xtm = Xhat[:, t - 1]
            xt = Xhat[:, t]
            Ptm = Ptt[t - 1]
            Pt = Ptt[t]
            Pcr = Ptt1[t - 1]   # = Sigma_{t, t-1}

            S00 += np.outer(xtm, xtm) + Ptm
            S11 += np.outer(xt, xt) + Pt
            S10 += np.outer(xt, xtm) + Pcr
            if have_exog:
                zt = Z[:, t - 1]
                S0z += np.outer(xtm, zt)
                S1z += np.outer(xt, zt)

        Syx = np.zeros((m, n))
        Sxx = np.zeros((n, n))
        Syy = np.zeros((m, m))
        if have_exog:
            Syz = np.zeros((m, k))
            Sxz = np.zeros((n, k))
            Szz_obs = np.zeros((k, k))
        for t_idx in fully_obs:
            t = t_idx + 1
            xt = Xhat[:, t]
            Pt = Ptt[t]
            Sxx += np.outer(xt, xt) + Pt
            yt = Y[:, t_idx]
            Syx += np.outer(yt, xt)
            Syy += np.outer(yt, yt)
            if have_exog:
                zt = Z[:, t_idx]
                Syz += np.outer(yt, zt)
                Sxz += np.outer(xt, zt)
                Szz_obs += np.outer(zt, zt)
        Tobs = len(fully_obs)

        # ---- M step ----
        if have_exog:
            M_aug = np.block([[S00,    S0z],
                              [S0z.T,  Szz]])
            RHS_a = np.concatenate([S10, S1z], axis=1)
            FB = np.linalg.solve((M_aug + jitter * np.eye(n + k)).T,
                                  RHS_a.T).T
            F_new = FB[:, :n]
            B_new = FB[:, n:]
            Q_new = _nearest_spd((S11 - FB @ RHS_a.T) / T, jitter)
        else:
            F_new = np.linalg.solve((S00 + jitter * np.eye(n)).T, S10.T).T
            Q_new = _nearest_spd((S11 - F_new @ S10.T) / T, jitter)
            B_new = None

        if Tobs > 0:
            if have_exog:
                M_aug_obs = np.block([[Sxx,      Sxz],
                                      [Sxz.T,    Szz_obs]])
                RHS_b = np.concatenate([Syx, Syz], axis=1)
                HD = np.linalg.solve((M_aug_obs + jitter * np.eye(n + k)).T,
                                      RHS_b.T).T
                H_new = HD[:, :n]
                D_new = HD[:, n:]
                R_new = _nearest_spd((Syy - HD @ RHS_b.T) / Tobs, jitter)
            else:
                H_new = np.linalg.solve((Sxx + jitter * np.eye(n)).T, Syx.T).T
                R_new = _nearest_spd((Syy - H_new @ Syx.T) / Tobs, jitter)
                D_new = None
        else:
            H_new = H
            R_new = R
            D_new = D if have_exog else None

        a0_new = Xhat[:, 0].copy()
        P0_new = _nearest_spd(Ptt[0], jitter)

        F = F_new; H = H_new; Q = Q_new; R = R_new
        a0 = a0_new; P0 = P0_new
        if have_exog:
            B = B_new; D = D_new

        # Log-likelihood.
        ll_kwargs = {}
        if have_exog:
            ll_kwargs.update(Z=Z, B=B, D=D)
        ll, _ = marginal_loglik(Y, F, H, Q, R, a0, P0, **ll_kwargs)
        hist.append(ll)
        if verbose:
            print(f"iter {it:3d}  loglik = {ll:.6f}")
        if it > 1:
            rel = abs(hist[-1] - hist[-2]) / (1 + abs(hist[-2]))
            if rel < tol:
                break

    params = dict(F=F, H=H, Q=Q, R=R, a0=a0, P0=P0)
    if have_exog:
        params["B"] = B
        params["D"] = D
    return params, Xhat, hist
