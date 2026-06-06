"""
Marginal log-likelihood log p(Y; theta) via the simultaneous formulation.
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
import scipy.sparse.linalg as spla

from .smooth import smooth_const


def marginal_loglik(Y, F, H, Q, R, a0, P0, *,
                    Z=None, B=None, D=None):
    """
    Compute the Gaussian marginal log-likelihood by completing the square in X.

    Closed form (constant-params, proper prior):
        log p(Y) = -1/2 [c_W' Omega^{-1} c_W + sum_t ytilde_t' R^{-1} ytilde_t - h' Xhat]
                   -1/2 [log det P0 + T log det Q + T log det R + log det J]
                   -(m T / 2) log(2 pi)

    where ytilde_t = y_t - D z_t in the exogenous case and
    c_W' Omega^{-1} c_W = a0' P0^{-1} a0 + sum_t (B z_t)' Q^{-1} (B z_t)
    (the second term vanishes without exogenous inputs).  Note the
    MINUS sign on log det J: a more concentrated posterior (larger
    det J) reduces the integrated mass.

    For the diffuse case (P0 = None), the prior contribution is omitted.

    Returns
    -------
    ll : float
    parts : dict
        Breakdown of contributions for diagnostics.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape
    diffuse = P0 is None

    Xhat, J, h, _ = smooth_const(Y, F, H, Q, R, a0, P0, Z=Z, B=B, D=D)
    Xvec = Xhat.T.reshape(-1)

    # log det J via dense banded LU is unstable; use the sparse Cholesky-like
    # decomposition implied by splu.  Diagonals of U give the relevant info.
    lu = spla.splu(J.tocsc(),
                   diag_pivot_thresh=0.0,  # symmetric/SPD: no pivoting required
                   options={"SymmetricMode": True})
    # Sum of log |U_ii| gives log |det J| up to a sign (positive for SPD).
    diagU = lu.U.diagonal()
    logdet_J = float(np.sum(np.log(np.abs(diagU))))

    Ri = np.linalg.inv(np.atleast_2d(R).astype(float))
    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z)
        if D is None:
            D = np.zeros((m, Z.shape[0]))
        D = np.atleast_2d(D)

    yRy = 0.0
    for t in range(T):
        yt = Y[:, t]
        ytilde = yt - D @ Z[:, t] if have_exog else yt
        yRy += float(ytilde @ Ri @ ytilde)

    # Exogenous state forcing contributes to the constant of the completed
    # square: c_W' Omega^{-1} c_W = a0' P0^{-1} a0 + sum_t (B z_t)' Q^{-1} (B z_t).
    bQb = 0.0
    if have_exog:
        B = np.atleast_2d(B).astype(float) if B is not None \
            else np.zeros((np.atleast_2d(Q).shape[0], Z.shape[0]))
        Qi = np.linalg.inv(np.atleast_2d(Q).astype(float))
        for t in range(T):
            bz = B @ Z[:, t]
            bQb += float(bz @ Qi @ bz)

    if diffuse:
        a_pri = 0.0
        logdet_P0 = 0.0
    else:
        a0 = np.atleast_1d(a0).astype(float).ravel()
        P0 = np.atleast_2d(P0).astype(float)
        P0i = np.linalg.inv(P0)
        a_pri = float(a0 @ P0i @ a0)
        sign, logdet_P0 = np.linalg.slogdet(P0)

    hX = float(h @ Xvec)
    _, logdet_Q = np.linalg.slogdet(np.atleast_2d(Q).astype(float))
    _, logdet_R = np.linalg.slogdet(np.atleast_2d(R).astype(float))

    ll = (-0.5 * (a_pri + bQb + yRy - hX)
          - 0.5 * (logdet_P0 + T * logdet_Q + T * logdet_R + logdet_J)
          - 0.5 * (m * T) * np.log(2 * np.pi))

    parts = dict(
        quad_prior=-0.5 * a_pri,
        quad_exog=-0.5 * bQb,
        quad_obs=-0.5 * yRy,
        quad_hXhat=+0.5 * hX,
        logdet_P0=-0.5 * logdet_P0,
        logdet_Q=-0.5 * T * logdet_Q,
        logdet_R=-0.5 * T * logdet_R,
        logdet_J=-0.5 * logdet_J,
        const=-0.5 * m * T * np.log(2 * np.pi),
        diffuse=diffuse,
        have_exog=have_exog,
    )
    return float(ll), parts
