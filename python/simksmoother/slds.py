"""
Switching Linear Dynamical System smoother via variational EM.

Exact coordinate updates for the factorization q(X, S) = q(X) prod_t q(s_t):

(a) q(X)-step: q(X) is Gaussian with precision/information assembled from
    responsibility-weighted EXPECTED quadratic transition terms.  The
    per-period blocks involve sum_k pi_{t,k} Q_k^{-1},
    sum_k pi_{t,k} Q_k^{-1} F_k, and sum_k pi_{t,k} F_k' Q_k^{-1} F_k
    (NOT the smoother run at averaged matrices Fbar_t, Qbar_t, which is a
    moment-matching approximation, not the variational update).  The
    resulting precision matrix is still block tridiagonal at every
    iteration, so the cost per iteration is unchanged.

(b) q(S)-step: forward-backward on the regime HMM with EXPECTED Gaussian
    transition log-likelihoods under q(X), i.e. including the posterior
    covariance corrections built from Sigma_{tt}, Sigma_{t-1,t-1}, and
    the lag-one block Sigma_{t,t-1} delivered by selected inversion ---
    not the density evaluated at the smoothed point path alone.
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .selected_inv import selected_inv


def _logsumexp(x):
    x = np.asarray(x)
    m = float(x.max())
    return m + float(np.log(np.sum(np.exp(x - m))))


def smooth_slds(Y, F_set, H, Q_set, R, a0, P0, *,
                A=None, pi0=None,
                Z=None, B_set=None, D=None,
                max_iter=30, tol=1e-5, verbose=False):
    """
    Variational EM smoother for switching linear dynamical systems.

    Model:
        s_t in {1,...,K} is a Markov chain with transition matrix A.
        x_t = F_{s_t} x_{t-1} + B_{s_t} z_t + w_t, w_t ~ N(0, Q_{s_t}).
        y_t = H x_t + D z_t + v_t, v_t ~ N(0, R).

    H, R, D are regime-invariant; F, Q, B are regime-dependent.

    Parameters
    ----------
    Y : (m, T) array.
    F_set : length-K list of (n, n) transition matrices.
    H : (m, n) observation matrix.
    Q_set : length-K list of (n, n) state innovation covariances.
    R : (m, m) measurement covariance.
    a0 : (n,) initial state mean.
    P0 : (n, n) initial state covariance.
    A : (K, K) transition matrix, rows sum to 1.  Default: near-diagonal.
    pi0 : (K,) initial regime distribution.  Default: uniform.
    Z : (k, T) optional exogenous.
    B_set : length-K list of (n, k) state loadings (if Z given).
    D : (m, k) measurement loading (regime-invariant).

    Returns
    -------
    Xhat : (n, T+1) smoothed state path.
    info : dict
        pi : (K, T) regime responsibilities (column t = pi_{t,:}).
        pi0_post : (K,) posterior over the initial regime.
        Ptt, Ptt1 : posterior covariance blocks of q(X) at convergence.
        iters, converged, delta_history.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape
    K = len(F_set)
    F_set = [np.atleast_2d(Fk).astype(float) for Fk in F_set]
    Q_set = [np.atleast_2d(Qk).astype(float) for Qk in Q_set]
    H = np.atleast_2d(H).astype(float)
    R = np.atleast_2d(R).astype(float)
    a0 = np.atleast_1d(a0).astype(float).ravel()
    P0 = np.atleast_2d(P0).astype(float)
    n = F_set[0].shape[0]

    if A is None:
        A = 0.95 * np.eye(K) + (0.05 / (K - 1)) * (np.ones((K, K)) - np.eye(K))
    A = np.atleast_2d(A).astype(float)

    if pi0 is None:
        pi0 = np.ones(K) / K
    pi0 = np.atleast_1d(pi0).astype(float).ravel()

    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z).astype(float)
        kZ = Z.shape[0]
        if B_set is None:
            B_set = [np.zeros((n, kZ)) for _ in range(K)]
        else:
            B_set = [np.atleast_2d(Bk).astype(float) for Bk in B_set]
        if D is None:
            D = np.zeros((m, kZ))
        D = np.atleast_2d(D).astype(float)

    # Initialize responsibilities uniformly.
    pi_r = np.ones((K, T)) / K
    pi_prev = pi_r.copy()
    delta_history = []
    converged = False
    iters_done = 0
    log_gamma = None

    # Per-regime precomputations for the expected quadratic forms.
    Qk_inv = [np.linalg.inv(Qk) for Qk in Q_set]
    QiF = [Qk_inv[k] @ F_set[k] for k in range(K)]                  # Q_k^{-1} F_k
    FQiF = [F_set[k].T @ Qk_inv[k] @ F_set[k] for k in range(K)]    # F_k' Q_k^{-1} F_k
    logdetQ = [float(np.linalg.slogdet(Qk)[1]) for Qk in Q_set]
    if have_exog:
        QiB = [Qk_inv[k] @ B_set[k] for k in range(K)]              # Q_k^{-1} B_k
        FQiB = [F_set[k].T @ Qk_inv[k] @ B_set[k] for k in range(K)]

    P0i = np.linalg.inv(P0)
    Ri_full = np.linalg.inv(R)

    # Per-period measurement contributions (missing entries dropped).
    HRH = []     # H_m' R_m^{-1} H_m
    HRy = []     # H_m' R_m^{-1} (y_t^obs - D_m z_t)
    for t in range(T):
        yt = Y[:, t]
        mask = ~np.isnan(yt)
        if mask.any():
            Hm = H[mask, :]
            Rm_i = np.linalg.inv(R[np.ix_(mask, mask)])
            resid = yt[mask]
            if have_exog:
                resid = resid - (D @ Z[:, t])[mask]
            HRH.append(Hm.T @ Rm_i @ Hm)
            HRy.append(Hm.T @ Rm_i @ resid)
        else:
            HRH.append(np.zeros((n, n)))
            HRy.append(np.zeros(n))

    N = (T + 1) * n
    Xhat = None
    Ptt = Ptt1 = None

    for it in range(1, max_iter + 1):
        iters_done = it

        # ---- q(X) step: exact assembly from expected quadratic forms ----
        # Responsibility-weighted transition blocks at each t = 1..T.
        Qbar = []    # sum_k pi Q_k^{-1}
        QFbar = []   # sum_k pi Q_k^{-1} F_k
        FQFbar = []  # sum_k pi F_k' Q_k^{-1} F_k
        if have_exog:
            QBbar = []   # sum_k pi Q_k^{-1} B_k
            FQBbar = []  # sum_k pi F_k' Q_k^{-1} B_k
        for t in range(T):
            w = pi_r[:, t]
            Qbar.append(sum(w[k] * Qk_inv[k] for k in range(K)))
            QFbar.append(sum(w[k] * QiF[k] for k in range(K)))
            FQFbar.append(sum(w[k] * FQiF[k] for k in range(K)))
            if have_exog:
                QBbar.append(sum(w[k] * QiB[k] for k in range(K)))
                FQBbar.append(sum(w[k] * FQiB[k] for k in range(K)))

        rows, cols, vals = [], [], []
        h = np.zeros(N)

        def add_block(bi, bj, M):
            r0, c0 = bi * n, bj * n
            for i in range(n):
                for j in range(n):
                    v = M[i, j]
                    if v != 0.0:
                        rows.append(r0 + i)
                        cols.append(c0 + j)
                        vals.append(v)

        # Diagonal blocks.  Block index b = 0..T corresponds to x_b;
        # transition t (= 1..T) lives at python index t-1 of the lists above.
        for b in range(T + 1):
            Jbb = np.zeros((n, n))
            if b == 0:
                Jbb += P0i
            else:
                Jbb += Qbar[b - 1] + HRH[b - 1]
            if b < T:
                Jbb += FQFbar[b]          # outgoing transition b -> b+1
            add_block(b, b, Jbb)
        # Off-diagonal blocks: J_{b, b-1} = -sum_k pi_{b,k} Q_k^{-1} F_k.
        for b in range(1, T + 1):
            Joff = -QFbar[b - 1]
            add_block(b, b - 1, Joff)
            add_block(b - 1, b, Joff.T)

        # Information vector.
        h[0:n] = P0i @ a0
        if have_exog:
            h[0:n] -= FQBbar[0] @ Z[:, 0]
        for b in range(1, T + 1):
            hb = HRy[b - 1].copy()
            if have_exog:
                hb += QBbar[b - 1] @ Z[:, b - 1]
                if b < T:
                    hb -= FQBbar[b] @ Z[:, b]
            h[b * n:(b + 1) * n] += hb

        J = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        J = (J + J.T) * 0.5
        lu = spla.splu(J)
        Xvec = lu.solve(h)
        Xhat = Xvec.reshape((T + 1, n)).T

        # Posterior covariance blocks for the expected-log-likelihood step.
        Ptt, Ptt1 = selected_inv(J, n)

        # ---- q(S) step: HMM forward-backward on EXPECTED log-likelihoods ----
        logL = np.zeros((K, T))
        for t in range(T):
            xt = Xhat[:, t + 1]
            xtm = Xhat[:, t]
            S_tt = Ptt[t + 1]            # Cov(x_t | Y)
            S_mm = Ptt[t]                # Cov(x_{t-1} | Y)
            S_tm = Ptt1[t]               # Cov(x_t, x_{t-1} | Y)
            for kk in range(K):
                Fk = F_set[kk]
                mu_k = Fk @ xtm
                if have_exog:
                    mu_k = mu_k + B_set[kk] @ Z[:, t]
                d = xt - mu_k
                Ctr = (S_tt - Fk @ S_tm.T - S_tm @ Fk.T
                       + Fk @ S_mm @ Fk.T)
                quad = float(d @ Qk_inv[kk] @ d)                        + float(np.trace(Qk_inv[kk] @ Ctr))
                logL[kk, t] = (-0.5 * quad
                                - 0.5 * logdetQ[kk]
                                - 0.5 * n * np.log(2 * np.pi))

        log_alpha = np.full((K, T), -np.inf)
        log_alpha[:, 0] = np.log(pi0 + 1e-300) + logL[:, 0]
        log_A = np.log(A + 1e-300)
        for t in range(1, T):
            for j in range(K):
                log_alpha[j, t] = (_logsumexp(log_alpha[:, t - 1] + log_A[:, j])
                                    + logL[j, t])
        log_beta = np.zeros((K, T))
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[i, t] = _logsumexp(log_A[i, :] + logL[:, t + 1]
                                             + log_beta[:, t + 1])

        log_gamma = log_alpha + log_beta
        pi_new = np.zeros((K, T))
        for t in range(T):
            lse = _logsumexp(log_gamma[:, t])
            pi_new[:, t] = np.exp(log_gamma[:, t] - lse)

        delta = float(np.max(np.abs(pi_new - pi_prev)))
        delta_history.append(delta)
        if verbose:
            print(f"  slds iter {it:2d}  |delta pi|_inf = {delta:.3e}")
        pi_prev = pi_r.copy()
        pi_r = pi_new
        if delta < tol and it > 1:
            converged = True
            break

    lse0 = _logsumexp(log_gamma[:, 0])
    pi0_post = np.exp(log_gamma[:, 0] - lse0)

    info = dict(
        pi=pi_r.copy(),
        pi0_post=pi0_post,
        Ptt=Ptt,
        Ptt1=Ptt1,
        iters=iters_done,
        converged=converged,
        delta_history=delta_history,
    )
    return Xhat, info
