"""
Simultaneous Kalman smoother: constant-parameter and time-varying versions.

Solves the smoothing problem  J X = h  where J is the block-tridiagonal
posterior precision matrix and h is the information vector, for the model

    x_t = F_t x_{t-1} + B_t z_t + w_t,   w_t ~ N(0, Q_t)
    y_t = H_t x_t       + D_t z_t + v_t, v_t ~ N(0, R_t)
    x_0 ~ N(a0, P0)            (or diffuse: P0 = None)

NaN entries in Y are treated as missing.
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


def _resolve_exog(Z, B, D, n, m, T):
    """Resolve optional exogenous arguments, returning (have_exog, Z, B, D, k)."""
    if Z is None:
        return False, None, None, None, 0
    Z = np.atleast_2d(Z)
    if Z.shape[1] != T:
        raise ValueError(f"Z must have T={T} columns, got {Z.shape[1]}")
    k = Z.shape[0]
    if B is None:
        B = np.zeros((n, k))
    else:
        B = np.atleast_2d(B)
        if B.shape != (n, k):
            raise ValueError(f"B must be ({n},{k}), got {B.shape}")
    if D is None:
        D = np.zeros((m, k))
    else:
        D = np.atleast_2d(D)
        if D.shape != (m, k):
            raise ValueError(f"D must be ({m},{k}), got {D.shape}")
    return True, Z, B, D, k


def _make_getter(X, T, name, expected_shape):
    """Accept either a length-T list of arrays or a single array (constant)."""
    if isinstance(X, (list, tuple)):
        if len(X) != T:
            raise ValueError(f"{name} must have length T={T}, got {len(X)}")
        return lambda t: np.atleast_2d(X[t])
    arr = np.atleast_2d(X)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} constant must be {expected_shape}, got {arr.shape}")
    return lambda t: arr


def smooth_const(Y, F, H, Q, R, a0, P0, *,
                 Z=None, B=None, D=None,
                 return_cov=False):
    """
    Simultaneous Kalman smoother with constant matrices.

    Parameters
    ----------
    Y : (m, T) array
        Observations.  NaN entries are treated as missing.
    F : (n, n) array
        State transition matrix.
    H : (m, n) array
        Observation matrix.
    Q : (n, n) array
        State innovation covariance.
    R : (m, m) array
        Measurement covariance.
    a0 : (n,) array
        Initial state mean.
    P0 : (n, n) array or None
        Initial state covariance.  Pass None for diffuse (drops the prior block).
    Z : (k, T) array or None
        Optional exogenous inputs.
    B : (n, k) array or None
        State-side exogenous loading.  Defaults to zeros if Z is given.
    D : (m, k) array or None
        Measurement-side exogenous loading.  Defaults to zeros if Z is given.
    return_cov : bool
        If True, compute posterior covariance blocks via selected inversion.

    Returns
    -------
    Xhat : (n, T+1) array
        Smoothed state path.
    J : sparse CSC matrix, ((T+1)*n, (T+1)*n)
        Posterior precision (block tridiagonal SPD).
    h : (N,) array
        Information vector.
    info : dict
        At minimum {'n', 'm', 'T', 'nnz', 'diffuse', 'have_exog'}.  If
        return_cov is True, also contains 'Ptt' (length T+1 list of (n,n))
        and 'Ptt1' (length T list of (n,n)) for the diagonal and
        sub-diagonal blocks of J^{-1}.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape

    F = np.atleast_2d(F).astype(float)
    H = np.atleast_2d(H).astype(float)
    Q = np.atleast_2d(Q).astype(float)
    R = np.atleast_2d(R).astype(float)
    a0 = np.atleast_1d(a0).astype(float).ravel()

    n = F.shape[0]
    if F.shape != (n, n):
        raise ValueError(f"F must be square, got {F.shape}")
    if H.shape != (m, n):
        raise ValueError(f"H must be ({m},{n}), got {H.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n},{n}), got {Q.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must be ({m},{m}), got {R.shape}")
    if a0.shape != (n,):
        raise ValueError(f"a0 must be ({n},), got {a0.shape}")

    have_exog, Z, B, D, k = _resolve_exog(Z, B, D, n, m, T)

    diffuse = P0 is None
    if diffuse:
        P0i = np.zeros((n, n))
    else:
        P0i = np.linalg.inv(np.atleast_2d(P0).astype(float))
    Qi = np.linalg.inv(Q)
    Ri = np.linalg.inv(R)

    N = (T + 1) * n

    # ----- Build J as triplets, h as a dense vector -----
    # Each dynamics step contributes 4 blocks of n*n entries; the prior
    # contributes 1 block; measurement contributes T blocks.  Total
    # entries: n*n + 4*T*n*n + T*n*n.  Add slack for safety.
    cap = n * n + 4 * T * n * n + T * n * n + 100
    ii = np.empty(cap, dtype=np.intp)
    jj = np.empty(cap, dtype=np.intp)
    vv = np.empty(cap, dtype=float)
    ptr = [0]

    def add_block(rows, cols, M):
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        sz = M.size
        ii[ptr[0]:ptr[0] + sz] = rr.ravel()
        jj[ptr[0]:ptr[0] + sz] = cc.ravel()
        vv[ptr[0]:ptr[0] + sz] = M.ravel()
        ptr[0] += sz

    def idx(t):
        return np.arange(t * n, (t + 1) * n)

    h = np.zeros(N)

    if not diffuse:
        add_block(idx(0), idx(0), P0i)
        h[idx(0)] += P0i @ a0

    FtQiF = F.T @ Qi @ F
    FtQi = F.T @ Qi
    QiF = Qi @ F
    for t in range(1, T + 1):
        add_block(idx(t - 1), idx(t - 1),  FtQiF)
        add_block(idx(t),     idx(t),      Qi)
        add_block(idx(t - 1), idx(t),     -FtQi)
        add_block(idx(t),     idx(t - 1), -QiF)
        if have_exog:
            Bz = B @ Z[:, t - 1]
            h[idx(t)]     += Qi @ Bz
            h[idx(t - 1)] += -FtQi @ Bz

    HtRiH_full = H.T @ Ri @ H
    HtRi = H.T @ Ri
    any_missing = np.isnan(Y).any()
    for t in range(1, T + 1):
        yt = Y[:, t - 1]
        ytilde = yt - D @ Z[:, t - 1] if have_exog else yt
        if not any_missing or not np.isnan(yt).any():
            add_block(idx(t), idx(t), HtRiH_full)
            h[idx(t)] += HtRi @ ytilde
        else:
            mask = ~np.isnan(yt)
            if mask.any():
                Hm = H[mask, :]
                Rm_i = np.linalg.inv(R[np.ix_(mask, mask)])
                add_block(idx(t), idx(t), Hm.T @ Rm_i @ Hm)
                h[idx(t)] += Hm.T @ Rm_i @ ytilde[mask]

    # Assemble sparse J (COO -> CSC).  Duplicate (i,j) entries are summed.
    coo = sp.coo_matrix(
        (vv[:ptr[0]], (ii[:ptr[0]], jj[:ptr[0]])),
        shape=(N, N),
    )
    J = coo.tocsc()
    # Symmetrize against accumulated floating-point asymmetry.
    J = (J + J.T) * 0.5

    # Solve J X = h via sparse Cholesky (scipy's splu suffices for block
    # tridiagonal; if scikit-sparse is installed users can swap in CHOLMOD).
    lu = spla.splu(J.tocsc())
    Xvec = lu.solve(h)
    Xhat = Xvec.reshape((T + 1, n)).T

    info = dict(n=n, m=m, T=T, nnz=int(J.nnz),
                diffuse=diffuse, have_exog=have_exog)

    if return_cov:
        Ptt, Ptt1 = selected_inv(J, n)
        info["Ptt"] = Ptt
        info["Ptt1"] = Ptt1

    return Xhat, J, h, info


def smooth_tv(Y, Fseq, Hseq, Qseq, Rseq, a0, P0, *,
              Z=None, Bseq=None, Dseq=None,
              return_cov=False):
    """
    Simultaneous Kalman smoother with time-varying matrices.

    F, H, Q, R can each be either a constant array OR a length-T list of
    arrays.  B and D, if given, can also be constant or per-period.

    Other arguments and outputs are as in smooth_const.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape

    a0 = np.atleast_1d(a0).astype(float).ravel()
    n = a0.shape[0]

    get_F = _make_getter(Fseq, T, "Fseq", (n, n))
    get_H = _make_getter(Hseq, T, "Hseq", (m, n))
    get_Q = _make_getter(Qseq, T, "Qseq", (n, n))
    get_R = _make_getter(Rseq, T, "Rseq", (m, m))

    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z).astype(float)
        if Z.shape[1] != T:
            raise ValueError(f"Z must have T={T} columns, got {Z.shape[1]}")
        kZ = Z.shape[0]
        if Bseq is None:
            Bseq = np.zeros((n, kZ))
        if Dseq is None:
            Dseq = np.zeros((m, kZ))
        get_B = _make_getter(Bseq, T, "Bseq", (n, kZ))
        get_D = _make_getter(Dseq, T, "Dseq", (m, kZ))

    diffuse = P0 is None
    if diffuse:
        P0i = np.zeros((n, n))
    else:
        P0i = np.linalg.inv(np.atleast_2d(P0).astype(float))

    N = (T + 1) * n
    cap = n * n + 4 * T * n * n + T * n * n + 100
    ii = np.empty(cap, dtype=np.intp)
    jj = np.empty(cap, dtype=np.intp)
    vv = np.empty(cap, dtype=float)
    ptr = [0]

    def add_block(rows, cols, M):
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        sz = M.size
        ii[ptr[0]:ptr[0] + sz] = rr.ravel()
        jj[ptr[0]:ptr[0] + sz] = cc.ravel()
        vv[ptr[0]:ptr[0] + sz] = M.ravel()
        ptr[0] += sz

    def idx(t):
        return np.arange(t * n, (t + 1) * n)

    h = np.zeros(N)
    if not diffuse:
        add_block(idx(0), idx(0), P0i)
        h[idx(0)] += P0i @ a0

    for t in range(1, T + 1):
        Ft = get_F(t - 1)
        Qti = np.linalg.inv(get_Q(t - 1))
        add_block(idx(t - 1), idx(t - 1),  Ft.T @ Qti @ Ft)
        add_block(idx(t),     idx(t),      Qti)
        add_block(idx(t - 1), idx(t),     -Ft.T @ Qti)
        add_block(idx(t),     idx(t - 1), -Qti @ Ft)
        if have_exog:
            Bz = get_B(t - 1) @ Z[:, t - 1]
            h[idx(t)]     += Qti @ Bz
            h[idx(t - 1)] += -Ft.T @ Qti @ Bz

    for t in range(1, T + 1):
        yt = Y[:, t - 1]
        Ht = get_H(t - 1)
        Rt = get_R(t - 1)
        ytilde = yt - get_D(t - 1) @ Z[:, t - 1] if have_exog else yt
        mask = ~np.isnan(yt)
        if mask.all():
            Rti = np.linalg.inv(Rt)
            add_block(idx(t), idx(t), Ht.T @ Rti @ Ht)
            h[idx(t)] += Ht.T @ Rti @ ytilde
        elif mask.any():
            Hm = Ht[mask, :]
            Rm_i = np.linalg.inv(Rt[np.ix_(mask, mask)])
            add_block(idx(t), idx(t), Hm.T @ Rm_i @ Hm)
            h[idx(t)] += Hm.T @ Rm_i @ ytilde[mask]

    coo = sp.coo_matrix(
        (vv[:ptr[0]], (ii[:ptr[0]], jj[:ptr[0]])),
        shape=(N, N),
    )
    J = coo.tocsc()
    J = (J + J.T) * 0.5

    lu = spla.splu(J.tocsc())
    Xvec = lu.solve(h)
    Xhat = Xvec.reshape((T + 1, n)).T

    info = dict(n=n, m=m, T=T, nnz=int(J.nnz),
                diffuse=diffuse, have_exog=have_exog)

    if return_cov:
        Ptt, Ptt1 = selected_inv(J, n)
        info["Ptt"] = Ptt
        info["Ptt1"] = Ptt1

    return Xhat, J, h, info
