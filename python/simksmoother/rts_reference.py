"""
Textbook Kalman filter + Rauch-Tung-Striebel smoother.

This is the recursive form, kept around purely to cross-check that the
simultaneous smoother in smooth.py agrees with the standard derivation
to machine precision.
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


def rts_smoother(Y, F, H, Q, R, a0, P0):
    """
    Run the textbook Kalman filter + RTS smoother on the no-exogenous,
    constant-parameter model.  Missing observations (NaN entries) are
    handled by zeroing the relevant rows of H R^{-1} H at each t.

    Returns
    -------
    XsT : (n, T+1) array
        Smoothed means.
    PsT : list of length T+1
        Smoothed covariances P_{t|T}.
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
    P0 = np.atleast_2d(P0).astype(float)
    n = F.shape[0]

    # Filtering pass.
    af = [None] * (T + 1)        # filtered means a_{t|t}
    Pf = [None] * (T + 1)        # filtered covariances P_{t|t}
    ap = [None] * (T + 1)        # predicted means a_{t|t-1}
    Pp = [None] * (T + 1)        # predicted covariances P_{t|t-1}

    af[0] = a0.copy()
    Pf[0] = P0.copy()

    for t in range(1, T + 1):
        # Predict
        ap[t] = F @ af[t - 1]
        Pp[t] = F @ Pf[t - 1] @ F.T + Q

        yt = Y[:, t - 1]
        mask = ~np.isnan(yt)
        if mask.any():
            Hm = H[mask, :]
            Rm = R[np.ix_(mask, mask)]
            y_obs = yt[mask]
            S = Hm @ Pp[t] @ Hm.T + Rm
            K = np.linalg.solve(S.T, (Pp[t] @ Hm.T).T).T   # = Pp Hm' S^{-1}
            innov = y_obs - Hm @ ap[t]
            af[t] = ap[t] + K @ innov
            Pf[t] = Pp[t] - K @ Hm @ Pp[t]
            Pf[t] = 0.5 * (Pf[t] + Pf[t].T)
        else:
            af[t] = ap[t]
            Pf[t] = Pp[t]

    # Smoothing pass (RTS).
    XsT = np.zeros((n, T + 1))
    PsT = [None] * (T + 1)
    XsT[:, T] = af[T]
    PsT[T] = Pf[T]
    for t in range(T - 1, -1, -1):
        Pp_next = F @ Pf[t] @ F.T + Q
        A = np.linalg.solve(Pp_next.T, (Pf[t] @ F.T).T).T   # smoother gain
        XsT[:, t] = af[t] + A @ (XsT[:, t + 1] - F @ af[t])
        PsT[t] = Pf[t] + A @ (PsT[t + 1] - Pp_next) @ A.T
        PsT[t] = 0.5 * (PsT[t] + PsT[t].T)

    return XsT, PsT
