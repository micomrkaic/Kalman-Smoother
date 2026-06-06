"""
Takahashi / Erisman-Tinney selected inversion for block-tridiagonal SPD J.

Returns the diagonal blocks Sigma_{tt} = (J^{-1})_{tt} and the sub-diagonal
blocks Sigma_{t,t-1} = (J^{-1})_{t,t-1}, which are the only blocks the
RTS smoother and the EM step actually need.

Method (Takahashi / Erisman-Tinney).  Factor J = L D L^T where L is unit
lower block-bidiagonal, and D is block-diagonal.  We write M_{t+1} for the
unit-lower factor block itself, M_{t+1} = L_{t+1,t} = J_{t+1,t} D_t^{-1}
(a RIGHT solve; for the standard state-space blocks this equals
-Q^{-1} F D_t^{-1}).  Texts that instead define L_{t,t-1} = -M_t flip the
sign of M and hence the sign in the Takahashi off-diagonal line below.

  Forward sweep:
    D_0       = J_{00}
    M_{t+1}   = J_{t+1,t} D_t^{-1}             (right solve)
    D_{t+1}   = J_{t+1,t+1} - M_{t+1} D_t M_{t+1}^T

  Backward sweep (Takahashi):
    Sigma_{T,T}     = D_T^{-1}
    Sigma_{t+1,t}   = -Sigma_{t+1,t+1} M_{t+1}
    Sigma_{t,t}     = D_t^{-1} + M_{t+1}^T Sigma_{t+1,t+1} M_{t+1}

Total cost O(T n^3).
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


def _block(J, t1, t2, n):
    return np.asarray(
        J[t1 * n:(t1 + 1) * n, t2 * n:(t2 + 1) * n].todense()
    )


def selected_inv(J, n):
    """
    Block Takahashi selected inversion for block-tridiagonal SPD J.

    Returns
    -------
    Ptt : list of length T+1, with Ptt[t] = (n, n) block Sigma_{tt}.
    Ptt1 : list of length T, with Ptt1[t] = (n, n) block Sigma_{t+1, t}.
    """
    N = J.shape[0]
    if N % n != 0:
        raise ValueError(f"J shape {J.shape} not divisible by n={n}")
    T1 = N // n
    T = T1 - 1

    D = [None] * T1
    M = [None] * T1
    D[0] = _block(J, 0, 0, n)
    D[0] = 0.5 * (D[0] + D[0].T)
    for t in range(1, T1):
        Jtm = _block(J, t, t - 1, n)
        Mt = np.linalg.solve(D[t - 1].T, Jtm.T).T
        M[t] = Mt
        D[t] = _block(J, t, t, n) - Mt @ D[t - 1] @ Mt.T
        D[t] = 0.5 * (D[t] + D[t].T)

    Ptt = [None] * T1
    Ptt1 = [None] * T
    Ptt[T] = np.linalg.inv(D[T])
    Ptt[T] = 0.5 * (Ptt[T] + Ptt[T].T)
    for t in range(T - 1, -1, -1):
        Mtp1 = M[t + 1]
        Stp1 = Ptt[t + 1]
        Ptt1[t] = -Stp1 @ Mtp1
        Ptt[t] = np.linalg.inv(D[t]) + Mtp1.T @ Stp1 @ Mtp1
        Ptt[t] = 0.5 * (Ptt[t] + Ptt[t].T)

    return Ptt, Ptt1
