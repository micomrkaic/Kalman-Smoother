"""
Simulate from the constant-parameter linear Gaussian state-space model
(optionally with exogenous inputs).
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


def simulate_const(T, F, H, Q, R, a0, P0, *,
                   Z=None, B=None, D=None, rng=None):
    """
    Simulate (Y, X) from x_t = F x_{t-1} + B z_t + w_t, y_t = H x_t + D z_t + v_t.

    Parameters
    ----------
    T : int
        Number of observation periods.
    F, H, Q, R, a0, P0 : arrays
        Model parameters.
    Z : (k, T) array or None
        Optional exogenous inputs.
    B, D : arrays or None
        Optional exogenous loadings (default zero).
    rng : numpy.random.Generator or None
        Random number generator.  Defaults to numpy.random.default_rng().

    Returns
    -------
    Y : (m, T) array
    X : (n, T+1) array, columns x_0, x_1, ..., x_T
    """
    if rng is None:
        rng = np.random.default_rng()

    F = np.atleast_2d(F).astype(float)
    H = np.atleast_2d(H).astype(float)
    Q = np.atleast_2d(Q).astype(float)
    R = np.atleast_2d(R).astype(float)
    a0 = np.atleast_1d(a0).astype(float).ravel()
    P0 = np.atleast_2d(P0).astype(float)

    n = F.shape[0]
    m = H.shape[0]

    have_exog = Z is not None
    if have_exog:
        Z = np.atleast_2d(Z).astype(float)
        k = Z.shape[0]
        if B is None:
            B = np.zeros((n, k))
        if D is None:
            D = np.zeros((m, k))
        B = np.atleast_2d(B)
        D = np.atleast_2d(D)

    CP0 = np.linalg.cholesky(P0)
    CQ = np.linalg.cholesky(Q)
    CR = np.linalg.cholesky(R)

    X = np.zeros((n, T + 1))
    Y = np.zeros((m, T))

    X[:, 0] = a0 + CP0 @ rng.standard_normal(n)
    for t in range(1, T + 1):
        w = CQ @ rng.standard_normal(n)
        v = CR @ rng.standard_normal(m)
        if have_exog:
            X[:, t] = F @ X[:, t - 1] + B @ Z[:, t - 1] + w
            Y[:, t - 1] = H @ X[:, t] + D @ Z[:, t - 1] + v
        else:
            X[:, t] = F @ X[:, t - 1] + w
            Y[:, t - 1] = H @ X[:, t] + v

    return Y, X
