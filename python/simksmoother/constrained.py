"""
Inequality-constrained simultaneous smoother via interior-point QP.

Solves
    min  (1/2) X' J X - h' X
    s.t. A_ineq X <= b_ineq

with (J, h) the precision matrix and information vector of the underlying
Gaussian smoothing problem.  Standard log-barrier Newton method; the
Hessian is J plus a low-rank PSD update that preserves the block-tridiagonal
sparsity of J for typical macro constraints.

When constraints are inactive at the unconstrained optimum, returns the
ordinary Kalman smoother solution unchanged.
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

from .smooth import smooth_const
from .selected_inv import selected_inv


def _phase1_feasible(A_ineq, b_ineq, X0, max_iter=50):
    """Iteratively project to find a strictly feasible X."""
    X = X0.copy()
    for _ in range(max_iter):
        slack = b_ineq - A_ineq @ X
        v = float(np.min(slack))
        if v > 1e-10:
            return X, True
        # Worst row.
        kw = int(np.argmin(slack))
        a_k = np.asarray(A_ineq[kw, :].todense()).ravel() \
              if sp.issparse(A_ineq) else A_ineq[kw, :]
        rhs = -v + 1e-4
        denom = float(a_k @ a_k)
        if denom < 1e-14:
            return X, False
        X = X - (rhs / denom) * a_k
    slack = b_ineq - A_ineq @ X
    return X, bool(np.all(slack > 0))


def smooth_constrained(Y, F, H, Q, R, a0, P0, A_ineq, b_ineq, *,
                       return_cov=False,
                       Z=None, B=None, D=None,
                       mu0=1.0, mu_factor=0.5,
                       max_outer=40, max_newton=3,
                       tol_feas=1e-8, tol_step=1e-10,
                       verbose=False):
    """
    Constrained simultaneous smoother.

    Parameters
    ----------
    Y, F, H, Q, R, a0, P0 : as in smooth_const.
    A_ineq : sparse matrix or array, shape (c, N) where N = (T+1)*n.
    b_ineq : (c,) array.
    Z, B, D : optional exogenous inputs.
    Interior-point parameters (mu0, mu_factor, max_outer, max_newton,
        tol_feas, tol_step).

    Returns
    -------
    Xhat : (n, T+1) array.
    info : dict
        lambda : approximate KKT multipliers (c,).
        active : boolean mask of active constraints.
        iters_outer, iters_newton.
        max_violation, converged.
        (Ptt, Ptt1 if return_cov.)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    m, T = Y.shape
    n = np.atleast_2d(F).shape[0]
    N = (T + 1) * n

    A_ineq = sp.csr_matrix(A_ineq).astype(float)
    b_ineq = np.asarray(b_ineq, dtype=float).ravel()
    c = A_ineq.shape[0]
    if A_ineq.shape[1] != N:
        raise ValueError(f"A_ineq must have {N} columns")
    if b_ineq.shape[0] != c:
        raise ValueError("b_ineq must match A_ineq rows")

    # Inner Gaussian smoother for J, h.
    kwargs = dict()
    if Z is not None:
        kwargs.update(Z=Z, B=B, D=D)
    X_unc, J, h, info_inner = smooth_const(Y, F, H, Q, R, a0, P0, **kwargs)
    X_unc_vec = X_unc.T.reshape(-1)

    slack_unc = b_ineq - A_ineq @ X_unc_vec
    if np.all(slack_unc > tol_feas):
        # No constraints bind; return the Gaussian solution.
        info = dict(
            lambda_=np.zeros(c),
            iters_outer=0,
            iters_newton=0,
            active=np.zeros(c, dtype=bool),
            max_violation=0.0,
            converged=True,
        )
        if return_cov:
            Ptt, Ptt1 = selected_inv(J, n)
            info["Ptt"] = Ptt
            info["Ptt1"] = Ptt1
        return X_unc, info

    # Find a strictly feasible starting point.
    X = X_unc_vec.copy()
    slack = b_ineq - A_ineq @ X
    if np.any(slack <= 0):
        X, ok = _phase1_feasible(A_ineq, b_ineq, X)
        if not ok:
            X = X * 0.5
            slack = b_ineq - A_ineq @ X
            if np.any(slack <= 0):
                import warnings
                warnings.warn(
                    "smooth_constrained: could not find a strictly feasible starting point."
                )
        slack = b_ineq - A_ineq @ X

    mu_cur = mu0
    total_newton = 0
    converged = False

    for outer in range(1, max_outer + 1):
        for inner in range(1, max_newton + 1):
            slack = b_ineq - A_ineq @ X
            slack = np.maximum(slack, 1e-12)
            one_over_s = 1.0 / slack
            one_over_s2 = one_over_s * one_over_s

            # Gradient of (1/2) X'JX - h'X - mu * sum_i log(b_i - a_i'X):
            # the barrier contributes d/dX [-mu log(b - a'X)] = +mu a/(b - a'X),
            # so the barrier term enters with a PLUS sign.
            g = J @ X - h + mu_cur * (A_ineq.T @ one_over_s)

            # Hessian: J + mu * A^T diag(1/s^2) A.
            Dmat = sp.diags(one_over_s2)
            AtDA = (A_ineq.T @ Dmat @ A_ineq).tocsc()
            H_mat = (J + mu_cur * AtDA).tocsc()
            H_mat = (H_mat + H_mat.T) * 0.5  # enforce symmetry

            try:
                lu = spla.splu(H_mat)
                dX = -lu.solve(g)
            except RuntimeError:
                # Singular; jitter.
                lu = spla.splu(H_mat + 1e-10 * sp.eye(N))
                dX = -lu.solve(g)

            step = 1.0
            A_dX = A_ineq @ dX
            pos = A_dX > 0
            if np.any(pos):
                max_step = 0.99 * float(np.min(slack[pos] / A_dX[pos]))
                step = min(step, max_step)
            X = X + step * dX

            step_norm = step * float(np.max(np.abs(dX)))
            total_newton += 1

            if verbose:
                viol = float(max(0.0, np.max(A_ineq @ X - b_ineq)))
                print(f"  outer {outer:2d}  inner {inner}  mu={mu_cur:.2e}"
                      f"  |dX|={step_norm:.2e}  viol={viol:.2e}")

            if step_norm < tol_step:
                break

        viol = float(max(0.0, np.max(A_ineq @ X - b_ineq)))
        if mu_cur < 1e-9 and viol < tol_feas:
            converged = True
            break
        mu_cur *= mu_factor
        if mu_cur < 1e-12:
            mu_cur = 1e-12

    # Dual variables.
    slack = b_ineq - A_ineq @ X
    lam = np.maximum(0.0, mu_cur / np.maximum(slack, 1e-14))

    Xhat = X.reshape((T + 1, n)).T

    info = dict(
        lambda_=lam,
        iters_outer=outer,
        iters_newton=total_newton,
        active=(lam > 1e-6),
        max_violation=float(max(0.0, np.max(A_ineq @ X - b_ineq))),
        converged=converged,
    )

    if return_cov:
        Ptt, Ptt1 = selected_inv(J, n)
        info["Ptt"] = Ptt
        info["Ptt1"] = Ptt1

    return Xhat, info
