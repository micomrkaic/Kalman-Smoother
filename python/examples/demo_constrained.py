"""
Demo: inequality-constrained smoother enforcing x_t >= 0 (NAIRU-style).
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
from _plot_helpers import plt, save_and_close
import simksmoother as sks

def main(seed=31):
    rng = np.random.default_rng(seed)
    T = 120
    F = np.array([[0.9]])
    H = np.array([[1.0]])
    Q = np.array([[0.05]])
    R = np.array([[0.15]])
    a0 = np.array([0.3]); P0 = np.array([[1.0]])

    _, X_true = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    true_trend = np.maximum(0.0, X_true[0, 1:])
    Y = (true_trend + float(np.sqrt(R[0, 0])) * rng.standard_normal(T)).reshape(1, T)

    X_unc = sks.smooth_const(Y, F, H, Q, R, a0, P0)[0]

    N = T + 1
    A_ineq = -sp.eye(N).tocsr()
    b_ineq = np.zeros(N)
    X_con, info_con = sks.smooth_constrained(Y, F, H, Q, R, a0, P0,
                                              A_ineq, b_ineq, verbose=False)

    print("Constrained smoother demo: x_t >= 0 constraint")
    print("===============================================")
    print(f"T = {T}, n = 1, c = {N} constraints (-x_t <= 0 for all t)")
    print()
    print("Unconstrained smoother:")
    print(f"  Number of periods where x_t < 0:      {int((X_unc[0, 1:] < -1e-6).sum())} / {T}")
    print(f"  Minimum smoothed value:               {float(X_unc[0, 1:].min()):.4f}")
    print()
    print("Constrained smoother:")
    print(f'  Newton iterations (total):            {info_con["iters_newton"]}')
    print(f'  Outer barrier reductions:             {info_con["iters_outer"]}')
    print(f'  Converged:                            {info_con["converged"]}')
    print(f'  Max constraint violation:             {info_con["max_violation"]:.2e}')
    print(f'  Number of active constraints:         {int(info_con["active"].sum())}')
    print(f'  Minimum smoothed value:               {float(X_con[0, 1:].min()):.4f}')
    print()
    rmse_u = float(np.sqrt(np.mean((X_unc[0, 1:] - true_trend)**2)))
    rmse_c = float(np.sqrt(np.mean((X_con[0, 1:] - true_trend)**2)))
    print(f"RMSE vs true (non-negative) trend:")
    print(f"  Unconstrained:  {rmse_u:.4f}")
    print(f"  Constrained:    {rmse_c:.4f}")

    # --- Plot ----------------------------------------------------------
    tt = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.axhline(0, color="0.4", lw=0.8, label="zero")
    ax.plot(tt, Y[0],         "o", ms=3, color="0.6", label="noisy observation")
    ax.plot(tt, true_trend,    "k-",  lw=1.4, label="true (non-negative) trend")
    ax.plot(tt, X_unc[0, 1:],  "b--", lw=1.2, label="unconstrained smoother")
    ax.plot(tt, X_con[0, 1:],  "r-",  lw=1.6, label="constrained smoother")
    neg_idx = np.where(X_unc[0, 1:] < -1e-6)[0]
    if neg_idx.size:
        ax.plot(neg_idx + 1, X_unc[0, neg_idx + 1], "o", ms=5,
                color="#2f5fd9", mfc="#2f5fd9", label="unconstrained < 0")
    ax.set_title("Constrained smoothing: enforcing x_t >= 0 via interior-point QP")
    ax.set_xlabel("t"); ax.set_ylabel("x_t")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_constrained")

if __name__ == "__main__":
    main()
