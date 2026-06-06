"""
Demo: smoothing with known parameters.
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
from _plot_helpers import plt, save_and_close
import simksmoother as sks

def main(seed=1):
    rng = np.random.default_rng(seed)

    n = 2
    T = 100
    F = np.array([[0.75, 0.15], [0.0, 0.5]])
    H = np.array([[1.0, 0.3], [0.2, 1.0]])
    Q = 0.1 * np.eye(2)
    R = 0.1 * np.eye(2)
    a0 = np.zeros(2)
    P0 = np.eye(2)

    Y, X = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)

    print("Known-parameter simultaneous smoother demo")
    print(f"State dimension n = {n}, observations T = {T}")
    print(f"Sparse precision size: {J.shape[0]} x {J.shape[1]}, nnz = {info['nnz']}")

    ll, _ = sks.marginal_loglik(Y, F, H, Q, R, a0, P0)
    print(f"Marginal log-likelihood = {ll:.4f}")

    rmse1 = float(np.sqrt(np.mean((Xhat[0, 1:] - X[0, 1:])**2)))
    rmse2 = float(np.sqrt(np.mean((Xhat[1, 1:] - X[1, 1:])**2)))
    print(f"RMSE state 1 = {rmse1:.4f}")
    print(f"RMSE state 2 = {rmse2:.4f}")

    # Empirical 95% coverage on state 1
    sd = np.array([np.sqrt(info["Ptt"][t][0, 0]) for t in range(1, T + 1)])
    z = (Xhat[0, 1:] - X[0, 1:]) / sd
    coverage = float(np.mean(np.abs(z) < 1.96))
    print(f"Empirical 95% coverage for state 1 = {coverage:.3f}")

    # --- Plot ----------------------------------------------------------
    # State trajectories with 95% posterior bands, plus observation overlay
    # on state 1 (it's the linearly-mostly-loaded one in this H).
    tt = np.arange(T + 1)
    sd1 = np.array([np.sqrt(info["Ptt"][t][0, 0]) for t in range(T + 1)])
    sd2 = np.array([np.sqrt(info["Ptt"][t][1, 1]) for t in range(T + 1)])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, idx, sd in [(axes[0], 0, sd1), (axes[1], 1, sd2)]:
        ax.fill_between(tt, Xhat[idx] - 1.96 * sd, Xhat[idx] + 1.96 * sd,
                        color="#d8e2f3", label="95% band")
        ax.plot(tt, X[idx], "k-", lw=1.2, label="true")
        ax.plot(tt, Xhat[idx], "r--", lw=1.2, label="smoothed")
        ax.set_ylabel(f"x_{idx+1}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    # Overlay observation on state 1 panel (using y_1 since H has 1.0 on x_1).
    axes[0].plot(np.arange(1, T + 1), Y[0], "o", ms=3,
                 color="0.6", label="obs (y_1)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("Smoothed states with 95% posterior bands")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    save_and_close(fig, "demo_const_known")

if __name__ == "__main__":
    main()
