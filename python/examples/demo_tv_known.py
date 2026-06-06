"""
Demo: time-varying parameters via cell-array interface.
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

def main(seed=3):
    rng = np.random.default_rng(seed)
    T = 80
    Fseq = []
    for t in range(T):
        phi = 0.5 + 0.3 * np.sin(2 * np.pi * t / T)
        Fseq.append(np.array([[phi, 0.1], [0.0, 0.6]]))
    H = np.array([[1.0, 0.4]])
    Q = 0.1 * np.eye(2)
    R = np.array([[0.1]])
    a0 = np.zeros(2)
    P0 = np.eye(2)

    # Simulate
    X = np.zeros((2, T + 1))
    Y = np.zeros((1, T))
    X[:, 0] = a0
    for t in range(T):
        X[:, t + 1] = Fseq[t] @ X[:, t] + np.sqrt(0.1) * rng.standard_normal(2)
        Y[:, t] = H @ X[:, t + 1] + np.sqrt(0.1) * rng.standard_normal(1)

    Xhat, J, h, info = sks.smooth_tv(Y, Fseq, H, Q, R, a0, P0)
    rmse1 = float(np.sqrt(np.mean((Xhat[0, 1:] - X[0, 1:])**2)))
    rmse2 = float(np.sqrt(np.mean((Xhat[1, 1:] - X[1, 1:])**2)))

    print("Time-varying simultaneous smoother demo")
    print(f"Sparse precision size: {J.shape[0]} x {J.shape[1]}, nnz = {info['nnz']}")
    print(f"RMSE state 1 = {rmse1:.4f}")
    print(f"RMSE state 2 = {rmse2:.4f}")

    # --- Plot ----------------------------------------------------------
    tt   = np.arange(T + 1)
    F11  = np.array([Ft[0, 0] for Ft in Fseq])
    F22  = np.array([Ft[1, 1] for Ft in Fseq])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(np.arange(1, T + 1), F11, "b-", lw=1.2, label="F_{11}(t)")
    axes[0].plot(np.arange(1, T + 1), F22, "r-", lw=1.2, label="F_{22}(t)")
    axes[0].set_title("Time-varying dynamics: diagonal entries of F_t")
    axes[0].set_ylabel("value"); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(tt, X[0], "k-",  lw=1.2, label="true x_1")
    axes[1].plot(tt, Xhat[0], "r--", lw=1.2, label="smoothed x_1")
    axes[1].plot(np.arange(1, T + 1), Y[0], "o", ms=3, color="0.6",
                 label="observation")
    axes[1].set_title("Smoothed vs true state under time-varying dynamics")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("x_1")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    save_and_close(fig, "demo_tv_known")

if __name__ == "__main__":
    main()
