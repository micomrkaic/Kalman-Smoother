"""
Demo: proper Gaussian EM with monotone log-likelihood check.
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

def main(seed=6):
    rng = np.random.default_rng(seed)
    T = 300
    F_true = np.array([[0.75, 0.15], [0.0, 0.5]])
    H_true = np.array([[1.0, 0.3], [0.2, 1.0]])
    Q_true = 0.1 * np.eye(2)
    R_true = np.diag([0.11, 0.16])
    a0 = np.zeros(2); P0 = np.eye(2)

    Y, X = sks.simulate_const(T, F_true, H_true, Q_true, R_true, a0, P0, rng=rng)

    params, Xhat, hist = sks.em_const(Y, n=2, max_iter=80, tol=1e-7, verbose=False)
    monotone = all(hist[i + 1] >= hist[i] - 1e-8 for i in range(len(hist) - 1))

    print(f"EM: {len(hist)} iterations, final loglik = {hist[-1]:.4f}")
    print(f"Monotone non-decreasing: {bool(monotone)}")
    print()
    print("True F:")
    print(F_true)
    print("Estimated F (up to rotation of latent state):")
    print(params["F"])
    print()
    print("Compare INVARIANTS of F (eigenvalues), not entries.")
    print(f'eig(F)_true = {sorted(np.linalg.eigvals(F_true).real)}')
    eig_est = np.linalg.eigvals(params["F"])
    print(f'eig(F)_est  = {sorted([(e.real, e.imag) for e in eig_est])}')

    # --- Plot ----------------------------------------------------------
    # Top: marginal log-likelihood ascending across EM iterations.
    # Bottom: final smoothed state 1 vs truth.
    tt = np.arange(T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(np.arange(1, len(hist) + 1), hist, "b-o", ms=3, lw=1.2)
    axes[0].set_title("Marginal log-likelihood across EM iterations")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("log p(Y; theta)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(tt, X[0],    "k-",  lw=1.2, label="true x_1")
    axes[1].plot(tt, Xhat[0], "r--", lw=1.2, label="smoothed x_1 (final EM fit)")
    axes[1].plot(np.arange(1, T + 1), Y[0], "o", ms=3, color="0.6",
                 label="observation (y_1)")
    axes[1].set_title("Smoothed state at final EM fit (latent rotation expected)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("x_1")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_const_em")

if __name__ == "__main__":
    main()
