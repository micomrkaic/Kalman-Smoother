"""
Demo: structural-break detection via state-side robust prior.
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

def main(seed=7):
    rng = np.random.default_rng(seed)
    T = 200
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.0025]])
    R = np.array([[0.04]])
    a0 = np.array([0.0]); P0 = np.array([[1.0]])

    x = np.zeros(T + 1)
    x[0] = float(np.sqrt(P0[0, 0]) * rng.standard_normal())
    for t in range(T):
        x[t + 1] = x[t] + float(np.sqrt(Q[0, 0])) * rng.standard_normal()
    break_periods = [60, 110, 160]
    break_sizes = [-1.5, 2.0, -1.0]
    for bp, sz in zip(break_periods, break_sizes):
        x[bp + 1:] += sz

    Y = (x[1:] + float(np.sqrt(R[0, 0])) * rng.standard_normal(T)).reshape(1, T)
    truth = x[1:]

    Xg = sks.smooth_const(Y, F, H, Q, R, a0, P0)[0]
    Xr, info_r = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "student-t",
                                    nu=4, robust_state=True, max_iter=100)

    rmse_g = float(np.sqrt(np.mean((Xg[0, 1:] - truth)**2)))
    rmse_r = float(np.sqrt(np.mean((Xr[0, 1:] - truth)**2)))

    print("Robust state-side smoother on a series with breaks")
    print(f'T = {T}, breaks at t = {break_periods}')
    print(f'  Gaussian smoother           {rmse_g:.4f}')
    print(f'  Robust state (Student-t)    {rmse_r:.4f}   [{info_r["iters"]} iters]')
    print()
    print("State innovation weights tau_w at the break periods:")
    for bp, sz in zip(break_periods, break_sizes):
        print(f'  t = {bp:3d} (break size {sz:+.1f}) :  tau_w = {info_r["tau_w"][bp]:.4f}')
    outside = np.setdiff1d(np.arange(T),
                            np.concatenate([np.arange(bp - 3, bp + 4)
                                            for bp in break_periods]))
    print(f'Mean tau_w outside ±3 of any break: {float(info_r["tau_w"][outside].mean()):.3f}')

    # --- Plot ----------------------------------------------------------
    # Top: data, true level (with breaks visible), Gaussian and robust paths.
    # Bottom: state-innovation weights tau_w, with dots at the three breaks.
    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    axes[0].plot(tt, Y[0], "o", ms=3, color="0.6", label="observation")
    axes[0].plot(tt, truth,    "k-",  lw=1.2, label="true level")
    axes[0].plot(tt, Xg[0, 1:],"b--", lw=1.2, label="Gaussian smoother")
    axes[0].plot(tt, Xr[0, 1:],"r-",  lw=1.5, label="Student-t state")
    for bp in break_periods:
        axes[0].axvline(bp, ls=":", color="0.5", lw=0.8)
    axes[0].set_title("Robust smoothing through structural breaks")
    axes[0].set_ylabel("level"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].plot(tt, info_r["tau_w"], "r-", lw=1.0, label="xi_t (state weight)")
    for bp in break_periods:
        axes[1].plot(bp, info_r["tau_w"][bp], "o", ms=6,
                     color="#e53322", mfc="#e53322")
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8, label="threshold 0.5")
    axes[1].set_title("State innovation weights xi_t plunge at the three "
                      "injected breaks")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("xi_t")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_robust_breaks")

if __name__ == "__main__":
    main()
