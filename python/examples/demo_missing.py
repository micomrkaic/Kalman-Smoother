"""
Demo: missing observations (NaN entries) and posterior variance in gaps.
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

def main(seed=4):
    rng = np.random.default_rng(seed)
    T = 150
    F = np.array([[0.85, 0.1], [0.0, 0.6]])
    H = np.array([[1.0, 0.3]])
    Q = 0.05 * np.eye(2)
    R = np.array([[0.05]])
    a0 = np.zeros(2)
    P0 = np.eye(2)

    Y, X = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    Y_miss = Y.copy()
    Y_miss[:, 60:80] = np.nan
    extra = rng.random(T) < 0.3
    Y_miss[:, extra] = np.nan

    Xhat, J, h, info = sks.smooth_const(Y_miss, F, H, Q, R, a0, P0,
                                          return_cov=True)
    rmse_full = float(np.sqrt(np.mean((Xhat[0, 1:] - X[0, 1:])**2)))
    in_gap = slice(60, 80)
    rmse_gap = float(np.sqrt(np.mean((Xhat[0, 1:][in_gap] - X[0, 1:][in_gap])**2)))
    var_in = float(np.mean([info["Ptt"][t][0, 0] for t in range(61, 81)]))
    var_out = float(np.mean([info["Ptt"][t][0, 0]
                              for t in range(1, T + 1)
                              if not (60 < t <= 80)]))

    n_nans = int(np.isnan(Y_miss).sum())
    print("Smoothing with missing data")
    print(f"T = {T}, total NaNs in Y = {n_nans} / {Y.size}")
    print(f"Sparse precision still SPD, nnz = {info['nnz']}")
    print(f"RMSE state 1 over full sample = {rmse_full:.4f}")
    print(f"RMSE state 1 inside the 60..80 gap = {rmse_gap:.4f}")
    print(f"Avg posterior variance, state 1, inside gap  = {var_in:.4f}")
    print(f"Avg posterior variance, state 1, outside gap = {var_out:.4f}")

    # --- Plot ----------------------------------------------------------
    # Single panel: state 1, contiguous gap shaded, posterior band wider
    # in the gap.
    tt = np.arange(T + 1)
    sd = np.array([np.sqrt(info["Ptt"][t][0, 0]) for t in range(T + 1)])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    yl = (float(X[0].min() - 1), float(X[0].max() + 1))
    ax.axvspan(60, 80, facecolor="#f5f0d8", alpha=0.7,
               label="missing gap (t=60..80)")
    ax.fill_between(tt, Xhat[0] - 1.96 * sd, Xhat[0] + 1.96 * sd,
                    color="#d8e2f3", label="95% band")
    obs_idx = np.where(~np.isnan(Y_miss[0]))[0]
    ax.plot(obs_idx + 1, Y_miss[0, obs_idx], "o", ms=3, color="0.4",
            label="observation")
    ax.plot(tt, X[0],    "k-",  lw=1.2, label="true x_1")
    ax.plot(tt, Xhat[0], "r--", lw=1.2, label="smoothed x_1")
    ax.set_title("Smoothing with missing data: bands widen where observations "
                 "are absent")
    ax.set_xlabel("t"); ax.set_ylabel("x_1")
    ax.set_ylim(*yl); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_missing")

if __name__ == "__main__":
    main()
