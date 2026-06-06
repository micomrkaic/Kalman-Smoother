"""
Demo: HP filter as a special case of the simultaneous smoother.
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

def main(seed=5):
    rng = np.random.default_rng(seed)
    T = 200
    lam = 1600

    # Simulate a series: trend + cycle.
    trend = np.cumsum(0.5 + 0.005 * np.arange(T))
    cycle = 5 * np.sin(2 * np.pi * np.arange(T) / 40)
    y = trend + cycle + 2 * rng.standard_normal(T)

    # HP via simultaneous smoother.
    eps_lvl = 1e-10
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([eps_lvl, 1.0 / lam])
    H = np.array([[1.0, 0.0]])
    R = np.array([[1.0]])
    a0 = np.array([y[0], 0.0])
    P0 = 1e6 * np.eye(2)

    Y = y.reshape(1, T)
    Xhat, _, _, _ = sks.smooth_const(Y, F, H, Q, R, a0, P0)
    trend_sim = Xhat[0, 1:]

    # Closed-form HP solve.
    D2 = np.zeros((T - 2, T))
    for t in range(T - 2):
        D2[t, t] = 1
        D2[t, t + 1] = -2
        D2[t, t + 2] = 1
    trend_cf = np.linalg.solve(np.eye(T) + lam * D2.T @ D2, y)

    err = float(np.max(np.abs(trend_sim - trend_cf)))
    print("HP filter via simultaneous smoother vs classical closed form")
    print(f"lambda = {lam}")
    print(f"max |smoothed - classical| = {err:.3e}")
    print("(Discrepancy is purely from the boundary treatment of the")
    print(" near-diffuse prior; raising P0 toward infinity drives it to zero.)")

    # --- Plot ----------------------------------------------------------
    tt = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(tt, y, "o-", color="0.6", ms=3, mfc="0.85",
            label="observed y_t")
    ax.plot(tt, trend,    "k-",  lw=1.5, label="true trend")
    ax.plot(tt, trend_sim, "b--", lw=1.5, label="HP trend (simultaneous)")
    ax.plot(tt, trend_cf,  "r:",  lw=1.5, label="HP trend (closed form)")
    ax.set_title(f"Hodrick--Prescott decomposition, lambda = {lam}")
    ax.set_xlabel("t"); ax.set_ylabel("value")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_hp_filter")

if __name__ == "__main__":
    main()
