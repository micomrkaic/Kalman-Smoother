"""
Demo: simultaneous smoother vs textbook RTS, agreement to machine precision.
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

def main(seed=2):
    rng = np.random.default_rng(seed)

    n = 3
    T = 80
    F = np.array([[0.8, 0.1, 0.0],
                  [0.0, 0.7, 0.2],
                  [0.0, 0.0, 0.5]])
    H = np.array([[1.0, 0.2, 0.3],
                  [0.4, 1.0, 0.1]])
    Q = 0.1 * np.eye(3)
    R = 0.05 * np.eye(2)
    a0 = np.zeros(3)
    P0 = np.eye(3)

    Y, X = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)
    XsT, PsT = sks.rts_smoother(Y, F, H, Q, R, a0, P0)

    mean_err = float(np.max(np.abs(Xhat - XsT)))
    cov_err = max(float(np.max(np.abs(info["Ptt"][t] - PsT[t]))) for t in range(T + 1))

    print("Simultaneous vs RTS reference")
    print(f"  max |Xhat - XsT| over all t  = {mean_err:.2e}")
    print(f"  max |P_tT,sim - P_tT,rts|    = {cov_err:.2e}")
    print(f'  (Both should be ~1e-15, since J X = h is just a different '
          f'sparse factorization of the same block-tridiagonal system.)')

    # --- Plot ----------------------------------------------------------
    # Top: the two paths overlaid (visually identical).
    # Bottom: the pointwise difference (machine-precision noise).
    tt = np.arange(T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(tt, Xhat[0], "b-",  lw=1.5, label="simultaneous (J\\h)")
    axes[0].plot(tt, XsT [0], "r--", lw=1.5, label="recursive (Kalman+RTS)")
    axes[0].set_title("Smoothed state 1: two methods overlaid")
    axes[0].set_ylabel("x_1"); axes[0].grid(alpha=0.3); axes[0].legend()

    diff = Xhat[0] - XsT[0]
    axes[1].plot(tt, diff, "k-", lw=1.0)
    axes[1].set_title(f"Pointwise difference, state 1 "
                      f"(max abs = {np.max(np.abs(diff)):.2e})")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Xsim - Xrts")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_and_close(fig, "demo_compare_recursive")

if __name__ == "__main__":
    main()
