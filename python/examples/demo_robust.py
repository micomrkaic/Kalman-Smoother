"""
Demo: robust smoother on contaminated observations.
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

def main(seed=42):
    rng = np.random.default_rng(seed)
    n = 2; m = 1; T = 200
    F = np.array([[0.95, 0.05], [0.0, 0.7]])
    H = np.array([[1.0, 0.3]])
    Q = 0.05 * np.eye(2)
    R = np.array([[0.1]])
    a0 = np.zeros(2); P0 = np.eye(2)

    Y_clean, X_true = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    outlier_mask = rng.random(T) < 0.10
    Y = Y_clean.copy()
    Y[:, outlier_mask] += 2 * rng.standard_normal((1, int(outlier_mask.sum())))

    Xg = sks.smooth_const(Y, F, H, Q, R, a0, P0)[0]
    Xt, info_t = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "student-t", nu=4)
    Xh, info_h = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "huber", c=1.345)
    Xl, info_l = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "laplace")

    truth = X_true[0, 1:]
    rmse = lambda Xh: float(np.sqrt(np.mean((Xh[0, 1:] - truth)**2)))

    print("Robust simultaneous smoother demo")
    print("=================================")
    print(f"T = {T}, n = {n}, m = {m}")
    print(f"Outliers: {int(outlier_mask.sum())} / {T} ({100 * outlier_mask.mean():.0f}%)")
    print()
    print("Smoothing RMSE on state 1")
    print(f"  Gaussian            {rmse(Xg):.4f}")
    print(f'  Student-t (nu=4)    {rmse(Xt):.4f}  [{info_t["iters"]} iters, converged={info_t["converged"]}]')
    print(f'  Huber (c=1.345)     {rmse(Xh):.4f}  [{info_h["iters"]} iters, converged={info_h["converged"]}]')
    print(f'  Laplace             {rmse(Xl):.4f}  [{info_l["iters"]} iters, converged={info_l["converged"]}]')

    threshold = 0.5
    flagged = info_t["tau_v"] < threshold
    tp = int((flagged & outlier_mask).sum())
    print()
    print(f"Outlier flagging via Student-t weights (tau < {threshold}):")
    print(f'  True positives:  {tp} / {int(outlier_mask.sum())}')
    print(f'  Mean tau_v at true outliers:  {float(info_t["tau_v"][outlier_mask].mean()):.3f}')
    print(f'  Mean tau_v at clean obs:      {float(info_t["tau_v"][~outlier_mask].mean()):.3f}')

    # --- Plot ----------------------------------------------------------
    # Top (large): data with outliers highlighted, true state, Gaussian
    # and Student-t smoothed paths.
    # Bottom (small): IRLS weights tau_v from Student-t.
    tt = np.arange(1, T + 1)
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(tt[~outlier_mask], Y[0, ~outlier_mask], "o", ms=3,
             color="0.6", label="clean obs")
    ax1.plot(tt[outlier_mask],  Y[0,  outlier_mask], "o", ms=5,
             color="#e53322", mfc="#e53322", label="outlier")
    ax1.plot(tt, X_true[0, 1:], "k-",  lw=1.2, label="true x_1")
    ax1.plot(tt, Xg[0, 1:],     "b--", lw=1.2, label="Gaussian smoother")
    ax1.plot(tt, Xt[0, 1:],     "r--", lw=1.2, label="Student-t smoother")
    ax1.set_title("Robust smoother on contaminated data")
    ax1.set_ylabel("x_1"); ax1.grid(alpha=0.3); ax1.legend(loc="best", fontsize=9)

    ax2.plot(tt, info_t["tau_v"], "b-", lw=1.0, label="tau_t (Student-t)")
    ax2.plot(tt[outlier_mask], info_t["tau_v"][outlier_mask], "o",
             ms=5, color="#e53322", mfc="#e53322", label="tau at true outliers")
    ax2.axhline(0.5, color="k", ls="--", lw=0.8, label="flag threshold 0.5")
    ax2.set_title("IRLS weights tau_t: low values flag outliers")
    ax2.set_xlabel("t"); ax2.set_ylabel("tau_t")
    ax2.grid(alpha=0.3); ax2.legend(loc="best", fontsize=9)
    save_and_close(fig, "demo_robust")

if __name__ == "__main__":
    main()
