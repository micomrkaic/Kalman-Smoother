"""
Flagship demo: HP filter with credit-cycle exogenous driver.

Combines HP-style trend extraction, an exogenous credit-gap driver,
Student-t state innovations for structural-break detection, and EM
estimation of (F, H, Q, R, B, D) on contaminated data.
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

def main(seed=11):
    rng = np.random.default_rng(seed)
    T = 200
    lam = 1600

    beta = 0.30
    gamma = 0.10
    eps_lvl = 1e-6
    sd_w = 1.0 / np.sqrt(lam)
    sd_v = 0.25

    F_true = np.array([[1.0, 1.0], [0.0, 1.0]])
    H_true = np.array([[1.0, 0.0]])
    Q_true = np.diag([eps_lvl, sd_w**2])
    R_true = np.array([[sd_v**2]])
    B_true = np.array([[0.0], [beta]])
    D_true = np.array([[gamma]])

    phi_z = 0.92
    sd_z = 0.6
    z = np.zeros(T)
    z[0] = sd_z * rng.standard_normal()
    for t in range(1, T):
        z[t] = phi_z * z[t - 1] + sd_z * rng.standard_normal()
    Z = z.reshape(1, T)

    Y_clean, X_true = sks.simulate_const(T, F_true, H_true, Q_true, R_true,
                                          np.array([0.0, 0.5]), 0.1 * np.eye(2),
                                          Z=Z, B=B_true, D=D_true, rng=rng)
    trend_true = X_true[0, 1:]

    # Inject a structural break at t = 80.
    break_t = 80
    slope_drop = -1.5
    X_true[1, break_t:] += slope_drop
    for t in range(break_t, T):
        X_true[0, t + 1] = X_true[0, t] + X_true[1, t]
    trend_true = X_true[0, 1:]
    Y = (X_true[0, 1:] + (D_true @ Z).ravel()
          + sd_v * rng.standard_normal(T)).reshape(1, T)

    outlier_mask = rng.random(T) < 0.10
    Y[:, outlier_mask] += 3 * rng.standard_normal((1, int(outlier_mask.sum())))

    P0_diff = 1e6 * np.eye(2)

    # Estimator A: classical HP, no exogenous, Gaussian
    X_a = sks.smooth_const(Y, F_true, H_true, Q_true, R_true,
                            np.array([0.0, 0.5]), P0_diff)[0]
    # Estimator B: Gaussian + exogenous
    X_b = sks.smooth_const(Y, F_true, H_true, Q_true, R_true,
                            np.array([0.0, 0.5]), P0_diff,
                            Z=Z, B=B_true, D=D_true)[0]
    # Estimator C: robust + exogenous
    X_c, info_c = sks.smooth_robust(Y, F_true, H_true, Q_true, R_true,
                                     np.array([0.0, 0.5]), P0_diff,
                                     "student-t", nu=4, robust_state=True,
                                     max_iter=100, Z=Z, B=B_true, D=D_true)
    # Estimator D: EM jointly estimates (F, H, Q, R, B, D)
    params_d, X_d, hist_d = sks.em_const(
        Y, n=2, max_iter=80, tol=1e-5,
        F=np.array([[1.0, 1.0], [0.0, 0.9]]),
        H=np.array([[1.0, 0.0]]),
        Q=np.diag([1e-6, 1e-3]),
        R=np.array([[1.0]]),
        a0=np.array([0.0, 0.0]),
        P0=100 * np.eye(2),
        Z=Z, B=np.zeros((2, 1)), D=np.zeros((1, 1)),
    )

    rmse = lambda X: float(np.sqrt(np.mean((X[0, 1:] - trend_true)**2)))

    print("Flagship demo: HP + credit cycle + structural break + outliers")
    print("================================================================")
    print(f"T = {T}, lambda = {lam}, break at t = {break_t} (slope drops by {slope_drop:+.1f})")
    print(f"True beta (credit -> slope) = {beta:.2f}")
    print(f"True gamma (credit -> output) = {gamma:.2f}")
    print(f"Outlier contamination rate = {100 * outlier_mask.mean():.0f}%")
    print()
    print("RMSE on the latent trend:")
    print(f"  (A) Gaussian, no exogenous           {rmse(X_a):.4f}")
    print(f"  (B) Gaussian, true exogenous         {rmse(X_b):.4f}")
    print(f"  (C) Student-t robust + true exog     {rmse(X_c):.4f}")
    print(f"  (D) EM (estimated params) + exog     {rmse(X_d):.4f}")
    print()
    print(f"EM convergence: {len(hist_d)} iterations, final loglik = {hist_d[-1]:.4f}")
    print(f"EM-estimated B (truth = [0; {beta:.2f}]):")
    print(params_d["B"])
    print(f"EM-estimated D (truth = {gamma:.2f}): {float(params_d['D'][0, 0]):.4f}")

    print()
    print("Structural-break detection by robust smoother (Student-t state):")
    print(f'  tau_w at break t = {break_t}:  {float(info_c["tau_w"][break_t]):.4f}')
    outside = np.setdiff1d(np.arange(T), np.arange(break_t - 3, break_t + 4))
    print(f'  mean tau_w outside ±3 of break:  {float(info_c["tau_w"][outside].mean()):.4f}')

    n_caught = int((info_c["tau_v"][outlier_mask] < 0.5).sum())
    n_out = int(outlier_mask.sum())
    print()
    print("Outlier detection (tau_v < 0.5):")
    print(f'  Outliers correctly flagged: {n_caught} / {n_out} ({100 * n_caught / max(n_out, 1):.0f}%)')
    print(f'  Mean tau_v at outliers:  {float(info_c["tau_v"][outlier_mask].mean()):.3f}')
    print(f'  Mean tau_v at clean obs: {float(info_c["tau_v"][~outlier_mask].mean()):.3f}')

    # --- Plot ----------------------------------------------------------
    # 3 panels: (a) data and all four trend estimators,
    #           (b) the credit-gap exogenous driver z_t,
    #           (c) EM log-likelihood trace + IRLS state weights, rescaled
    #               to a common [0,1] axis.
    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(tt, Y[0],         "o", ms=3, color="0.6", label="observation")
    axes[0].plot(tt, trend_true,   "k-",  lw=1.5, label="true trend")
    axes[0].plot(tt, X_a[0, 1:],   ":",   color="0.4", lw=1.2, label="A: HP no exog")
    axes[0].plot(tt, X_b[0, 1:],   "--",  color="#2980d9", lw=1.2, label="B: HP + exog")
    axes[0].plot(tt, X_c[0, 1:],   "-",   color="#e53322", lw=1.5, label="C: robust + exog")
    axes[0].plot(tt, X_d[0, 1:],   "-.",  color="#1b964c", lw=1.5, label="D: EM + exog")
    axes[0].axvline(break_t, ls=":", color="k")
    axes[0].set_title(f"HP + credit cycle: structural break at t={break_t}, "
                      f"{100 * outlier_mask.mean():.0f}% outliers")
    axes[0].set_ylabel("output level"); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(tt, z, "-", color="#2980d9", lw=1.2)
    axes[1].axhline(0, color="k", ls=":", lw=0.8)
    axes[1].set_title("Exogenous driver: credit gap z_t")
    axes[1].set_ylabel("z_t"); axes[1].grid(alpha=0.3)

    # Rescale EM log-likelihood to [0,1] so we can overlay it on the same axis
    # as the IRLS state weights for a compact view of both convergence and
    # break detection.
    ll = np.asarray(hist_d, dtype=float)
    ll_norm = (ll - ll.min()) / max(1e-12, ll.max() - ll.min())
    ll_t = np.linspace(1, T, len(ll))
    axes[2].plot(ll_t, ll_norm, "b-o", ms=3, lw=1.0,
                 label="EM log-lik (rescaled to [0,1])")
    axes[2].plot(tt, info_c["tau_w"], "-", color="#e53322", lw=1.0,
                 label="xi_t (state weight)")
    axes[2].axvline(break_t, ls=":", color="k", label=f"break at t={break_t}")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("EM convergence trace and IRLS state weights "
                      "(overlaid, rescaled)")
    axes[2].set_xlabel("t (or iteration, for the EM trace)")
    axes[2].grid(alpha=0.3); axes[2].legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_hp_credit")

if __name__ == "__main__":
    main()
