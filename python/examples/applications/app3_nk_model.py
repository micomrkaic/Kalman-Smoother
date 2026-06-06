"""
Application 3: Semi-structural 3-equation New Keynesian model.

Backward-looking reduced form used as the empirical workhorse for
semi-structural projections at central banks (Berg-Karam-Laxton 2006).
Three equations link the output gap, inflation, and policy rate.

The smoother decomposes observed series into demand, cost-push, and
monetary shocks; EM recovers the structural slopes (kappa, the IS
sensitivity, the Taylor coefficients) from raw data.

Pedagogical points exercised:
  (1) Multi-equation simultaneous system.
  (2) State = (ytilde, pi, i); structural shocks read off as innovations.
  (3) Smoothed shock decomposition (the central-bank narrative tool).
  (4) EM with B,D estimation recovers structural slopes.
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
# --- Allow `import simksmoother` without a pip install ---
import sys as _sys
from pathlib import Path as _Path
import importlib.util as _ilu
if _ilu.find_spec('simksmoother') is None:
    _here = _Path(__file__).resolve().parent
    _python_dir = _here.parent.parent  # examples/applications/ -> python/
    if (_python_dir / 'simksmoother' / '__init__.py').is_file():
        _sys.path.insert(0, str(_python_dir))
# --- end path fix ---
import simksmoother as sks
# --- matplotlib setup: show figures live when a display is available -------
import os as _os
import matplotlib as _mpl
def _have_display():
    return (_sys.platform.startswith("win") or _sys.platform == "darwin"
            or bool(_os.environ.get("DISPLAY")))
_DISPLAY = _have_display()
if not _DISPLAY:
    _mpl.use("Agg")
import matplotlib.pyplot as plt
if _DISPLAY:
    try: plt.ion()
    except Exception: pass

def _save_and_show(fig, name):
    """Save under ../figures/{name}.png and display if a GUI is available."""
    d = _Path(__file__).resolve().parent.parent / "figures"
    d.mkdir(exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if _DISPLAY:
        plt.show(block=False)
        try: plt.pause(0.05)
        except Exception: pass
    else:
        plt.close(fig)
    print(f"Figure saved: {path}")



def main(seed=21):
    rng = np.random.default_rng(seed)
    T = 200

    # True parameters (quarterly).
    alpha_1, alpha_2 = 0.70, -0.10
    beta_1, beta_2 = 0.55, 0.15
    rho_i, phi_pi, phi_y = 0.80, 1.50, 0.50
    r_bar, pi_bar = 0.5, 0.5
    sigma_y, sigma_pi, sigma_i = 0.50, 0.40, 0.30
    sigma_meas_y, sigma_meas_pi, sigma_meas_i = 0.20, 0.05, 0.05

    n = 3
    F = np.zeros((n, n))
    F[0, 0] = alpha_1; F[0, 1] = -alpha_2; F[0, 2] = alpha_2
    F[1, 0] = beta_2;  F[1, 1] = beta_1
    F[2, 0] = (1 - rho_i) * phi_y
    F[2, 1] = (1 - rho_i) * phi_pi
    F[2, 2] = rho_i

    B = np.zeros((n, 1))
    B[0, 0] = -alpha_2 * r_bar
    B[1, 0] = (1 - beta_1) * pi_bar
    B[2, 0] = (1 - rho_i) * (r_bar + pi_bar - phi_pi * pi_bar)
    Z = np.ones((1, T))

    Q = np.diag([sigma_y**2, sigma_pi**2, sigma_i**2])

    m = 3
    H = np.eye(m, n)
    D = np.zeros((m, 1))
    R = np.diag([sigma_meas_y**2, sigma_meas_pi**2, sigma_meas_i**2])

    a0 = np.array([0.0, pi_bar, r_bar + pi_bar])
    P0 = np.diag([1.0, 0.5, 0.5])

    Y, Xtrue = sks.simulate_const(T, F, H, Q, R, a0, P0, Z=Z, B=B, D=D, rng=rng)
    ytilde_true = Xtrue[0, 1:]
    pi_true = Xtrue[1, 1:]
    i_true = Xtrue[2, 1:]

    # True structural shocks.
    W_true = np.zeros((n, T))
    for t in range(T):
        W_true[:, t] = Xtrue[:, t + 1] - F @ Xtrue[:, t] - B[:, 0]

    # Estimator A: Gaussian smoother with true parameters.
    X_a, J, h, info_a = sks.smooth_const(Y, F, H, Q, R, a0, P0, Z=Z, B=B, D=D)

    W_smooth = np.zeros((n, T))
    for t in range(T):
        W_smooth[:, t] = X_a[:, t + 1] - F @ X_a[:, t] - B[:, 0]

    # Estimator B: EM
    em_kwargs = dict(
        Z=Z,
        F=0.7 * np.eye(n) + 0.05 * rng.standard_normal((n, n)),
        H=H,
        Q=0.5 * np.eye(n),
        R=np.diag([0.1, 0.01, 0.01]),
        B=np.zeros((n, 1)),
        D=np.zeros((m, 1)),
        a0=a0.copy(),
        P0=P0.copy(),
        max_iter=120,
        tol=1e-7,
    )
    params_b, X_b, hist_b = sks.em_const(Y, n=n, **em_kwargs)

    F_est = params_b["F"]
    alpha_1_est = F_est[0, 0]
    alpha_2_est = F_est[0, 2]
    beta_1_est = F_est[1, 1]
    beta_2_est = F_est[1, 0]
    rho_i_est = F_est[2, 2]
    if abs(1 - rho_i_est) > 0.01:
        phi_y_est = F_est[2, 0] / (1 - rho_i_est)
        phi_pi_est = F_est[2, 1] / (1 - rho_i_est)
    else:
        phi_y_est = phi_pi_est = float("nan")

    print("Application 3: Semi-structural 3-equation NK model")
    print("===================================================")
    print(f"T = {T} quarters (~{T // 4} years)")
    print()
    print("True structural parameters:")
    print(f"  IS curve:    alpha_1 = {alpha_1:.2f}  alpha_2 = {alpha_2:.2f}")
    print(f"  Phillips:    beta_1  = {beta_1:.2f}  beta_2  = {beta_2:.2f}")
    print(f"  Taylor:      rho_i   = {rho_i:.2f}  phi_pi  = {phi_pi:.2f}  phi_y = {phi_y:.2f}")

    print()
    print("--- Estimator A: Gaussian smoother, true parameters ---")
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b)**2)))
    print(f"  RMSE on ytilde:  {rmse(X_a[0, 1:], ytilde_true):.4f}  (signal sd = {float(ytilde_true.std()):.4f})")
    print(f"  RMSE on pi:      {rmse(X_a[1, 1:], pi_true):.4f}")
    print(f"  RMSE on i:       {rmse(X_a[2, 1:], i_true):.4f}")

    rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print()
    print("  Correlation of smoothed vs true structural shocks:")
    print(f"    Demand    (eps^y) : {rho(W_smooth[0], W_true[0]):.3f}")
    print(f"    Cost-push (eps^pi): {rho(W_smooth[1], W_true[1]):.3f}")
    print(f"    Monetary  (eps^i) : {rho(W_smooth[2], W_true[2]):.3f}")

    print()
    print("--- Estimator B: EM, all structural parameters unknown ---")
    print(f"  EM iterations: {len(hist_b)}")
    print(f"  Final marginal log-likelihood: {hist_b[-1]:.4f}")
    monotone = all(hist_b[i] >= hist_b[i - 1] - 1e-6
                   for i in range(1, len(hist_b)))
    print(f"  Monotone non-decreasing: {monotone}")
    print()
    print("  Parameter recovery (truth -> EM estimate):")
    print(f"    alpha_1  : {alpha_1:.3f} -> {alpha_1_est:.3f}")
    print(f"    alpha_2  : {alpha_2:.3f} -> {alpha_2_est:.3f}")
    print(f"    beta_1   : {beta_1:.3f} -> {beta_1_est:.3f}")
    print(f"    beta_2   : {beta_2:.3f} -> {beta_2_est:.3f}  (=kappa, the Phillips slope)")
    print(f"    rho_i    : {rho_i:.3f} -> {rho_i_est:.3f}")
    print(f"    phi_pi   : {phi_pi:.3f} -> {phi_pi_est:.3f}")
    print(f"    phi_y    : {phi_y:.3f} -> {phi_y_est:.3f}")

    print()
    print("Key takeaways:")
    print("* The smoother decomposes observed series into the three latent")
    print("  structural shocks (demand, cost-push, monetary).  This is")
    print("  exactly what semi-structural projections at central banks need.")
    print("* EM recovers the structural parameters from the data.  The")
    print("  Phillips slope beta_2 (= kappa) is the famously hard one;")
    print("  identification rests on the contemporaneous link from output")
    print("  gap to inflation.")
    print("* No bespoke 'DSGE estimation routine' is needed.  The state-space")
    print("  formulation + standard EM does all the work, on the same sparse")
    print("  J that the basic smoother used.")

    # --- Plot ----------------------------------------------------------
    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    # Left column: observables vs smoothed states
    obs_data = [(ytilde_true, X_a[0, 1:], "Output gap $\\tilde y_t$"),
                (pi_true,     X_a[1, 1:], "Inflation $\\pi_t$"),
                (i_true,      X_a[2, 1:], "Policy rate $i_t$")]
    for row, (true_path, sm_path, title) in enumerate(obs_data):
        ax = axes[row, 0]
        ax.plot(tt, true_path, "k-",  lw=1.2, label="true")
        ax.plot(tt, sm_path,   "r--", lw=1.0, label="smoothed")
        ax.set_title(title)
        ax.set_xlabel("quarter")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(loc="best", fontsize=9)

    # Right column: structural shocks decomposition (truth vs smoothed)
    shock_names = ["Demand shock $\\epsilon_t^y$",
                   "Cost-push shock $\\epsilon_t^\\pi$",
                   "Monetary shock $\\epsilon_t^i$"]
    for k in range(3):
        ax = axes[k, 1]
        ax.plot(tt, W_true[k, :],   "k-",  lw=1.0)
        ax.plot(tt, W_smooth[k, :], "r--", lw=1.0)
        ax.axhline(0, color="0.5", ls=":", lw=0.8)
        rho_k = rho(W_smooth[k], W_true[k])
        ax.set_title(f"{shock_names[k]}   (corr = {rho_k:.2f})")
        ax.set_xlabel("quarter")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_and_show(fig, "app3_nk_model")


if __name__ == "__main__":
    main()
