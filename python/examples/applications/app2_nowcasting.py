"""
Application 2: Mixed-frequency dynamic factor model for nowcasting GDP.

A canonical central-bank/IMF country-desk problem.  Several monthly
indicators are driven by a single latent monthly factor; quarterly
GDP enters as a third-month aggregate.  We construct realistic
"jagged-edge" data with publication lags and nowcast the current
quarter's GDP as the quarter unfolds.

Pedagogical points exercised:
  (1) Factor-model state-space mapping (small state, many indicators).
  (2) Mixed-frequency: GDP loads on (1/3)(f_t + f_{t-1} + f_{t-2}).
  (3) Jagged-edge missing data: different release lags per indicator.
  (4) Sequential nowcasts as new data arrive within the current quarter.
  (5) Nowcast standard error from the smoother's posterior covariance.
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



def main(seed=33):
    rng = np.random.default_rng(seed)
    T_months = 120
    phi = 0.70
    sigma_e = 0.5

    K = 5
    lambda_mon = np.array([1.0, 0.8, 0.6, 1.2, 0.4])
    R_mon = np.diag(np.array([0.30, 0.50, 0.80, 0.40, 1.50])**2)

    lambda_gdp = 1.0
    R_gdp = 0.30**2

    # State = (f_t, f_{t-1}, f_{t-2}).
    n = 3
    F = np.zeros((n, n))
    F[0, 0] = phi
    F[1, 0] = 1
    F[2, 1] = 1

    Q = np.zeros((n, n))
    Q[0, 0] = sigma_e**2
    Q[1, 1] = 1e-10; Q[2, 2] = 1e-10

    m = K + 1
    H = np.zeros((m, n))
    H[:K, 0] = lambda_mon
    H[K, :] = lambda_gdp * np.array([1/3, 1/3, 1/3])

    R = np.zeros((m, m))
    R[:K, :K] = R_mon
    R[K, K] = R_gdp

    a0 = np.zeros(n)
    P0 = (sigma_e**2 / (1 - phi**2)) * np.eye(n)

    Y, Xtrue = sks.simulate_const(T_months, F, H, Q, R, a0, P0, rng=rng)
    f_true = Xtrue[0, 1:]

    # Apply jagged-edge release pattern.
    release_lags = [0, 1, 1, 0, 0]
    Y_observed = Y.copy()
    for k, lag in enumerate(release_lags):
        if lag > 0:
            Y_observed[k, T_months - lag:T_months] = np.nan

    # GDP missing except at end-of-quarter, and not yet for current month.
    for t in range(T_months):
        if (t + 1) % 3 != 0:
            Y_observed[K, t] = np.nan
    Y_observed[K, T_months - 1] = np.nan

    # ---- Full-sample factor recovery ----
    X_smooth, J, h, info = sks.smooth_const(Y_observed, F, H, Q, R, a0, P0,
                                              return_cov=True)
    f_smooth = X_smooth[0, 1:]

    # ---- Sequential nowcasts ----
    scenarios = [
        "after 1 month of quarter",
        "after 2 months of quarter",
        "after 3 months of quarter (end of quarter)",
    ]
    nowcasts = np.zeros(3)
    nowcast_sd = np.zeros(3)
    true_gdp = lambda_gdp * (f_true[T_months - 3] + f_true[T_months - 2]
                              + f_true[T_months - 1]) / 3

    for scen in range(3):
        Y_scen = Y_observed.copy()
        cutoff = T_months - 3 + scen + 1   # last fully observed month (1-indexed)
        for k in range(K):
            Y_scen[k, cutoff:T_months] = np.nan
        Xs, _, _, info_s = sks.smooth_const(Y_scen, F, H, Q, R, a0, P0,
                                              return_cov=True)

        # In Python's 0-indexed convention, Xs[:, t+1] is x_{t+1} (with t=0..T-1)
        # so monthly index t corresponds to column t+1.
        t1, t2, t3 = T_months - 3, T_months - 2, T_months - 1
        three = np.array([Xs[0, t1 + 1], Xs[0, t2 + 1], Xs[0, t3 + 1]])
        nowcasts[scen] = lambda_gdp * three.mean()

        # The augmented state at the end-of-quarter month t3 is
        # x_{t3} = (f_{t3}, f_{t3-1}, f_{t3-2})', so the full covariance
        # of the three monthly factor values within the quarter is the
        # SINGLE diagonal block Sigma_{t3,t3} of J^{-1}: its off-diagonal
        # ELEMENTS are the cross-month covariances (including the lag-2
        # term).  No off-diagonal time blocks are needed.
        Sigma_3 = info_s["Ptt"][t3 + 1]
        w = np.ones(3) / 3
        nowcast_sd[scen] = lambda_gdp * float(np.sqrt(w @ Sigma_3 @ w))

    # ---- Report ----
    print("Application 2: Nowcasting current-quarter GDP")
    print("==============================================")
    print(f"Monthly indicators K = {K}, factor AR(1) phi = {phi:.2f}")
    print(f"Quarterly GDP aggregation: (1/3)(f_t + f_{{t-1}} + f_{{t-2}})")
    print(f"T_months = {T_months} ({T_months/12:.1f} years)")
    print()
    n_nan = int(np.isnan(Y_observed).sum())
    print(f"Total NaN entries in Y_observed: {n_nan} / {Y_observed.size}")
    print(f"Jagged edge: indicator release lags = {release_lags}")

    print()
    print("--- Full-sample factor recovery ---")
    rmse = float(np.sqrt(np.mean((f_smooth - f_true)**2)))
    print(f"RMSE on monthly factor f_t = {rmse:.4f}")
    rho = float(np.corrcoef(f_smooth, f_true)[0, 1])
    print(f"Correlation(smoothed, true factor) = {rho:.4f}")

    print()
    print("--- Real-time GDP nowcast ---")
    print(f"True current-quarter GDP value = {true_gdp:.4f}")
    print()
    print("Nowcast as data accumulates within the quarter:")
    for scen, label in enumerate(scenarios):
        print(f"  {label:<48s}  nowcast = {nowcasts[scen]:+.4f}  (sd {nowcast_sd[scen]:.4f})")

    print()
    print("Nowcast error vs truth:")
    for scen, label in enumerate(scenarios):
        err = nowcasts[scen] - true_gdp
        z = abs(err) / nowcast_sd[scen]
        print(f"  {label:<48s}  err = {err:+.4f}  (z = {z:.2f})")

    print()
    print("Key takeaways:")
    print("* All 'missing' observations (jagged edge, mixed frequency) are")
    print("  literal NaN entries in Y.  The smoother handles them by")
    print("  dropping rows of H R^-1 H at each t.  No bespoke algorithm.")
    print("* The nowcast sd shrinks monotonically as more monthly data come in.")
    print("  This is the smoother's built-in measure of nowcast precision.")
    print("* The factor is identified up to sign and scale.  Here we fix the")
    print("  scale through GDP's loading; in practice one fixes lambda_1 = 1.")

    # --- Plot ----------------------------------------------------------
    mm = np.arange(1, T_months + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(mm, f_true,   "k-",  lw=1.2, label="true factor")
    axes[0, 0].plot(mm, f_smooth, "r--", lw=1.2, label="smoothed factor")
    axes[0, 0].set_title("Latent monthly factor $f_t$")
    axes[0, 0].set_xlabel("month"); axes[0, 0].set_ylabel("f")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(loc="best", fontsize=9)

    # Indicator with the largest release lag.
    k_lag = int(np.argmax(release_lags))
    obs_idx = ~np.isnan(Y_observed[k_lag, :])
    axes[0, 1].plot(mm[obs_idx], Y_observed[k_lag, obs_idx], "o",
                    color="0.4", ms=4)
    nan_idx = np.isnan(Y_observed[k_lag, :])
    if nan_idx.any():
        first_nan = mm[np.where(nan_idx)[0][0]]
        axes[0, 1].axvspan(first_nan, T_months,
                            facecolor="#f5f0d8", alpha=0.7)
    axes[0, 1].set_title(f"Indicator {k_lag+1} (release lag = {release_lags[k_lag]} months)")
    axes[0, 1].set_xlabel("month"); axes[0, 1].set_ylabel("y")
    axes[0, 1].grid(alpha=0.3)

    xpts = np.arange(1, 4)
    for k in range(3):
        axes[1, 0].plot([xpts[k], xpts[k]],
                        [nowcasts[k] - 1.96*nowcast_sd[k],
                         nowcasts[k] + 1.96*nowcast_sd[k]],
                        "-", color="#2f5fd9", lw=1.5,
                        label="95% interval" if k == 0 else None)
    axes[1, 0].plot(xpts, nowcasts, "o", color="#2f5fd9", lw=1.5, ms=8,
                    mfc="#2f5fd9", label="nowcast")
    axes[1, 0].axhline(true_gdp, color="k", lw=1.4,
                        label="true current-quarter GDP")
    axes[1, 0].set_xticks(xpts)
    axes[1, 0].set_xticklabels(["after 1 mo", "after 2 mo", "end of qtr"])
    axes[1, 0].set_xlim(0.5, 3.5)
    axes[1, 0].set_title("GDP nowcast and 95% interval as the quarter progresses")
    axes[1, 0].set_ylabel("GDP value")
    axes[1, 0].grid(alpha=0.3); axes[1, 0].legend(loc="best", fontsize=9)

    axes[1, 1].plot(xpts, nowcast_sd, "o-", color="#e53322", lw=1.5,
                    ms=8, mfc="#e53322")
    axes[1, 1].set_xticks(xpts)
    axes[1, 1].set_xticklabels(["after 1 mo", "after 2 mo", "end of qtr"])
    axes[1, 1].set_xlim(0.5, 3.5)
    axes[1, 1].set_title("Nowcast posterior std shrinks as data accumulate")
    axes[1, 1].set_ylabel("posterior std")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    _save_and_show(fig, "app2_nowcasting")


if __name__ == "__main__":
    main()
