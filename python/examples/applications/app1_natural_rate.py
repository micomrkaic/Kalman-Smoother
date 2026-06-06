"""
Application 1: Natural rate of interest (HLW-style).

A simplified Holston-Laubach-Williams setup with latent trend output
y*, trend growth g, and an AR(2) output gap ytilde.  The natural rate
is tied to trend growth: r*_t = c g_t (Laubach-Williams identification).
The IS curve feeds the output gap with the lagged real-rate gap; the
Phillips curve identifies the output-gap effect on inflation.

Pedagogical points exercised:
  (1) Multi-equation state-space mapping with state augmentation.
  (2) Exogenous inputs: the observed policy rate enters through B.
  (3) Cross-equation restrictions: r* = c g links measurement and state.
  (4) EM identification (the "pile-up" problem on sigma_g).
  (5) Constrained smoothing: enforce r*_t >= 0 (ZLB-relevant).
  (6) Diffuse initialization for unit-root trend components.

This is the same model as app1_natural_rate.m in the Octave reference;
running both should give identical results given the same seed.
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
import scipy.sparse as sp
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


def main(seed=7):
    rng = np.random.default_rng(seed)

    # ---- True parameters (HLW-calibrated, quarterly) ----
    T = 200
    a1, a2 = 1.55, -0.6
    a_r = -0.08          # IS curve real-rate gap sensitivity
    c = 1.0              # r* = c * g
    b_pi = 0.6
    b_y = 0.10
    sigma_ys = 0.55
    sigma_g = 0.08       # larger sigma_g -> r* wanders negative in places
    sigma_yt = 0.40
    sigma_pi = 0.80

    # Observed exogenous: policy rate r_t (AR(1) around 2.5).
    r_mean, phi_r, sigma_r = 2.5, 0.92, 0.40
    r = np.zeros(T + 2)
    r[0] = r_mean; r[1] = r_mean
    for t in range(2, T + 2):
        r[t] = r_mean + phi_r * (r[t - 1] - r_mean) + sigma_r * rng.standard_normal()

    # ---- State-space matrices ----
    # x_t = (y*_t, g_t, g_{t-1}, ytilde_t, ytilde_{t-1})'
    n = 5
    # In x_{t-1} = (y*_{t-1}, g_{t-1}, g_{t-2}, ytilde_{t-1}, ytilde_{t-2})',
    # g_{t-1} is component 2 (index 1) and g_{t-2} is component 3 (index 2).
    F = np.zeros((n, n))
    F[0, 0] = 1; F[0, 1] = 1            # y*_t = y*_{t-1} + g_{t-1}
    F[1, 1] = 1                          # g_t  = g_{t-1}
    F[2, 1] = 1                          # carries g_{t-1} into next state
    # IS curve (HLW): ytilde_t = a1 ytilde_{t-1} + a2 ytilde_{t-2}
    #   + (a_r/2) [(r_{t-1} - r*_{t-1}) + (r_{t-2} - r*_{t-2})] + eps,
    # with r*_t = c g_t.  The policy-rate average enters through B z_t;
    # the natural-rate average loads -a_r c / 2 on BOTH g_{t-1} and g_{t-2}.
    F[3, 3] = a1; F[3, 4] = a2
    F[3, 1] = -0.5 * a_r * c
    F[3, 2] = -0.5 * a_r * c
    F[4, 3] = 1                          # ytilde_{t-1} (lag)

    B = np.zeros((n, 1))
    B[3, 0] = a_r                        # IS curve loading on average real rate

    m = 2
    H = np.zeros((m, n))
    H[0, 0] = 1; H[0, 3] = 1             # y_t = y*_t + ytilde_t
    H[1, 4] = b_y                        # pi_t (reduced) = b_y * ytilde_{t-1}
    D = np.zeros((m, 1))

    Q = np.diag([sigma_ys**2, sigma_g**2, 1e-10, sigma_yt**2, 1e-10])
    R = np.diag([0.001, sigma_pi**2])

    a0_x = np.array([100.0, 0.5, 0.5, 0.0, 0.0])
    P0_x = np.diag([100.0, 1.0, 1.0, 5.0, 5.0])

    # Exogenous: average of last two policy rates.
    Z = np.zeros((1, T))
    for t in range(T):
        Z[0, t] = 0.5 * (r[t + 1] + r[t])

    # ---- Simulate ----
    Y, Xtrue = sks.simulate_const(T, F, H, Q, R, a0_x, P0_x,
                                    Z=Z, B=B, D=D, rng=rng)
    # NOTE: we use Y exactly as simulated.  Earlier versions of this
    # script post-processed Y[1, :] with an AR(1) on inflation to make
    # it look more empirically realistic, but that created a mismatch
    # between the data-generating process and the smoother's model and
    # produced a systematic bias on y*.  The pedagogical point that
    # the inflation row identifies the Phillips slope b_y is fully made
    # by the H[1, 4] = b_y entry in the measurement equation as-is.

    ystar_true = Xtrue[0, 1:]
    g_true = Xtrue[1, 1:]
    rstar_true = c * g_true

    # ---- Estimator A: Gaussian smoother, true parameters ----
    X_a, J, h, info_a = sks.smooth_const(Y, F, H, Q, R, a0_x, P0_x,
                                           Z=Z, B=B, D=D, return_cov=True)
    g_a = X_a[1, 1:]
    rstar_a = c * g_a
    sd_rstar_a = np.array([c * np.sqrt(info_a["Ptt"][t + 1][1, 1]) for t in range(T)])

    # ---- Estimator B: constrained smoother, r* >= 0 ----
    # Each row picks out -c * g_t at one time step.
    rows = np.arange(T + 1)
    cols = np.array([t * n + 1 for t in range(T + 1)])      # g is index 1
    data = -c * np.ones(T + 1)
    N = (T + 1) * n
    A_ineq = sp.csr_matrix((data, (rows, cols)), shape=(T + 1, N))
    b_ineq = np.zeros(T + 1)

    X_b, info_b = sks.smooth_constrained(
        Y, F, H, Q, R, a0_x, P0_x, A_ineq, b_ineq,
        Z=Z, B=B, D=D,
        max_outer=80, max_newton=5, tol_feas=1e-9,
    )
    g_b = X_b[1, 1:]
    rstar_b = c * g_b

    # ---- Estimator C: EM with all parameters unknown ----
    em_kwargs = dict(
        Z=Z,
        F=F + 0.01 * rng.standard_normal((n, n)),
        H=H,
        Q=np.diag([0.3, 0.1, 1e-8, 0.3, 1e-8]),
        R=np.eye(m) * 0.5,
        B=np.zeros((n, 1)),
        D=np.zeros((m, 1)),
        a0=a0_x.copy(),
        P0=P0_x.copy(),
        max_iter=100,
        tol=1e-6,
    )
    try:
        params_c, X_c, hist_c = sks.em_const(Y, n=n, **em_kwargs)
        em_converged = True
    except Exception as err:
        print(f"  EM failed: {err}")
        X_c = X_a
        hist_c = [info_a["nnz"]]
        em_converged = False

    # ---- Report ----
    print("Application 1: Natural rate of interest (HLW-style)")
    print("====================================================")
    print(f"T = {T} quarters (~{T/4:.1f} years)")
    print(f"True a_r (IS curve real-rate sensitivity) = {a_r:.3f}")
    print(f"True c   (r* = c * g, identification map) = {c:.2f}")
    print(f"True sigma_g (trend growth innov sd)      = {sigma_g:.3f}")

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b)**2)))

    print()
    print("--- Estimator A: Gaussian smoother, true parameters ---")
    print(f"  RMSE on y*       = {rmse(X_a[0, 1:], ystar_true):.4f}")
    print(f"  RMSE on g        = {rmse(g_a, g_true):.4f}")
    print(f"  RMSE on r* = c g = {rmse(rstar_a, rstar_true):.4f}")
    print(f"  Posterior sd of r* at end of sample = {float(sd_rstar_a[-1]):.3f}")
    print(f"  Avg posterior sd of r*              = {float(sd_rstar_a.mean()):.3f}")

    print()
    print("--- Estimator B: constrained smoother, r* >= 0 ---")
    neg_a = int((rstar_a < 0).sum())
    neg_b = int((rstar_b < -1e-8).sum())
    print(f"  Periods where unconstrained r* < 0:   {neg_a} / {T}")
    print(f"  Periods where constrained   r* < 0:   {neg_b} / {T}  (should be 0)")
    print(f"  Constraint converged: {info_b['converged']}   "
          f"max violation = {info_b['max_violation']:.2e}")
    print(f"  RMSE on r* (constrained)             = {rmse(rstar_b, rstar_true):.4f}")
    neg_true = int((rstar_true < 0).sum())
    if neg_true > 0:
        print(f"  NOTE: true r* < 0 for {neg_true} periods in this draw.")
        print(f"  Imposing r* >= 0 thus contradicts truth here and *hurts* RMSE.")
        print(f"  This is the right diagnostic: a constraint is only informative")
        print(f"  when it reflects the true data-generating process.")

    if em_converged:
        print()
        print("--- Estimator C: EM, all parameters unknown ---")
        print(f"  EM iterations: {len(hist_c)}")
        print(f"  Final marginal log-likelihood: {hist_c[-1]:.4f}")
        monotone = all(hist_c[i] >= hist_c[i - 1] - 1e-6
                       for i in range(1, len(hist_c)))
        print(f"  Monotone non-decreasing: {monotone}")
        ystar_em = X_c[0, 1:]
        rho = float(np.corrcoef(ystar_em, ystar_true)[0, 1])
        print(f"  Correlation(EM y*, true y*): {rho:.4f}")

    print()
    print("Key takeaways:")
    print("* The natural rate r* is identified through the cross-equation")
    print("  restriction r*_t = c g_t.  This pins down what would otherwise")
    print("  be one extra unidentified latent path.")
    print("* The constraint r* >= 0 (ZLB-relevant) is enforced via the")
    print("  interior-point QP on the SAME sparse J that the Gaussian")
    print("  smoother used.  No bespoke algorithm.")
    print("* Posterior bands on r* are large near the end of sample because")
    print("  the diffuse prior on g leaves the level of trend growth weakly")
    print("  identified.  This is the \"pile-up\" problem in disguise.")

    # --- Plot ----------------------------------------------------------
    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(tt, ystar_true,   "k-",  lw=1.4, label="true y*")
    axes[0, 0].plot(tt, X_a[0, 1:],   "r--", lw=1.2, label="estimator A")
    axes[0, 0].set_title("Trend output $y_t^*$")
    axes[0, 0].set_xlabel("quarter"); axes[0, 0].set_ylabel("log output")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(loc="best", fontsize=9)

    axes[0, 1].plot(tt, g_true, "k-",  lw=1.4, label="true g")
    axes[0, 1].plot(tt, g_a,    "r--", lw=1.2, label="estimator A")
    axes[0, 1].set_title("Trend growth $g_t$")
    axes[0, 1].set_xlabel("quarter"); axes[0, 1].set_ylabel("g")
    axes[0, 1].grid(alpha=0.3); axes[0, 1].legend(loc="best", fontsize=9)

    axes[1, 0].fill_between(tt, rstar_a - 1.96*sd_rstar_a, rstar_a + 1.96*sd_rstar_a,
                             color="#d8e2f3", label="95% band (A)")
    axes[1, 0].axhline(0, color="0.4", lw=0.8, label="zero")
    axes[1, 0].plot(tt, rstar_true, "k-",  lw=1.4, label="true r*")
    axes[1, 0].plot(tt, rstar_a,    "r--", lw=1.2, label="A: unconstrained")
    axes[1, 0].plot(tt, rstar_b,    "b-",  lw=1.5, label="B: r* >= 0 (constrained)")
    axes[1, 0].set_title(r"Natural rate $r_t^* = c \cdot g_t$")
    axes[1, 0].set_xlabel("quarter"); axes[1, 0].set_ylabel("r*")
    axes[1, 0].grid(alpha=0.3); axes[1, 0].legend(loc="best", fontsize=9)

    if em_converged and len(hist_c) > 1:
        axes[1, 1].plot(np.arange(1, len(hist_c) + 1), hist_c, "b-o",
                        lw=1.2, ms=3)
        axes[1, 1].set_title("EM marginal log-likelihood")
        axes[1, 1].set_xlabel("iteration")
        axes[1, 1].set_ylabel(r"$\log p(Y; \theta)$")
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "EM did not converge",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].axis("off")

    fig.tight_layout()
    _save_and_show(fig, "app1_natural_rate")


if __name__ == "__main__":
    main()
