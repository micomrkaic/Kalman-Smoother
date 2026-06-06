"""
Demo: Switching Linear Dynamical System on a two-regime AR(1).
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

def _portable_draws(seed, T):
    """Park-Miller LCG + Box-Muller, implemented identically in
    demo_slds.py and demo_slds.m so that Python, Octave, and MATLAB
    simulate BIT-IDENTICAL data (the uniforms are exact in double
    precision; the normals agree to the last ulp of libm).  This makes
    the demo's printed numbers directly comparable across languages."""
    m_lcg, a_lcg = 2147483647, 16807
    state = int(seed)
    nU = 5 * T
    U = np.empty(nU)
    for i in range(nU):
        state = (a_lcg * state) % m_lcg
        U[i] = state / m_lcg
    u_reg = U[:T - 1]                          # regime-switch uniforms
    z = np.sqrt(-2.0 * np.log(U[T - 1:5 * T - 1:2])) \
        * np.cos(2.0 * np.pi * U[T:5 * T:2])   # 2T Box-Muller normals
    e_state = z[:T]                            # state innovations
    e_meas = z[T:2 * T]                        # measurement noise
    return u_reg, e_state, e_meas


def main(seed=17):
    T = 240
    u_reg, e_state, e_meas = _portable_draws(seed, T)
    F_set = [np.array([[0.85]]), np.array([[0.50]])]
    Q_set = [np.array([[0.02]]), np.array([[0.20]])]
    H = np.array([[1.0]])
    # Measurement noise must be small relative to the quiet-regime
    # innovation variance (here sigma^2_1 = 0.02) for regime membership
    # to be sharply identified: the exact variational q(S)-update uses
    # EXPECTED transition log-likelihoods, which correctly account for
    # posterior state uncertainty.  With R comparable to sigma^2_1, even
    # an oracle observing the true states classifies regimes imperfectly.
    R = np.array([[0.01]])
    A_true = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi0 = np.array([1.0, 0.0])

    s = np.zeros(T, dtype=int)
    s[0] = 0
    for t in range(1, T):
        s[t] = 1 if u_reg[t - 1] < A_true[s[t - 1], 1] else 0
    x = np.zeros(T + 1)
    for t in range(T):
        x[t + 1] = (float(F_set[s[t]][0, 0]) * x[t]
                     + float(np.sqrt(Q_set[s[t]][0, 0])) * e_state[t])
    Y = (x[1:] + float(np.sqrt(R[0, 0])) * e_meas).reshape(1, T)

    # Single-regime Gaussian smoother for baseline
    F_avg = np.mean([float(F[0, 0]) for F in F_set]) * np.eye(1)
    Q_avg = np.mean([float(Q[0, 0]) for Q in Q_set]) * np.eye(1)
    Xa = sks.smooth_const(Y, F_avg, H, Q_avg, R,
                          np.array([0.0]), np.array([[1.0]]))[0]

    Xb, info_b = sks.smooth_slds(Y, F_set, H, Q_set, R,
                                  np.array([0.0]), np.array([[1.0]]),
                                  A=A_true, pi0=pi0, max_iter=50, tol=1e-5)

    truth = x[1:]
    rmse_a = float(np.sqrt(np.mean((Xa[0, 1:] - truth)**2)))
    rmse_b = float(np.sqrt(np.mean((Xb[0, 1:] - truth)**2)))

    print("SLDS smoother demo: regime-switching AR(1)")
    print("==========================================")
    print(f"T = {T}, K = 2 regimes")
    print(f"True F per regime: phi_1 = {float(F_set[0][0,0]):.2f}, phi_2 = {float(F_set[1][0,0]):.2f}")
    print(f"True Q per regime: sigma^2_1 = {float(Q_set[0][0,0]):.3f}, sigma^2_2 = {float(Q_set[1][0,0]):.3f}")
    print(f"Regime composition: {int((s == 0).sum())} expansion, {int((s == 1).sum())} recession")
    print()
    print("Smoothing RMSE:")
    print(f"  (A) Gaussian, averaged single regime  {rmse_a:.4f}")
    print(f'  (B) SLDS smoother                     {rmse_b:.4f}  [{info_b["iters"]} iters, converged={info_b["converged"]}]')

    pred = (info_b["pi"][1, :] > 0.5).astype(int)
    accuracy = float(np.mean(pred == s))
    rec_recall = float(np.logical_and(pred == 1, s == 1).sum()) / max(int((s == 1).sum()), 1)
    rec_prec = float(np.logical_and(pred == 1, s == 1).sum()) / max(int((pred == 1).sum()), 1)

    print()
    print(f"Regime classification (pi_2 > 0.5 means recession):")
    print(f'  Accuracy:                           {accuracy:.3f}')
    print(f'  Recall (true recessions caught):    {rec_recall:.3f}')
    print(f'  Precision (flag was a recession):   {rec_prec:.3f}')
    print(f'  Mean pi_2 during true recessions:   {float(info_b["pi"][1, s == 1].mean()):.3f}')
    print(f'  Mean pi_2 during true expansions:   {float(info_b["pi"][1, s == 0].mean()):.3f}')

    # --- Plot ----------------------------------------------------------
    # Top: data, true x_t, Gaussian and SLDS smoothed paths, recessions shaded.
    # Bottom: estimated recession probability vs true regime indicator.
    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Shade recession periods (s==1) lightly in the top panel.  Only label
    # the first shaded span so the legend doesn't end up with N copies.
    yl = (float(x.min() - 0.5), float(x.max() + 0.5))
    in_rec = False
    rec_start = 0
    first_shade = True
    for t in range(T):
        if s[t] == 1 and not in_rec:
            rec_start, in_rec = t, True
        elif s[t] == 0 and in_rec:
            axes[0].axvspan(rec_start + 0.5, t + 0.5,
                            color="#fcdcdc", alpha=0.7,
                            label="recessions (truth)" if first_shade else None)
            first_shade = False
            in_rec = False
    if in_rec:
        axes[0].axvspan(rec_start + 0.5, T + 0.5, color="#fcdcdc", alpha=0.7,
                        label="recessions (truth)" if first_shade else None)

    axes[0].plot(tt, Y[0], "o", ms=3, color="0.6", label="observation")
    axes[0].plot(tt, truth,    "k-",  lw=1.2, label="true x_t")
    axes[0].plot(tt, Xa[0, 1:],"b--", lw=1.2, label="single-regime Gaussian")
    axes[0].plot(tt, Xb[0, 1:],"r--", lw=1.2, label="SLDS smoother")
    axes[0].set_ylim(*yl)
    axes[0].set_title("SLDS smoothing: high-volatility recessions and "
                      "low-volatility expansions")
    axes[0].set_ylabel("output gap")
    axes[0].grid(alpha=0.3); axes[0].legend(loc="lower left", fontsize=9)

    axes[1].step(tt, s, "k-", where="post", lw=1.2,
                 label="true regime (0=exp, 1=rec)")
    axes[1].plot(tt, info_b["pi"][1, :], "r-", lw=1.2,
                 label="estimated pi_t(recession)")
    axes[1].axhline(0.5, color="k", ls=":", lw=0.8, label="threshold 0.5")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title("Smoothed regime probability vs ground truth")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("probability")
    axes[1].grid(alpha=0.3); axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_and_close(fig, "demo_slds")

if __name__ == "__main__":
    main()
