"""
demo_main: Five-minute interactive tour of the simultaneous Kalman smoother.

Usage:
    cd python/examples
    python demo_main.py
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

import importlib
import sys


DEMOS = [
    ("demo_const_known",
     "Smoothing with known parameters.  Posterior bands and empirical coverage check on a 2D state.",
     "How the basic smoother is called, and that the posterior covariances are well-calibrated."),
    ("demo_tv_known",
     "Time-varying parameters.  Sinusoidally varying F_t demonstrates the cell-array interface.",
     "How to pass time-varying matrices through lists, with the same call shape."),
    ("demo_compare_recursive",
     "Simultaneous vs textbook RTS smoother.  Should agree to machine precision (~1e-15).",
     "That J X = h is mathematically the same object as the recursive Kalman filter + RTS."),
    ("demo_missing",
     "Missing observations (NaN entries).  Posterior variance grows inside the gap.",
     "How NaN observations are dropped from H R^-1 H, with no special-case code."),
    ("demo_hp_filter",
     "Hodrick-Prescott filter as a special case (F = [[1,1],[0,1]], Q = diag(eps, 1/lambda)).",
     "That HP, Beveridge-Nelson, and unobserved-components are all the same matrix problem."),
    ("demo_const_em",
     "EM with smoothed sufficient statistics.  Monotone log-likelihood verified.",
     "The proper Gaussian EM formulas with posterior-covariance corrections."),
    ("demo_robust",
     "Robust smoothing on data with outliers.  Student-t, Huber, Laplace all cut RMSE.",
     "IRLS on the same sparse J, handling heavy-tailed measurement noise."),
    ("demo_robust_breaks",
     "Structural-break detection.  Robust prior on state innovations flags level shifts.",
     "How heavy-tailed Q catches one-off jumps without smearing them across periods."),
    ("demo_hp_credit",
     "Flagship: HP + credit-cycle exogenous + Student-t innovations + EM with unknown B,D.",
     "All of the above stacked: a serious applied econometric tool in ~100 lines."),
    ("demo_slds",
     "Switching Linear Dynamical System.  Recovers regime probabilities and states jointly.",
     "How J stays block-tridiagonal in every iteration of a discrete-regime mixture."),
    ("demo_constrained",
     "Inequality-constrained smoother.  NAIRU-style non-negativity via interior-point QP on J.",
     "How the same sparse J extends to constrained problems via a low-rank Hessian update."),
]


def run_one(idx):
    name, desc, insight = DEMOS[idx]
    print()
    print("-" * 65)
    print(f"Running demo {idx + 1}: {name}")
    print(f"  {desc}")
    print("-" * 65)
    print()
    try:
        mod = importlib.import_module(name)
        mod.main()
    except Exception as e:
        print(f"\n[Demo {name} failed: {e}]")
    print()
    print(f"What to take away: {insight}")
    print("-" * 65)


def main():
    while True:
        print()
        print("=" * 65)
        print("  Simultaneous Kalman Smoother -- five-minute demo tour")
        print("=" * 65)
        print()
        print("Available demos:")
        print()
        for i, (name, desc, _) in enumerate(DEMOS, start=1):
            print(f"  [{i:2d}]  {name}")
            print(f"        {desc}")
            print()
        print("  [ 0]  Run all in sequence.")
        print("  [ q]  Quit.")
        print()
        choice = input("Select demo number (or q to quit): ").strip()
        if not choice:
            continue
        if choice.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            return
        try:
            num = int(choice)
        except ValueError:
            print(f'  (Did not recognize "{choice}".  Enter a number or q to quit.)')
            continue
        if num == 0:
            for i in range(len(DEMOS)):
                run_one(i)
        elif 1 <= num <= len(DEMOS):
            run_one(num - 1)
        else:
            print(f"  (Selection out of range; pick 0..{len(DEMOS)} or q.)")


if __name__ == "__main__":
    main()
