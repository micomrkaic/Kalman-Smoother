"""
Smoke test: run each demo's main() and confirm it doesn't raise.

Also runs cross-check tests for numerical agreement of:
 - simultaneous smoother vs RTS reference (mean and covariance)
 - selected_inv vs direct inverse on small J
 - marginal_loglik value reproducibility
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
import os
import sys

import numpy as np


# Make sure python/ is on the path so simksmoother and the demos import.
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, ".."))
sys.path.insert(0, os.path.join(THIS, "..", "examples"))


def test_smooth_const_vs_rts():
    """Simultaneous smoother and textbook RTS must agree to ~1e-13."""
    import simksmoother as sks
    rng = np.random.default_rng(123)
    T = 100
    F = np.array([[0.8, 0.1], [0.0, 0.6]])
    H = np.array([[1.0, 0.3]])
    Q = 0.05 * np.eye(2)
    R = np.array([[0.05]])
    a0 = np.zeros(2); P0 = np.eye(2)

    Y, _ = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)
    XsT, PsT = sks.rts_smoother(Y, F, H, Q, R, a0, P0)
    assert np.max(np.abs(Xhat - XsT)) < 1e-13, \
        f"Mean disagreement {np.max(np.abs(Xhat - XsT)):.2e}"
    cov_err = max(np.max(np.abs(info["Ptt"][t] - PsT[t])) for t in range(T + 1))
    assert cov_err < 1e-13, f"Cov disagreement {cov_err:.2e}"


def test_selected_inv():
    """Selected inversion matches direct inverse on small J."""
    import simksmoother as sks
    rng = np.random.default_rng(456)
    T = 30
    F = np.array([[0.7, 0.2], [0.1, 0.6]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = 0.1 * np.eye(2)
    R = 0.1 * np.eye(2)
    a0 = np.zeros(2); P0 = np.eye(2)
    Y, _ = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    _, J, _, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)
    Jinv = np.linalg.inv(J.toarray())
    n = 2
    for t in range(T + 1):
        sub = Jinv[t * n:(t + 1) * n, t * n:(t + 1) * n]
        assert np.allclose(sub, info["Ptt"][t], atol=1e-12), \
            f"Block ({t},{t}) selected_inv disagrees by {np.max(np.abs(sub - info['Ptt'][t])):.2e}"
    for t in range(T):
        sub = Jinv[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n]
        assert np.allclose(sub, info["Ptt1"][t], atol=1e-12), \
            f"Off-diag block disagrees by {np.max(np.abs(sub - info['Ptt1'][t])):.2e}"


def test_exogenous_consistency():
    """smooth_const with B=D=0 (explicit) equals smooth_const without exogenous."""
    import simksmoother as sks
    rng = np.random.default_rng(789)
    T = 50
    F = np.array([[0.8]])
    H = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.1]])
    a0 = np.zeros(1); P0 = np.array([[1.0]])
    Y, _ = sks.simulate_const(T, F, H, Q, R, a0, P0, rng=rng)
    Z = rng.standard_normal((1, T))
    X1 = sks.smooth_const(Y, F, H, Q, R, a0, P0)[0]
    X2 = sks.smooth_const(Y, F, H, Q, R, a0, P0,
                          Z=Z, B=np.zeros((1, 1)), D=np.zeros((1, 1)))[0]
    assert np.allclose(X1, X2, atol=1e-12)


def test_em_monotone():
    """EM marginal log-likelihood must be (almost) monotone non-decreasing."""
    import simksmoother as sks
    rng = np.random.default_rng(99)
    T = 200
    F = np.array([[0.85, 0.1], [0.0, 0.6]])
    H = np.array([[1.0, 0.3]])
    Q = 0.05 * np.eye(2)
    R = np.array([[0.1]])
    Y, _ = sks.simulate_const(T, F, H, Q, R, np.zeros(2), np.eye(2), rng=rng)
    _, _, hist = sks.em_const(Y, n=2, max_iter=40, tol=1e-7)
    for i in range(1, len(hist)):
        assert hist[i] >= hist[i - 1] - 1e-7, \
            f"Loglik decreased at iter {i}: {hist[i]:.6f} < {hist[i-1]:.6f}"


def test_all_demos_run():
    """Each demo's main() must run without raising."""
    demo_names = [
        "demo_const_known", "demo_tv_known", "demo_compare_recursive",
        "demo_missing", "demo_hp_filter", "demo_const_em",
        "demo_robust", "demo_robust_breaks", "demo_hp_credit",
        "demo_slds", "demo_constrained",
    ]
    for name in demo_names:
        mod = importlib.import_module(name)
        mod.main()


def test_all_applications_run():
    """Each appendix application's main() must run without raising."""
    sys.path.insert(0, os.path.join(THIS, "..", "examples", "applications"))
    app_names = ["app1_natural_rate", "app2_nowcasting", "app3_nk_model"]
    for name in app_names:
        mod = importlib.import_module(name)
        mod.main()


if __name__ == "__main__":
    tests = [
        test_smooth_const_vs_rts,
        test_selected_inv,
        test_exogenous_consistency,
        test_em_monotone,
        test_all_demos_run,
        test_all_applications_run,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        try:
            t()
            print(f"  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            sys.exit(1)
    print()
    print("All tests passed.")
