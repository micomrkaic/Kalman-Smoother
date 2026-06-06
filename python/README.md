# simksmoother (Python)

Python implementation of the simultaneous Kalman smoother:
the Kalman smoother formulated as a single sparse block-tridiagonal
linear system $JX = h$, instead of as a forward-backward recursion.

This is a direct port of the Octave reference implementation in
`../octave/`.  The function names and APIs match closely; argument
shapes follow the NumPy convention (rows = state coordinates,
columns = time).

## Install

```bash
cd python
pip install -e .
```

Requires Python 3.9+, NumPy 1.20+, SciPy 1.7+.

Optional: install `scikit-sparse` (CHOLMOD wrapper) for faster sparse
factorization on large problems.

## Quick tour

The fastest way to see what the package does is to run the interactive
demo menu:

```bash
cd python/examples
python demo_main.py
```

Then pick demos by number, or hit `0` to run all 11 in sequence.

## Worked applications

For end-to-end demonstrations of the framework on three classical
empirical-macro problems, see `python/examples/applications/`:

```bash
cd python/examples/applications
python app1_natural_rate.py    # HLW natural rate of interest
python app2_nowcasting.py      # mixed-frequency factor model
python app3_nk_model.py        # 3-equation NK model with EM
```

Each application is a complete walkthrough (model -> state-space mapping
-> simulation -> smoothing -> EM estimation -> interpretation) and is
discussed in detail in the appendix of the accompanying note.

## Usage

```python
import numpy as np
import simksmoother as sks

# Define and simulate from a model.
F = np.array([[0.8, 0.1], [0.0, 0.6]])
H = np.array([[1.0, 0.3]])
Q = 0.05 * np.eye(2); R = np.array([[0.1]])
a0 = np.zeros(2); P0 = np.eye(2)

Y, X = sks.simulate_const(T=200, F=F, H=H, Q=Q, R=R,
                            a0=a0, P0=P0,
                            rng=np.random.default_rng(0))

# Smoothing.
Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0)

# Smoothing with posterior covariances via selected inversion.
Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)
P_t = info["Ptt"]      # list of (n, n), one per t = 0..T

# Marginal log-likelihood for MLE.
ll, _ = sks.marginal_loglik(Y, F, H, Q, R, a0, P0)

# Missing data: just pass NaN entries in Y.
Y[:, 50:70] = np.nan
Xhat, _, _, _ = sks.smooth_const(Y, F, H, Q, R, a0, P0)

# Diffuse prior: pass P0=None.
Xhat, _, _, _ = sks.smooth_const(Y, F, H, Q, R, a0, P0=None)

# Exogenous inputs.
Z = np.random.default_rng(1).standard_normal((1, 200))
B = np.array([[0.0], [0.5]])
D = np.array([[0.1]])
Xhat, _, _, _ = sks.smooth_const(Y, F, H, Q, R, a0, P0, Z=Z, B=B, D=D)

# Robust smoothing (Student-t, Laplace, Huber).
Xhat, info = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "student-t", nu=4)
# State-side robust for structural-break detection:
Xhat, info = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "student-t",
                                robust_state=True)

# EM for unknown parameters.
params, Xhat, hist = sks.em_const(Y, n=2, max_iter=100, verbose=False)

# Switching linear dynamical system.
F_set = [np.array([[0.85]]), np.array([[0.50]])]
Q_set = [np.array([[0.02]]), np.array([[0.20]])]
A_true = np.array([[0.95, 0.05], [0.10, 0.90]])
Xhat, info = sks.smooth_slds(Y, F_set, H, Q_set, R, a0, P0, A=A_true)

# Constrained smoother (NAIRU >= 0).
import scipy.sparse as sp
A_ineq = -sp.eye(201).tocsr()
b_ineq = np.zeros(201)
Xhat, info = sks.smooth_constrained(Y, F, H, Q, R, a0, P0, A_ineq, b_ineq)
```

## Tests

```bash
cd python
python tests/test_all_demos.py
```

Five tests:
1. Simultaneous smoother vs textbook RTS (agreement to ~1e-13).
2. Selected inversion vs direct dense inverse.
3. Exogenous-consistency: `B=D=0` reproduces the no-exogenous solution.
4. EM marginal log-likelihood is monotone non-decreasing.
5. All 11 demos run without errors.

## Module reference

| Module              | Function                | Purpose                                       |
|---------------------|-------------------------|-----------------------------------------------|
| `smooth`            | `smooth_const`          | constant-parameter smoother                   |
| `smooth`            | `smooth_tv`             | time-varying smoother                         |
| `selected_inv`      | `selected_inv`          | Takahashi recursion for J^{-1} blocks         |
| `marginal_loglik`   | `marginal_loglik`       | log p(Y; theta)                               |
| `simulate`          | `simulate_const`        | draw from the model                           |
| `rts_reference`     | `rts_smoother`          | textbook Kalman + RTS (for cross-checks)      |
| `robust`            | `smooth_robust`         | IRLS for Student-t, Laplace, Huber            |
| `em`                | `em_const`              | Gaussian EM with smoothed sufficient stats    |
| `slds`              | `smooth_slds`           | switching linear dynamical system (vEM)       |
| `constrained`       | `smooth_constrained`    | interior-point QP for inequality-constrained  |

## Differences from the Octave version

The math is identical.  Surface differences:

- 0-based indexing.  In Octave column `t+1` of `Xhat` is `x_t`; in
  Python column `t` is `x_t`.  The internal representation is unchanged.
- Options are passed as keyword arguments instead of an `opts` struct.
- `info` is a Python `dict`, not a struct.
- Diffuse prior: pass `P0=None` rather than `P0=[]`.
- Random number generation uses `numpy.random.Generator` (the modern API),
  passed in via the optional `rng=` parameter to `simulate_const`.

## See also

The full mathematical derivation is in `../doc/simultaneous_kalman_smoother.pdf`
in the parent package.  The Octave reference implementation in `../octave/`
provides the cross-check for numerical agreement.

