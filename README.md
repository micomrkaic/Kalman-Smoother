# Simultaneous Kalman Smoother — v3

**Author:** Mico Mrkaic

*Produced under the guidance and direction of Mico Mrkaic, with the assistance
of AI (Claude, Anthropic). The author is responsible for all content, errors,
and omissions.*

A teaching package on the Kalman smoother written as one sparse block-tridiagonal
linear system

    J Xhat = h

instead of as a forward-backward recursion.  The note and the code are aimed at
graduate students of economics and applied macroeconomists who already know
matrix algebra but find the textbook Kalman/RTS derivation more obscure than the
object it computes.

## What v3 changes

v3 responds to a full external peer review: six code bugs fixed across both
languages (barrier gradient sign, exogenous term in the marginal likelihood,
exact variational SLDS updates, covariance-corrected robust E-step option,
HLW conventions in the natural-rate app, nowcast variance block), the `.m`
sources made MATLAB-compatible, `demo_slds` given a portable RNG so all
languages print identical numbers, and the note re-synchronized with the
validated code throughout.  See `CHANGELOG.md` for the full list.

## What v2 adds over v1

Six things that v1 either lacked or got wrong.

1. **Posterior covariances.**  v1 returned the smoothed mean only.  v2 adds
   `simks_selected_inv.m`, a Takahashi/Erisman–Tinney recursion that returns
   the diagonal blocks `P_{t|T}` and sub-diagonal blocks `P_{t,t-1|T}` of
   `J^{-1}` in `O(T n^3)` operations.  No full dense inverse, no Cholesky
   back-substitution per column.

2. **Marginal log-likelihood.**  v1's `simks_loglik_const.m` evaluated the
   *joint* negative log density at a fixed `X`, which is not what you want for
   maximum likelihood.  v2 has `simks_marginal_loglik.m`, which evaluates the
   marginal log p(Y; theta) by completing the square in `X`.  The old file is
   renamed `simks_objective_const.m` and clearly labelled as the joint
   objective.

3. **Missing data.**  `simks_smooth_const` and `simks_smooth_tv` now accept
   `NaN` entries in `Y`.  At each `t`, only the non-missing rows of `y_t`
   contribute to `H' R^{-1} H` and `H' R^{-1} y_t`.  Partial missingness and
   whole-period gaps both work.

4. **Diffuse prior.**  Pass `P0 = []` and the prior block is just dropped from
   `J`.  This is the `P_0^{-1} -> 0` limit and is the cleanest thing to do
   when you don't want to commit to a prior on `x_0`.

5. **Proper EM.**  v1's `simks_em_const.m` was an alternating
   smoother/regression algorithm with biased `Q` and `R` updates.  v2 does the
   real Gaussian EM using smoothed sufficient statistics

       S_00 = sum_t E[x_{t-1} x_{t-1}']
       S_11 = sum_t E[x_t     x_t']
       S_10 = sum_t E[x_t     x_{t-1}']

   with posterior covariance corrections from selected inversion.  The
   marginal log-likelihood is monitored and verified monotone.

6. **Robust smoother for non-Gaussian disturbances.**  v2 adds
   `simks_smooth_robust.m`, which extends the framework to heavy-tailed
   measurement and/or state noise (Student-t, Laplace, Huber) via Gaussian
   scale mixtures and EM/IRLS on the same sparse system.  The sparsity
   pattern of `J` never changes; only the per-period weights `tau_t` do.
   Two demos: `demo_robust.m` (outlier handling, cuts RMSE by ~40% versus
   Gaussian on contaminated data) and `demo_robust_breaks.m` (structural-
   break detection, cuts RMSE by ~60% and flags the breaks via weights
   `tau_w < 0.03`).

## File layout

The Octave/MATLAB and Python sides mirror each other: each language has
its own top-level directory with a `simksmoother/` source folder and a
parallel `examples/` tree.

    octave/
      simksmoother/                source routines
        simks_smooth_const.m       smoother, constant matrices
        simks_smooth_tv.m          smoother, time-varying matrices (cells or const)
        simks_selected_inv.m       Takahashi recursion for blocks of J^{-1}
        simks_marginal_loglik.m    marginal log p(Y; theta)
        simks_em_const.m           proper Gaussian EM
        simks_smooth_robust.m      IRLS smoother for Student-t / Laplace / Huber
        simks_smooth_slds.m        switching linear dynamical system (variational EM)
        simks_smooth_constrained.m inequality-constrained smoother (interior-point QP)
        simks_objective_const.m    joint negative log density at fixed X
        simks_simulate_const.m     generate (X, Y) from the model
        simks_rts_reference.m      textbook Kalman + RTS, for cross-checking
        nearest_spd.m              symmetric PD repair, used by EM
      examples/
        demo_main.m                interactive menu, runs any demo or all in sequence
        demo_const_known.m         smoothing + posterior bands + coverage
        demo_tv_known.m            time-varying parameters
        demo_const_em.m            proper EM, monotone loglik check
        demo_compare_recursive.m   simultaneous vs RTS, agreement to ~1e-15
        demo_missing.m             NaN handling, posterior var inside gaps
        demo_hp_filter.m           HP filter as the special case F=[1,1;0,1]
        demo_robust.m              robust smoothing on data with outliers
        demo_robust_breaks.m       structural-break detection via state-side robust
        demo_hp_credit.m           flagship: HP + credit-cycle exogenous; compares robust smoothing (true params) and Gaussian EM
        demo_slds.m                switching linear dynamical system (regime detection)
        demo_constrained.m         inequality-constrained smoother (NAIRU >= 0)
        figures/                   demos write PNGs here
        applications/
          app1_natural_rate.m      HLW-style natural rate of interest (r* = c g)
          app2_nowcasting.m        mixed-frequency dynamic factor model
          app3_nk_model.m          3-equation New Keynesian, EM-estimated

    python/
      simksmoother/                Python package (same routines as octave/simksmoother)
      examples/                    demo_*.py mirroring the Octave demos
        applications/              app*.py mirroring the Octave applications
        figures/                   demos write PNGs here
      tests/                       test_all_demos.py
      pyproject.toml

    doc/
      simultaneous_kalman_smoother.tex   teaching note
      simultaneous_kalman_smoother.pdf

    LICENSE                        GPLv3 full text
    octave_to_matlab.py            Octave-to-MATLAB syntax converter

## Usage

### Octave / MATLAB

    addpath('octave/simksmoother');

    % Basic smoothing.
    [Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0);

    % Smoothing + posterior covariance blocks.
    opts.return_cov = true;
    [Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0, opts);
    P_tT = info.Ptt;       % cell, P_{t|T} for t=0..T

    % Diffuse prior.
    [Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, []);

    % Marginal log-likelihood (for MLE, model comparison, etc.).
    ll = simks_marginal_loglik(Y, F, H, Q, R, a0, P0);

    % Time-varying: F, H, Q, R can be cell arrays length T (or constant).
    [Xhat, J, h, info] = simks_smooth_tv(Y, Fcell, Hcell, Qcell, Rcell, a0, P0);

    % Robust smoothing (Student-t measurement noise, nu = 4).
    opts = struct('nu', 4);
    [Xhat, info] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'student-t', opts);
    %   info.tau_v(t) is the per-period IRLS weight on the measurement;
    %   values << 1 flag observations the smoother treated as outliers.

    % Robust state innovations too (for structural-break detection).
    opts = struct('nu', 4, 'robust_state', true);
    [Xhat, info] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'student-t', opts);
    %   info.tau_w(t) << 1 marks t as the timing of a structural break.

    % EM for unknown F, H, Q, R.
    [Fhat, Hhat, Qhat, Rhat, ll_path] = simks_em_const(Y, F0, H0, Q0, R0, a0, P0, 100);

### Python

    # From the python/ directory: pip install -e .   (or just run the demos,
    # which add the package to sys.path automatically if it isn't installed).
    import simksmoother as sks

    # Basic smoothing (+ posterior covariance blocks).
    Xhat, J, h, info = sks.smooth_const(Y, F, H, Q, R, a0, P0, return_cov=True)
    P_tT = info["Ptt"]          # list of P_{t|T}, t = 0..T

    # Diffuse prior: pass P0=None.  Missing data: NaN entries in Y.
    # Marginal log-likelihood.
    ll, _ = sks.marginal_loglik(Y, F, H, Q, R, a0, P0)

    # Robust smoothing (Student-t, nu = 4); info["tau_v"] flags outliers.
    Xhat, info = sks.smooth_robust(Y, F, H, Q, R, a0, P0, "student-t", nu=4)

    # EM for unknown parameters.
    params, Xhat, hist = sks.em_const(Y, n=2)

## On the "LAPACK is faster than the recursion" claim

The recursive Kalman filter / RTS smoother and the simultaneous system
`J Xhat = h` both cost `O(T n^3)` operations.  They are the same complexity,
because sparse block-tridiagonal Cholesky on `J` is just block-LDL^T sweep,
which reproduces the Kalman recursion exactly.  Calling `J \ h` does not buy
you a complexity gain.

What the simultaneous formulation does buy is:

* **Conceptual clarity.**  The estimator is one sentence: minimize a quadratic
  penalty in `X`, get the posterior mean from solving `J Xhat = h`.  No
  filter, no smoother, no separate prediction/update step.

* **Robustness.**  Modern sparse Cholesky (CHOLMOD via `\`) is far more
  numerically robust than a hand-coded forward-backward recursion that subtracts
  large covariance matrices.  Stability-conscious recursive smoothers (square-root,
  UDU, Bierman–Thornton) exist, but you don't need them here — Cholesky does
  the work.

* **Modularity.**  Missing data, diffuse priors, irregular sampling, partially
  observed states, and non-stationary parameters are all small edits to `J`
  and `h`.  No need to redesign the recursion.

* **Extensibility.**  Non-Gaussian errors (Laplace, Student-t, robust losses,
  inequality constraints) lose the closed-form solve but keep the locality.
  IRLS or majorization-minimization on the same sparse system gets you a
  large class of robust state-space estimators almost for free.

The right way to teach the recursive Kalman filter is *as a special factorization*
of `J`, namely the block-LDL^T forward sweep with `D_t = P_{t|t}^{-1}`.  The note
shows this derivation.

## Tested on

GNU Octave 8.4.0 and MATLAB R2023a on Linux and Windows.  No toolboxes
required.  The Octave/MATLAB sources use only common dialect features
(plain `end`, `fprintf`, `~=`, etc.); a small `octave_to_matlab.py`
converter is included at the package root for anyone forking the code
and writing in Octave dialect.

## License

Copyright (C) 2026 Mico Mrkaic.

`simksmoother` is free software, released under the GNU General Public
License version 3 (or, at your option, any later version).  See the
`LICENSE` file in this directory for the full text.

The short version: you can use, modify, and redistribute this code
freely, including for commercial purposes.  If you distribute a
modified version (or any derivative work), you must also distribute it
under GPLv3, with the source available and the same license terms
intact.  See <https://www.gnu.org/licenses/gpl-3.0.html> for details.

If you use the package in published work, a citation is appreciated
but not required.

## References

* Rauch, Tung, Striebel (1965) — original recursive smoother
* Durbin and Koopman (2012) — *Time Series Analysis by State Space Methods*
* de Jong (1989) — diffuse Kalman filter
* McCausland, Miller, Pelletier (2011) — simulation smoothing via sparse precision
* Chan and Jeliazkov (2009) — efficient state-space inference via banded matrices
* Rue and Held (2005) — *Gaussian Markov Random Fields*; same precision-matrix view
* Takahashi, Fagan, Chen (1973) — selected inversion of sparse matrices
* Erisman and Tinney (1975) — selected inversion algorithm
* Roweis and Ghahramani (1999) — unifying review of linear-Gaussian models
