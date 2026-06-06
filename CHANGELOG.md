# Changelog

## v3 (June 2026)

Revision responding to a full external peer review (20 numbered comments
plus overall feedback).  Every comment was independently verified and
addressed.  The dominant theme: the LaTeX had drifted from the validated
code; v3 re-synchronizes them and fixes the genuine bugs the review
surfaced on both sides.

### Code fixes (Python and Octave/MATLAB, mirrored)

- **Constrained smoother: log-barrier gradient sign.**  The barrier
  contribution to the gradient is `+mu * A' * (1./slack)`, not minus.
  The old sign turned the interior-point method into boundary clipping,
  masked by the fraction-to-boundary step cap.
  (`constrained.py`, `simks_smooth_constrained.m`)
- **Marginal log-likelihood: missing exogenous quadratic.**  The
  completed-square constant now includes
  `sum_t (B z_t)' Q^{-1} (B z_t)`; previously the likelihood level was
  wrong whenever exogenous state forcing was present, distorting EM
  convergence monitoring.  EM monotonicity re-verified with exogenous
  inputs.  (`marginal_loglik.py`, `simks_marginal_loglik.m`)
- **SLDS: exact variational EM.**  The q(X)-step now assembles the
  precision matrix from responsibility-weighted *expected* quadratic
  forms (`sum_k pi Q_k^{-1}`, `sum_k pi Q_k^{-1} F_k`,
  `sum_k pi F_k' Q_k^{-1} F_k` — still block tridiagonal), replacing
  the previous moment-matching shortcut of smoothing under averaged
  `F_t`, `Q_t`.  The q(S)-step scores regimes on *expected* transition
  log-likelihoods, with covariance corrections built from the
  `Sigma_tt` and lag-one `Sigma_{t,t-1}` blocks of the Takahashi sweep,
  instead of evaluating densities at the smoothed point path.
  Regime-dependent exogenous loadings (`B_set`) are handled in the
  exact information vector.  (`slds.py`, `simks_smooth_slds.m`)
- **Robust smoother: honest framing + optional exact E-step.**  Plug-in
  IRLS is documented as computing the posterior mode (MAP).  A new
  `exact_estep` option (Student-t only) replaces plug-in squared
  Mahalanobis distances with their posterior expectations, adding the
  trace corrections `tr(R^{-1} H Sigma_tt H')` and the state-side
  analogue.  Huber is documented as an IRLS working weight
  (Holland-Welsch), not a Gaussian scale mixture.
  (`robust.py`, `simks_smooth_robust.m`)
- **Natural-rate application: HLW conventions.**  The transition matrix
  now implements `y*_t = y*_{t-1} + g_{t-1}` (previously loaded
  `g_{t-2}`) and the IS curve carries the two-lag real-rate-gap average
  with `-a_r c / 2` on each of `g_{t-1}` and `g_{t-2}` (previously a
  single `-a_r c` on the wrong lag).  Both growth lags were in the
  state all along; no approximation needed.
  (`app1_natural_rate.py/.m`)
- **Nowcasting application: nowcast variance.**  The within-quarter
  factor-average variance now uses the single 3x3 diagonal block of the
  augmented state at the end-of-quarter month, whose off-diagonal
  elements carry all cross-month covariances including the lag-two
  term (previously assembled from scalar blocks at three times and the
  lag-two covariance was silently dropped).
  (`app2_nowcasting.py/.m`)

### MATLAB compatibility

- Removed the lone Octave-only keyword in the source tree
  (`endswitch` in `simks_smooth_robust.m`), which made MATLAB reject
  the file with a mixed function/`end` conventions error and broke
  every robust demo.  The whole tree was swept for other
  MATLAB-incompatible constructs (compound assignments, `!`, `#`
  comments, `printf`, `rows`/`columns`, `do/until`, Octave end-keywords)
  and is clean: the `.m` files now run unmodified in both Octave and
  MATLAB.  `octave_to_matlab.py` gained the missing `endswitch` mapping.

### Cross-language reproducibility

- `demo_slds` now draws its data from a portable RNG (Park-Miller LCG
  plus Box-Muller) implemented identically in the `.py` and `.m`
  files, so Python, Octave, and MATLAB simulate the same data and
  print the same numbers.  Verified: smoothed paths agree to ~1e-16
  and regime responsibilities to ~1e-15 across languages on the shared
  data.  Other demos still use each language's native RNG, so their
  printed numbers differ across languages while remaining qualitatively
  identical; the headline numbers quoted in the note are from the
  Python runs.

### Note (LaTeX/PDF)

- §4.4 LDL'/Kalman equivalence rewritten: the unit-lower factor block
  is obtained by a *right* solve, `M_t = J_{t,t-1} D_{t-1}^{-1}`
  (matching the code's convention); interior pivots satisfy
  `D_t = P_{t|t}^{-1} + F' Q^{-1} F`, with the filtered precision
  recovered as `C_t = D_t - F' Q^{-1} F` via Woodbury; the forward
  solve yields filtered means in information form,
  `u_t = P_{t|t}^{-1} xhat_{t|t}`.
- §4.5 Takahashi recursion: sign-convention dependence made explicit,
  with a scalar AR(1) sanity check.
- §4.6 marginal likelihood: corrected signs
  (`+ h'J^{-1}h / 2 - log det J / 2`), the exogenous-adjusted residual
  `Y - DZ`, and the full stacked offset `c_W` including `B z_t` terms.
- §6.2: initial-state M-step (`a0 = xhat_{0|T}`, `P0 = Sigma_00`)
  documented; the EM appendix no longer overstates or understates what
  is estimated.
- §7.1 and §7.2 rewritten to match the corrected robust and SLDS
  algorithms, including when and why the covariance corrections matter.
- §7.3: barrier gradient sign corrected, with a sanity-check argument.
- Computational claims in §§1.4, 2.1, 8, 9 rescoped: sparse Cholesky on
  J is presented as the robust, teachable frame for well-posed batch
  Gaussian smoothing with SPD precision, while conceding the
  normal-equations conditioning penalty and the documented advantages
  of square-root/UDU filters for ill-conditioned online problems.
- Singular-Q honesty: the epsilon-regularizer is disclosed as changing
  the model used for likelihoods and EM (constant shift for fixed
  epsilon), with exact alternatives (deterministic-state elimination,
  pseudo-determinants) stated; formal claims scoped to SPD P0, Q, R.
- HP-filter details: penalty summation limits consistent with the
  paper's own second-difference definition; posterior variance carries
  the sigma_eps^2 scale; the improper/intrinsic prior is acknowledged
  (posterior remains proper).
- Natural-rate application: equations, transition matrix, prose, and
  code now share one HLW normalization; Q correctly described as three
  substantive innovation variances plus lag-shift regularizers; the
  r* >= 0 exercise reframed as a maintained restriction, not the zero
  lower bound (which binds the nominal rate).
- All demo and application numbers re-measured from the corrected code.

## v2

Initial public package: teaching note, Octave and Python
implementations, eleven demos, three worked macro applications
(natural rate, nowcasting, 3-equation NK model), cross-validation of
the simultaneous smoother against RTS (~1e-13) and of selected
inversion against the dense inverse (~1e-15).
