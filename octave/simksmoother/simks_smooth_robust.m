function [Xhat, info] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, family, opts)
% SIMKS_SMOOTH_ROBUST  Robust simultaneous smoother for non-Gaussian noise.
%
% Generalizes the Gaussian simultaneous smoother to heavy-tailed
% disturbances by interpreting them as Gaussian scale mixtures and
% running iteratively reweighted least squares (IRLS) on the same
% sparse block-tridiagonal system that the Gaussian smoother uses.
%
% Model:
%   x_t = F x_{t-1} + w_t,    t = 1,...,T
%   y_t = H x_t       + v_t,  t = 1,...,T
%   x_0 ~ N(a0, P0)           (or diffuse: P0 = [])
%
% with v_t (and optionally w_t) drawn from a heavy-tailed family
% representable as a Gaussian scale mixture:
%
%   v_t | tau_t ~ N(0, R / tau_t),    tau_t ~ p(tau | family).
%
% For Student-t and Laplace, marginalizing out tau_t gives the desired
% heavy-tailed marginal:
%   * Student-t:  tau ~ Gamma(nu/2, nu/2),  weight (nu+m)/(nu+delta^2)
%   * Laplace:    tau ~ Exp,                 weight 1/sqrt(delta^2)
%   * Huber:      weight min(1, c/delta).  Huber is NOT a proper scale
%                 mixture; min(1, c/delta) is the IRLS working weight
%                 implied by the Huber loss (Holland-Welsch), applied
%                 to the same sparse system.
%
% where delta^2 = r_t' R^{-1} r_t is the SQUARED Mahalanobis distance
% of the residual and m is the dimension of v_t (or the number of
% observed components at t under missing data).
%
% By default the weights are evaluated by plugging in residuals at the
% current smoothed mean; the resulting fixed point is the MAP /
% posterior mode of the heavy-tailed model (IRLS = majorize-minimize).
% For Student-t, opts.exact_estep = true replaces the plug-in squared
% distance by its posterior expectation under the current Gaussian
% approximation,
%     E[delta_t^2 | Y] = delta2_plugin + trace(R^{-1} H Sigma_tt H')
% (and the state-side analogue using Sigma_tt, Sigma_{t-1,t-1}, and the
% lag-one block Sigma_{t,t-1}), which is the E-step quantity required
% by latent-scale EM.  Costs one selected inversion per iteration.
%
% Inputs:
%   Y, F, H, Q, R, a0, P0 : as in simks_smooth_const. Y may contain
%       NaN for missing observations.
%   family : 'gaussian' | 'student-t' | 'laplace' | 'huber'
%   opts (optional struct):
%     .nu           : df for Student-t (default 4)
%     .c            : Huber threshold on Mahalanobis residual (default 1.345,
%                     the Holland-Welsch tuning for 95% efficiency at the
%                     normal in scalar Huber regression)
%     .max_iter     : IRLS iterations (default 50)
%     .tol          : convergence on max |Delta Xhat| (default 1e-6)
%     .robust_state : also robustify w_t with the same family (default false)
%     .exact_estep  : Student-t only; covariance-corrected E-step
%                     (default false = plug-in IRLS / posterior mode)
%     .return_cov   : posterior covariance blocks at the final iterate
%                     (default false)
%     .verbose      : print iteration progress (default false)
%
% Outputs:
%   Xhat : n x (T+1) smoothed state path
%   info : struct
%     .tau_v         : 1 x T  measurement IRLS weights at convergence
%                      (tau ~ 1 means full Gaussian weight; tau << 1 means
%                       observation was treated as an outlier)
%     .tau_w         : 1 x T  state-innovation IRLS weights (if robust_state)
%     .iters         : iterations performed
%     .converged     : true if the tolerance was reached before max_iter
%     .delta_history : max |Delta Xhat| per iteration
%     .family        : echo of family
%     .Ptt, .Ptt1    : if return_cov, posterior cov blocks at the final pass
%
% Algorithmic note:
%   This is the EM algorithm for the scale-mixture latent variable.
%   Each E-step computes tau_t from the current smoothed residuals;
%   each M-step solves the same sparse block-tridiagonal system with
%   weights R_t = R / tau_t (and Q_t = Q / xi_t if robust_state).  The
%   sparsity pattern of J never changes; only the values.  Per
%   iteration cost is one O(T n^3) sparse Cholesky.
%
%   Convergence rates differ across families.  Student-t and Huber
%   typically converge in 10-25 iterations to tol = 1e-6.  Laplace is
%   inherently non-smooth at zero residual and the IRLS weights have
%   unbounded range, so the algorithm converges only linearly; bump
%   max_iter to 200-500 or relax tol if tight convergence is needed,
%   or use the smoother Huber alternative.
%
% References:
%   Lange, Little, Taylor (1989) JASA -- Student-t regression via EM
%   Andrews, Mallows (1974) JRSS-B -- scale mixtures of normals
%   Holland, Welsch (1977) Comm. Stat. -- Huber tuning constants

% simksmoother --- a simultaneous Kalman smoother in sparse linear algebra form.
% Copyright (C) 2026 Mico Mrkaic.
%
% Produced under the guidance and direction of Mico Mrkaic, with the
% assistance of AI (Claude, Anthropic).
%
% This file is part of the simksmoother package.
%
% simksmoother is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; see the LICENSE file in the package root,
% or <https://www.gnu.org/licenses/>.
%

  if nargin < 9, opts = struct(); end

  nu           = getfield_default(opts, 'nu',           4);
  c_huber      = getfield_default(opts, 'c',            1.345);
  max_iter     = getfield_default(opts, 'max_iter',     50);
  tol          = getfield_default(opts, 'tol',          1e-6);
  robust_state = getfield_default(opts, 'robust_state', false);
  exact_estep  = getfield_default(opts, 'exact_estep',  false);
  if exact_estep && ~strcmpi(family, 'student-t')
    error('simks_smooth_robust: exact_estep is only available for student-t');
  end
  return_cov   = getfield_default(opts, 'return_cov',   false);
  verbose      = getfield_default(opts, 'verbose',      false);

  [m, T] = size(Y);
  n      = size(F, 1);

  % Exogenous inputs (optional).  Precompute Bz, Dz once for the IRLS
  % residual updates; the inner smoother receives them via opts.
  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    k = size(Z, 1);
    if isfield(opts, 'B') && ~isempty(opts.B), B = opts.B; else, B = zeros(n,k); end
    if isfield(opts, 'D') && ~isempty(opts.D), D = opts.D; else, D = zeros(m,k); end
    BZ = B * Z;       % n x T
    DZ = D * Z;       % m x T
  end

  % Pass-through for the Gaussian case so callers can use this function
  % uniformly regardless of family.
  if strcmpi(family, 'gaussian')
    sub_opts = struct('return_cov', return_cov);
    if have_exog
      sub_opts.Z = Z; sub_opts.B = B; sub_opts.D = D;
    end
    [Xhat, ~, ~, info0] = simks_smooth_const(Y, F, H, Q, R, a0, P0, sub_opts);
    info             = info0;
    info.tau_v       = ones(1, T);
    info.iters       = 1;
    info.converged   = true;
    info.delta_history = [];
    info.family      = 'gaussian';
    return;
  end

  Qi_full = inv(Q);

  % Initialize at the Gaussian smoother (all weights = 1).
  tau_v = ones(1, T);
  if robust_state
    tau_w = ones(1, T);
  end

  Xhat_prev     = [];
  delta_history = zeros(1, max_iter);
  converged     = false;
  iters_done    = 0;

  for it = 1:max_iter
    iters_done = it;

    % Assemble per-period covariances from current weights.
    Rcells = cell(1, T);
    for t = 1:T
      Rcells{t} = R / tau_v(t);
    end
    if robust_state
      Qcells = cell(1, T);
      for t = 1:T
        Qcells{t} = Q / tau_w(t);
      end
    else
      Qcells = Q;
    end

    % One sparse solve: the same J pattern as the Gaussian smoother.
    sub_opts = struct();
    if have_exog
      sub_opts.Z = Z; sub_opts.Bseq = B; sub_opts.Dseq = D;
    end
    if exact_estep
      sub_opts.return_cov = true;
    end
    [Xhat, ~, ~, info_it] = simks_smooth_tv(Y, F, H, Qcells, Rcells, a0, P0, sub_opts);
    if exact_estep
      Ptt_it  = info_it.Ptt;
      Ptt1_it = info_it.Ptt1;
    end

    % Convergence check (skip the first iteration).
    if ~isempty(Xhat_prev)
      delta = max(abs(Xhat(:) - Xhat_prev(:)));
      delta_history(it) = delta;
      if verbose
        fprintf('  iter %2d   |Delta X|_inf = %.3e\n', it, delta);
      end
      if delta < tol
        converged = true;
        break;
      end
    end
    Xhat_prev = Xhat;

    % E-step: update measurement weights from the smoothed residuals.
    % With exogenous, the model residual is r_t = y_t - H x_t - D z_t.
    for t = 1:T
      yt   = Y(:, t);
      mask = ~isnan(yt);
      if any(mask)
        Hm     = H(mask, :);
        Rm     = R(mask, mask);
        Rm_i   = inv(Rm);
        r      = yt(mask) - Hm * Xhat(:, t+1);
        if have_exog
          r = r - DZ(mask, t);
        end
        delta2 = r' * Rm_i * r;
        if exact_estep
          % E[delta^2 | Y] = plug-in + trace(R_m^{-1} H_m Sigma_tt H_m').
          delta2 = delta2 + trace(Rm_i * Hm * Ptt_it{t+1} * Hm');
        end
        m_eff  = sum(mask);
        tau_v(t) = robust_weight(family, delta2, m_eff, nu, c_huber);
      else
        tau_v(t) = 1;
      end
    end

    % E-step: same for state innovations if requested.
    % State innovation is w_t = x_t - F x_{t-1} - B z_t.
    if robust_state
      for t = 1:T
        w = Xhat(:, t+1) - F * Xhat(:, t);
        if have_exog
          w = w - BZ(:, t);
        end
        eta2     = w' * Qi_full * w;
        if exact_estep
          % Var(x_t - F x_{t-1} | Y) trace correction.
          S_tt = Ptt_it{t+1};  S_mm = Ptt_it{t};  S_tm = Ptt1_it{t};
          Ctr  = S_tt - F * S_tm' - S_tm * F' + F * S_mm * F';
          eta2 = eta2 + trace(Qi_full * Ctr);
        end
        tau_w(t) = robust_weight(family, eta2, n, nu, c_huber);
      end
    end
  end

  delta_history = delta_history(1:iters_done);

  info               = struct();
  info.tau_v         = tau_v;
  if robust_state
    info.tau_w = tau_w;
  end
  info.iters         = iters_done;
  info.converged     = converged;
  info.delta_history = delta_history;
  info.family        = family;

  % Final pass with posterior covariance, if requested.  Cheaper to do
  % this once at the end than every iteration.
  if return_cov
    Rcells_final = cell(1, T);
    for t = 1:T
      Rcells_final{t} = R / tau_v(t);
    end
    if robust_state
      Qcells_final = cell(1, T);
      for t = 1:T
        Qcells_final{t} = Q / tau_w(t);
      end
    else
      Qcells_final = Q;
    end
    sub_opts = struct('return_cov', true);
    if have_exog
      sub_opts.Z = Z; sub_opts.Bseq = B; sub_opts.Dseq = D;
    end
    [Xhat, ~, ~, info_final] = simks_smooth_tv(Y, F, H, Qcells_final, ...
                                                Rcells_final, a0, P0, sub_opts);
    info.Ptt  = info_final.Ptt;
    info.Ptt1 = info_final.Ptt1;
  end
end

function w = robust_weight(family, delta2, m, nu, c)
% Weight for the EM/IRLS update under each Gaussian scale mixture
% family.  delta2 is the Mahalanobis squared residual; m is the
% dimension of the observation (or its observed sub-dimension at t).
  switch lower(family)
    case 'student-t'
      w = (nu + m) / (nu + delta2);
    case 'laplace'
      % Multivariate Laplace via Gaussian scale mixture; weight is
      % 1 / sqrt(delta2), floored to avoid blow-up at zero residual.
      w = 1 / max(sqrt(delta2), 1e-10);
    case 'huber'
      delta = sqrt(delta2);
      if delta <= c
        w = 1;
      else
        w = c / delta;
      end
    otherwise
      error('simks_smooth_robust: unknown family "%s" (expected gaussian, student-t, laplace, or huber)', family);
  end
end

function v = getfield_default(s, fname, default)
  if isfield(s, fname)
    v = s.(fname);
  else
    v = default;
  end
end
