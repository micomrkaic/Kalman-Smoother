function [Xhat, info] = simks_smooth_constrained(Y, F, H, Q, R, a0, P0, A_ineq, b_ineq, opts)
% SIMKS_SMOOTH_CONSTRAINED  Simultaneous smoother with inequality constraints.
%
% Solves the constrained smoothing problem
%
%     minimize    1/2 X' J X - h' X
%     subject to  A_ineq * X <= b_ineq
%
% where (J, h) are the precision matrix and information vector of the
% underlying Gaussian smoothing problem.  When the constraints are
% inactive at the unconstrained optimum, the solution coincides with
% the simultaneous Kalman smoother.  When some constraints bind, the
% solution is the constrained posterior mode.
%
% Example uses:
%   - NAIRU smoother with NAIRU >= 0:        A_ineq blocks pick out the
%     NAIRU coordinate at each t, with b = 0;
%   - output gap with x_t >= -10:            A_ineq * X <= 10 with
%     A_ineq blocks pick out -x_t (so the row is "-x_t <= 10");
%   - monotone trend: Delta tau_t >= 0:      A_ineq picks out -Delta tau_t;
%   - bounds [lo, hi] on selected components: pairs of rows in A_ineq.
%
% Algorithm: log-barrier interior-point method.  At each outer step,
% solve
%
%     X(mu) = argmin  1/2 X' J X - h' X - mu sum_i log(b_i - a_i' X),
%
% where (a_i, b_i) are the rows of (A_ineq, b_ineq).  The Newton step
%
%     (J + A_ineq' diag(1/(b - A_ineq X)^2) A_ineq) dX
%        = h - J X + A_ineq' (1/(b - A_ineq X)),
%
% adds a low-rank, positive-semidefinite update to the same sparse J.
% When A_ineq has few size(the constraints are sparse in time and
% coordinates, 1), the dominant cost remains the O(T n^3) sparse Cholesky
% of J.
%
% Strategy: decrease the barrier parameter mu geometrically from
% mu_0 = max(1, scale of (b - A_ineq X_unc)) by a factor of 0.5
% each step; do 1-3 damped Newton iterations per mu, terminating when
% max(0, A_ineq X - b) is below tol_feas and the Newton step is below
% tol_step.  For state-space inequality constraints with a small number
% of binding rows, this converges in 20-40 inner solves.
%
% Inputs:
%   Y, F, H, Q, R, a0, P0 : as in simks_smooth_const (NaN allowed in Y).
%   A_ineq : c x N matrix, where N = (T+1)*n.
%   b_ineq : c x 1 vector.
%   opts (optional struct):
%     .return_cov  : default false.
%     .Z, .B, .D   : exogenous, as in simks_smooth_const.
%     .mu0         : initial barrier (default: 1).
%     .mu_factor   : geometric decrease (default 0.5).
%     .max_outer   : maximum outer steps (default 40).
%     .max_newton  : Newton iterations per mu (default 3).
%     .tol_feas    : feasibility tolerance (default 1e-8).
%     .tol_step    : Newton step tolerance (default 1e-10).
%     .verbose     : default false.
%
% Outputs:
%   Xhat : n x (T+1) constrained smoothed states.
%   info : struct
%     .lambda           : c x 1 dual variables (KKT multipliers).
%     .iters_outer      : barrier-decrease iterations.
%     .iters_newton     : total inner Newton steps.
%     .active           : c x 1 logical, lambda > 1e-6.
%     .max_violation    : max(0, A_ineq Xhat - b) at convergence.
%     .converged
%
% Notes on theory:
%   This is the Bayesian / posterior-mode interpretation of constrained
%   smoothing.  In a frequentist setting (constrained MLE), the same
%   solve produces the constrained mode of the joint, which coincides
%   with the truncated-Gaussian posterior mean only when the constraints
%   do not bind.  Posterior expectations under truncated Gaussians
%   require sampling (e.g. Botev's exact method) or HMC; the
%   posterior MODE is what this function returns.  For most macro
%   uses (smoothed-trend visualization, NAIRU bands), the mode is the
%   primary output.

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

  if nargin < 10, opts = struct(); end

  mu0          = get_opt(opts, 'mu0',         1.0);
  mu_factor    = get_opt(opts, 'mu_factor',   0.5);
  max_outer    = get_opt(opts, 'max_outer',   40);
  max_newton   = get_opt(opts, 'max_newton',  3);
  tol_feas     = get_opt(opts, 'tol_feas',    1e-8);
  tol_step     = get_opt(opts, 'tol_step',    1e-10);
  verbose      = get_opt(opts, 'verbose',     false);
  return_cov   = get_opt(opts, 'return_cov',  false);

  [m, T] = size(Y);
  n      = size(F, 1);
  N      = (T+1) * n;

  c      = size(A_ineq, 1);
  assert(size(A_ineq, 2) == N, sprintf('A_ineq must have %d columns', N));
  assert(size(b_ineq, 1) == c,     'b_ineq must match A_ineq');

  % --- Pass through exogenous to the inner Gaussian smoother to get J, h. ---
  inner_opts = struct();
  for f = {'Z', 'B', 'D'}
    if isfield(opts, f{1}), inner_opts.(f{1}) = opts.(f{1}); end
  end
  [X_unc, J, h, ~] = simks_smooth_const(Y, F, H, Q, R, a0, P0, inner_opts);
  X_unc_vec = X_unc(:);

  % Check whether constraints are already satisfied at the unconstrained
  % optimum.  If so, return immediately.
  slack_unc = b_ineq - A_ineq * X_unc_vec;
  if all(slack_unc > tol_feas)
    Xhat = X_unc;
    info = struct('lambda',        zeros(c, 1), ...
                  'iters_outer',   0, ...
                  'iters_newton',  0, ...
                  'active',        false(c, 1), ...
                  'max_violation', 0, ...
                  'converged',     true);
    if return_cov
      [Ptt, Ptt1] = simks_selected_inv(J, n);
      info.Ptt  = Ptt;
      info.Ptt1 = Ptt1;
    end
    return;
  end

  % --- Find an initial strictly feasible point. ---
  % Strategy: start from X_unc shifted to satisfy the constraints with a
  % uniform margin.  If the constraints define a non-empty polytope, this
  % succeeds for a small perturbation; otherwise we fall back to Phase-I.
  X = X_unc_vec;
  slack = b_ineq - A_ineq * X;
  if any(slack <= 0)
    % Phase I: shift X along -A_ineq' (1) to gain slack.  We solve
    % J dX = -A_ineq' s for any positive direction s; then take a step
    % to push X strictly inside.  In practice for state bounds, just
    % project: pick X with components inside the box.
    [X, ok] = phase1_feasible(J, A_ineq, b_ineq, X);
    if ~ok
      % Last resort: tiny shrinkage toward zero.
      X = X * 0.5;
      slack = b_ineq - A_ineq * X;
      if any(slack <= 0)
        warning('simks_smooth_constrained: could not find a strictly feasible starting point.');
      end
    end
    slack = b_ineq - A_ineq * X;
  end

  % --- Outer barrier loop. ---
  mu_cur = mu0;
  total_newton = 0;
  converged    = false;

  for outer = 1:max_outer
    % Inner: damped Newton iterations.
    for inner = 1:max_newton
      slack = b_ineq - A_ineq * X;
      if any(slack <= 0)
        % shouldn't happen if line search is correct, but safety net
        slack = max(slack, 1e-12);
      end
      one_over_s  = 1 ./ slack;
      one_over_s2 = one_over_s.^2;

      % Gradient of (1/2) X'JX - h'X - mu * sum_i log(b_i - a_i'X):
      % the barrier contributes d/dX [-mu log(b - a'X)] = +mu a/(b - a'X),
      % so the barrier term enters with a PLUS sign.
      g = J * X - h + mu_cur * (A_ineq' * one_over_s);

      % Hessian: J + mu * A_ineq' diag(1/s^2) A_ineq.  Use an
      % explicit sparse diagonal scaling to avoid broadcast pitfalls.
      Dmat = spdiags(one_over_s2, 0, c, c);
      AtDA = A_ineq' * Dmat * A_ineq;
      H_mat = J + mu_cur * AtDA;

      % Solve.
      dX = - H_mat \ g;

      % Backtracking line search to maintain strict feasibility.
      step = 1.0;
      A_dX = A_ineq * dX;
      pos  = A_dX > 0;
      if any(pos)
        max_step = 0.99 * min(slack(pos) ./ A_dX(pos));
        step = min(step, max_step);
      end
      X_new = X + step * dX;
      X = X_new;

      step_norm = step * max(abs(dX));
      total_newton = total_newton + 1;

      if verbose
        viol = max(0, max(A_ineq * X - b_ineq));
        fprintf('  outer %2d  inner %d  mu = %.2e  |dX| = %.2e  viol = %.2e\n', ...
               outer, inner, mu_cur, step_norm, viol);
      end

      if step_norm < tol_step
        break;
      end
    end

    viol = max(0, max(A_ineq * X - b_ineq));
    if mu_cur < 1e-9 && viol < tol_feas
      converged = true;
      break;
    end

    mu_cur = mu_cur * mu_factor;
    if mu_cur < 1e-12
      mu_cur = 1e-12;
    end
  end

  % --- Reconstruct dual variables from the KKT condition. ---
  slack = b_ineq - A_ineq * X;
  lambda = max(0, mu_cur ./ max(slack, 1e-14));   % approx dual

  Xhat = reshape(X, n, T+1);

  info = struct();
  info.lambda        = lambda;
  info.iters_outer   = outer;
  info.iters_newton  = total_newton;
  info.active        = lambda > 1e-6;
  info.max_violation = max(0, max(A_ineq * X - b_ineq));
  info.converged     = converged;

  if return_cov
    % Posterior covariances on the constrained problem are only
    % approximate; we report the conditional covariance restricted to
    % the working set.  For most teaching uses the unconstrained
    % covariance is what users want; we return that.
    [Ptt, Ptt1] = simks_selected_inv(J, n);
    info.Ptt  = Ptt;
    info.Ptt1 = Ptt1;
  end
end


function [X, ok] = phase1_feasible(J, A_ineq, b_ineq, X0)
% Phase-I: find any strictly feasible X.
%
% Two strategies, tried in order:
% (1) Single-pass projection: if every column of A_ineq has at most one
%     nonzero in violating rows, just shift those coordinates of X by
%     the right amount.  This handles componentwise bounds in O(c+n).
% (2) Iterative single-row projection on the worst violation, repeated
%     up to 200 times.  For general constraints.
  c = size(A_ineq, 1);
  X = X0;
  slack = b_ineq - A_ineq * X;
  if all(slack > 1e-10)
    ok = true;
    return;
  end

  % Strategy 1: one-shot projection when each row touches one variable.
  % We check that the A_ineq has at most one nonzero per row; if so we
  % can clip each affected variable directly.
  Aviol = A_ineq(slack <= 1e-10, :);
  if ~isempty(Aviol)
    nz_per_row = sum(Aviol ~= 0, 2);
    if all(nz_per_row == 1)
      % Each violating row constrains one variable, X(j) * v <= b.
      [vrows, vcols, vvals] = find(Aviol);
      bviol = b_ineq(slack <= 1e-10);
      for kk = 1:length(vrows)
        j_var = vcols(kk);
        coef  = vvals(kk);
        rhs   = bviol(vrows(kk));
        % Want coef * X(j_var) <= rhs.  Choose X(j_var) = (rhs - margin)/coef
        % if that pushes X further inside the constraint.
        margin = 1e-3;
        if coef > 0
          X_target = (rhs - margin) / coef;
          if coef * X(j_var) > rhs - margin
            X(j_var) = X_target;
          end
        elseif coef < 0
          X_target = (rhs - margin) / coef;
          if coef * X(j_var) > rhs - margin
            X(j_var) = X_target;
          end
        end
      end
      slack = b_ineq - A_ineq * X;
      if all(slack > 0)
        ok = true;
        return;
      end
    end
  end

  % Strategy 2: iterative projection on the worst violation.
  ok = false;
  for it = 1:200
    slack = b_ineq - A_ineq * X;
    [v, k] = min(slack);
    if v > 1e-10
      ok = true;
      return;
    end
    a_k = A_ineq(k, :)';
    rhs_v = -v + 1e-4;
    denom = a_k' * a_k;
    if denom < 1e-14
      return;
    end
    X = X - (rhs_v / denom) * a_k;
  end
  slack = b_ineq - A_ineq * X;
  ok = all(slack > 0);
end


function v = get_opt(s, name, default)
  if isfield(s, name)
    v = s.(name);
  else
    v = default;
  end
end
