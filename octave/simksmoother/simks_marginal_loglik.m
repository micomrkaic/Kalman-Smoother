function [ll, parts] = simks_marginal_loglik(Y, F, H, Q, R, a0, P0, opts)
% SIMKS_MARGINAL_LOGLIK  Gaussian marginal log-likelihood log p(Y; theta).
%
% Computes the data log-likelihood by integrating X out of the joint
% p(Y, X; theta), using the simultaneous formulation.  Derivation:
%
%   log p(Y, X) = -1/2 X' J X + h' X + g(Y)
% where g(Y) collects all X-independent terms.  Integrating out X gives
%   log p(Y) = g(Y) + 1/2 h' J^{-1} h + (N/2) log 2*pi - 1/2 log det J
%
% Closed form (constant-params, all observations present, proper prior):
%   log p(Y) = -1/2 [ a0' P0^{-1} a0 + sum_t ytilde_t' R^{-1} ytilde_t - h' Xhat ]
%             -1/2 [ log det P0 + T log det Q + T log det R + log det J ]
%             -(m T / 2) log(2*pi)
% where ytilde_t = y_t - D z_t if exogenous inputs are supplied.
%
% For the diffuse case (P0 = []), the prior contribution is omitted; the
% result is the diffuse log-likelihood (Koopman) up to additive constants.
%
% Inputs / outputs as in simks_smooth_const.  opts may contain Z, B, D
% for exogenous inputs (see simks_smooth_const).  parts struct breaks
% the contributions for diagnostics.

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

  if nargin < 8, opts = struct(); end

  [m, T] = size(Y);
  n = size(F, 1);
  diffuse = isempty(P0);

  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    if isfield(opts, 'B') && ~isempty(opts.B)
      Bx = opts.B;
    else
      Bx = zeros(n, size(Z, 1));
    end
    if isfield(opts, 'D') && ~isempty(opts.D)
      D = opts.D;
    else
      D = zeros(m, size(Z, 1));
    end
  end

  % Run the smoother to get Xhat, J, h.
  [Xhat, J, h, ~] = simks_smooth_const(Y, F, H, Q, R, a0, P0, opts);
  Xvec = Xhat(:);

  % log det J from a sparse Cholesky -- O(T n^3) on band-tridiagonal J.
  Lc = chol(J, 'lower');           % SPD sparse chol
  logdet_J = 2 * sum(log(diag(Lc)));

  % Quadratic terms.  With exogenous input D z_t, the measurement
  % residual is ytilde_t = y_t - D z_t and the constant in the joint
  % density depends on ytilde rather than y.
  yRy  = 0;
  Ri   = inv(R);
  for t = 1:T
    if have_exog
      ytilde = Y(:,t) - D * Z(:,t);
    else
      ytilde = Y(:,t);
    end
    yRy = yRy + ytilde' * Ri * ytilde;
  end

  % Exogenous state forcing contributes to the constant of the completed
  % square: c_W' Omega^{-1} c_W = a0' P0^{-1} a0 + sum_t (B z_t)' Q^{-1} (B z_t).
  bQb = 0;
  if have_exog
    Qi_const = inv(Q);
    for t = 1:T
      bz  = Bx * Z(:, t);
      bQb = bQb + bz' * Qi_const * bz;
    end
  end

  if diffuse
    a_pri = 0;
    logdet_P0 = 0;
  else
    P0i   = inv(P0);
    a_pri = a0' * P0i * a0;
    logdet_P0 = log(det(P0));
  end

  hX = h' * Xvec;

  logdet_Q = log(det(Q));
  logdet_R = log(det(R));

  ll = -0.5 * (a_pri + bQb + yRy - hX) ...
       -0.5 * (logdet_P0 + T*logdet_Q + T*logdet_R + logdet_J) ...
       -0.5 * (m*T) * log(2*pi);

  parts = struct( ...
    'quad_prior',  -0.5*a_pri, ...
    'quad_exog',   -0.5*bQb, ...
    'quad_obs',    -0.5*yRy, ...
    'quad_hXhat',  +0.5*hX, ...
    'logdet_P0',   -0.5*logdet_P0, ...
    'logdet_Q',    -0.5*T*logdet_Q, ...
    'logdet_R',    -0.5*T*logdet_R, ...
    'logdet_J',    -0.5*logdet_J, ...
    'const',       -0.5*m*T*log(2*pi), ...
    'diffuse',     diffuse, ...
    'have_exog',   have_exog);
end
