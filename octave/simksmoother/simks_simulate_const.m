function [Y, X] = simks_simulate_const(T, F, H, Q, R, a0, P0, opts)
% SIMKS_SIMULATE_CONST  Simulate a constant linear Gaussian state-space system.
%
%   x_0 ~ N(a0, P0)
%   x_t = F x_{t-1} + B z_t + w_t,    w_t ~ N(0, Q)
%   y_t = H x_t       + D z_t + v_t,  v_t ~ N(0, R)
%
% Exogenous inputs are optional.  Pass opts.Z (k x T) and optionally
% opts.B (n x k, default zeros) and opts.D (m x k, default zeros).
%
% Returns Y (m x T) and X (n x (T+1), columns x_0,...,x_T).

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

  n = size(F, 1);
  m = size(H, 1);

  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    assert(size(Z, 2) == T, 'opts.Z must be k x T');
    k = size(Z, 1);
    if isfield(opts, 'B') && ~isempty(opts.B)
      B = opts.B;
    else
      B = zeros(n, k);
    end
    if isfield(opts, 'D') && ~isempty(opts.D)
      D = opts.D;
    else
      D = zeros(m, k);
    end
  end

  X = zeros(n, T+1);
  Y = zeros(m, T);

  CP0 = chol(P0, 'lower');
  CQ  = chol(Q,  'lower');
  CR  = chol(R,  'lower');

  X(:,1) = a0 + CP0 * randn(n,1);
  for t = 1:T
    if have_exog
      X(:,t+1) = F * X(:,t) + B * Z(:,t) + CQ * randn(n,1);
      Y(:,t)   = H * X(:,t+1) + D * Z(:,t) + CR * randn(m,1);
    else
      X(:,t+1) = F * X(:,t) + CQ * randn(n,1);
      Y(:,t)   = H * X(:,t+1) + CR * randn(m,1);
    end
  end
end
