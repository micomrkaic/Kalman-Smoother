function obj = simks_objective_const(Y, X, F, H, Q, R, a0, P0)
% SIMKS_OBJECTIVE_CONST  Negative joint log-posterior kernel at fixed X.
%
% Evaluates -2 * (joint log p(Y, X; theta) up to constants) for diagnostics:
%
%   obj = (x_0 - a_0)' P_0^{-1} (x_0 - a_0) + log det P_0
%       + sum_t [ (x_t - F x_{t-1})' Q^{-1} (x_t - F x_{t-1}) + log det Q ]
%       + sum_t [ (y_t - H x_t)'   R^{-1} (y_t - H x_t)       + log det R ]
%
% This is NOT the marginal data log-likelihood log p(Y; theta).  For that,
% use simks_marginal_loglik.  This routine is for monitoring the inner
% smoother solve, which is exactly the quadratic minimizer of the X-part.

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

  [m, T] = size(Y);
  n = size(F, 1);

  Qi  = inv(Q);
  Ri  = inv(R);
  P0i = inv(P0);

  e0  = X(:,1) - a0;
  obj = e0' * P0i * e0 + log(det(P0));

  for t = 1:T
    ew = X(:,t+1) - F * X(:,t);
    obj = obj + ew' * Qi * ew + log(det(Q));
    yt = Y(:,t);
    mask = ~isnan(yt);
    if all(mask)
      ev = yt - H * X(:,t+1);
      obj = obj + ev' * Ri * ev + log(det(R));
    elseif any(mask)
      Hm = H(mask,:);
      Rm = R(mask,mask);
      ev = yt(mask) - Hm * X(:,t+1);
      obj = obj + ev' * inv(Rm) * ev + log(det(Rm));
    end
  end
end
