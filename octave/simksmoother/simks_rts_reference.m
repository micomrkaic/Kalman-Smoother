function [Xhat, P] = simks_rts_reference(Y, F, H, Q, R, a0, P0)
% SIMKS_RTS_REFERENCE  Textbook Kalman filter + Rauch-Tung-Striebel smoother.
%
% Provided as a reference implementation for the recursive formulation,
% so that readers can compare it line-by-line to the simultaneous smoother
% and see that the two compute the same object.  No NaN handling here:
% this is the classical algorithm in its simplest form.
%
% Returns Xhat (n x (T+1), columns x_{0|T},...,x_{T|T}) and P (1 x (T+1)
% cell with P{t+1} = Cov(x_t | y_{1:T})).

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

  a_pred = zeros(n, T+1);
  P_pred = cell(1, T+1);
  a_filt = zeros(n, T+1);
  P_filt = cell(1, T+1);

  a_filt(:,1) = a0;
  P_filt{1}   = P0;

  % --- forward Kalman filter ---
  for t = 1:T
    a_pred(:, t+1) = F * a_filt(:, t);
    P_pred{t+1}    = F * P_filt{t} * F' + Q;

    S = H * P_pred{t+1} * H' + R;
    K = P_pred{t+1} * H' / S;
    innov = Y(:,t) - H * a_pred(:, t+1);
    a_filt(:, t+1) = a_pred(:, t+1) + K * innov;
    P_filt{t+1}    = (eye(n) - K*H) * P_pred{t+1};
    P_filt{t+1}    = 0.5 * (P_filt{t+1} + P_filt{t+1}');
  end

  % --- RTS backward smoother ---
  Xhat = a_filt;
  P    = P_filt;
  for t = T-1:-1:0
    Jt = P_filt{t+1} * F' / P_pred{t+2};
    Xhat(:, t+1) = a_filt(:, t+1) + Jt * (Xhat(:, t+2) - a_pred(:, t+2));
    P{t+1}       = P_filt{t+1} + Jt * (P{t+2} - P_pred{t+2}) * Jt';
    P{t+1}       = 0.5 * (P{t+1} + P{t+1}');
  end
end
