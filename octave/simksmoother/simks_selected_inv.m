function [Ptt, Ptt1, logdetJ] = simks_selected_inv(J, n)
% SIMKS_SELECTED_INV  Tridiagonal blocks of J^{-1} for block-tridiagonal SPD J.
%
% Inputs:
%   J : (T+1)n x (T+1)n sparse SPD block tridiagonal, block size n.
%   n : block size.
%
% Outputs:
%   Ptt   : 1 x (T+1) cell, Ptt{t+1}  = (J^{-1})_{tt}   for t = 0,...,T.
%   Ptt1  : 1 x T     cell, Ptt1{t}   = (J^{-1})_{t,t-1} for t = 1,...,T.
%   logdetJ : scalar, log det J (free byproduct of the LDL sweep).
%
% Method (Takahashi / Erisman-Tinney).  Factor J = L D L' where L is unit
% lower block-bidiagonal with L_{t+1,t} = M_{t+1}, and D is block-diagonal.
%
%   Forward sweep:
%     D_0       = J_{00}
%     M_{t+1}   = J_{t+1,t} D_t^{-1}
%     D_{t+1}   = J_{t+1,t+1} - M_{t+1} D_t M_{t+1}'      (= Schur complement)
%
%   Backward sweep (Takahashi):
%     Sigma_{T,T}   = D_T^{-1}
%     Sigma_{t+1,t} = -Sigma_{t+1,t+1} M_{t+1}
%     Sigma_{t,t}   = D_t^{-1} + M_{t+1}' Sigma_{t+1,t+1} M_{t+1}
%
% Interpretation:
%   D_t            = P_{t|t}^{-1}   (filtered precision)
%   -M_{t+1}       = the negative analogue of the RTS smoother gain
%   the backward sweep is the RTS smoother covariance recursion in
%   information form.

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

  N = size(J, 1);
  T = N / n - 1;
  assert(mod(N, n) == 0, 'J size must be divisible by n');

  idx = @(t) (t*n+1):((t+1)*n);

  % --- Forward block LDL^T sweep ---
  D = cell(1, T+1);
  M = cell(1, T);   % M{t} = L_{t, t-1}, for t = 1,...,T
  D{1} = full(J(idx(0), idx(0)));
  D{1} = 0.5 * (D{1} + D{1}');
  logdetJ = log(abs(det(D{1})));
  for t = 1:T
    Jtm = full(J(idx(t),   idx(t-1)));
    Jtt = full(J(idx(t),   idx(t)));
    M{t}   = Jtm / D{t};
    D{t+1} = Jtt - M{t} * D{t} * M{t}';
    D{t+1} = 0.5 * (D{t+1} + D{t+1}');
    logdetJ = logdetJ + log(abs(det(D{t+1})));
  end

  % --- Backward Takahashi sweep ---
  Ptt  = cell(1, T+1);
  Ptt1 = cell(1, T);
  Ptt{T+1} = inv(D{T+1});
  Ptt{T+1} = 0.5 * (Ptt{T+1} + Ptt{T+1}');
  for t = T-1:-1:0
    Stp1 = Ptt{t+2};
    Mtp1 = M{t+1};                                % L_{t+1, t}
    Sigma_tp1_t = -Stp1 * Mtp1;                   % = Sigma_{t+1, t}
    Ptt1{t+1} = Sigma_tp1_t;                      % store Sigma_{t+1, t}
    Ptt{t+1}  = inv(D{t+1}) + Mtp1' * Stp1 * Mtp1;
    Ptt{t+1}  = 0.5 * (Ptt{t+1} + Ptt{t+1}');
  end
end
