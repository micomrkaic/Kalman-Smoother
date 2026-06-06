function Ahat = nearest_spd(A, jitter)
% NEAREST_SPD  Symmetric positive-definite cleanup for covariance matrices.
% Symmetrizes A, clips eigenvalues from below at `jitter`, and reconstructs.
% Cheap and stable; used after M-step covariance updates to guard against
% roundoff producing eigenvalues just below zero.

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

  if nargin < 2
    jitter = 1e-8;
  end

  A = 0.5 * (A + A');
  [V,D] = eig(A);
  d = diag(D);
  d = max(d, jitter);
  Ahat = V * diag(d) * V';
  Ahat = 0.5 * (Ahat + Ahat');
end
