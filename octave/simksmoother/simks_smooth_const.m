function [Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0, opts)
% SIMKS_SMOOTH_CONST  Simultaneous Kalman smoother with constant matrices.
%
% Model:
%   x_t = F x_{t-1} + B z_t + w_t,   w_t ~ N(0,Q),  t = 1,...,T
%   y_t = H x_t       + D z_t + v_t, v_t ~ N(0,R),  t = 1,...,T
%   x_0 ~ N(a0, P0)            (or diffuse: P0 = [])
%
% The exogenous input z_t is optional; if omitted, the model collapses
% to the standard form x_t = F x_{t-1} + w_t and y_t = H x_t + v_t.
%
% Inputs:
%   Y  : m x T observations. NaN entries are treated as missing.
%   F  : n x n transition matrix
%   H  : m x n observation matrix
%   Q  : n x n state innovation covariance
%   R  : m x m measurement covariance
%   a0 : n x 1 initial mean   (ignored if diffuse prior)
%   P0 : n x n initial covariance, or [] for diffuse (P0i = 0)
%   opts (optional struct):
%     .return_cov  : if true, also return blocks of J^{-1} (default: false)
%     .Z           : k x T exogenous inputs.  Omit for no exogenous.
%     .B           : n x k state-loading matrix (default: zeros).
%     .D           : m x k measurement-loading matrix (default: zeros).
%
% Outputs:
%   Xhat : n x (T+1) smoothed states, columns x_0, ..., x_T
%   J    : sparse posterior precision (T+1)n x (T+1)n, block tridiagonal
%   h    : posterior information vector
%   info : struct.  Always has n, m, T, nnz.  If opts.return_cov:
%          info.Ptt   - 1 x (T+1) cell, P_{t|T} = block (t,t) of J^{-1}
%          info.Ptt1  - 1 x T cell, P_{t,t-1|T} = block (t,t-1) of J^{-1}
%
% Pedagogical note:
%   Exogenous inputs leave J entirely unchanged.  They only modify h, by
%   adding a deterministic offset.  The block-tridiagonal structure and
%   the per-iteration cost are identical to the no-exogenous case.  This
%   is the precision-based formulation's clearest payoff: the geometry of
%   the smoothing problem is invariant to deterministic forcing terms.
%
% Implementation notes:
%   - J is assembled by triplet arrays (i, j, v) with one terminal call to
%     sparse(i, j, v, N, N).  Octave's sparse(...) with the 5-arg form
%     accumulates duplicate entries, so off-diagonal contributions from
%     adjacent transitions add correctly.
%   - Missing observations (whole period or partial) are handled by zeroing
%     the relevant rows of H' R^{-1} H and H' R^{-1} y_t at each t.

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
  return_cov = isfield(opts, 'return_cov') && opts.return_cov;

  [m, T] = size(Y);
  n = size(F, 1);

  assert(size(F, 2) == n,                 'F must be n x n');
  assert(size(H, 1) == m && size(H, 2) == n, 'H must be m x n');
  assert(all(size(Q) == [n,n]),           'Q must be n x n');
  assert(all(size(R) == [m,m]),           'R must be m x m');
  assert(all(size(a0) == [n,1]),          'a0 must be n x 1');

  % Exogenous inputs.  Default to no-exogenous if Z is absent or empty.
  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    k = size(Z, 1);
    assert(size(Z, 2) == T, 'Z must be k x T (same T as Y)');
    if isfield(opts, 'B') && ~isempty(opts.B)
      B = opts.B;
      assert(all(size(B) == [n,k]), 'B must be n x k');
    else
      B = zeros(n, k);
    end
    if isfield(opts, 'D') && ~isempty(opts.D)
      D = opts.D;
      assert(all(size(D) == [m,k]), 'D must be m x k');
    else
      D = zeros(m, k);
    end
  end

  diffuse = isempty(P0);
  if diffuse
    P0i = zeros(n,n);   % no prior term in J
  else
    assert(all(size(P0) == [n,n]),       'P0 must be n x n');
    P0i = inv(P0);
  end
  Qi  = inv(Q);
  Ri  = inv(R);

  N = (T+1)*n;
  idx = @(t) (t*n+1):((t+1)*n);    % t = 0,...,T

  % ----- assemble J by triplets ----------------------------------------------
  % The assembly is fully vectorized: with constant (F, H, Q, R) the same
  % four n x n dynamics blocks repeat at regularly spaced offsets, so all
  % triplets are built with broadcasting and ONE call to sparse().
  % (An earlier version looped over t calling a nested add_block helper;
  % in Octave/MATLAB the per-call interpreter overhead on tiny blocks
  % dominated total runtime by an order of magnitude.)
  FtQiF = F' * Qi * F;
  FtQi  = F' * Qi;
  QiF   = Qi  * F;
  HtRiH_full = H' * Ri * H;
  HtRi       = H' * Ri;

  [bi, bj] = ndgrid(1:n, 1:n);
  bi = bi(:);  bj = bj(:);             % n^2 x 1 within-block offsets
  off  = n * (0:T-1);                  % 1 x T block offsets for t = 1..T
  rows_d  = bi + off;                  % n^2 x T (implicit expansion)
  cols_d  = bj + off;
  rows_d1 = bi + n + off;              % shifted one block down/right
  cols_d1 = bj + n + off;

  % Dynamics: each transition x_t = F x_{t-1} + B z_t + w_t contributes
  %   F' Qi F at (t-1, t-1), Qi at (t,t), and -F' Qi / -Qi F off-diagonal.
  ii = [rows_d(:);  rows_d1(:); rows_d(:);  rows_d1(:)];
  jj = [cols_d(:);  cols_d1(:); cols_d1(:); cols_d(:)];
  vv = [repmat( FtQiF(:), T, 1);
        repmat( Qi(:),    T, 1);
        repmat(-FtQi(:),  T, 1);
        repmat(-QiF(:),   T, 1)];

  % Prior on x_0.
  h = zeros(N, 1);
  if ~diffuse
    ii = [ii; bi];  jj = [jj; bj];  vv = [vv; P0i(:)];
    h(idx(0)) = P0i * a0;
  end

  % Exogenous state forcing in h:
  %   h(idx(t))   += Qi * B z_t        (from the +B z_t side)
  %   h(idx(t-1)) += -F' Qi * B z_t    (from the F x_{t-1} side)
  if have_exog
    BZ = B * Z;                                        % n x T
    h(n+1:end)   = h(n+1:end)   + reshape(Qi * BZ, [], 1);
    h(1:end-n)   = h(1:end-n)   - reshape(FtQi * BZ, [], 1);
  end

  % Measurement: handle missing entries.  If row k of y_t is NaN, drop it.
  % With exogenous input D z_t, we replace y_t with the residual
  %   ytilde_t = y_t - D z_t.  Complete-data periods are vectorized; only
  % periods with missing entries take the slow per-period path.
  if have_exog
    Ytilde = Y - D * Z;                                % m x T
  else
    Ytilde = Y;
  end
  miss_t = find(any(isnan(Y), 1));                     % periods with any NaN
  full_t = 1:T;
  full_t(miss_t) = [];

  if ~isempty(full_t)
    off_f  = n * full_t;                               % block offsets idx(t)
    rows_m = bi + off_f;
    cols_m = bj + off_f;
    ii = [ii; rows_m(:)];
    jj = [jj; cols_m(:)];
    vv = [vv; repmat(HtRiH_full(:), numel(full_t), 1)];
    hm   = HtRi * Ytilde(:, full_t);                   % n x numel(full_t)
    hidx = (1:n)' + n * full_t;                        % n x numel(full_t)
    h(hidx(:)) = h(hidx(:)) + hm(:);
  end
  for t = miss_t
    yt   = Y(:,t);
    mask = ~isnan(yt);
    if any(mask)
      Hm   = H(mask,:);
      Rm_i = inv(R(mask,mask));
      ytil = Ytilde(mask, t);
      blk  = Hm' * Rm_i * Hm;
      ii = [ii; bi + n*t];
      jj = [jj; bj + n*t];
      vv = [vv; blk(:)];
      h(idx(t)) = h(idx(t)) + Hm' * Rm_i * ytil;
    end
    % else: no information at time t; that period contributes nothing.
  end

  J = sparse(ii, jj, vv, N, N);   % duplicate (i,j) entries are summed.
  J = (J + J') / 2;               % enforce numerical symmetry.

  % ----- solve ---------------------------------------------------------------
  Xvec = J \ h;

  Xhat = reshape(Xvec, n, T+1);

  info = struct('n',n,'m',m,'T',T,'nnz',nnz(J),'diffuse',diffuse, ...
                'have_exog', have_exog);

  if return_cov
    [Ptt, Ptt1] = simks_selected_inv(J, n);
    info.Ptt  = Ptt;
    info.Ptt1 = Ptt1;
  end
end
