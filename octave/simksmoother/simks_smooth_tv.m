function [Xhat, J, h, info] = simks_smooth_tv(Y, Fseq, Hseq, Qseq, Rseq, a0, P0, opts)
% SIMKS_SMOOTH_TV  Simultaneous Kalman smoother with time-varying matrices.
%
% Model:
%   x_t = F_t x_{t-1} + B_t z_t + w_t,   w_t ~ N(0, Q_t)
%   y_t = H_t x_t       + D_t z_t + v_t, v_t ~ N(0, R_t)
%   x_0 ~ N(a0, P0)                       (or diffuse: P0 = [])
%
% The exogenous input z_t is optional; if omitted, the model collapses
% to x_t = F_t x_{t-1} + w_t and y_t = H_t x_t + v_t.
%
% Inputs:
%   Y     : m x T observations. NaN entries are treated as missing.
%   Fseq  : 1xT cell with Fseq{t} = n x n, OR a single n x n matrix
%           (in which case F_t = Fseq for all t).
%   Hseq  : 1xT cell with Hseq{t} = m x n, OR a single m x n matrix.
%   Qseq  : 1xT cell with Qseq{t} = n x n, OR a single n x n matrix.
%   Rseq  : 1xT cell with Rseq{t} = m x m, OR a single m x m matrix.
%   a0,P0 : initial mean and covariance (P0 = [] for diffuse).
%   opts  : .return_cov  (default false), see simks_smooth_const.
%           .Z           : k x T exogenous inputs (default: none).
%           .Bseq        : 1xT cell of n x k OR single n x k matrix
%                          (default: zeros if Z given).
%           .Dseq        : 1xT cell of m x k OR single m x k matrix
%                          (default: zeros if Z given).

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
  n = size(a0, 1);

  get_F = make_getter(Fseq, T, 'Fseq', [n n]);
  get_H = make_getter(Hseq, T, 'Hseq', [m n]);
  get_Q = make_getter(Qseq, T, 'Qseq', [n n]);
  get_R = make_getter(Rseq, T, 'Rseq', [m m]);

  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    k = size(Z, 1);
    assert(size(Z, 2) == T, 'Z must be k x T');
    if isfield(opts, 'Bseq') && ~isempty(opts.Bseq)
      get_B = make_getter(opts.Bseq, T, 'Bseq', [n k]);
    else
      Bzero = zeros(n, k);
      get_B = @(t) Bzero;
    end
    if isfield(opts, 'Dseq') && ~isempty(opts.Dseq)
      get_D = make_getter(opts.Dseq, T, 'Dseq', [m k]);
    else
      Dzero = zeros(m, k);
      get_D = @(t) Dzero;
    end
  end

  diffuse = isempty(P0);
  if diffuse
    P0i = zeros(n,n);
  else
    P0i = inv(P0);
  end

  N = (T+1)*n;
  idx = @(t) (t*n+1):((t+1)*n);

  % Triplet assembly.  Indices for all blocks are precomputed with
  % implicit expansion; the per-period loops only fill value columns of
  % preallocated arrays (matrix inverses must be computed per period
  % anyway).  An earlier version called a nested add_block helper with
  % ndgrid per block; in Octave/MATLAB that per-call overhead dominated
  % runtime for small n.
  [bi, bj] = ndgrid(1:n, 1:n);
  bi = bi(:);  bj = bj(:);             % n^2 x 1 within-block offsets
  off = n * (0:T-1);                   % 1 x T

  V_dprev = zeros(n*n, T);             % F_t' Qti F_t   at (t-1, t-1)
  V_dcur  = zeros(n*n, T);             % Qti            at (t,   t)
  V_oUp   = zeros(n*n, T);             % -F_t' Qti      at (t-1, t)
  V_oLo   = zeros(n*n, T);             % -Qti F_t       at (t,   t-1)
  V_meas  = zeros(n*n, T);             % measurement information at (t, t)

  h = zeros(N, 1);

  for t = 1:T
    Ft  = get_F(t);
    Qti = inv(get_Q(t));
    FtQti = Ft' * Qti;
    V_dprev(:,t) = reshape(FtQti * Ft, [], 1);
    V_dcur(:,t)  = Qti(:);
    Mup          = -FtQti;
    V_oUp(:,t)   = Mup(:);
    Mlo          = -Qti * Ft;
    V_oLo(:,t)   = Mlo(:);

    if have_exog
      Bz = get_B(t) * Z(:,t);
      h(idx(t)) = h(idx(t)) + Qti * Bz;
      h(idx(t-1)) = h(idx(t-1)) - FtQti * Bz;
    end
  end

  for t = 1:T
    yt = Y(:,t);
    Ht = get_H(t);
    Rt = get_R(t);
    if have_exog
      ytilde = yt - get_D(t) * Z(:,t);
    else
      ytilde = yt;
    end
    mask = ~isnan(yt);
    if all(mask)
      Rti = inv(Rt);
      HtRti = Ht' * Rti;
      V_meas(:,t) = reshape(HtRti * Ht, [], 1);
      h(idx(t)) = h(idx(t)) + HtRti * ytilde;
    elseif any(mask)
      Hm   = Ht(mask,:);
      Rm_i = inv(Rt(mask,mask));
      HmRm = Hm' * Rm_i;
      V_meas(:,t) = reshape(HmRm * Hm, [], 1);
      h(idx(t)) = h(idx(t)) + HmRm * ytilde(mask);
    end
    % all-missing periods leave V_meas(:,t) at zero: no information.
  end

  rows_d  = bi + off;       cols_d  = bj + off;        % blocks (t-1, t-1)
  rows_d1 = bi + n + off;   cols_d1 = bj + n + off;    % blocks (t,   t)

  ii = [rows_d(:); rows_d1(:); rows_d(:);  rows_d1(:); rows_d1(:)];
  jj = [cols_d(:); cols_d1(:); cols_d1(:); cols_d(:);  cols_d1(:)];
  vv = [V_dprev(:); V_dcur(:); V_oUp(:);   V_oLo(:);   V_meas(:)];

  if ~diffuse
    ii = [ii; bi];  jj = [jj; bj];  vv = [vv; P0i(:)];
    h(idx(0)) = h(idx(0)) + P0i * a0;
  end

  J = sparse(ii, jj, vv, N, N);
  J = (J + J') / 2;

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

function fn = make_getter(X, T, name, expected_size)
% Accept either a 1xT cell of matrices or one matrix (constant).
  if iscell(X)
    assert(numel(X) == T, sprintf('%s must have T = %d entries', name, T));
    fn = @(t) X{t};
  else
    assert(all(size(X) == expected_size), ...
           sprintf('%s constant must be %dx%d', name, expected_size(1), expected_size(2)));
    fn = @(t) X;
  end
end
