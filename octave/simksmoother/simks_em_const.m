function [params, Xhat, hist] = simks_em_const(Y, n, opts)
% SIMKS_EM_CONST  Gaussian EM for constant linear state-space systems.
%
% Iterates between an E-step (simultaneous smoother + selected inversion)
% and an M-step that uses the full smoothed sufficient statistics.  This is
% the correct EM; the M-step formulas account for posterior covariance of X,
% so Q-hat and R-hat are no longer biased downward as in alternating
% regression.
%
% Optional exogenous inputs z_t enter both the state and the observation
% equation: x_t = F x_{t-1} + B z_t + w_t, y_t = H x_t + D z_t + v_t.
% When Z is supplied, the M-step jointly estimates [F B] and [H D] by
% regressing on the augmented regressor [x_{t-1}; z_t] and [x_t; z_t]
% respectively.
%
% E-step (per iteration):
%   Xhat       = E[X | Y]                    -- from simks_smooth_const
%   P_{t|T}    = Cov(x_t | Y)                -- block (t,t) of J^{-1}
%   P_{t,t-1|T}= Cov(x_t, x_{t-1} | Y)       -- block (t,t-1) of J^{-1}
%
% Sufficient statistics (sums over t = 1..T):
%   S00 = sum E[x_{t-1} x_{t-1}']
%   S11 = sum E[x_t x_t']
%   S10 = sum E[x_t x_{t-1}']
% Additionally with exogenous Z:
%   S0z = sum E[x_{t-1}] z_t'
%   S1z = sum E[x_t]     z_t'
%   Syz = sum y_t z_t',  Szz = sum z_t z_t'
%
% M-step without exogenous:
%   F = S10 / S00
%   H = Syx / Sxx
%   Q = (S11 - F S10') / T
%   R = (Syy - H Syx') / T
% M-step with exogenous (joint over (F,B) and (H,D)):
%   [F B] = S10_aug / S00_aug,   where the augmented stats stack S00 with
%   S0z, S0z', Szz, etc.  See code for the explicit formulas.
%
% Inputs:
%   Y   : m x T observations (NaNs allowed; see note below)
%   n   : state dimension
%   opts: optional struct
%         .max_iter (default 200), .tol (1e-6), .jitter (1e-8),
%         .F, .H, .Q, .R, .a0, .P0 : initializations,
%         .verbose (false),
%         .Z (k x T): exogenous inputs (optional),
%         .B, .D : initializations for exogenous loadings.
%
% Output:
%   params : struct with F, H, Q, R, a0, P0 (and B, D if exogenous)
%   Xhat   : final smoothed states, n x (T+1)
%   hist   : marginal log-likelihood history (should be monotone non-decreasing)
%
% Note on missing data: simks_smooth_const already handles NaN observations;
% the M-step formulas for H and R involve sums over only the periods where
% y_t (or rows of it) are observed, which the current implementation does
% in the simplest possible way -- it skips periods where the whole y_t is
% missing.  Partial missingness in y_t is not used to update H here; for
% that, one needs to accumulate per-coordinate sums.

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

  [m,T] = size(Y);
  if nargin < 3, opts = struct(); end

  max_iter = get_opt(opts, 'max_iter', 200);
  tol      = get_opt(opts, 'tol', 1e-6);
  jitter   = get_opt(opts, 'jitter', 1e-8);
  verbose  = get_opt(opts, 'verbose', false);

  % Exogenous inputs.
  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    k = size(Z, 1);
    assert(size(Z, 2) == T, 'opts.Z must be k x T');
    B = get_opt(opts, 'B', zeros(n, k));
    D = get_opt(opts, 'D', zeros(m, k));
    Szz = Z * Z';   % deterministic, computed once
  end

  % Crude defaults.
  F = get_opt(opts, 'F', 0.8 * eye(n));
  if isfield(opts, 'H')
    H = opts.H;
  else
    C = cov(Y(:, all(~isnan(Y),1))');
    [V,D_eig] = eig(C);
    [~,ord] = sort(diag(D_eig), 'descend');
    V = V(:,ord);
    H = zeros(m,n);
    H(:,1:min(m,n)) = V(:,1:min(m,n));
  end
  Q  = get_opt(opts, 'Q', eye(n));
  R  = get_opt(opts, 'R', 0.2 * eye(m));
  a0 = get_opt(opts, 'a0', zeros(n,1));
  P0 = get_opt(opts, 'P0', 10 * eye(n));

  obs_periods = find(any(~isnan(Y),1));    % at least one row observed
  fully_obs   = find(all(~isnan(Y),1));    % no missing rows in y_t

  hist = zeros(max_iter, 1);

  for it = 1:max_iter

    % ---------- E step ----------
    sopts.return_cov = true;
    if have_exog
      sopts.Z = Z;  sopts.B = B;  sopts.D = D;
    end
    [Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0, sopts);
    Ptt  = info.Ptt;     % cell 1 x (T+1)
    Ptt1 = info.Ptt1;    % cell 1 x T

    % Sufficient statistics with posterior-covariance corrections.
    S00 = zeros(n,n);
    S11 = zeros(n,n);
    S10 = zeros(n,n);
    if have_exog
      S0z = zeros(n,k);
      S1z = zeros(n,k);
    end
    for t = 1:T
      xtm = Xhat(:, t);            % column index t in Octave = x_{t-1}
      xt  = Xhat(:, t+1);          % = x_t
      Ptm = Ptt{t};                % P_{t-1|T}
      Pt  = Ptt{t+1};              % P_{t|T}
      Pcr = Ptt1{t};               % P_{t,t-1|T}

      S00 = S00 + xtm * xtm' + Ptm;
      S11 = S11 + xt  * xt'  + Pt;
      S10 = S10 + xt  * xtm' + Pcr;
      if have_exog
        S0z = S0z + xtm * Z(:,t)';
        S1z = S1z + xt  * Z(:,t)';
      end
    end

    % Observation sums use only periods where y_t is fully observed.
    Syx = zeros(m,n);
    Sxx = zeros(n,n);
    Syy = zeros(m,m);
    if have_exog
      Syz = zeros(m,k);
      Sxz = zeros(n,k);
      Szz_obs = zeros(k,k);
    end
    for t = fully_obs
      xt = Xhat(:, t+1);
      Pt = Ptt{t+1};
      Sxx = Sxx + xt * xt' + Pt;
      Syx = Syx + Y(:,t) * xt';
      Syy = Syy + Y(:,t) * Y(:,t)';
      if have_exog
        zt = Z(:,t);
        Syz = Syz + Y(:,t) * zt';
        Sxz = Sxz + xt * zt';
        Szz_obs = Szz_obs + zt * zt';
      end
    end
    Tobs = numel(fully_obs);

    % ---------- M step ----------
    if have_exog
      % Joint update of [F B] by regressing x_t on [x_{t-1}; z_t].
      M_aug = [S00,      S0z;
               S0z',     Szz];
      RHS_a = [S10, S1z];
      FB = RHS_a / (M_aug + jitter * eye(n+k));
      F_new = FB(:, 1:n);
      B_new = FB(:, n+1:end);
      % Q update with cross-term correction.
      Q_new = (S11 - FB * RHS_a') / T;
      Q_new = nearest_spd(Q_new, jitter);
    else
      F_new = S10 / (S00 + jitter * eye(n));
      Q_new = (S11 - F_new * S10') / T;
      Q_new = nearest_spd(Q_new, jitter);
      B_new = [];
    end

    if Tobs > 0
      if have_exog
        M_aug_obs = [Sxx,       Sxz;
                     Sxz',      Szz_obs];
        RHS_b = [Syx, Syz];
        HD = RHS_b / (M_aug_obs + jitter * eye(n+k));
        H_new = HD(:, 1:n);
        D_new = HD(:, n+1:end);
        R_new = (Syy - HD * RHS_b') / Tobs;
        R_new = nearest_spd(R_new, jitter);
      else
        H_new = Syx / (Sxx + jitter * eye(n));
        R_new = (Syy - H_new * Syx') / Tobs;
        R_new = nearest_spd(R_new, jitter);
        D_new = [];
      end
    else
      H_new = H;  R_new = R;
      if have_exog, D_new = D; end
    end

    % Initial state: posterior mean and covariance at t = 0.
    a0_new = Xhat(:,1);
    P0_new = nearest_spd(Ptt{1}, jitter);

    F = F_new;  H = H_new;  Q = Q_new;  R = R_new;
    a0 = a0_new;  P0 = P0_new;
    if have_exog
      B = B_new;  D = D_new;
    end

    % Track marginal log-likelihood for monitoring.
    ll_opts = struct();
    if have_exog
      ll_opts.Z = Z;  ll_opts.B = B;  ll_opts.D = D;
    end
    hist(it) = simks_marginal_loglik(Y, F, H, Q, R, a0, P0, ll_opts);
    if verbose
      fprintf('iter %3d  loglik = %.6f\n', it, hist(it));
    end

    if it > 1
      rel = abs(hist(it) - hist(it-1)) / (1 + abs(hist(it-1)));
      if rel < tol
        hist = hist(1:it);
        break;
      end
    end
  end

  params = struct('F',F,'H',H,'Q',Q,'R',R,'a0',a0,'P0',P0);
  if have_exog
    params.B = B;
    params.D = D;
  end
end

function val = get_opt(s, name, default)
  if isfield(s, name)
    val = s.(name);
  else
    val = default;
  end
end
