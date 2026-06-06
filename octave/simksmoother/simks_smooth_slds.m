function [Xhat, info] = simks_smooth_slds(Y, F_set, H, Q_set, R, a0, P0, opts)
% SIMKS_SMOOTH_SLDS  Switching Linear Dynamical System smoother.
%
% Model:
%   s_t in {1,...,K} is a discrete regime variable with transition
%   probabilities Pr(s_t = j | s_{t-1} = i) = A(i,j).
%   x_t = F_{s_t} x_{t-1} + B_{s_t} z_t + w_t,    w_t ~ N(0, Q_{s_t})
%   y_t = H x_t + D z_t + v_t,                    v_t ~ N(0, R)
%
% Currently the measurement matrices H and R, and the loadings D, are
% regime-independent.  Regime-dependent H, R, D are a straightforward
% extension; the demo focuses on regime-switching dynamics, which is
% the common case in macro (regimes affect persistence or volatility,
% but observation is the same).
%
% Inference is by variational EM with the factorization
% q(X, S) = q(X) prod_t q(s_t), using EXACT coordinate updates:
%   1. q(X) step: q(X) is Gaussian with precision assembled from
%      responsibility-weighted EXPECTED quadratic transition terms.  The
%      per-period blocks involve sum_k pi_{t,k} Q_k^{-1},
%      sum_k pi_{t,k} Q_k^{-1} F_k, and sum_k pi_{t,k} F_k' Q_k^{-1} F_k
%      (NOT the smoother run at averaged matrices Fbar_t, Qbar_t, which
%      would be a moment-matching approximation rather than the
%      variational update).  The precision matrix is still block
%      tridiagonal at every iteration, so the cost is unchanged.
%   2. q(S) step: forward-backward on the regime HMM with EXPECTED
%      Gaussian transition log-likelihoods under q(X), i.e. including
%      the posterior covariance corrections built from Sigma_{tt},
%      Sigma_{t-1,t-1}, and the lag-one block Sigma_{t,t-1} delivered
%      by selected inversion --- not the density evaluated at the
%      smoothed point path alone.
%
% This is a single-Gaussian-mixture posterior approximation; the exact
% posterior is a Gaussian mixture of K^T components, intractable for any
% T > 20.  Variational EM is the standard scalable approach (see Ghahramani
% and Hinton 2000 for the canonical reference).
%
% Inputs:
%   Y     : m x T observations (NaN entries allowed)
%   F_set : 1 x K cell of n x n transition matrices, one per regime
%   H     : m x n observation matrix (regime-invariant for now)
%   Q_set : 1 x K cell of n x n state innovation covariances
%   R     : m x m measurement covariance
%   a0,P0 : initial mean and covariance
%   opts  : optional struct
%           .A           : K x K regime transition matrix (rows sum to 1).
%                          Default: 0.95 diagonal, uniform off-diagonal.
%           .pi0         : K x 1 initial regime distribution (default uniform).
%           .B_set       : 1 x K cell of n x k state-loading (default zeros).
%           .Z           : k x T exogenous inputs (optional).
%           .D           : m x k measurement loading (optional, regime-invariant).
%           .max_iter    : default 30.
%           .tol         : convergence on max |Delta pi| (default 1e-5).
%           .verbose     : default false.
%
% Outputs:
%   Xhat : n x (T+1) smoothed state path
%   info : struct
%     .pi        : K x T smoothed regime responsibilities (column t = pi_{t,:})
%     .pi0_post  : K x 1 posterior over initial regime
%     .Ptt, .Ptt1: posterior covariance blocks of q(X) at convergence
%     .iters
%     .converged
%     .delta_history
%
% References:
%   Ghahramani, Z., and Hinton, G. E. (2000).  Variational learning for
%   switching state-space models.  Neural Computation, 12, 831-864.

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
  n      = size(F_set{1}, 1);
  K      = numel(F_set);

  % --- options ---
  if isfield(opts, 'A') && ~isempty(opts.A)
    A = opts.A;
  else
    A = 0.95 * eye(K) + (0.05/(K-1)) * (ones(K) - eye(K));
  end
  if isfield(opts, 'pi0') && ~isempty(opts.pi0)
    pi0 = opts.pi0(:);
  else
    pi0 = ones(K,1) / K;
  end

  have_exog = isfield(opts, 'Z') && ~isempty(opts.Z);
  if have_exog
    Z = opts.Z;
    kZ = size(Z, 1);
    if isfield(opts, 'B_set') && ~isempty(opts.B_set)
      B_set = opts.B_set;
    else
      B_set = cell(1,K);
      for k = 1:K, B_set{k} = zeros(n, kZ); end
    end
    if isfield(opts, 'D') && ~isempty(opts.D), D = opts.D; else, D = zeros(m, kZ); end
  else
    B_set = cell(1,K);
    for k = 1:K, B_set{k} = zeros(n, 0); end
  end

  max_iter = getfield_def(opts, 'max_iter', 30);
  tol      = getfield_def(opts, 'tol',      1e-5);
  verbose  = getfield_def(opts, 'verbose',  false);

  % --- initialize responsibilities uniformly ---
  pi_r = ones(K, T) / K;
  pi_prev = pi_r;
  delta_history = zeros(1, max_iter);
  converged = false;
  iters_done = 0;

  % --- per-regime precomputations for the expected quadratic forms ---
  Qk_inv  = cell(1, K);
  QiF     = cell(1, K);   % Q_k^{-1} F_k
  FQiF    = cell(1, K);   % F_k' Q_k^{-1} F_k
  logdetQ = zeros(1, K);
  if have_exog
    QiB  = cell(1, K);    % Q_k^{-1} B_k
    FQiB = cell(1, K);    % F_k' Q_k^{-1} B_k
  end
  for k = 1:K
    Qk_inv{k}  = inv(Q_set{k});
    QiF{k}     = Qk_inv{k} * F_set{k};
    FQiF{k}    = F_set{k}' * Qk_inv{k} * F_set{k};
    logdetQ(k) = 2 * sum(log(diag(chol(Q_set{k}, 'lower'))));
    if have_exog
      QiB{k}  = Qk_inv{k} * B_set{k};
      FQiB{k} = F_set{k}' * Qk_inv{k} * B_set{k};
    end
  end

  P0i = inv(P0);

  % --- per-period measurement contributions (missing entries dropped) ---
  HRH = cell(1, T);   % H_m' R_m^{-1} H_m
  HRy = cell(1, T);   % H_m' R_m^{-1} (y_t^obs - D_m z_t)
  for t = 1:T
    yt   = Y(:, t);
    mask = ~isnan(yt);
    if any(mask)
      Hm   = H(mask, :);
      Rm_i = inv(R(mask, mask));
      resid = yt(mask);
      if have_exog
        dz = D * Z(:, t);
        resid = resid - dz(mask);
      end
      HRH{t} = Hm' * Rm_i * Hm;
      HRy{t} = Hm' * Rm_i * resid;
    else
      HRH{t} = zeros(n, n);
      HRy{t} = zeros(n, 1);
    end
  end

  N    = (T + 1) * n;
  bidx = @(b) (b*n+1):((b+1)*n);    % block b = 0..T  <->  x_b

  Ptt  = [];
  Ptt1 = [];

  for it = 1:max_iter
    iters_done = it;

    % ---- q(X) step: exact assembly from expected quadratic forms ----
    Qbar  = cell(1, T);   % sum_k pi Q_k^{-1}
    QFbar = cell(1, T);   % sum_k pi Q_k^{-1} F_k
    FQFbar = cell(1, T);  % sum_k pi F_k' Q_k^{-1} F_k
    if have_exog
      QBbar  = cell(1, T);
      FQBbar = cell(1, T);
    end
    for t = 1:T
      Qb = zeros(n,n); QFb = zeros(n,n); FQFb = zeros(n,n);
      if have_exog, QBb = zeros(n,kZ); FQBb = zeros(n,kZ); end
      for k = 1:K
        w = pi_r(k,t);
        Qb   = Qb   + w * Qk_inv{k};
        QFb  = QFb  + w * QiF{k};
        FQFb = FQFb + w * FQiF{k};
        if have_exog
          QBb  = QBb  + w * QiB{k};
          FQBb = FQBb + w * FQiB{k};
        end
      end
      Qbar{t} = Qb;  QFbar{t} = QFb;  FQFbar{t} = FQFb;
      if have_exog, QBbar{t} = QBb; FQBbar{t} = FQBb; end
    end

    % Sparse triplet assembly of J and the information vector h.
    nb   = (T + 1) + 2 * T;            % number of blocks
    ii   = zeros(nb * n * n, 1);
    jj   = zeros(nb * n * n, 1);
    vv   = zeros(nb * n * n, 1);
    ptr  = 0;
    h    = zeros(N, 1);

    [rg, cg] = ndgrid(1:n, 1:n);

    for b = 0:T
      Jbb = zeros(n, n);
      if b == 0
        Jbb = Jbb + P0i;
      else
        Jbb = Jbb + Qbar{b} + HRH{b};
      end
      if b < T
        Jbb = Jbb + FQFbar{b+1};       % outgoing transition b -> b+1
      end
      rows = b*n + rg(:);  cols = b*n + cg(:);
      ii(ptr+1:ptr+n*n) = rows;  jj(ptr+1:ptr+n*n) = cols;
      vv(ptr+1:ptr+n*n) = Jbb(:);  ptr = ptr + n*n;
    end
    for b = 1:T
      Joff = -QFbar{b};                % J_{b, b-1}
      rows = b*n + rg(:);      cols = (b-1)*n + cg(:);
      ii(ptr+1:ptr+n*n) = rows;  jj(ptr+1:ptr+n*n) = cols;
      vv(ptr+1:ptr+n*n) = Joff(:);  ptr = ptr + n*n;
      JoffT = Joff';
      rows = (b-1)*n + rg(:);  cols = b*n + cg(:);
      ii(ptr+1:ptr+n*n) = rows;  jj(ptr+1:ptr+n*n) = cols;
      vv(ptr+1:ptr+n*n) = JoffT(:);  ptr = ptr + n*n;
    end
    J = sparse(ii(1:ptr), jj(1:ptr), vv(1:ptr), N, N);
    J = 0.5 * (J + J');

    h(bidx(0)) = P0i * a0;
    if have_exog
      h(bidx(0)) = h(bidx(0)) - FQBbar{1} * Z(:, 1);
    end
    for b = 1:T
      hb = HRy{b};
      if have_exog
        hb = hb + QBbar{b} * Z(:, b);
        if b < T
          hb = hb - FQBbar{b+1} * Z(:, b+1);
        end
      end
      h(bidx(b)) = h(bidx(b)) + hb;
    end

    Xvec = J \ h;
    Xhat = reshape(Xvec, n, T + 1);

    % Posterior covariance blocks for the expected-log-likelihood step.
    [Ptt, Ptt1] = simks_selected_inv(J, n);

    % ---- q(S) step: HMM forward-backward on EXPECTED log-likelihoods ----
    logL = zeros(K, T);
    for t = 1:T
      xt   = Xhat(:, t+1);
      xtm  = Xhat(:, t);
      S_tt = Ptt{t+1};                 % Cov(x_t | Y)
      S_mm = Ptt{t};                   % Cov(x_{t-1} | Y)
      S_tm = Ptt1{t};                  % Cov(x_t, x_{t-1} | Y)
      if have_exog
        zt = Z(:, t);
      end
      for k = 1:K
        Fk   = F_set{k};
        mu_k = Fk * xtm;
        if have_exog
          mu_k = mu_k + B_set{k} * zt;
        end
        d    = xt - mu_k;
        Ctr  = S_tt - Fk * S_tm' - S_tm * Fk' + Fk * S_mm * Fk';
        quad = d' * Qk_inv{k} * d + trace(Qk_inv{k} * Ctr);
        logL(k, t) = -0.5 * quad - 0.5 * logdetQ(k) - 0.5 * n * log(2*pi);
      end
    end

    % Forward-backward in log-space for stability.
    log_alpha = zeros(K, T);   % alpha(k, t) = log Pr(s_t = k, x_{1:t})
    log_alpha(:, 1) = log(pi0 + 1e-300) + logL(:, 1);
    log_A = log(A + 1e-300);
    for t = 2:T
      for j = 1:K
        log_alpha(j, t) = logsumexp(log_alpha(:, t-1) + log_A(:, j)) + logL(j, t);
      end
    end
    log_beta = zeros(K, T);
    log_beta(:, T) = 0;
    for t = T-1:-1:1
      for i = 1:K
        log_beta(i, t) = logsumexp(log_A(i, :)' + logL(:, t+1) + log_beta(:, t+1));
      end
    end

    log_gamma = log_alpha + log_beta;
    pi_new = zeros(K, T);
    for t = 1:T
      pi_new(:, t) = exp(log_gamma(:, t) - logsumexp(log_gamma(:, t)));
    end

    % --- convergence ---
    delta = max(abs(pi_new(:) - pi_prev(:)));
    delta_history(it) = delta;
    if verbose
      fprintf('  slds iter %2d  |delta pi|_inf = %.3e\n', it, delta);
    end
    pi_prev = pi_r;
    pi_r = pi_new;
    if delta < tol && it > 1
      converged = true;
      break;
    end
  end

  delta_history = delta_history(1:iters_done);

  info = struct();
  info.pi            = pi_r;
  info.pi0_post      = exp(log_gamma(:, 1) - logsumexp(log_gamma(:, 1)));
  info.Ptt           = Ptt;
  info.Ptt1          = Ptt1;
  info.iters         = iters_done;
  info.converged     = converged;
  info.delta_history = delta_history;
end


function y = logsumexp(x)
% Numerically stable log-sum-exp.
  m = max(x);
  y = m + log(sum(exp(x - m)));
end


function v = getfield_def(s, name, default)
  if isfield(s, name)
    v = s.(name);
  else
    v = default;
  end
end
