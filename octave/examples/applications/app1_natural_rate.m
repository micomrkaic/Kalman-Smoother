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

% APP1_NATURAL_RATE  Estimating the natural rate of interest.
%
% A simplified Holston-Laubach-Williams (HLW) model for the natural rate.
%
% Conceptual structure.  Three latent objects: trend output y*, trend
% growth g, and the output gap ytilde (an AR(2) process).  The natural
% rate r* is tied to trend growth deterministically, r*_t = c g_t,
% under the "Laubach-Williams identification" assumption.  The IS curve
% feeds the output gap with lagged real rate gaps, where the real rate
% gap is r_{t-1} - r*_{t-1}.  Inflation follows a Phillips curve in
% the output gap.
%
% State equations (latent dynamics):
%   y*_t      = y*_{t-1} + g_{t-1} + eps_y*_t                                                       (trend level)
%   g_t       = g_{t-1} + eps_g_t                                                                    (trend growth)
%   ytilde_t  = a1 ytilde_{t-1} + a2 ytilde_{t-2} + (a_r/2)(rgap_{t-1} + rgap_{t-2}) + eps_yt_t      (IS curve)
%
% where rgap_t = r_t - c g_t (real rate gap).  The state augments to
%
%   x_t = (y*_t, g_t, g_{t-1}, ytilde_t, ytilde_{t-1})'  in R^5.
%
% Measurement (two observables):
%   y_t       = y*_t + ytilde_t                                                  (log output)
%   pi_t      = b_pi pi_{t-1} + (1-b_pi) pi_avg_t + b_y ytilde_{t-1} + v_t       (Phillips curve)
%
% Inflation has its own AR(1) structure that we put in the state as well,
% but for simplicity we take inflation as an AR(1) shock onto ytilde_{t-1};
% the b_y ytilde_{t-1} coefficient is what identifies the Phillips slope.
%
% Pedagogical points exercised:
% (1) Multi-equation state-space mapping with state augmentation.
% (2) Exogenous inputs: the observed policy rate enters through B.
% (3) Cross-equation restrictions: r* = c g links measurement and state.
% (4) EM identification: the IS-curve slope a_r is famously near-zero,
%     leading to a "pile-up" problem on sigma_g.  We illustrate this.
% (5) Constrained smoothing: enforce r*_t >= 0, relevant at ZLB.
% (6) Diffuse initialization for the unit-root trend components.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this application file lives.  Works regardless of the user's CWD.
app_dir = fileparts(mfilename('fullpath'));
oct_dir = fullfile(app_dir, '..', '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 7);
rand('seed',  7);

% ----- True parameters (HLW-calibrated, quarterly) -----
T         = 200;          % 50 years of quarterly data
a1        = 1.55;         % IS curve AR(1) on output gap
a2        = -0.6;         % IS curve AR(2) on output gap
a_r       = -0.08;        % IS curve sensitivity to real rate gap (negative: higher rate -> lower gap)
c         = 1.0;          % r* = c * g (one-for-one with trend growth)
b_pi      = 0.6;          % Phillips curve persistence
b_y       = 0.10;         % Phillips curve slope (sacrifice ratio)
sigma_ys  = 0.55;         % sd of innovations to trend output
sigma_g   = 0.08;         % sd of innovations to trend growth (larger: lets r* wander negative)
sigma_yt  = 0.40;         % sd of innovations to output gap
sigma_pi  = 0.80;         % sd of innovations to inflation

% Observed exogenous inputs: the policy rate r_t (we just simulate it as
% an AR(1) around 2.5 with shocks).
r_mean    = 2.5;
phi_r     = 0.92;
sigma_r   = 0.40;
r = zeros(1, T+2);
r(1) = r_mean; r(2) = r_mean;
for t = 3:T+2
  r(t) = r_mean + phi_r*(r(t-1) - r_mean) + sigma_r*randn;
end

% ----- Map to state-space matrices -----
% State: x_t = (y*_t, g_t, g_{t-1}, ytilde_t, ytilde_{t-1})'
%
% Dynamics:
%   y*_t          = y*_{t-1} + g_{t-1} + eps_y*
%   g_t           = g_{t-1} + eps_g
%   g_{t-1}       = g_{t-1}   (lag)
%   ytilde_t      = a1 ytilde_{t-1} + a2 ytilde_{t-2} + (a_r/2)(r_{t-1}+r_{t-2})
%                    - (a_r c/2)(g_{t-1} + g_{t-2}) + eps_yt
%   ytilde_{t-1}  = ytilde_{t-1}  (lag)
%
% In x_{t-1} = (y*_{t-1}, g_{t-1}, g_{t-2}, ytilde_{t-1}, ytilde_{t-2})',
% g_{t-1} is component 2 and g_{t-2} is component 3, so BOTH lags the
% HLW IS curve needs are already in the state: no approximation is
% required.  The natural-rate average loads -a_r c / 2 on each.

n = 5;
F = zeros(n,n);
% y*_t  <- y*_{t-1} + g_{t-1}
F(1,1) = 1;  F(1,2) = 1;
% g_t   <- g_{t-1}
F(2,2) = 1;
% carries g_{t-1} into the next state's third component
F(3,2) = 1;
% ytilde_t <- a1 ytilde_{t-1} + a2 ytilde_{t-2} - (a_r c/2)(g_{t-1} + g_{t-2})
F(4,4) = a1;  F(4,5) = a2;
F(4,2) = -0.5 * a_r * c;
F(4,3) = -0.5 * a_r * c;
% ytilde_{t-1} <- ytilde_{t-1}
F(5,4) = 1;

% Exogenous: z_t = (r_{t-1} + r_{t-2})/2 (mean of last two lags).  The
% IS-curve loading is a_r on this scalar regressor.
%
%   ytilde_t <- ... + a_r * z_t
%
% B is 5x1, only the ytilde_t row is nonzero.
B = zeros(n, 1);
B(4) = a_r;

% Measurement: y_t = y*_t + ytilde_t,  pi_t = ...
% For the Phillips curve, we use a simplified observable equation in which
% inflation is observed and follows pi_t = b_pi pi_{t-1} + b_y ytilde_{t-1} + v_t.
% To keep the framework single-equation observation for clarity, we treat
% inflation as a SECOND measurement equation, with its own state component
% would be needed for true AR(1) persistence.  For pedagogy we use the
% reduced form: pi_t = b_y * ytilde_{t-1} + v_t (after partialling out the
% pi_{t-1} lag).  This isolates the Phillips-slope identification problem.

m = 2;   % output and inflation
H = zeros(m, n);
H(1, 1) = 1; H(1, 4) = 1;          % y_t = y*_t + ytilde_t
H(2, 5) = b_y;                      % pi_t = b_y ytilde_{t-1} (reduced)
D = zeros(m, 1);
% No direct dependence of measurement on z (= avg lagged r).

% Innovation covariance Q:
% Order:           y*    g       g_{-1}  ytilde  ytilde_{-1}
Q = diag([sigma_ys^2, sigma_g^2, 1e-10, sigma_yt^2, 1e-10]);

% Measurement covariance R:
R = diag([0.001, sigma_pi^2]);     % output measured precisely; inflation noisy

% Initial conditions: diffuse on trends and growth, but proper on AR(2) gap.
a0_x = [100; 0.5; 0.5; 0; 0];      % 100 = log GDP level start, 0.5 = quarterly trend growth
P0_x = diag([100, 1, 1, 5, 5]);

% Exogenous Z: average of the previous two observed rates.
Z = zeros(1, T);
for t = 1:T
  Z(1, t) = 0.5 * (r(t+1) + r(t));   % uses r_{t-1} and r_{t-2} via 1-based indexing
end

% ----- Simulate the system -----
[Y, Xtrue] = simks_simulate_const(T, F, H, Q, R, a0_x, P0_x, ...
                                   struct('Z', Z, 'B', B, 'D', D));

% NOTE: we use Y exactly as simulated.  Earlier versions of this script
% post-processed Y(2, :) with an AR(1) on inflation to make the series
% look more empirically realistic, but that created a mismatch between
% the data-generating process and the smoother's model, which produced
% a systematic bias on y*.  The pedagogical point that the inflation
% row identifies the Phillips slope b_y is fully made by the
% H(2, 5) = b_y entry in the measurement equation as-is.

% Extract truth.
ystar_true   = Xtrue(1, 2:end);
g_true       = Xtrue(2, 2:end);
ytilde_true  = Xtrue(4, 2:end);
rstar_true   = c * g_true;
rgap_true    = r(3:T+2) - rstar_true;

% ----- Estimator A: Gaussian smoother, true parameters -----
[X_a, J, h, info_a] = simks_smooth_const(Y, F, H, Q, R, a0_x, P0_x, ...
                                          struct('Z', Z, 'B', B, 'D', D, 'return_cov', true));
ystar_a  = X_a(1, 2:end);
g_a      = X_a(2, 2:end);
rstar_a  = c * g_a;

% Extract posterior standard deviation of r* = c g, i.e. sd of c g_t.
% Cov(g_t) is element (2,2) of Ptt{t+1}.
sd_rstar_a = zeros(1, T);
for t = 1:T
  sd_rstar_a(t) = c * sqrt(info_a.Ptt{t+1}(2,2));
end

% ----- Estimator B: constrained smoother, enforce r* >= 0 -----
% Constraint: c * g_t >= 0, i.e. -c * g_t <= 0 for t = 0,...,T.
% A_ineq is (T+1) x ((T+1)*n) with rows that pick out -c * g_t.
N = (T+1) * n;
ii = zeros(T+1, 1);
jj = zeros(T+1, 1);
vv = zeros(T+1, 1);
for t = 0:T
  ii(t+1) = t+1;
  jj(t+1) = t*n + 2;    % g is the 2nd state component
  vv(t+1) = -c;
end
A_ineq = sparse(ii, jj, vv, T+1, N);
b_ineq = zeros(T+1, 1);

[X_b, info_b] = simks_smooth_constrained(Y, F, H, Q, R, a0_x, P0_x, ...
                                          A_ineq, b_ineq, ...
                                          struct('Z', Z, 'B', B, 'D', D, ...
                                                 'max_outer', 80, 'max_newton', 5, ...
                                                 'tol_feas', 1e-9));
g_b      = X_b(2, 2:end);
rstar_b  = c * g_b;

% ----- Estimator C: EM with all parameters unknown -----
% We initialize at "reasonable economist's prior" guesses and let EM
% refine them.  Starting too far from a sensible region exposes EM to
% bad local maxima, which is a known property of state-space EM.
em_opts = struct( ...
  'Z',        Z, ...
  'F',        F + 0.01*randn(n,n), ...
  'H',        H, ...
  'Q',        diag([0.3, 0.1, 1e-8, 0.3, 1e-8]), ...
  'R',        eye(m)*0.5, ...
  'B',        zeros(n,1), ...
  'D',        zeros(m,1), ...
  'a0',       a0_x, ...
  'P0',       P0_x, ...
  'max_iter', 100, ...
  'tol',      1e-6);

% We do NOT impose the cross-equation restriction r* = c g here.  EM
% just estimates an unrestricted linear-Gaussian model.  This illustrates
% the identification challenge: without restrictions, the latent state
% direction "trend growth" is not separately identified from "trend level."
try
  [params_c, X_c, hist_c] = simks_em_const(Y, n, em_opts);
  em_converged = true;
catch err
  fprintf('  EM failed: %s\n', err.message);
  X_c = X_a;
  hist_c = [info_a.nnz];
  em_converged = false;
end

% ----- Report -----
fprintf('Application 1: Natural rate of interest (HLW-style)\n');
fprintf('====================================================\n');
fprintf('T = %d quarters (~%.1f years)\n', T, T/4);
fprintf('True a_r (IS curve real-rate sensitivity) = %.3f\n', a_r);
fprintf('True c   (r* = c * g, identification map) = %.2f\n', c);
fprintf('True sigma_g (trend growth innov sd)      = %.3f\n', sigma_g);

rmse_g    = sqrt(mean((g_a - g_true).^2));
rmse_rstar= sqrt(mean((rstar_a - rstar_true).^2));
rmse_ys   = sqrt(mean((ystar_a - ystar_true).^2));

fprintf('\n--- Estimator A: Gaussian smoother, true parameters ---\n');
fprintf('  RMSE on y*       = %.4f\n', rmse_ys);
fprintf('  RMSE on g        = %.4f\n', rmse_g);
fprintf('  RMSE on r* = c g = %.4f\n', rmse_rstar);
fprintf('  Posterior sd of r* at end of sample = %.3f\n', sd_rstar_a(end));
fprintf('  Avg posterior sd of r*              = %.3f\n', mean(sd_rstar_a));

fprintf('\n--- Estimator B: constrained smoother, r* >= 0 ---\n');
neg_periods_a = sum(rstar_a < 0);
neg_periods_b = sum(rstar_b < -1e-8);
fprintf('  Periods where unconstrained r* < 0:   %d / %d\n', neg_periods_a, T);
fprintf('  Periods where constrained   r* < 0:   %d / %d  (should be 0)\n', neg_periods_b, T);
fprintf('  Constraint converged: %d   max violation = %.2e\n', ...
       info_b.converged, info_b.max_violation);
rmse_rstar_b = sqrt(mean((rstar_b - rstar_true).^2));
fprintf('  RMSE on r* (constrained)             = %.4f\n', rmse_rstar_b);
neg_true = sum(rstar_true < 0);
if neg_true > 0
  fprintf('  NOTE: true r* < 0 for %d periods in this draw.\n', neg_true);
  fprintf('  Imposing r* >= 0 thus contradicts truth here and *hurts* RMSE.\n');
  fprintf('  This is the right diagnostic: a constraint is only informative\n');
  fprintf('  when it reflects the true data-generating process.\n');
end

if em_converged
  fprintf('\n--- Estimator C: EM, all parameters unknown ---\n');
  fprintf('  EM iterations: %d\n', length(hist_c));
  fprintf('  Final marginal log-likelihood: %.4f\n', hist_c(end));
  fprintf('  Monotone non-decreasing: %d\n', ...
         all(diff(hist_c) >= -1e-6));

  % The trend output should be recoverable up to identification rotation.
  % We report the correlation between smoothed and true y*.
  X_em = X_c(:, 2:end);
  ystar_em = X_em(1, :);  % first state component
  rho_ystar = corr(ystar_em', ystar_true');
  fprintf('  Correlation(EM y*, true y*): %.4f\n', rho_ystar);
end

fprintf('\nKey takeaways:\n');
fprintf('* The natural rate r* is identified through the cross-equation\n');
fprintf('  restriction r*_t = c g_t.  This pins down what would otherwise\n');
fprintf('  be one extra unidentified latent path.\n');
fprintf('* The constraint r* >= 0 (ZLB-relevant) is enforced via the\n');
fprintf('  interior-point QP on the SAME sparse J that the Gaussian\n');
fprintf('  smoother used.  No bespoke algorithm.\n');
fprintf('* Posterior bands on r* are large near the end of sample because\n');
fprintf('  the diffuse prior on g leaves the level of trend growth weakly\n');
fprintf('  identified.  This is the "pile-up" problem in disguise.\n');

% --- Plot ----------------------------------------------------------------
% Four panels:
%   (1) Trend output y*: true vs Estimator A.
%   (2) Trend growth g: true vs Estimator A.
%   (3) Natural rate r*: true, A (unconstrained, with 95% band),
%       B (r* >= 0 constrained).
%   (4) EM marginal log-likelihood across iterations (if EM ran).
tt = 1:T;
fig = figure('visible', 'on', 'position', [100 100 1000 800]);

subplot(2,2,1); hold on; box on;
plot(tt, ystar_true, 'k-',  'linewidth', 1.4);
plot(tt, ystar_a,    'r--', 'linewidth', 1.2);
legend({'true y*', 'estimator A'}, 'location', 'best');
title('Trend output y_t^*');
xlabel('quarter'); ylabel('log output');

subplot(2,2,2); hold on; box on;
plot(tt, g_true, 'k-',  'linewidth', 1.4);
plot(tt, g_a,    'r--', 'linewidth', 1.2);
legend({'true g', 'estimator A'}, 'location', 'best');
title('Trend growth g_t');
xlabel('quarter'); ylabel('g');

subplot(2,2,3); hold on; box on;
fill([tt, fliplr(tt)], ...
     [rstar_a - 1.96*sd_rstar_a, fliplr(rstar_a + 1.96*sd_rstar_a)], ...
     [0.85 0.90 0.98], 'edgecolor', 'none');
plot([1 T], [0 0], '-', 'color', [0.4 0.4 0.4]);
plot(tt, rstar_true, 'k-',  'linewidth', 1.4);
plot(tt, rstar_a,    'r--', 'linewidth', 1.2);
plot(tt, rstar_b,    'b-',  'linewidth', 1.5);
legend({'95% band (A)', 'zero', 'true r*', 'A: unconstrained', 'B: r* >= 0 (constrained)'}, ...
       'location', 'best');
title('Natural rate r_t^* = c \cdot g_t');
xlabel('quarter'); ylabel('r*');

subplot(2,2,4); hold on; box on;
if em_converged && length(hist_c) > 1
  plot(1:length(hist_c), hist_c, 'b-o', 'linewidth', 1.2, 'markersize', 3);
  title('EM marginal log-likelihood');
  xlabel('iteration'); ylabel('log p(Y; \theta)');
else
  text(0.5, 0.5, 'EM did not converge', 'units', 'normalized', ...
       'horizontalalignment', 'center');
  title('EM log-likelihood (n/a)');
  axis off;
end

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'app1_natural_rate.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
