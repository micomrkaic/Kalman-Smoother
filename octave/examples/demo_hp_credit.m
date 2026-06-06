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

% DEMO_HP_CREDIT  HP filter augmented with a financial cycle driver.
%
% This is the flagship demo: it combines several features of the
% simultaneous smoother in a setting that is recognizable to applied
% macroeconomists.
%
% Model.  The latent state is the trend-slope pair (tau_t, Delta tau_t)
% as in the textbook HP filter.  But the trend is also pushed by an
% observed credit-gap variable z_t -- credit booms raise the latent
% growth rate of output via the slope, and the residual cycle is what
% we want to extract:
%
%   tau_t        = tau_{t-1} + Delta tau_{t-1}                + eps_lvl_t
%   Delta tau_t  = Delta tau_{t-1}        + beta * z_t + e_t
%   y_t          = tau_t                  + gamma * z_t + v_t
%
% Two genuine macroeconomic features are layered on:
% (1) at t = 80 the economy experiences a structural break: the
%     slope drops by 1.5 percentage points (e.g. a productivity shock,
%     a pandemic, a regime change).  Student-t innovations on Delta tau
%     are needed to catch this; Gaussian Q would smear it across many
%     periods;
% (2) 10% of observations are contaminated with outlier noise
%     (e.g. data revisions, miscoded prints, one-off shocks).
%     Student-t on v_t handles these without distorting trend and slope.
%
% We then estimate (F, H, Q, R, B, D) by EM with the credit-gap
% exogenous variable as a known input, on noisy contaminated data,
% comparing three estimators:
%   (a) classical Gaussian HP with no exogenous (the textbook bad fit)
%   (b) Gaussian smoother with the credit-gap exogenous correctly used
%   (c) Robust + exogenous: Student-t innovations on both observation
%       and state, with credit-gap exogenous correctly used
%
% Outputs are reported as RMSE on the latent trend, plus diagnostic
% IRLS weights showing how the robust algorithm identifies the
% structural break and the outliers.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 11);
rand('seed',  11);

% ----- Generate data ---------------------------------------------------
T      = 200;
lambda = 1600;          % textbook quarterly HP smoothing

% True parameters.
beta   = 0.30;          % credit gap raises slope of trend
gamma  = 0.10;          % small direct effect of credit on output
eps_lvl = 1e-6;         % level-side regularizer
sd_w   = 1 / sqrt(lambda);
sd_v   = 0.25;

F_true = [1 1; 0 1];
H_true = [1 0];
Q_true = diag([eps_lvl, sd_w^2]);
R_true = sd_v^2;
B_true = [0; beta];     % credit gap enters only the slope equation
D_true = gamma;         % small direct effect on measurement

% Generate a credit-gap series: a persistent AR(1) cycle.
phi_z = 0.92;
sd_z  = 0.6;
z     = zeros(1, T);
z(1)  = sd_z * randn;
for t = 2:T
  z(t) = phi_z * z(t-1) + sd_z * randn;
end

% Simulate true model with exogenous input.
sim_opts = struct('Z', z, 'B', B_true, 'D', D_true);
[Y_clean, X_true] = simks_simulate_const(T, F_true, H_true, Q_true, R_true, ...
                                          [0; 0.5], 0.1*eye(2), sim_opts);
trend_true = X_true(1, 2:end);

% Inject a structural break in the slope at t = 80.
break_t    = 80;
slope_drop = -1.5;
for t = break_t : T
  X_true(2, t+1) = X_true(2, t+1) + slope_drop;
end
% Propagate the broken slope through the trend.
for t = break_t : T
  X_true(1, t+1) = X_true(1, t) + X_true(2, t);
end
trend_true = X_true(1, 2:end);
Y = X_true(1, 2:end) + D_true * z + sd_v * randn(1, T);

% Contaminate 10% of observations with outlier noise.
outlier_mask = rand(1, T) < 0.10;
Y(outlier_mask) = Y(outlier_mask) + 3 * randn(1, sum(outlier_mask));

% ----- Estimator A: classical HP, no exogenous, Gaussian -------------
P0_diff = 1e6 * eye(2);
[Xa] = simks_smooth_const(Y, F_true, H_true, Q_true, R_true, [0;0.5], P0_diff);
trend_a = Xa(1, 2:end);

% ----- Estimator B: Gaussian + exogenous --------------------------------
optsB = struct('Z', z, 'B', B_true, 'D', D_true);
[Xb] = simks_smooth_const(Y, F_true, H_true, Q_true, R_true, [0;0.5], P0_diff, optsB);
trend_b = Xb(1, 2:end);

% ----- Estimator C: Student-t robust + exogenous -----------------------
optsC = struct('Z', z, 'B', B_true, 'D', D_true, ...
               'nu', 4, 'robust_state', true, 'max_iter', 100);
[Xc, infoC] = simks_smooth_robust(Y, F_true, H_true, Q_true, R_true, ...
                                   [0;0.5], P0_diff, 'student-t', optsC);
trend_c = Xc(1, 2:end);

% ----- Estimator D: EM (unknown params) + exogenous --------------------
% Initialize at simple guesses; let EM find F, H, Q, R, B, D.
em_opts = struct('Z', z, ...
                 'F', [1 1; 0 0.9], ...
                 'H', [1 0], ...
                 'Q', diag([1e-6, 1e-3]), ...
                 'R', 1.0, ...
                 'B', [0; 0], ...
                 'D', 0, ...
                 'a0', [0; 0], ...
                 'P0', 100 * eye(2), ...
                 'max_iter', 80, ...
                 'tol', 1e-5);
[paramsD, Xd, hist_d] = simks_em_const(Y, 2, em_opts);
trend_d = Xd(1, 2:end);

% ----- Report ----------------------------------------------------------
rmse  = @(x) sqrt(mean((x - trend_true).^2));

fprintf('Flagship demo: HP + credit cycle + structural break + outliers\n');
fprintf('===============================================================\n');
fprintf('T = %d, lambda = %d, break at t = %d (slope drops by %.1f)\n', ...
       T, lambda, break_t, slope_drop);
fprintf('True beta (credit -> slope) = %.2f\n', beta);
fprintf('True gamma (credit -> output) = %.2f\n', gamma);
fprintf('Outlier contamination rate = %.0f%%\n\n', 100 * mean(outlier_mask));

fprintf('RMSE on the latent trend:\n');
fprintf('  (A) Gaussian, no exogenous           %.4f\n', rmse(trend_a));
fprintf('  (B) Gaussian, true exogenous         %.4f\n', rmse(trend_b));
fprintf('  (C) Student-t robust + true exog     %.4f\n', rmse(trend_c));
fprintf('  (D) EM (estimated params) + exog     %.4f\n', rmse(trend_d));

fprintf('\nEM convergence: %d iterations, final loglik = %.4f\n', ...
       length(hist_d), hist_d(end));
fprintf('EM-estimated B (truth = [0; %.2f]):\n', beta);
disp(paramsD.B);
fprintf('EM-estimated D (truth = %.2f): %.4f\n', gamma, paramsD.D);
fprintf('Eigenvalues of estimated F (truth = [1; 1]):\n');
disp(sort(eig(paramsD.F)));

fprintf('\nStructural-break detection by robust smoother (Student-t state):\n');
fprintf('  tau_w at break period t = %d:  %.4f\n', break_t, infoC.tau_w(break_t));
fprintf('  mean tau_w outside ±3 of break:  %.4f\n', ...
       mean(infoC.tau_w(setdiff(1:T, break_t-3:break_t+3))));

n_outliers_caught = sum(infoC.tau_v(outlier_mask) < 0.5);
n_outliers        = sum(outlier_mask);
fprintf('\nOutlier detection (tau_v < 0.5):\n');
fprintf('  Outliers correctly flagged: %d / %d (%.0f%%)\n', ...
       n_outliers_caught, n_outliers, 100*n_outliers_caught/max(n_outliers,1));
fprintf('  Mean tau_v at outlier observations:    %.3f\n', ...
       mean(infoC.tau_v(outlier_mask)));
fprintf('  Mean tau_v at clean observations:      %.3f\n', ...
       mean(infoC.tau_v(~outlier_mask)));

% --- Plot ----------------------------------------------------------------
% (a) Observed y_t, true trend, all four estimated trends.
% (b) The credit-gap exogenous driver z_t.
% (c) EM log-likelihood trace + IRLS state weights from estimator C.
tt = 1:T;
fig = figure('visible', 'on', 'position', [100 100 900 900]);

subplot(3,1,1); hold on; box on;
plot(tt, Y,          'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
plot(tt, trend_true, 'k-',  'linewidth', 1.5);
plot(tt, trend_a,    ':',   'color', [0.4 0.4 0.4],  'linewidth', 1.2);
plot(tt, trend_b,    '--',  'color', [0.2 0.5 0.9],  'linewidth', 1.2);
plot(tt, trend_c,    '-',   'color', [0.9 0.2 0.2],  'linewidth', 1.5);
plot(tt, trend_d,    '-.',  'color', [0.1 0.6 0.3],  'linewidth', 1.5);
plot([break_t break_t], ylim, 'k:');
legend({'observation', 'true trend', 'A: HP no exog', 'B: HP + exog', ...
        'C: robust + exog', 'D: EM + exog'}, 'location', 'best');
title(sprintf('HP + credit cycle: structural break at t=%d, %.0f%% outliers', ...
              break_t, 100*mean(outlier_mask)));
xlabel('t'); ylabel('output level');

subplot(3,1,2); hold on; box on;
plot(tt, z, '-', 'color', [0.2 0.4 0.7], 'linewidth', 1.2);
plot([1 T], [0 0], 'k:');
title('Exogenous driver: credit gap z_t');
xlabel('t'); ylabel('z_t');

subplot(3,1,3); hold on; box on;
% EM log-likelihood and IRLS state weights overlaid via a rescaled
% secondary plot.  Octave's yyaxis is not universally available, so we
% normalize both to [0,1] and indicate via the legend what each is.
ll_norm = (hist_d - min(hist_d)) / max(1e-12, max(hist_d) - min(hist_d));
ll_t    = linspace(1, T, length(hist_d));   % squash iters onto t-axis
plot(ll_t, ll_norm,       'b-o', 'linewidth', 1.0, 'markersize', 3);
plot(tt,  infoC.tau_w,    '-', 'color', [0.9 0.2 0.2], 'linewidth', 1.0);
plot([break_t break_t], [0 1.05], 'k:');
ylim([0 1.05]);
legend({'EM log-lik (rescaled to [0,1])', '\xi_t (state weight)', ...
        'break at t=80'}, 'location', 'best');
title('EM convergence trace and IRLS state weights (overlaid, rescaled)');
xlabel('t (or iteration, for the EM trace)');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_hp_credit.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
