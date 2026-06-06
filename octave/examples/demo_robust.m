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

% DEMO_ROBUST  Robust simultaneous smoother on data with outliers.
%
% Generates a 2D state, 1D observation system with clean Gaussian
% dynamics and contaminated measurement noise (10% of observations
% receive extra noise from a much wider distribution).  Compares the
% Gaussian smoother, a Student-t smoother, and a Huber smoother.
%
% The Gaussian smoother is pulled toward outliers because it treats
% large residuals as ordinary noise.  The robust smoothers downweight
% them via IRLS on the same sparse system that the Gaussian smoother
% uses.  The Mahalanobis residuals at outlier periods receive small
% weights tau_t << 1, so those observations effectively drop out at
% the next iteration.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 42);
rand('seed',  42);

% ----- Model -----------------------------------------------------------
n  = 2;
m  = 1;
F  = [0.95  0.05;
      0     0.7];
H  = [1     0.3];
Q  = 0.05 * eye(n);
R  = 0.10;
a0 = zeros(n, 1);
P0 = eye(n);
T  = 200;

% ----- Simulate --------------------------------------------------------
[Y_clean, X_true] = simks_simulate_const(T, F, H, Q, R, a0, P0);

% Contaminate: 10% of observations get extra noise from N(0, 4) (std 2,
% compared with clean std ~0.32).  These are the outliers.
outlier_rate   = 0.10;
outlier_mask   = rand(1, T) < outlier_rate;
extra_noise    = 2 * randn(1, T);
Y              = Y_clean;
Y(:, outlier_mask) = Y_clean(:, outlier_mask) + extra_noise(outlier_mask);

% ----- Smoothers -------------------------------------------------------
[X_g] = simks_smooth_const(Y, F, H, Q, R, a0, P0);

opts_t = struct('nu', 4, 'verbose', false);
[X_t, info_t] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'student-t', opts_t);

opts_h = struct('c', 1.345);
[X_h, info_h] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'huber', opts_h);

opts_l = struct();
[X_l, info_l] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'laplace', opts_l);

% Sanity check: family='gaussian' must agree with the constant smoother.
[X_g2, info_g2] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'gaussian');
disagreement_g  = max(abs(X_g(:) - X_g2(:)));

% ----- Compare on state 1 ---------------------------------------------
truth = X_true(1, 2:end);                 % x_1, ..., x_T (T columns)
err_g = X_g(1, 2:end) - truth;
err_t = X_t(1, 2:end) - truth;
err_h = X_h(1, 2:end) - truth;
err_l = X_l(1, 2:end) - truth;

rmse = @(e) sqrt(mean(e.^2));

fprintf('Robust simultaneous smoother demo\n');
fprintf('=================================\n');
fprintf('T = %d, n = %d, m = %d\n', T, n, m);
fprintf('Outliers: %d / %d (%.0f%%)\n\n', sum(outlier_mask), T, 100 * mean(outlier_mask));

fprintf('Smoothing RMSE on state 1\n');
fprintf('  Gaussian          %.4f\n', rmse(err_g));
fprintf('  Student-t (nu=4)  %.4f   [%d iters, converged=%d]\n', ...
       rmse(err_t), info_t.iters, info_t.converged);
fprintf('  Huber  (c=1.345)  %.4f   [%d iters, converged=%d]\n', ...
       rmse(err_h), info_h.iters, info_h.converged);
fprintf('  Laplace           %.4f   [%d iters, converged=%d]\n', ...
       rmse(err_l), info_l.iters, info_l.converged);

fprintf('\nGaussian self-consistency check\n');
fprintf('  family=''gaussian'' agrees with simks_smooth_const to %.2e\n', disagreement_g);

% ----- Diagnostic: which observations were flagged as outliers? -------
threshold = 0.5;
flagged_t = info_t.tau_v < threshold;
TP  = sum(flagged_t &  outlier_mask);
FP  = sum(flagged_t & ~outlier_mask);
FN  = sum(~flagged_t &  outlier_mask);
TN  = sum(~flagged_t & ~outlier_mask);

precision = TP / max(TP + FP, 1);
recall    = TP / max(TP + FN, 1);

fprintf('\nOutlier flagging using Student-t weights, threshold tau < %.2f\n', threshold);
fprintf('  True positives  : %3d\n', TP);
fprintf('  False positives : %3d\n', FP);
fprintf('  False negatives : %3d\n', FN);
fprintf('  Precision       : %.3f\n', precision);
fprintf('  Recall          : %.3f\n', recall);
fprintf('  Mean tau at true outliers   : %.3f\n', mean(info_t.tau_v(outlier_mask)));
fprintf('  Mean tau at clean observations: %.3f\n', mean(info_t.tau_v(~outlier_mask)));

% ----- A note on what this demonstrates --------------------------------
% The same sparse system J X = h is solved at every iteration. Only
% the per-period measurement covariance R_t = R / tau_t changes. The
% sparsity pattern of J is identical to the Gaussian case, and the
% per-iteration cost is one O(T n^3) sparse Cholesky. This is the
% IRLS-on-the-same-J claim of the LaTeX note, in concrete form.

% --- Plot ----------------------------------------------------------------
% Top: observed data (outliers marked), true state, Gaussian and Student-t
% smoothed paths.  The Gaussian path bends toward outliers; the robust
% path does not.
% Bottom: per-period IRLS weights tau_v from Student-t.  Should be ~1
% at clean periods and << 1 at outliers.
tt   = 1:T;
fig  = figure('visible', 'on', 'position', [100 100 900 700]);

subplot(3,1,[1 2]); hold on; box on;
% Outliers in red, clean in grey, so they stand out.
plot(tt(~outlier_mask), Y(~outlier_mask), 'o', 'color', [0.6 0.6 0.6], ...
     'markersize', 3);
plot(tt( outlier_mask), Y( outlier_mask), 'o', 'color', [0.9 0.2 0.2], ...
     'markersize', 5, 'markerfacecolor', [0.9 0.2 0.2]);
plot(tt, X_true(1, 2:end), 'k-',  'linewidth', 1.2);
plot(tt, X_g  (1, 2:end),  'b--', 'linewidth', 1.2);
plot(tt, X_t  (1, 2:end),  'r--', 'linewidth', 1.2);
legend({'clean obs', 'outlier', 'true x_1', ...
        'Gaussian smoother', 'Student-t smoother'}, 'location', 'best');
title('Robust smoother on contaminated data');
xlabel('t'); ylabel('x_1');

subplot(3,1,3); hold on; box on;
plot(tt, info_t.tau_v, 'b-', 'linewidth', 1.0);
plot(tt(outlier_mask), info_t.tau_v(outlier_mask), 'o', ...
     'color', [0.9 0.2 0.2], 'markersize', 5, ...
     'markerfacecolor', [0.9 0.2 0.2]);
plot([1 T], [0.5 0.5], 'k--');
legend({'\tau_t (Student-t)', '\tau_t at true outliers', ...
        'flag threshold 0.5'}, 'location', 'best');
title('IRLS weights \tau_t: low values flag outliers');
xlabel('t'); ylabel('\tau_t');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_robust.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
