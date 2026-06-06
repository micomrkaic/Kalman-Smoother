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

% DEMO_CONST_EM  Estimate a constant system by proper Gaussian EM.
%
% The estimator uses smoothed sufficient statistics (with posterior
% covariance corrections from selected inversion), so Q-hat and R-hat are
% not biased downward.  The monitor is the marginal log-likelihood
% log p(Y; theta), which should be non-decreasing across iterations.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 3);

T     = 200;
Ftrue = [0.75 0.15; 0.00 0.50];
Htrue = [1.0 0.3; 0.2 1.0];
Qtrue = [0.08 0.01; 0.01 0.05];
Rtrue = [0.15 0.02; 0.02 0.12];
a0 = [0;0];
P0 = eye(2);

[Y, Xtrue] = simks_simulate_const(T, Ftrue, Htrue, Qtrue, Rtrue, a0, P0);

opts          = struct();
opts.max_iter = 200;
opts.tol      = 1e-6;
opts.P0       = 10*eye(2);

[params, Xhat, hist] = simks_em_const(Y, 2, opts);

fprintf('Constant-system EM demo (proper Gaussian EM)\n');
fprintf('Iterations: %d\n', numel(hist));
fprintf('Initial marginal log-lik: %.4f\n', hist(1));
fprintf('Final   marginal log-lik: %.4f\n', hist(end));
mono = all(diff(hist) >= -1e-6);
fprintf('Monotone non-decreasing:  %d\n', mono);

fprintf('\nTrue F:\n');         disp(Ftrue);
fprintf('Estimated F (up to rotation of latent state):\n'); disp(params.F);
fprintf('\nTrue H:\n');          disp(Htrue);
fprintf('Estimated H (up to rotation of latent state):\n'); disp(params.H);
fprintf('\nTrue Q (eigenvalues):'); disp(sort(eig(Qtrue))');
fprintf('Est. Q (eigenvalues):  ');  disp(sort(eig(params.Q))');
fprintf('True R (eigenvalues):'); disp(sort(eig(Rtrue))');
fprintf('Est. R (eigenvalues):  ');  disp(sort(eig(params.R))');
fprintf('\nReminder: F, H are not uniquely identified without restrictions.\n');
fprintf('Compare INVARIANTS (eigenvalues of F, log-likelihood) rather than entries.\n');
fprintf('eig(F)_true = '); disp(sort(eig(Ftrue))');
fprintf('eig(F)_est  = '); disp(sort(eig(params.F))');

% --- Plot ----------------------------------------------------------------
% Top: marginal log-likelihood across EM iterations (should be monotone).
% Bottom: smoothed state 1 from the final EM fit vs truth.
tt = 0:size(Xtrue,2)-1;
fig = figure('visible', 'on', 'position', [100 100 900 600]);

subplot(2,1,1); hold on; box on;
plot(1:length(hist), hist, 'b-o', 'linewidth', 1.2, 'markersize', 3);
title('Marginal log-likelihood across EM iterations');
xlabel('iteration'); ylabel('log p(Y; \theta)');

subplot(2,1,2); hold on; box on;
plot(tt, Xtrue(1,:), 'k-',  'linewidth', 1.2);
plot(tt, Xhat (1,:), 'r--', 'linewidth', 1.2);
plot(1:size(Y,2), Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
legend({'true x_1', 'smoothed x_1 (final EM fit)', 'observation'}, ...
       'location', 'best');
title('Smoothed state at final EM fit (latent rotation expected)');
xlabel('t'); ylabel('x_1');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_const_em.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
