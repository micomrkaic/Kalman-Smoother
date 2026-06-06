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

% DEMO_ROBUST_BREAKS  Robust smoother on data with structural breaks.
%
% Generates a near-random-walk process with three sudden level shifts
% (structural breaks).  The Gaussian smoother is forced to spread each
% break across many periods because its prior penalizes large
% innovations Gaussian-tail-heavily.  A Student-t smoother on the state
% innovations w_t accommodates occasional large jumps without smearing
% them, because the heavy tails of the prior make a single huge w_t
% much more likely than ten medium ones.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 7);
rand('seed',  7);

% ----- Model -----------------------------------------------------------
% State is a single scalar (the level we want to track).
n  = 1;
m  = 1;
F  = 1.0;                 % random walk
H  = 1.0;
Q  = 0.0025;              % small ordinary innovations (sd 0.05)
R  = 0.04;                % measurement noise sd 0.20
a0 = 0;
P0 = 1;
T  = 200;

% Simulate ordinary Gaussian dynamics, then inject three big breaks.
x      = zeros(1, T+1);
x(1)   = a0 + sqrt(P0) * randn;
for t = 1:T
  x(t+1) = F * x(t) + sqrt(Q) * randn;
end
break_periods = [60, 110, 160];
break_sizes   = [-1.5, 2.0, -1.0];
for k = 1:length(break_periods)
  x(break_periods(k)+1 : end) = x(break_periods(k)+1 : end) + break_sizes(k);
end
X_true = x;
Y      = H * x(2:end) + sqrt(R) * randn(1, T);

% ----- Smoothers -------------------------------------------------------
X_g = simks_smooth_const(Y, F, H, Q, R, a0, P0);

% Robust on state innovations only.
opts_rs = struct('nu', 4, 'robust_state', true, 'max_iter', 100);
[X_rs, info_rs] = simks_smooth_robust(Y, F, H, Q, R, a0, P0, 'student-t', opts_rs);

% ----- Compare ---------------------------------------------------------
truth   = X_true(2:end);
err_g   = X_g(1, 2:end)  - truth;
err_rs  = X_rs(1, 2:end) - truth;
rmse    = @(e) sqrt(mean(e.^2));

fprintf('Robust state-side smoother on a series with breaks\n');
fprintf('==================================================\n');
fprintf('T = %d, %d structural breaks injected at t = %s\n', ...
       T, length(break_periods), mat2str(break_periods));
fprintf('\nRMSE on the latent level\n');
fprintf('  Gaussian smoother           %.4f\n', rmse(err_g));
fprintf('  Robust state (Student-t)    %.4f   [%d iters, converged=%d]\n', ...
       rmse(err_rs), info_rs.iters, info_rs.converged);

% The state-innovation weights tau_w should be small (downweighted)
% only at the break periods.
fprintf('\nState innovation weights tau_w at the break periods\n');
for k = 1:length(break_periods)
  bp = break_periods(k);
  fprintf('  t = %3d (break size %+.1f) :  tau_w = %.3f\n', ...
         bp, break_sizes(k), info_rs.tau_w(bp));
end

fprintf('\nMean tau_w outside any break window (|t-bp| > 3): %.3f\n', ...
       mean(info_rs.tau_w(setdiff(1:T, [break_periods-3, break_periods-2, ...
            break_periods-1, break_periods, break_periods+1, break_periods+2, ...
            break_periods+3]))));

% A low tau_w at a break period means "the model interpreted this as a
% one-off large innovation rather than a series of normal-sized ones".
% In the Gaussian smoother, that interpretation is not available and
% the break is smeared across several adjacent periods.

% --- Plot ----------------------------------------------------------------
% Top: data with injected breaks, plus Gaussian and robust smoothed paths.
% Bottom: state-innovation weights tau_w; should plunge at break periods.
tt    = 1:T;
X_g_path  = simks_smooth_const(Y, F, H, Q, R, a0, P0);

fig = figure('visible', 'on', 'position', [100 100 900 650]);

subplot(2,1,1); hold on; box on;
plot(tt, Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
plot(tt, X_true(2:end),    'k-',  'linewidth', 1.2);
plot(tt, X_g_path(1,2:end),'b--', 'linewidth', 1.2);
plot(tt, X_rs(1,2:end),    'r--', 'linewidth', 1.4);
for k = 1:length(break_periods)
  xline_t = break_periods(k);
  plot([xline_t xline_t], ylim, ':', 'color', [0.5 0.5 0.5]);
end
legend({'observation', 'true level', 'Gaussian', 'Student-t state'}, ...
       'location', 'best');
title('Robust smoothing through structural breaks');
xlabel('t'); ylabel('level');

subplot(2,1,2); hold on; box on;
plot(tt, info_rs.tau_w, 'r-', 'linewidth', 1.0);
for k = 1:length(break_periods)
  bp = break_periods(k);
  plot(bp, info_rs.tau_w(bp), 'o', 'color', [0.9 0.2 0.2], ...
       'markersize', 6, 'markerfacecolor', [0.9 0.2 0.2]);
end
plot([1 T], [0.5 0.5], 'k--');
legend({'\xi_t (state weight)', 'at injected breaks', 'threshold 0.5'}, ...
       'location', 'best');
title('State innovation weights \xi_t plunge at the three injected breaks');
xlabel('t'); ylabel('\xi_t');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_robust_breaks.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
