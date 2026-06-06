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

% DEMO_CONST_KNOWN  Known-parameter simultaneous Kalman smoothing.
% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 1);

T  = 100;
F  = [0.8 0.2; 0.0 0.6];
H  = [1.0 0.0];
Q  = [0.10 0.02; 0.02 0.08];
R  = 0.25;
a0 = [0;0];
P0 = eye(2);

[Y, Xtrue] = simks_simulate_const(T, F, H, Q, R, a0, P0);

% Smoother with posterior covariance blocks via selected inversion.
opts.return_cov = true;
[Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0, opts);

% Marginal log-likelihood at the true parameters.
ll = simks_marginal_loglik(Y, F, H, Q, R, a0, P0);

fprintf('Known-parameter simultaneous smoother demo\n');
fprintf('State dimension n = %d, observations T = %d\n', size(F, 1), T);
fprintf('Sparse precision size: %d x %d, nnz = %d\n', size(J, 1), size(J, 2), info.nnz);
fprintf('Marginal log-likelihood = %.4f\n', ll);
fprintf('RMSE state 1 = %.4f\n', sqrt(mean((Xhat(1,:) - Xtrue(1,:)).^2)));
fprintf('RMSE state 2 = %.4f\n', sqrt(mean((Xhat(2,:) - Xtrue(2,:)).^2)));

% Empirical coverage of 95% smoothed bands for state 1.
band = zeros(1, T+1);
for t = 0:T
  band(t+1) = 1.96 * sqrt(info.Ptt{t+1}(1,1));
end
cov_state1 = mean(abs(Xhat(1,:) - Xtrue(1,:)) <= band);
fprintf('Empirical 95%% coverage for state 1 = %.3f\n', cov_state1);

% --- Plot ----------------------------------------------------------------
% State trajectories: true, smoothed, +/-2 sigma bands.  State 1 also has
% the noisy observation overlaid (state 2 is unobserved).
band2 = zeros(1, T+1);
for t = 0:T
  band2(t+1) = 1.96 * sqrt(info.Ptt{t+1}(2,2));
end
tt = 0:T;
fig = figure('visible', 'on', 'position', [100 100 900 600]);

subplot(2,1,1);
hold on; box on;
fill([tt, fliplr(tt)], [Xhat(1,:)-band, fliplr(Xhat(1,:)+band)], ...
     [0.85 0.90 0.98], 'edgecolor', 'none');
plot(1:T, Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
plot(tt, Xtrue(1,:), 'k-',  'linewidth', 1.2);
plot(tt, Xhat (1,:), 'r--', 'linewidth', 1.2);
legend({'95% band', 'observation', 'true', 'smoothed'}, 'location', 'best');
title('State 1: true (observed) coordinate');
xlabel('t'); ylabel('x_1');

subplot(2,1,2);
hold on; box on;
fill([tt, fliplr(tt)], [Xhat(2,:)-band2, fliplr(Xhat(2,:)+band2)], ...
     [0.85 0.90 0.98], 'edgecolor', 'none');
plot(tt, Xtrue(2,:), 'k-',  'linewidth', 1.2);
plot(tt, Xhat (2,:), 'r--', 'linewidth', 1.2);
legend({'95% band', 'true', 'smoothed'}, 'location', 'best');
title('State 2: unobserved coordinate');
xlabel('t'); ylabel('x_2');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_const_known.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
