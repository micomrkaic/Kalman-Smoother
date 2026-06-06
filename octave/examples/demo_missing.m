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

% DEMO_MISSING  Smoothing with missing observations (ragged ends, gaps).
%
% Macro data have NaNs.  The simultaneous formulation handles them
% trivially: at each missing observation, the corresponding H' R^{-1} H
% block contributes nothing to J, and the corresponding H' R^{-1} y_t
% entry contributes nothing to h.  Identification is then a property of
% the residual sparsity pattern of J -- the system is still well-posed
% as long as enough observations remain to identify the state path.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 5);

T  = 150;
F  = [0.9 0.1; 0.0 0.8];
H  = [1.0 0.0; 1.0 1.0];
Q  = [0.04 0.00; 0.00 0.02];
R  = [0.10 0.00; 0.00 0.05];
a0 = [0;0];
P0 = eye(2);

[Yfull, Xtrue] = simks_simulate_const(T, F, H, Q, R, a0, P0);

% Introduce three kinds of missingness:
%   (i)   a contiguous gap in the middle of the sample,
%   (ii)  a ragged front edge (one series missing for first 30 periods),
%   (iii) random scattered NaNs.
Y = Yfull;
Y(:, 60:80)   = NaN;          % whole rows missing
Y(2, 1:30)    = NaN;          % ragged front edge for series 2
rand('seed', 4);
scatter = rand(size(Y)) < 0.05;
Y(scatter)    = NaN;

opts.return_cov = true;
[Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0, opts);

fprintf('Smoothing with missing data\n');
fprintf('T = %d, total NaNs in Y = %d / %d\n', T, sum(isnan(Y(:))), numel(Y));
fprintf('Sparse precision still SPD, nnz = %d\n', info.nnz);
fprintf('RMSE state 1 over full sample = %.4f\n', ...
       sqrt(mean((Xhat(1,:) - Xtrue(1,:)).^2)));
fprintf('RMSE state 1 inside the 60..80 gap = %.4f\n', ...
       sqrt(mean((Xhat(1,61:81) - Xtrue(1,61:81)).^2)));

% Inside the gap, posterior uncertainty should be visibly larger.
var_in_gap  = mean(arrayfun(@(t) info.Ptt{t+1}(1,1), 60:80));
var_outside = mean(arrayfun(@(t) info.Ptt{t+1}(1,1), [1:59, 82:T]));
fprintf('Avg posterior variance, state 1, inside gap  = %.4f\n', var_in_gap);
fprintf('Avg posterior variance, state 1, outside gap = %.4f\n', var_outside);

% --- Plot ----------------------------------------------------------------
% State 1 vs observation, with the contiguous-gap region shaded and the
% 95% posterior band visibly wider inside it.
tt   = 0:T;
band = zeros(1, T+1);
for t = 0:T
  band(t+1) = 1.96 * sqrt(info.Ptt{t+1}(1,1));
end

fig = figure('visible', 'on', 'position', [100 100 900 500]);
hold on; box on;

% Shade the contiguous gap (t = 60..80 in the observation index).
yl = [min(Xtrue(1,:)) - 2, max(Xtrue(1,:)) + 2];
fill([60 80 80 60], [yl(1) yl(1) yl(2) yl(2)], ...
     [0.95 0.95 0.85], 'edgecolor', 'none');

% Posterior band.
fill([tt, fliplr(tt)], [Xhat(1,:)-band, fliplr(Xhat(1,:)+band)], ...
     [0.85 0.90 0.98], 'edgecolor', 'none');

% Observed and missing markers for series 1.
obs_idx = find(~isnan(Y(1,:)));
plot(obs_idx, Y(1, obs_idx), 'o', 'color', [0.4 0.4 0.4], 'markersize', 3);

% True and smoothed state 1.
plot(tt, Xtrue(1,:), 'k-',  'linewidth', 1.2);
plot(tt, Xhat (1,:), 'r--', 'linewidth', 1.2);

ylim(yl);
legend({'missing gap (t=60..80)', '95% band', 'observation (series 1)', ...
        'true x_1', 'smoothed x_1'}, 'location', 'best');
title('Smoothing with missing data: bands widen where observations are absent');
xlabel('t'); ylabel('x_1');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_missing.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
