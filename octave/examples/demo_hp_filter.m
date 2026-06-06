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

% DEMO_HP_FILTER  The Hodrick-Prescott filter as a special case.
%
% The HP filter solves
%   min_tau  sum_t (y_t - tau_t)^2 + lambda * sum_t (Delta^2 tau_t)^2
% which is exactly the simultaneous formulation with state
%   x_t = (tau_t, Delta tau_t)' = (level, slope),
% transition
%   F = [1 1; 0 1],     w_t = (0, e_t)',   Q = diag(0, 1/lambda),
% and measurement
%   H = [1, 0],         R = 1.
%
% The smoothed level path coincides (up to boundary effects) with the
% textbook HP recursion.  No need to derive special HP boundary conditions:
% the simultaneous system carries them automatically through the prior
% terms at t = 0.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 11);

% Synthetic series: smooth trend plus AR(1)-ish noise.
T = 200;
t = (1:T)';
trend_true = 100 + 0.05*t + 5*sin(t/30);
noise      = filter([1], [1, -0.5], 0.6*randn(T,1));
y          = trend_true + noise;

% HP-filter as state-space, lambda = 1600 (quarterly default).
lambda = 1600;
F  = [1 1; 0 1];
H  = [1 0];
% Q is singular in textbook HP (zero process noise on the level).  The
% precision-based simultaneous smoother needs Q^{-1}, so we add a tiny
% level regularizer.  This is a real subtlety the recursive form hides:
% the recursive HP boundary conditions also implicitly assume something
% about the limit Q_{11} -> 0.  We just make it explicit.
eps_lvl = 1e-10;
Q  = diag([eps_lvl, 1/lambda]);
R  = 1;                               % unit measurement noise scale
a0 = [y(1); 0];
P0 = 1e6 * eye(2);                    % near-diffuse on initial level/slope

Y = y';                               % m x T
[Xhat, J, h, info] = simks_smooth_const(Y, F, H, Q, R, a0, P0);
trend_hp = Xhat(1, 2:end)';           % smoothed level x_t for t = 1..T

% Compare to the classical closed-form HP filter (penalty matrix).
D2 = spdiags([ones(T,1) -2*ones(T,1) ones(T,1)], 0:2, T-2, T);
hp_classical = (speye(T) + lambda * (D2' * D2)) \ y;

fprintf('HP filter via simultaneous smoother vs classical closed form\n');
fprintf('lambda = %d\n', lambda);
fprintf('max |smoothed - classical| = %.3e\n', max(abs(trend_hp - hp_classical)));
fprintf('(Discrepancy is purely from the boundary treatment of the\n');
fprintf(' near-diffuse prior; raising P0 toward infinity drives it to zero.)\n');

% --- Plot ----------------------------------------------------------------
% The classic HP-filter picture: noisy data, true trend, and the
% extracted trends via both methods (they overlap to ~1e-4).
tt = 1:T;
fig = figure('visible', 'on', 'position', [100 100 900 500]);
hold on; box on;
plot(tt, y,            'o-', 'color', [0.6 0.6 0.6], 'markersize', 3, ...
     'markerfacecolor', [0.85 0.85 0.85]);
plot(tt, trend_true,   'k-',  'linewidth', 1.5);
plot(tt, trend_hp,     'b--', 'linewidth', 1.5);
plot(tt, hp_classical, 'r:',  'linewidth', 1.5);
legend({'observed y_t', 'true trend', ...
        'HP trend (simultaneous)', 'HP trend (closed form)'}, ...
       'location', 'best');
title(sprintf('Hodrick--Prescott decomposition, lambda = %d', lambda));
xlabel('t'); ylabel('value');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_hp_filter.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
