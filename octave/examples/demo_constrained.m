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

% DEMO_CONSTRAINED  Smoother with inequality constraints on the state.
%
% Macro use case: a "NAIRU-like" latent rate that we know must be
% non-negative (e.g. the natural rate of unemployment, the equilibrium
% real interest rate floor, the trend output gap floor in a regime
% with structural slack).  The data may push the smoothed value
% slightly negative due to noise; constrained smoothing enforces
% non-negativity directly via the QP
%
%     min  1/2 X' J X - h' X
%     s.t. -x_t <= 0  for all t.
%
% We simulate a near-boundary latent process that the Gaussian
% smoother places below zero in some periods, then run the constrained
% smoother and verify (a) the constraint is satisfied and (b) the
% constrained estimate is otherwise close to the unconstrained one.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 31);
rand('seed',  31);

% ----- Model: AR(1) NAIRU floating near zero -----
T  = 120;
F  = 0.9;
H  = 1;
Q  = 0.05;
R  = 0.15;                                  % noisy measurement
a0 = 0.3;
P0 = 1;

% Simulate with a positive mean so the true latent path is non-negative
% but close to the boundary in some periods.
[Y, X_true] = simks_simulate_const(T, F, H, Q, R, a0, P0);

% Add a deterministic positive shift to keep truth positive.
true_trend = max(0, X_true(1, 2:end));
Y = true_trend + sqrt(R) * randn(1, T);     % synthetic noisy NAIRU data

% ----- Unconstrained smoother -----
[X_unc, J, h, ~] = simks_smooth_const(Y, F, H, Q, R, a0, P0);

% ----- Constrained smoother: -x_t <= 0 for t = 0, 1, ..., T -----
n = 1;
N = (T+1) * n;
A_ineq = -speye(N);             % each row: -x_i <= 0
b_ineq = zeros(N, 1);

[X_con, info_con] = simks_smooth_constrained(Y, F, H, Q, R, a0, P0, ...
                                              A_ineq, b_ineq, ...
                                              struct('verbose', false));

% ----- Report -----
fprintf('Constrained smoother demo: x_t >= 0 constraint\n');
fprintf('===============================================\n');
fprintf('T = %d, n = 1, c = %d constraints (-x_t <= 0 for all t)\n\n', T, N);

n_neg_unc = sum(X_unc(1, 2:end) < -1e-6);
min_unc   = min(X_unc(1, 2:end));
min_con   = min(X_con(1, 2:end));

fprintf('Unconstrained smoother:\n');
fprintf('  Number of periods where x_t < 0:       %d / %d\n', n_neg_unc, T);
fprintf('  Minimum smoothed value (constraint = 0): %.4f\n', min_unc);

fprintf('\nConstrained smoother:\n');
fprintf('  Newton iterations (total):            %d\n', info_con.iters_newton);
fprintf('  Outer barrier reductions:             %d\n', info_con.iters_outer);
fprintf('  Converged:                            %d\n', info_con.converged);
fprintf('  Max constraint violation:             %.2e\n', info_con.max_violation);
fprintf('  Number of active constraints:         %d\n', sum(info_con.active));
fprintf('  Minimum smoothed value:               %.4f\n', min_con);

% Where the constraint binds, the constrained estimate is at 0; elsewhere
% it should closely track the unconstrained estimate.
diff_norm = norm(X_con(1, 2:end) - X_unc(1, 2:end), inf);
fprintf('\nMax |X_con - X_unc| across periods:    %.4f\n', diff_norm);
fprintf('  (Large where the constraint binds; zero elsewhere.)\n');

% Quality of recovery vs truth
rmse_u = sqrt(mean((X_unc(1, 2:end) - true_trend).^2));
rmse_c = sqrt(mean((X_con(1, 2:end) - true_trend).^2));
fprintf('\nRMSE vs true (non-negative) trend:\n');
fprintf('  Unconstrained:  %.4f\n', rmse_u);
fprintf('  Constrained:    %.4f\n', rmse_c);

% --- Plot ----------------------------------------------------------------
% Single panel: observed data, true non-negative trend, unconstrained
% smoother (often goes negative), constrained smoother (pinned at 0).
tt   = 1:T;
fig  = figure('visible', 'on', 'position', [100 100 900 500]);
hold on; box on;

% Reference line at zero.
plot([0 T], [0 0], '-', 'color', [0.4 0.4 0.4]);

plot(tt, Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
plot(tt, true_trend,        'k-',  'linewidth', 1.4);
plot(tt, X_unc(1, 2:end),   'b--', 'linewidth', 1.2);
plot(tt, X_con(1, 2:end),   'r-',  'linewidth', 1.6);

% Highlight the periods where the unconstrained estimate goes below zero.
neg_idx = find(X_unc(1, 2:end) < -1e-6);
if ~isempty(neg_idx)
  plot(neg_idx, X_unc(1, neg_idx + 1), 'o', 'color', [0.2 0.4 0.9], ...
       'markersize', 5, 'markerfacecolor', [0.2 0.4 0.9]);
end

legend({'zero', 'noisy observation', 'true (non-negative) trend', ...
        'unconstrained smoother', 'constrained smoother', ...
        'unconstrained < 0'}, 'location', 'best');
title('Constrained smoothing: enforcing x_t \geq 0 via interior-point QP');
xlabel('t'); ylabel('x_t');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_constrained.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
