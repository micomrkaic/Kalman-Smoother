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

% DEMO_COMPARE_RECURSIVE  Side-by-side check: simultaneous smoother vs
% textbook Kalman filter + Rauch-Tung-Striebel smoother.
%
% This demo serves two purposes:
%   (i)  it confirms the equivalence of the two formulations to numerical
%        precision;
%   (ii) the reference implementation in simks_rts_reference.m documents
%        the recursive algorithm explicitly, so that readers familiar with
%        the textbook presentation can match it line-by-line to the
%        sparse block-tridiagonal LDL' factorization derived in the note.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 7);

T  = 200;
F  = [0.8 0.1; 0.0 0.7];
H  = [1.0 0.0; 0.0 1.0];
Q  = [0.05 0.01; 0.01 0.04];
R  = [0.20 0.00; 0.00 0.15];
a0 = [0;0];
P0 = eye(2);

[Y, Xtrue] = simks_simulate_const(T, F, H, Q, R, a0, P0);

% Simultaneous smoother with posterior covariance.
sopts.return_cov = true;
[Xsim, ~, ~, info_sim] = simks_smooth_const(Y, F, H, Q, R, a0, P0, sopts);

% Classical Kalman filter + RTS smoother (reference implementation).
[Xrts, Prts] = simks_rts_reference(Y, F, H, Q, R, a0, P0);

% Compare means.
err_mean = max(max(abs(Xsim - Xrts)));

% Compare diagonals of posterior covariance for state 1.
diag_sim = zeros(1, T+1);
diag_rts = zeros(1, T+1);
for t = 0:T
  diag_sim(t+1) = info_sim.Ptt{t+1}(1,1);
  diag_rts(t+1) = Prts{t+1}(1,1);
end
err_cov = max(abs(diag_sim - diag_rts));

fprintf('Simultaneous smoother vs textbook RTS, T = %d\n', T);
fprintf('max |Xsim - Xrts|              = %.3e\n', err_mean);
fprintf('max |Var_sim - Var_rts| state 1 = %.3e\n', err_cov);
if err_mean < 1e-9 && err_cov < 1e-9
  fprintf('PASS: the two methods agree to within roundoff.\n');
else
  fprintf('Discrepancy larger than roundoff -- check implementation.\n');
end
fprintf('\nThis is what is meant by "the recursive filter is a specialized\n');
fprintf('sparse factorization of the same block-tridiagonal system".\n');

% --- Plot ----------------------------------------------------------------
% Top: the two smoothed paths on top of each other (should look identical).
% Bottom: their pointwise difference, which is at machine precision.
tt = 0:T;
fig = figure('visible', 'on', 'position', [100 100 900 600]);

subplot(2,1,1); hold on; box on;
plot(tt, Xsim(1,:), 'b-',  'linewidth', 1.5);
plot(tt, Xrts(1,:), 'r--', 'linewidth', 1.5);
legend({'simultaneous (J\\h)', 'recursive (Kalman+RTS)'}, 'location', 'best');
title('Smoothed state 1: two methods overlaid');
xlabel('t'); ylabel('x_1');

subplot(2,1,2); hold on; box on;
plot(tt, Xsim(1,:) - Xrts(1,:), 'k-', 'linewidth', 1.0);
title(sprintf('Pointwise difference, state 1  (max abs = %.2e)', ...
              max(abs(Xsim(1,:) - Xrts(1,:)))));
xlabel('t'); ylabel('Xsim - Xrts');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_compare_recursive.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
