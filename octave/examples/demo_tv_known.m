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

% DEMO_TV_KNOWN  Time-varying known-parameter simultaneous smoother.
% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 2);

T = 80;
n = 2;
m = 1;
a0 = [0;0];
P0 = eye(n);

Fcell = cell(1,T);
Hcell = cell(1,T);
Qcell = cell(1,T);
Rcell = cell(1,T);

X = zeros(n,T+1);
Y = zeros(m,T);
X(:,1) = a0 + chol(P0,'lower') * randn(n,1);

for t = 1:T
  Ft = [0.75 + 0.10*sin(t/15), 0.2; 0, 0.55 + 0.05*cos(t/20)];
  Ht = [1, 0.2*sin(t/10)];
  Qt = 0.08 * eye(n);
  Rt = 0.20;

  Fcell{t} = Ft;
  Hcell{t} = Ht;
  Qcell{t} = Qt;
  Rcell{t} = Rt;

  X(:,t+1) = Ft * X(:,t) + chol(Qt,'lower') * randn(n,1);
  Y(:,t)   = Ht * X(:,t+1) + sqrt(Rt) * randn(m,1);
end

[Xhat, J, h, info] = simks_smooth_tv(Y, Fcell, Hcell, Qcell, Rcell, a0, P0);

fprintf('Time-varying simultaneous smoother demo\n');
fprintf('Sparse precision size: %d x %d, nnz = %d\n', size(J, 1), size(J, 2), info.nnz);
fprintf('RMSE state 1 = %.4f\n', sqrt(mean((Xhat(1,:) - X(1,:)).^2)));
fprintf('RMSE state 2 = %.4f\n', sqrt(mean((Xhat(2,:) - X(2,:)).^2)));

% --- Plot ----------------------------------------------------------------
% Top: the time-varying F_t entries we used.
% Bottom: smoothed vs true state 1.
tt    = 0:T;
F11_t = cellfun(@(M) M(1,1), Fcell);
F22_t = cellfun(@(M) M(2,2), Fcell);

fig = figure('visible', 'on', 'position', [100 100 900 600]);

subplot(2,1,1); hold on; box on;
plot(1:T, F11_t, 'b-', 'linewidth', 1.2);
plot(1:T, F22_t, 'r-', 'linewidth', 1.2);
legend({'F_{11}(t)', 'F_{22}(t)'}, 'location', 'best');
title('Time-varying dynamics: diagonal entries of F_t');
xlabel('t'); ylabel('value');

subplot(2,1,2); hold on; box on;
plot(tt, X(1,:),    'k-',  'linewidth', 1.2);
plot(tt, Xhat(1,:), 'r--', 'linewidth', 1.2);
plot(1:T, Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
legend({'true state 1', 'smoothed state 1', 'observation'}, 'location', 'best');
title('Smoothed vs true state under time-varying dynamics');
xlabel('t'); ylabel('x_1');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_tv_known.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
