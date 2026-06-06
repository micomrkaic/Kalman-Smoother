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

% DEMO_SLDS  Regime-switching state-space smoother demonstration.
%
% A scalar output gap follows an AR(1) law, but with two regimes:
%   "expansion":  persistence phi_1 = 0.85, low innovation variance
%   "recession":  persistence phi_2 = 0.50 (mean reversion strengthens),
%                 high innovation variance
%
% We simulate a series with three regime episodes interleaved, then
% run the SLDS smoother to recover both the latent path and the
% smoothed regime probabilities.
%
% This is the canonical kind of model where a single-regime Gaussian
% smoother gives biased estimates: it averages out the volatility
% differences and the persistence shifts, smearing the recession
% episodes.  The SLDS smoother recovers the regime structure directly
% from the data.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this demo file lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

% Portable RNG (Park-Miller LCG + Box-Muller), implemented identically
% in demo_slds.m and demo_slds.py so that Python, Octave, and MATLAB
% simulate BIT-IDENTICAL data (the uniforms are exact in double
% precision; the normals agree to the last ulp of libm).  This makes
% the demo's printed numbers directly comparable across languages.
seed = 17;

% ----- Model: two regimes -----
n  = 1;
m  = 1;
H  = 1;
% Measurement noise must be small relative to the quiet-regime
% innovation variance (here sigma^2_1 = 0.02) for regime membership to
% be sharply identified: the exact variational q(S)-update uses EXPECTED
% transition log-likelihoods, which correctly account for posterior
% state uncertainty.  With R comparable to sigma^2_1, even an oracle
% observing the true states classifies regimes imperfectly.
R  = 0.01;

F_set = {0.85, 0.50};                                % expansion, recession
Q_set = {0.02, 0.20};                                % volatility 10x in recession
A_true = [0.95, 0.05;                                 % regime transition
          0.10, 0.90];
pi0    = [1; 0];                                      % start in expansion

T = 240;                                              % 60 years of quarterly data

% ----- Portable draws (must mirror _portable_draws in demo_slds.py) -----
m_lcg = 2147483647;  a_lcg = 16807;
state = seed;
nU = 5 * T;
U  = zeros(1, nU);
for i = 1:nU
  state = mod(a_lcg * state, m_lcg);    % exact in double (< 2^53)
  U(i)  = state / m_lcg;
end
u_reg   = U(1:T-1);                                       % regime-switch uniforms
z       = sqrt(-2 * log(U(T:2:5*T-2))) .* cos(2*pi * U(T+1:2:5*T-1));
e_state = z(1:T);                                          % state innovations
e_meas  = z(T+1:2*T);                                      % measurement noise

% ----- Simulate true regimes and states -----
s     = zeros(1, T);
s(1)  = 1;
for t = 2:T
  s(t) = (u_reg(t-1) < A_true(s(t-1), 2)) + 1;             % 1 or 2
end

x = zeros(1, T+1);
x(1) = 0;
for t = 1:T
  x(t+1) = F_set{s(t)} * x(t) + sqrt(Q_set{s(t)}) * e_state(t);
end
Y = H * x(2:end) + sqrt(R) * e_meas;

% ----- Estimator A: Gaussian smoother, one regime (averaged) -----
F_avg = mean(cell2mat(F_set));
Q_avg = mean(cell2mat(Q_set));
[Xa] = simks_smooth_const(Y, F_avg, H, Q_avg, R, 0, 1);

% ----- Estimator B: SLDS smoother -----
opts_b = struct('A', A_true, 'pi0', pi0, 'max_iter', 50, 'tol', 1e-5);
[Xb, infoB] = simks_smooth_slds(Y, F_set, H, Q_set, R, 0, 1, opts_b);

% ----- Results -----
truth = x(2:end);
err_a = Xa(1, 2:end) - truth;
err_b = Xb(1, 2:end) - truth;
rmse  = @(e) sqrt(mean(e.^2));

fprintf('SLDS smoother demo: regime-switching AR(1)\n');
fprintf('==========================================\n');
fprintf('T = %d, K = 2 regimes\n', T);
fprintf('True F per regime: phi_1 = %.2f (expansion), phi_2 = %.2f (recession)\n', ...
       F_set{1}, F_set{2});
fprintf('True Q per regime: sigma^2_1 = %.3f, sigma^2_2 = %.3f\n', Q_set{1}, Q_set{2});
fprintf('True transition matrix:\n');
disp(A_true);

fprintf('Regime composition of truth: %d expansion, %d recession periods\n', ...
       sum(s == 1), sum(s == 2));

fprintf('\nSmoothing RMSE:\n');
fprintf('  (A) Gaussian, averaged single regime  %.4f\n', rmse(err_a));
fprintf('  (B) SLDS smoother                     %.4f   [%d iters, converged=%d]\n', ...
       rmse(err_b), infoB.iters, infoB.converged);

% ----- Regime classification: pi(2,:) > 0.5 means "recession" predicted -----
pred_regime = 1 + (infoB.pi(2,:) > 0.5);
accuracy    = mean(pred_regime == s);
recession_recall = sum(pred_regime == 2 & s == 2) / max(sum(s == 2), 1);
recession_prec   = sum(pred_regime == 2 & s == 2) / max(sum(pred_regime == 2), 1);

fprintf('\nRegime classification (predicted vs true, threshold pi_2 > 0.5):\n');
fprintf('  Overall accuracy:               %.3f\n', accuracy);
fprintf('  Recall  (correctly flagged):    %.3f\n', recession_recall);
fprintf('  Precision (flag was a recession): %.3f\n', recession_prec);
fprintf('  Mean pi_2 during true recessions:    %.3f\n', mean(infoB.pi(2, s == 2)));
fprintf('  Mean pi_2 during true expansions:    %.3f\n', mean(infoB.pi(2, s == 1)));

% --- Plot ----------------------------------------------------------------
% Top: state trajectory (true, Gaussian, SLDS) with recession periods shaded.
% Bottom: SLDS-estimated recession probability pi_2 vs the true regime.
tt   = 1:T;
fig  = figure('visible', 'on', 'position', [100 100 900 700]);

subplot(2,1,1); hold on; box on;
% Shade recession periods (s==2) lightly.
yl = [min(x) - 0.5, max(x) + 0.5];
for t = 1:T
  if s(t) == 2
    fill([t-0.5 t+0.5 t+0.5 t-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
         [1.00 0.85 0.85], 'edgecolor', 'none');
  end
end
plot(tt, Y, 'o', 'color', [0.6 0.6 0.6], 'markersize', 3);
plot(tt, x(2:end),         'k-',  'linewidth', 1.2);
plot(tt, Xa(1, 2:end),    'b--', 'linewidth', 1.2);
plot(tt, Xb(1, 2:end),    'r--', 'linewidth', 1.2);
ylim(yl);
legend({'recessions (truth)', 'observation', 'true x_t', ...
        'single-regime Gaussian', 'SLDS smoother'}, ...
       'location', 'southwest');
title('SLDS smoothing: high-volatility recessions and low-volatility expansions');
xlabel('t'); ylabel('output gap');

subplot(2,1,2); hold on; box on;
% True regime as a step plot at 0 (expansion) or 1 (recession).
stairs(tt, s - 1, 'k-', 'linewidth', 1.2);
plot(tt, infoB.pi(2, :), 'r-', 'linewidth', 1.2);
plot([1 T], [0.5 0.5], 'k:');
ylim([-0.1 1.1]);
legend({'true regime (0=exp, 1=rec)', 'estimated \pi_t(recession)', ...
        'threshold 0.5'}, 'location', 'best');
title('Smoothed regime probability vs ground truth');
xlabel('t'); ylabel('probability');

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'demo_slds.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
