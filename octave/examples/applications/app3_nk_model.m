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

% APP3_NK_MODEL  Semi-structural 3-equation New Keynesian model.
%
% Backward-looking reduced form (the empirical workhorse used at central
% banks for semi-structural projections; see e.g. Berg-Karam-Laxton 2006,
% IMF WP 06/80).  Three equations link the output gap, inflation, and
% the policy rate, with structural shocks that we treat as latent.
%
%   ytilde_t = alpha_1 ytilde_{t-1} + alpha_2 (i_{t-1} - pi_{t-1} - r_bar)
%               + eps^y_t                                        (IS curve)
%   pi_t     = beta_1 pi_{t-1} + (1 - beta_1) pi_bar
%               + beta_2 ytilde_{t-1} + eps^pi_t                 (NK Phillips)
%   i_t      = rho_i i_{t-1}
%               + (1 - rho_i) (r_bar + pi_bar + phi_pi(pi_t - pi_bar)
%                              + phi_y ytilde_t)
%               + eps^i_t                                        (Taylor rule)
%
% Calibration: r_bar = 0.5 (quarterly natural rate), pi_bar = 0.5
% (quarterly inflation target, 2% annualized).
%
% Pedagogical points exercised:
% (1) Multi-equation simultaneous system: the Taylor rule depends on
%     CONTEMPORANEOUS pi_t and ytilde_t, requiring solving the system
%     forward at each t.  We handle this by working with the reduced
%     form after eliminating the simultaneity.
% (2) State = (ytilde_t, pi_t, i_t), observation = noisy versions of all three.
% (3) Smoothing decomposes observed series into demand, supply, and
%     policy shocks (the three components of the innovation w_t).
% (4) EM with B,D estimation recovers the structural slopes (kappa, sigma,
%     phi_pi) from raw data.
% (5) The smoothed shocks are useful in their own right -- e.g. for
%     story-telling around recessions: "this episode was 2/3 demand,
%     1/3 monetary."
%
% Reduced-form transformation.  Substituting the Taylor rule into the
% definitions, we get a contemporaneously closed system.  Let
%   alpha = 1 - rho_i,  k_p = alpha * phi_pi,  k_y = alpha * phi_y.
% The instantaneous Taylor effect on (ytilde, pi, i) can be eliminated
% by solving for i_t analytically and absorbing it into the F matrix.
% For pedagogical simplicity, we use a slightly modified Taylor rule
% that depends on LAGGED pi and ytilde (so the system is purely
% backward-looking and F is straightforward):
%
%   i_t = rho_i i_{t-1} + (1-rho_i)(r_bar + pi_bar
%         + phi_pi (pi_{t-1} - pi_bar) + phi_y ytilde_{t-1}) + eps^i_t.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this application file lives.  Works regardless of the user's CWD.
app_dir = fileparts(mfilename('fullpath'));
oct_dir = fullfile(app_dir, '..', '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 21);
rand('seed',  21);

% ----- True parameters (quarterly, conventional values) -----
T        = 200;          % 50 years of quarterly data
alpha_1  = 0.70;         % IS curve persistence
alpha_2  = -0.10;        % IS curve real-rate sensitivity (negative)
beta_1   = 0.55;         % Phillips curve persistence
beta_2   = 0.15;         % Phillips curve slope (=kappa)
rho_i    = 0.80;         % Taylor rule smoothing
phi_pi   = 1.50;         % Taylor on inflation
phi_y    = 0.50;         % Taylor on output gap
r_bar    = 0.5;          % quarterly natural rate (~2% annual)
pi_bar   = 0.5;          % quarterly inflation target (~2% annual)

sigma_y  = 0.50;         % demand-shock sd
sigma_pi = 0.40;         % cost-push shock sd
sigma_i  = 0.30;         % monetary shock sd

% Measurement noise
sigma_meas_y  = 0.20;    % output gap measured with error (e.g. via real-time GDP)
sigma_meas_pi = 0.05;    % CPI inflation is observed essentially exactly
sigma_meas_i  = 0.05;    % policy rate is observed essentially exactly

% ----- State-space mapping -----
% State: x_t = (ytilde_t, pi_t, i_t)' in R^3
% F matrix (uses backward-looking Taylor rule with lagged pi, ytilde):
n = 3;
F = zeros(n, n);

% ytilde_t = alpha_1 ytilde_{t-1} + alpha_2 (i_{t-1} - pi_{t-1} - r_bar)
F(1, 1) = alpha_1;
F(1, 2) = -alpha_2;       % -alpha_2 * pi_{t-1}
F(1, 3) =  alpha_2;       %  alpha_2 * i_{t-1}
% Constant term: -alpha_2 * r_bar -- absorbed via the exogenous input below.

% pi_t = beta_1 pi_{t-1} + beta_2 ytilde_{t-1} + (1-beta_1) pi_bar
F(2, 1) = beta_2;
F(2, 2) = beta_1;
% Constant: (1-beta_1) pi_bar -- absorbed below.

% i_t = rho_i i_{t-1} + (1-rho_i)(r_bar + pi_bar + phi_pi(pi_{t-1} - pi_bar) + phi_y ytilde_{t-1})
F(3, 1) = (1 - rho_i) * phi_y;
F(3, 2) = (1 - rho_i) * phi_pi;
F(3, 3) = rho_i;
% Constant: (1-rho_i)(r_bar + pi_bar - phi_pi*pi_bar)  -- absorbed below.

% Exogenous z_t = 1 (constant), with B carrying the deterministic intercepts.
B = zeros(n, 1);
B(1) = -alpha_2 * r_bar;
B(2) = (1 - beta_1) * pi_bar;
B(3) = (1 - rho_i) * (r_bar + pi_bar - phi_pi * pi_bar);

Z = ones(1, T);

% Innovation covariance: structural shocks assumed mutually uncorrelated.
Q = diag([sigma_y^2, sigma_pi^2, sigma_i^2]);

% Measurement: y_t = x_t + measurement_noise (we observe all three with noise).
m = 3;
H = eye(m, n);
D = zeros(m, 1);
R = diag([sigma_meas_y^2, sigma_meas_pi^2, sigma_meas_i^2]);

% Initial conditions
a0 = [0; pi_bar; r_bar + pi_bar];
P0 = diag([1, 0.5, 0.5]);

% ----- Simulate -----
[Y, Xtrue] = simks_simulate_const(T, F, H, Q, R, a0, P0, ...
                                   struct('Z', Z, 'B', B, 'D', D));

ytilde_true = Xtrue(1, 2:end);
pi_true     = Xtrue(2, 2:end);
i_true      = Xtrue(3, 2:end);

% Compute the latent structural shocks (residuals).
% eps_y_t   = ytilde_t - F(1,:) * x_{t-1} - B(1)
% eps_pi_t  = pi_t     - F(2,:) * x_{t-1} - B(2)
% eps_i_t   = i_t      - F(3,:) * x_{t-1} - B(3)
W_true = zeros(n, T);
for t = 1:T
  W_true(:, t) = Xtrue(:, t+1) - F * Xtrue(:, t) - B;
end

% ----- Estimator A: Gaussian smoother with TRUE parameters -----
[X_a, J, h, info_a] = simks_smooth_const(Y, F, H, Q, R, a0, P0, ...
                                          struct('Z', Z, 'B', B, 'D', D));

% Recover the smoothed structural shocks.
W_smooth = zeros(n, T);
for t = 1:T
  W_smooth(:, t) = X_a(:, t+1) - F * X_a(:, t) - B;
end

% ----- Estimator B: EM with parameters unknown -----
em_opts = struct( ...
  'Z',        Z, ...
  'F',        0.7 * eye(n) + 0.05 * randn(n, n), ...
  'H',        H, ...
  'Q',        0.5 * eye(n), ...
  'R',        diag([0.1, 0.01, 0.01]), ...
  'B',        zeros(n, 1), ...
  'D',        zeros(m, 1), ...
  'a0',       a0, ...
  'P0',       P0, ...
  'max_iter', 120, ...
  'tol',      1e-7);

[params_b, X_b, hist_b] = simks_em_const(Y, n, em_opts);

% Recover EM-implied structural parameters from the estimated F and B.
F_est = params_b.F;
B_est = params_b.B;

% Read parameters back from F:
alpha_1_est = F_est(1, 1);
alpha_2_est = F_est(1, 3);                              % coefficient on i_{t-1}
beta_1_est  = F_est(2, 2);
beta_2_est  = F_est(2, 1);
rho_i_est   = F_est(3, 3);
% Taylor coefficients via the (1-rho_i) factor:
if abs(1 - rho_i_est) > 0.01
  phi_y_est  = F_est(3, 1) / (1 - rho_i_est);
  phi_pi_est = F_est(3, 2) / (1 - rho_i_est);
else
  phi_y_est = NaN; phi_pi_est = NaN;
end

% ----- Report -----
fprintf('Application 3: Semi-structural 3-equation NK model\n');
fprintf('===================================================\n');
fprintf('T = %d quarters (~%.0f years)\n', T, T/4);
fprintf('\nTrue structural parameters:\n');
fprintf('  IS curve:    alpha_1 = %.2f  alpha_2 = %.2f\n', alpha_1, alpha_2);
fprintf('  Phillips:    beta_1  = %.2f  beta_2  = %.2f\n', beta_1, beta_2);
fprintf('  Taylor:      rho_i   = %.2f  phi_pi  = %.2f  phi_y = %.2f\n', ...
       rho_i, phi_pi, phi_y);

fprintf('\n--- Estimator A: Gaussian smoother, true parameters ---\n');
rmse_ytilde = sqrt(mean((X_a(1, 2:end) - ytilde_true).^2));
rmse_pi     = sqrt(mean((X_a(2, 2:end) - pi_true).^2));
rmse_i      = sqrt(mean((X_a(3, 2:end) - i_true).^2));
fprintf('  RMSE on ytilde:  %.4f  (signal sd = %.4f)\n', ...
       rmse_ytilde, std(ytilde_true));
fprintf('  RMSE on pi:      %.4f\n', rmse_pi);
fprintf('  RMSE on i:       %.4f\n', rmse_i);

% Correlation between recovered and true structural shocks.
rho_eps_y  = corr(W_smooth(1,:)', W_true(1,:)');
rho_eps_pi = corr(W_smooth(2,:)', W_true(2,:)');
rho_eps_i  = corr(W_smooth(3,:)', W_true(3,:)');
fprintf('\n  Correlation of smoothed vs true structural shocks:\n');
fprintf('    Demand    (eps^y) : %.3f\n', rho_eps_y);
fprintf('    Cost-push (eps^pi): %.3f\n', rho_eps_pi);
fprintf('    Monetary  (eps^i) : %.3f\n', rho_eps_i);

fprintf('\n--- Estimator B: EM, all structural parameters unknown ---\n');
fprintf('  EM iterations: %d\n', length(hist_b));
fprintf('  Final marginal log-likelihood: %.4f\n', hist_b(end));
fprintf('  Monotone non-decreasing: %d\n', all(diff(hist_b) >= -1e-6));
fprintf('\n  Parameter recovery (truth -> EM estimate):\n');
fprintf('    alpha_1  : %.3f -> %.3f\n', alpha_1, alpha_1_est);
fprintf('    alpha_2  : %.3f -> %.3f\n', alpha_2, alpha_2_est);
fprintf('    beta_1   : %.3f -> %.3f\n', beta_1, beta_1_est);
fprintf('    beta_2   : %.3f -> %.3f  (=kappa, the Phillips slope)\n', beta_2, beta_2_est);
fprintf('    rho_i    : %.3f -> %.3f\n', rho_i, rho_i_est);
fprintf('    phi_pi   : %.3f -> %.3f\n', phi_pi, phi_pi_est);
fprintf('    phi_y    : %.3f -> %.3f\n', phi_y, phi_y_est);

fprintf('\nKey takeaways:\n');
fprintf('* The smoother decomposes observed series into the three latent\n');
fprintf('  structural shocks (demand, cost-push, monetary).  This is\n');
fprintf('  exactly what semi-structural projections at central banks need.\n');
fprintf('* EM recovers the structural parameters from the data.  The\n');
fprintf('  Phillips slope beta_2 (= kappa) is the famously hard one;\n');
fprintf('  identification rests on the contemporaneous link from output\n');
fprintf('  gap to inflation.\n');
fprintf('* No bespoke "DSGE estimation routine" is needed.  The state-space\n');
fprintf('  formulation + standard EM does all the work, on the same sparse\n');
fprintf('  J that the basic smoother used.\n');

% --- Plot ----------------------------------------------------------------
% Six panels in a 3x2 layout:
%   Left column: latent observables (output gap, inflation, policy rate)
%                showing truth vs smoothed.
%   Right column: smoothed structural shocks (demand, cost-push, monetary)
%                 vs truth -- the central-bank-narrative quantities.
tt  = 1:T;
fig = figure('visible', 'on', 'position', [100 100 1100 900]);

X_a_path = X_a(:, 2:end);

% Left column: observables vs smoothed states
subplot(3,2,1); hold on; box on;
plot(tt, ytilde_true,    'k-',  'linewidth', 1.2);
plot(tt, X_a_path(1, :), 'r--', 'linewidth', 1.0);
legend({'true', 'smoothed'}, 'location', 'best');
title('Output gap \tilde y_t');
xlabel('quarter');

subplot(3,2,3); hold on; box on;
plot(tt, pi_true,        'k-',  'linewidth', 1.2);
plot(tt, X_a_path(2, :), 'r--', 'linewidth', 1.0);
title('Inflation \pi_t');
xlabel('quarter');

subplot(3,2,5); hold on; box on;
plot(tt, i_true,         'k-',  'linewidth', 1.2);
plot(tt, X_a_path(3, :), 'r--', 'linewidth', 1.0);
title('Policy rate i_t');
xlabel('quarter');

% Right column: structural shock decomposition
shock_names = {'Demand shock \epsilon_t^y', ...
               'Cost-push shock \epsilon_t^\pi', ...
               'Monetary shock \epsilon_t^i'};
shock_panels = [2 4 6];
for k = 1:3
  subplot(3, 2, shock_panels(k)); hold on; box on;
  plot(tt, W_true(k, :),   'k-',  'linewidth', 1.0);
  plot(tt, W_smooth(k, :), 'r--', 'linewidth', 1.0);
  plot([1 T], [0 0], ':', 'color', [0.5 0.5 0.5]);
  rho = corr(W_smooth(k, :)', W_true(k, :)');
  title(sprintf('%s   (corr = %.2f)', shock_names{k}, rho));
  xlabel('quarter');
end

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'app3_nk_model.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
