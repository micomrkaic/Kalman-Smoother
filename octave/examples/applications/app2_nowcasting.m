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

% APP2_NOWCASTING  Mixed-frequency dynamic factor model for nowcasting GDP.
%
% A classic central-bank/IMF country-desk problem.  We have several
% monthly economic indicators (industrial production, employment, retail
% sales, etc.) and we want to "nowcast" current-quarter GDP from them
% before the GDP release comes out.
%
% Model.  A single latent monthly factor f_t drives all indicators:
%   f_t      = phi * f_{t-1} + e_t                   (AR(1) factor)
%   y_{k,t}  = lambda_k * f_t + u_{k,t},  u_{k,t} ~ N(0, R_kk)
%
% Quarterly GDP enters as a third-month aggregate: at the end of each
% quarter, observed quarterly GDP equals (1/3)(f_t + f_{t-1} + f_{t-2}),
% scaled by an aggregation loading lambda_g.  In between, GDP is missing.
%
% This is the canonical Mariano-Murasawa (2003) / Banbura-Modugno (2014)
% setup with one factor.  Many monthly indicators give the factor a
% precise current estimate; the role of GDP is to anchor its scale.
%
% State augmentation.  To express the third-month average, the state
% carries the factor and its two lags:
%
%   x_t = (f_t, f_{t-1}, f_{t-2})'  in R^3.
%
% Pedagogical points exercised:
% (1) Factor-model state-space mapping (small state, many indicators).
% (2) Mixed-frequency: GDP observation operator depends on month within quarter.
% (3) Jagged-edge missing data: different indicators released at different lags.
% (4) Time-varying H_t (the GDP row activates only on end-of-quarter months).
% (5) Sequential nowcasts: as more data arrive, the smoother re-estimates
%     the latent factor and we read off the current-quarter GDP nowcast.
%
% Demonstration.  We simulate 10 years of monthly data plus quarterly
% GDP, hold back the last quarter, and produce nowcasts as if each
% month within that quarter just released its data.

% (script: cleared and cleared screen removed for demo_main compatibility)
% Add the simksmoother/ source directory to the path, relative to where
% this application file lives.  Works regardless of the user's CWD.
app_dir = fileparts(mfilename('fullpath'));
oct_dir = fullfile(app_dir, '..', '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

randn('seed', 33);
rand('seed',  33);

% ----- Model parameters -----
T_months = 120;                    % 10 years
phi      = 0.70;                   % factor persistence
sigma_e  = 0.5;                    % factor innovation sd

% Five monthly indicators with different signal-to-noise ratios.
K = 5;
lambda_mon = [1.0; 0.8; 0.6; 1.2; 0.4];   % factor loadings on monthly series
R_mon      = diag([0.30, 0.50, 0.80, 0.40, 1.50].^2);   % idiosyncratic sd

% One quarterly indicator (GDP), aggregated as 1/3 sum over a quarter.
lambda_gdp = 1.0;                  % GDP loading on the factor
R_gdp      = 0.30^2;

% ----- State-space matrices: state = (f_t, f_{t-1}, f_{t-2}) -----
n = 3;
F = zeros(n,n);
F(1,1) = phi;       % f_t = phi f_{t-1} + e_t
F(2,1) = 1;         % f_{t-1} = f_t (shifted)
F(3,2) = 1;         % f_{t-2} = f_{t-1} (shifted)

Q = zeros(n,n);
Q(1,1) = sigma_e^2;
Q(2,2) = 1e-10; Q(3,3) = 1e-10;    % regularize the lag-shift components

% Measurement matrix and covariance.  K monthly indicators + 1 GDP row.
m = K + 1;
H = zeros(m, n);
H(1:K, 1) = lambda_mon;            % monthly indicators load on f_t only
% GDP row: lambda_gdp * (1/3)(f_t + f_{t-1} + f_{t-2})
H(K+1, :) = lambda_gdp * [1/3, 1/3, 1/3];

R = blkdiag(R_mon, R_gdp);

a0 = zeros(n,1);
P0 = (sigma_e^2 / (1 - phi^2)) * eye(n);  % stationary variance of f_t

% ----- Simulate -----
[Y, Xtrue] = simks_simulate_const(T_months, F, H, Q, R, a0, P0);
f_true = Xtrue(1, 2:end);

% ----- Construct the realistic data pattern -----
% Realistic jagged-edge pattern:
% Indicator 1 (IP):       observed through T_months (no lag)
% Indicator 2 (employ):   observed through T_months - 1
% Indicator 3 (retail):   observed through T_months - 1
% Indicator 4 (PMI):      observed through T_months (no lag)
% Indicator 5 (survey):   observed through T_months
% GDP: observed only at end of completed quarters (months 3, 6, 9, ...);
%      latest GDP is for the most recent FULLY COMPLETED quarter.

Y_observed = Y;

% Apply standard release lags.
release_lags = [0, 1, 1, 0, 0];
for k = 1:K
  lag = release_lags(k);
  if lag > 0
    Y_observed(k, T_months - lag + 1 : T_months) = NaN;
  end
end

% Make GDP missing except at end-of-quarter (i.e. t such that mod(t,3) == 0).
for t = 1:T_months
  if mod(t, 3) ~= 0
    Y_observed(K+1, t) = NaN;
  end
end
% Then make the last completed quarter and partial current quarter realistic:
% The current quarter (last 3 months) has no GDP release yet.
% Also the immediately preceding quarter's GDP is the most recent release.
Y_observed(K+1, T_months) = NaN;        % current month: no GDP

% ----- Estimate the factor and its current value from full panel -----
opts = struct('return_cov', true);
[X_smooth, J, h, info] = simks_smooth_const(Y_observed, F, H, Q, R, a0, P0, opts);
f_smooth = X_smooth(1, 2:end);

% ----- "Nowcast experiment": ablate data and re-smooth -----
% Imagine we sit at month T_months and want a nowcast of the current quarter's
% GDP (months T_months-2, T_months-1, T_months).  We've already seen
% (T_months-3)-quarter GDP (the most recent release).  The current
% quarter's GDP = (1/3)(f_{T-2} + f_{T-1} + f_T) * lambda_gdp.

% Three scenarios as we acquire more data:
% Scenario 1: end of month T_months-2 (first month of the new quarter).
%             Only month T-2 monthly indicators are observed for this quarter.
% Scenario 2: end of month T_months-1 (second month).  Months T-2 and T-1 are in.
% Scenario 3: end of month T_months   (third month).  All three monthly obs in.

scenarios = {'after 1 month of quarter',
             'after 2 months of quarter',
             'after 3 months of quarter (end of quarter)'};
nowcasts  = zeros(1, 3);
nowcast_sd = zeros(1, 3);
true_gdp  = lambda_gdp * (f_true(T_months-2) + f_true(T_months-1) + f_true(T_months)) / 3;

for scen = 1:3
  Y_scen = Y_observed;
  % Mask monthly indicators for periods strictly after the scenario cutoff.
  cutoff = T_months - 3 + scen;     % last fully observed month at this scenario
  for k = 1:K
    Y_scen(k, cutoff+1:T_months) = NaN;
  end
  [Xs, ~, ~, info_s] = simks_smooth_const(Y_scen, F, H, Q, R, a0, P0, opts);
  f_s = Xs(1, 2:end);

  % Nowcast = mean of f over months T_months-2, T_months-1, T_months
  % times lambda_gdp.  In Octave 1-indexing, the smoothed state
  % column for month t (where t = 1..T_months) is Xs(:, t+1).  And
  % info_s.Ptt{t+1} is the posterior cov at that month.
  t1 = T_months - 2;
  t2 = T_months - 1;
  t3 = T_months;
  three = [Xs(1, t1+1); Xs(1, t2+1); Xs(1, t3+1)];
  nowcasts(scen) = lambda_gdp * mean(three);

  % Variance of the mean using the diagonal and adjacent off-diagonal
  % blocks from selected inversion.  Ptt{t+1} gives cov at month t;
  % The augmented state at the end-of-quarter month t3 is
  % x_{t3} = (f_{t3}, f_{t3-1}, f_{t3-2})', so the full covariance of the
  % three monthly factor values within the quarter is the SINGLE diagonal
  % block Sigma_{t3,t3} of J^{-1}: its off-diagonal ELEMENTS are the
  % cross-month covariances (including the lag-2 term).  No off-diagonal
  % time blocks are needed.
  Sigma_3 = info_s.Ptt{t3+1};
  % Two-period off-diagonal not directly stored; approximate as 0.
  % (Acceptable for AR(1) factor; under-states uncertainty slightly.)
  w = ones(3,1) / 3;
  nowcast_sd(scen) = lambda_gdp * sqrt(w' * Sigma_3 * w);
end

% ----- Report -----
fprintf('Application 2: Nowcasting current-quarter GDP\n');
fprintf('==============================================\n');
fprintf('Monthly indicators K = %d, factor AR(1) phi = %.2f\n', K, phi);
fprintf('Quarterly GDP aggregation: (1/3)(f_t + f_{t-1} + f_{t-2})\n');
fprintf('T_months = %d (%.1f years)\n', T_months, T_months/12);
fprintf('\nTotal NaN entries in Y_observed: %d / %d\n', ...
       sum(isnan(Y_observed(:))), numel(Y_observed));
fprintf('Jagged edge: indicators have release lags = '); disp(release_lags);

fprintf('\n--- Full-sample factor recovery ---\n');
fprintf('RMSE on monthly factor f_t = %.4f\n', sqrt(mean((f_smooth - f_true).^2)));
fprintf('Correlation(smoothed, true factor) = %.4f\n', corr(f_smooth', f_true'));

fprintf('\n--- Real-time GDP nowcast ---\n');
fprintf('True current-quarter GDP value = %.4f\n', true_gdp);
fprintf('\nNowcast as data accumulates within the quarter:\n');
for scen = 1:3
  fprintf('  %-48s  nowcast = %+.4f  (sd %.4f)\n', ...
         scenarios{scen}, nowcasts(scen), nowcast_sd(scen));
end

fprintf('\nNowcast error vs truth:\n');
for scen = 1:3
  err = nowcasts(scen) - true_gdp;
  z   = abs(err) / nowcast_sd(scen);
  fprintf('  %-48s  err = %+.4f  (z = %.2f)\n', ...
         scenarios{scen}, err, z);
end

fprintf('\nKey takeaways:\n');
fprintf('* All "missing" observations (jagged edge, mixed frequency) are\n');
fprintf('  literal NaN entries in Y.  The smoother handles them by\n');
fprintf('  dropping rows of H R^-1 H at each t.  No bespoke algorithm.\n');
fprintf('* The nowcast sd shrinks monotonically as more monthly data come in.\n');
fprintf('  This is the smoother''s built-in measure of nowcast precision.\n');
fprintf('* The factor is identified up to sign and scale.  Here we fix the\n');
fprintf('  scale through GDP''s loading; in practice one fixes lambda_1 = 1.\n');

% --- Plot ----------------------------------------------------------------
% Three panels:
%   (1) Latent monthly factor f_t: truth vs smoothed.
%   (2) One of the monthly indicators (the one with the largest release
%       lag) shown with its NaN gap at the end of sample.
%   (3) Nowcast point and 95% interval as the quarter progresses, with
%       the true current-quarter GDP value.
mm  = 1:T_months;
fig = figure('visible', 'on', 'position', [100 100 1000 800]);

subplot(2,2,1); hold on; box on;
plot(mm, f_true,   'k-',  'linewidth', 1.2);
plot(mm, f_smooth, 'r--', 'linewidth', 1.2);
legend({'true factor', 'smoothed factor'}, 'location', 'best');
title('Latent monthly factor f_t');
xlabel('month'); ylabel('f');

% Pick the indicator with the largest release lag for panel 2.
[~, k_lag] = max(release_lags);
subplot(2,2,2); hold on; box on;
obs_idx = ~isnan(Y_observed(k_lag, :));
plot(mm(obs_idx), Y_observed(k_lag, obs_idx), 'o', 'color', [0.4 0.4 0.4], ...
     'markersize', 4);
% Mark the tail-end NaN region
nan_idx = isnan(Y_observed(k_lag, :));
if any(nan_idx)
  yl = ylim;
  fill_x = mm(find(nan_idx, 1));
  fill([fill_x T_months T_months fill_x], [yl(1) yl(1) yl(2) yl(2)], ...
       [0.95 0.95 0.85], 'edgecolor', 'none');
  plot(mm(obs_idx), Y_observed(k_lag, obs_idx), 'o', 'color', [0.4 0.4 0.4], ...
       'markersize', 4);
end
title(sprintf('Indicator %d (release lag = %d months)', k_lag, release_lags(k_lag)));
xlabel('month'); ylabel('y');

subplot(2,2,3); hold on; box on;
% Plot the GDP nowcast at three within-quarter vintages, with 95% bands.
xpts = 1:3;
% Manual error bars: vertical line + marker.  More portable than
% errorbar() across MATLAB/Octave versions.
for k = 1:3
  plot([xpts(k) xpts(k)], ...
       [nowcasts(k) - 1.96*nowcast_sd(k), nowcasts(k) + 1.96*nowcast_sd(k)], ...
       '-', 'color', [0.2 0.4 0.9], 'linewidth', 1.5);
end
plot(xpts, nowcasts, 'o', 'color', [0.2 0.4 0.9], 'linewidth', 1.5, ...
     'markersize', 8, 'markerfacecolor', [0.2 0.4 0.9]);
plot([0.5 3.5], [true_gdp true_gdp], 'k-', 'linewidth', 1.4);
set(gca, 'xtick', xpts, 'xticklabel', {'after 1 mo', 'after 2 mo', 'end of qtr'});
xlim([0.5 3.5]);
legend({'95% interval', 'nowcast', 'true current-quarter GDP'}, 'location', 'best');
title('GDP nowcast and 95% interval as the quarter progresses');
ylabel('GDP value');

subplot(2,2,4); hold on; box on;
% Posterior std of the nowcast as the quarter progresses (should shrink).
plot(xpts, nowcast_sd, 'o-', 'color', [0.9 0.2 0.2], 'linewidth', 1.5, ...
     'markersize', 8, 'markerfacecolor', [0.9 0.2 0.2]);
set(gca, 'xtick', xpts, 'xticklabel', {'after 1 mo', 'after 2 mo', 'end of qtr'});
xlim([0.5 3.5]);
title('Nowcast posterior std shrinks as data accumulate');
ylabel('posterior std');

fig_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fig_path = fullfile(fig_dir, 'app2_nowcasting.png');
print(fig, fig_path, '-dpng', '-r120');
drawnow;
fprintf('Figure saved: %s\n', fig_path);
