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

% DEMO_MAIN  Five-minute interactive tour of the simultaneous Kalman smoother.
%
% Run this file from the examples/ directory with no arguments.  It will
% present a menu of demos, run any selection (or all), and print a brief
% summary of what each one shows.  No other Octave knowledge needed.
%
% Usage:
%   cd examples
%   demo_main
%
% Then select demos by number.  Select 0 to run all demos in sequence.

clear; clc;

% Add the simksmoother/ source directory to the path, relative to where
% this script lives.  Works regardless of the user's CWD.
demo_dir = fileparts(mfilename('fullpath'));
oct_dir  = fullfile(demo_dir, '..', 'simksmoother');
if isfolder(oct_dir), addpath(oct_dir); end

% ---- Demo catalogue ---------------------------------------------------
% Each entry: name, one-line description, what it teaches.
demos = {
  'demo_const_known',       ...
  'Smoothing with known parameters.  Posterior bands and empirical coverage check on a 2D state.', ...
  'How the basic smoother is called, and that the posterior covariances are well-calibrated.';

  'demo_tv_known',          ...
  'Time-varying parameters.  Sinusoidally varying F_t demonstrates the cell-array interface.', ...
  'How to pass time-varying matrices through cells, with the same call shape.';

  'demo_compare_recursive', ...
  'Simultaneous vs textbook RTS smoother.  Should agree to machine precision (1e-15).', ...
  'That J X = h is mathematically the same object as the recursive Kalman filter + RTS.';

  'demo_missing',           ...
  'Missing observations (NaN entries).  Posterior variance grows inside the gap.', ...
  'How NaN observations are dropped from H R^-1 H, with no special-case code.';

  'demo_hp_filter',         ...
  'Hodrick-Prescott filter as a special case (F = [1 1; 0 1], Q = diag(eps, 1/lambda)).', ...
  'That HP, Beveridge-Nelson, and unobserved-components are all the same matrix problem.';

  'demo_const_em',          ...
  'EM with smoothed sufficient statistics.  Monotone log-likelihood verified.', ...
  'The proper Gaussian EM formulas with posterior-covariance corrections.';

  'demo_robust',            ...
  'Robust smoothing on data with outliers.  Student-t, Huber, Laplace all cut RMSE.', ...
  'IRLS on the same sparse J, handling heavy-tailed measurement noise.';

  'demo_robust_breaks',     ...
  'Structural-break detection.  Robust prior on state innovations flags level shifts.', ...
  'How heavy-tailed Q catches one-off jumps without smearing them across periods.';

  'demo_hp_credit',         ...
  'Flagship: HP + credit-cycle exogenous + Student-t innovations + EM with unknown B,D.', ...
  'All of the above stacked: a serious applied econometric tool in 100 lines.';

  'demo_slds',              ...
  'Switching Linear Dynamical System.  Recovers regime probabilities and states jointly.', ...
  'How J stays block-tridiagonal in every iteration of a discrete-regime mixture.';

  'demo_constrained',       ...
  'Inequality-constrained smoother.  NAIRU-style non-negativity via interior-point QP on J.', ...
  'How the same sparse J extends to constrained problems via a low-rank Hessian update.';
};

% ---- Main menu loop ---------------------------------------------------
N = size(demos, 1);

while true
  fprintf('\n');
  fprintf('===========================================================\n');
  fprintf('  Simultaneous Kalman Smoother -- five-minute demo tour\n');
  fprintf('===========================================================\n\n');
  fprintf('Available demos:\n\n');
  for i = 1:N
    fprintf('  [%d]  %s\n       %s\n\n', i, demos{i,1}, demos{i,2});
  end
  fprintf('  [0]  Run all in sequence.\n');
  fprintf('  [q]  Quit.\n\n');

  choice = input('Select demo number (or q to quit): ', 's');
  choice = strtrim(choice);

  if isempty(choice)
    continue;
  elseif strcmpi(choice, 'q') || strcmpi(choice, 'quit') || strcmpi(choice, 'exit')
    fprintf('Goodbye!\n');
    break;
  end

  num = str2double(choice);
  if isnan(num)
    fprintf('  (Did not recognize "%s".  Enter a number or q to quit.)\n', choice);
    continue;
  end

  if num == 0
    for i = 1:N
      fprintf('\n-----------------------------------------------------------\n');
      fprintf('Running demo %d: %s\n', i, demos{i,1});
      fprintf('  %s\n', demos{i,2});
      fprintf('-----------------------------------------------------------\n\n');
      try
        feval(demos{i,1});
      catch err
        fprintf('\n[Demo %s failed: %s]\n', demos{i,1}, err.message);
      end
      fprintf('\nWhat to take away: %s\n', demos{i,3});
      fprintf('-----------------------------------------------------------\n');
    end
  elseif num >= 1 && num <= N
    i = num;
    fprintf('\n-----------------------------------------------------------\n');
    fprintf('Running demo %d: %s\n', i, demos{i,1});
    fprintf('  %s\n', demos{i,2});
    fprintf('-----------------------------------------------------------\n\n');
    try
      feval(demos{i,1});
    catch err
      fprintf('\n[Demo %s failed: %s]\n', demos{i,1}, err.message);
    end
    fprintf('\nWhat to take away: %s\n', demos{i,3});
    fprintf('-----------------------------------------------------------\n');
  else
    fprintf('  (Selection out of range; pick 0..%d or q.)\n', N);
  end
end
