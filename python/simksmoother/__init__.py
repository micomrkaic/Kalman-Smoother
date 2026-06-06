"""
simksmoother: Simultaneous Kalman smoother.

The Kalman smoother formulated as a single sparse block-tridiagonal
linear system J X = h, instead of as a forward-backward recursion.

Core functions:
    smooth_const, smooth_tv      basic smoothing
    selected_inv                 Takahashi recursion for blocks of J^{-1}
    marginal_loglik              Gaussian log p(Y; theta)
    simulate_const               draw from the model
    rts_smoother                 textbook Kalman+RTS, for cross-checking
    smooth_robust                IRLS for Student-t, Laplace, Huber
    smooth_slds                  switching linear dynamical system (variational EM)
    smooth_constrained           inequality-constrained smoother (interior-point QP)
    em_const                     Gaussian EM with smoothed sufficient stats

See the documentation note `simultaneous_kalman_smoother.pdf` for the
full mathematical derivation and the README for usage examples.
"""

# simksmoother --- a simultaneous Kalman smoother in sparse linear algebra form.
# Copyright (C) 2026 Mico Mrkaic.
#
# Produced under the guidance and direction of Mico Mrkaic, with the
# assistance of AI (Claude, Anthropic).
#
# This file is part of the simksmoother package.
#
# simksmoother is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the LICENSE file in the package root,
# or <https://www.gnu.org/licenses/>.
#

from .smooth import smooth_const, smooth_tv
from .selected_inv import selected_inv
from .marginal_loglik import marginal_loglik
from .simulate import simulate_const
from .rts_reference import rts_smoother
from .robust import smooth_robust
from .em import em_const
from .slds import smooth_slds
from .constrained import smooth_constrained

__all__ = [
    "smooth_const",
    "smooth_tv",
    "selected_inv",
    "marginal_loglik",
    "simulate_const",
    "rts_smoother",
    "smooth_robust",
    "em_const",
    "smooth_slds",
    "smooth_constrained",
]

__version__ = "2.0.0"
