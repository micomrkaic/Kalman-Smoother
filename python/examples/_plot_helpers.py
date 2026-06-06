"""
Plot-saving helpers shared by all demos.

This module does three things:

  1. Sets up matplotlib to display figures live when a GUI backend is
     available (e.g. when run from IPython, Spyder, or a terminal with
     a working display), and to fall back to the non-interactive Agg
     backend when no display is available (e.g. on a headless CI box).
  2. Ensures that ``import simksmoother`` works even when the package
     hasn't been pip-installed.  If `simksmoother` isn't importable, the
     parent directory (which contains the `simksmoother/` source folder)
     is prepended to `sys.path`.  This lets users run the demos straight
     out of the unzipped tree without first running `pip install -e .`.
  3. Provides ``save_and_show(fig, name)``: always saves to PNG under
     ``examples/figures/``, and additionally displays the figure on
     screen if a GUI backend is in use.

Importing this module first in every demo (before any
``import simksmoother``) is therefore enough to make the demos
"just work" from a freshly unzipped copy of the package.
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

from pathlib import Path
import importlib.util
import os
import sys


# --- sys.path fixup: make `simksmoother` importable without pip install -----
def _ensure_simksmoother_importable() -> None:
    """If `simksmoother` is not already importable, prepend the project's
    `python/` directory (the parent of this examples folder) to sys.path."""
    if importlib.util.find_spec("simksmoother") is not None:
        return  # already on the path (pip-installed or otherwise)
    parent = Path(__file__).resolve().parent.parent  # .../python/
    if (parent / "simksmoother" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))


_ensure_simksmoother_importable()


# --- matplotlib backend selection ------------------------------------------
# We want figures to appear on screen when a display is available, and
# to fall back gracefully (write PNG only) when not.  The strategy:
#
#   * If a display is plausibly available, let matplotlib pick its
#     default backend.  On Windows this is TkAgg or Qt5Agg; on macOS,
#     macosx; on Linux with X11, TkAgg/Qt5Agg.  All of these support
#     interactive display.
#   * If we appear to be headless (no DISPLAY env var on Linux, and
#     not running on Windows/macOS), force Agg.
import matplotlib

def _have_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY"))


_DISPLAY_AVAILABLE = _have_display()
if not _DISPLAY_AVAILABLE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Turn on interactive mode so plt.show(block=False) returns immediately
# and the figure window actually renders before the next demo line runs.
if _DISPLAY_AVAILABLE:
    try:
        plt.ion()
    except Exception:
        pass


def figures_dir() -> Path:
    """Return the path to examples/figures/, creating it if needed."""
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(exist_ok=True)
    return d


def save_and_show(fig, name: str) -> Path:
    """Save figure to figures/{name}.png and, if a GUI backend is in use,
    display it on screen.  When no display is available, the figure is
    closed after saving.

    The figure window stays open after this function returns so the user
    can interact with it; closing it is the user's responsibility (or
    happens implicitly at script exit).
    """
    if not name.endswith(".png"):
        name = name + ".png"
    path = figures_dir() / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if _DISPLAY_AVAILABLE:
        # Show non-blocking; pause briefly so the GUI event loop flushes
        # and the window appears before the next demo prints to stdout.
        plt.show(block=False)
        try:
            plt.pause(0.05)
        except Exception:
            pass
    else:
        plt.close(fig)
    print(f"Figure saved: {path}")
    return path


# Back-compat alias: older demos may import `save_and_close`.
save_and_close = save_and_show
