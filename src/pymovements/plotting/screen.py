# Copyright (c) 2023-2026 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Create an empty display-space canvas."""
from __future__ import annotations

import matplotlib.pyplot as plt


def screen(
        width_px: int | None,
        height_px: int | None,
        *,
        origin: str | None = 'upper left',
        ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create an empty axes configured for display-space coordinates.

    Screen coordinates increase downward for both supported origins. Therefore,
    images drawn on the returned axes should use ``origin='upper'``.

    Parameters
    ----------
    width_px : int | None
        Width of the display in pixels.
    height_px : int | None
        Height of the display in pixels.
    origin : str | None
        Coordinate-system origin. Supported values are ``'upper left'`` and
        ``'center'``. (default: ``'upper left'``)
    ax : plt.Axes | None
        Existing axes to configure. A new figure and axes are created when unset.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and configured axes.

    Raises
    ------
    ValueError
        If either display dimension is unset.
    ValueError
        If the origin is unset or unsupported.
    """
    if width_px is None or height_px is None:
        raise ValueError('screen width_px and height_px must be set.')

    if origin not in {'upper left', 'center'}:
        raise ValueError(
            f'screen origin must be "upper left" or "center", got "{origin}".',
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if origin == 'upper left':
        ax.set_xlim(0, width_px)
        ax.set_ylim(height_px, 0)
    else:
        ax.set_xlim(-width_px / 2, width_px / 2)
        ax.set_ylim(height_px / 2, -height_px / 2)

    ax.set_aspect('equal', adjustable='box')
    return fig, ax
