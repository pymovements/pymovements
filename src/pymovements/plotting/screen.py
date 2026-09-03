# Copyright (c) 2026 The pymovements Project Authors
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
"""Provides an empty display-space canvas."""
from __future__ import annotations

import matplotlib.pyplot as plt

SUPPORTED_ORIGINS = ('upper left', 'center')

# imshow origin corresponding to both supported screen origins. See screen().
IMSHOW_ORIGIN = 'upper'


def screen(
    width_px: int | None,
    height_px: int | None,
    *,
    origin: str | None = 'upper left',
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create an empty axes spanning a display's pixel extent.

    The returned axes carries the extent, aspect ratio and y-orientation of the
    display, but no content: the axes frame is the screen boundary. Stimulus and
    gaze plotters draw onto the returned axes.

    Parameters
    ----------
    width_px : int | None
        Screen width in pixels.
    height_px : int | None
        Screen height in pixels.
    origin : str | None
        Origin of the screen coordinate system. Supported values are
        ``'upper left'`` and ``'center'``. (default: 'upper left')
    ax : plt.Axes | None
        Axes to set up. A new figure and axes are created if ``None``.
        (default: None)

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and the display-space axes.

    Raises
    ------
    ValueError
        If ``width_px`` or ``height_px`` is ``None``.
    ValueError
        If ``origin`` is ``None`` or not a supported origin.

    Notes
    -----
    This function owns the mapping between the screen origin vocabulary
    (``'upper left'`` / ``'center'``) and the imshow origin vocabulary
    (``'upper'`` / ``'lower'``). Both supported screen origins keep screen y
    increasing downward, which corresponds to imshow origin ``'upper'``, exposed
    as the module constant ``IMSHOW_ORIGIN``. Stimulus plotters should pass that value when
    drawing an image onto a canvas created here, so that the image and the gaze
    data drawn over it share one orientation.

    Examples
    --------
    >>> fig, ax = screen(1280, 1024)
    >>> tuple(float(value) for value in ax.get_xlim())
    (0.0, 1280.0)
    >>> tuple(float(value) for value in ax.get_ylim())
    (1024.0, 0.0)

    The ``'center'`` origin puts the coordinate origin at the display center,
    keeping screen y increasing downward:

    >>> fig, ax = screen(1280, 1024, origin='center')
    >>> tuple(float(value) for value in ax.get_xlim())
    (-640.0, 640.0)
    >>> tuple(float(value) for value in ax.get_ylim())
    (512.0, -512.0)
    """
    if width_px is None or height_px is None:
        raise ValueError(
            'screen width and height must be set, '
            f'got width_px={width_px} and height_px={height_px}.',
        )

    if origin not in SUPPORTED_ORIGINS:
        supported = ', '.join(f'"{supported_origin}"' for supported_origin in SUPPORTED_ORIGINS)
        raise ValueError(f'screen origin must be one of {supported}, got "{origin}".')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if origin == 'center':
        ax.set_xlim(-width_px / 2, width_px / 2)
        ax.set_ylim(height_px / 2, -height_px / 2)
    else:
        ax.set_xlim(0, width_px)
        ax.set_ylim(height_px, 0)

    ax.set_aspect('equal', adjustable='box')

    return fig, ax
