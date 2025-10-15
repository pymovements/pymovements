# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Internal helpers for consistent matplotlib figure handling.

Not part of the public API.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from typing import Union
from warnings import warn

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib import scale as mpl_scale
from matplotlib.collections import LineCollection
from typing_extensions import TypeAlias

from pymovements.gaze.experiment import Screen

LinearSegmentedColormapType: TypeAlias = dict[
    Literal['red', 'green', 'blue', 'alpha'],
    Sequence[tuple[float, ...]],
]

DEFAULT_SEGMENTDATA: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

DEFAULT_SEGMENTDATA_TWOSLOPE: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 1.0, 1.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

CmapNormType: TypeAlias = Union[
    matplotlib.colors.TwoSlopeNorm,
    matplotlib.colors.Normalize,
    matplotlib.colors.NoNorm,
]

MatplotlibSetupType: TypeAlias = tuple[
    plt.figure,
    plt.Axes,
    matplotlib.colors.Colormap,
    CmapNormType,
    np.ndarray,
    bool,
]


def prepare_figure(
    ax: plt.Axes | None, figsize: tuple[int, int] | tuple[float, float] | None,
    *, func_name: str,
) -> tuple[plt.Figure, plt.Axes, bool]:
    """Prepare a matplotlib figure and axes.

    Create or reuse a matplotlib Figure/Axes pair. If an external Axes is provided,
    the given figsize is ignored and a warning is emitted.

    Parameters
    ----------
    ax : plt.Axes | None
        Existing matplotlib Axes to use. If None, a new Axes will be created.
    figsize : tuple[int, int] | tuple[float, float] | None
        Figure size in inches as (width, height). Ignored if `ax` is provided.
    func_name : str
        Name of the calling function, used for generating warnings.

    Returns
    -------
    tuple[plt.Figure, plt.Axes, bool]
        A tuple ``(fig, ax, own)`` where ``own`` indicates whether the Axes was
        created internally (True) or provided externally (False).
    """
    if ax is None:
        # figsize may be None for some callers
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
        own = True
    else:
        fig = ax.figure
        own = False
        if figsize is not None:
            warn(
                f'{func_name}: "figsize" is ignored because an external Axes was provided.',
                UserWarning,
                stacklevel=2,
            )
    return fig, ax, own


def finalize_figure(
    fig: plt.Figure,
    *,
    show: bool,
    savepath: str | None,
    closefig: bool | None,
    own_figure: bool,
    func_name: str,
) -> None:
    """Finalize a matplotlib figure (save/show/close).

    Manage saving, showing, and closing behavior consistently. When plotting into an
    external Axes, ``show=True`` and ``closefig=True`` are ignored with a warning.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to finalize.
    show : bool
        Whether to display the figure.
    savepath : str | None
        File path to save the figure to. If None, the figure is not saved.
    closefig : bool | None
        Whether to close the figure. If None, close only when the figure is owned
        by the current function (``own_figure=True``).
    own_figure : bool
        Indicates whether the figure was created by the current function.
    func_name : str
        Name of the calling function, used in warning messages.
    """
    if savepath is not None:
        fig.savefig(savepath)

    if show:
        if own_figure:
            plt.show()
        else:
            warn(
                f'{func_name}: "show=True" has no effect if plotting into an external Axes.',
                UserWarning,
                stacklevel=2,
            )

    if closefig is None:
        do_close = own_figure
    else:
        if not own_figure and closefig:
            warn(
                f'{func_name}: "closefig=True" is ignored if an external Axes is provided.',
                UserWarning,
                stacklevel=2,
            )
        do_close = bool(closefig) and own_figure

    if do_close:
        plt.close(fig)


def _setup_axes_and_colormap(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    figsize: tuple[int, int] | tuple[float, float],
    cmap: matplotlib.colors.Colormap | None = None,
    cmap_norm: matplotlib.colors.Normalize | str | None = None,
    cmap_segmentdata: LinearSegmentedColormapType | None = None,
    cval: np.ndarray | None = None,
    show_cbar: bool = False,
    add_stimulus: bool = False,
    path_to_image_stimulus: str | None = None,
    stimulus_origin: str = 'upper',
    padding: float | None = None,
    pad_factor: float | None = 0.05,
    ax: plt.Axes | None = None,
) -> MatplotlibSetupType:
    """Prepare axes limits and colormap configuration for 2D positional data.

    Returns fig, ax, cmap, cmap_norm, cval, show_cbar.
    """
    n = len(x_signal)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.figure
        if figsize is not None:
            warn(
                'figsize is ignored because an external Axes was provided.',
                UserWarning,
                stacklevel=2,
            )

    if add_stimulus:
        img = PIL.Image.open(path_to_image_stimulus)
        ax.imshow(img, origin=stimulus_origin, extent=None)
    else:
        if padding is None:
            x_pad = (np.nanmax(x_signal) - np.nanmin(x_signal)) * pad_factor
            y_pad = (np.nanmax(y_signal) - np.nanmin(y_signal)) * pad_factor
        else:
            x_pad = padding
            y_pad = padding

        ax.set_xlim(np.nanmin(x_signal) - x_pad, np.nanmax(x_signal) + x_pad)
        ax.set_ylim(np.nanmin(y_signal) - y_pad, np.nanmax(y_signal) + y_pad)
        ax.invert_yaxis()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    cval_max = np.nanmax(np.abs(cval))
    cval_min = np.nanmin(cval).astype(float)

    if cmap_norm is None:
        if cval_max and cval_min < 0:
            cmap_norm = 'twoslope'
        elif cval_max:
            cmap_norm = 'normalize'
        else:
            cmap_norm = 'nonorm'

    if cmap is None:
        if cmap_segmentdata is None:
            if cmap_norm == 'twoslope':
                cmap_segmentdata = DEFAULT_SEGMENTDATA_TWOSLOPE
            else:
                cmap_segmentdata = DEFAULT_SEGMENTDATA

        cmap = matplotlib.colors.LinearSegmentedColormap(
            'line_cmap', segmentdata=cmap_segmentdata, N=512,
        )

    if cmap_norm == 'twoslope':
        cmap_norm = matplotlib.colors.TwoSlopeNorm(
            vcenter=0, vmin=-cval_max, vmax=cval_max,
        )
    elif cmap_norm == 'normalize':
        cmap_norm = matplotlib.colors.Normalize(
            vmin=cval_min, vmax=cval_max,
        )
    elif cmap_norm == 'nonorm':
        cmap_norm = matplotlib.colors.NoNorm()

    elif isinstance(cmap_norm, str):
        # pylint: disable=protected-access
        if (
            scale_class := mpl_scale._scale_mapping.get(cmap_norm, None)  # type: ignore
        ) is None:
            raise ValueError(f'cmap_norm string {cmap_norm} is not supported')

        norm_class = matplotlib.colors.make_norm_from_scale(scale_class)
        cmap_norm = norm_class(matplotlib.colors.Normalize)()

    return fig, ax, cmap, cmap_norm, cval, show_cbar


def _draw_line_data(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    ax: plt.Axes,
    cmap: matplotlib.colors.Colormap | None = None,
    cmap_norm: matplotlib.colors.Normalize | str | None = None,
    cval: np.ndarray | None = None,
) -> LineCollection:
    """Draw line data as a colored LineCollection and return the collection."""
    points = np.array([x_signal, y_signal]).T.reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = LineCollection(segments, cmap=cmap, norm=cmap_norm)
    line_collection.set_array(cval)
    line_collection.set_linewidth(2)
    line = ax.add_collection(line_collection)
    return line


def _set_screen_axes(
    ax: plt.Axes,
    screen: Screen | None,
    *,
    func_name: str,
) -> None:
    """Set axes limits and aspect ratio from gaze.experiment.screen, if available.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to modify.
    screen : Screen | None
        Screen object from a Gaze's Experiment. If None, no changes are made.
    func_name : str
        Name of the plotting function, used in error messages.

    Raises
    ------
    ValueError
        If the screen origin is not 'upper left'.
    ValueError
        If the screen width or height is not positive.
    """
    if screen is None:
        return

    # If screen has no pixel info, skip silently
    if screen.width_px is None or screen.height_px is None:
        return

    if (
        screen.width_px is None or screen.height_px is None
        or screen.width_px <= 0 or screen.height_px <= 0
    ):
        raise ValueError(
            f'{func_name}: screen width and height must be positive, '
            f'got width={screen.width_px}, height={screen.height_px}.',
        )

    if screen.origin != 'upper left':
        raise ValueError(
            f'{func_name}: screen origin must be "upper left", got "{screen.origin}".',
        )

    ax.set_xlim(0, screen.width_px)
    ax.set_ylim(screen.height_px, 0)
    ax.set_aspect('equal', adjustable='box')
