# Copyright (c) 2024-2025 The pymovements Project Authors
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
"""Functions for plotting.

.. deprecated:: v0.22.0
   This module will be removed in v0.27.0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
from deprecated.sphinx import deprecated

from pymovements.plotting._matplotlib import _draw_line_data
from pymovements.plotting._matplotlib import _setup_axes_and_colormap
from pymovements.plotting._matplotlib import LinearSegmentedColormapType
from pymovements.plotting._matplotlib import MatplotlibSetupType
from pymovements.stimulus.image import _draw_image_stimulus


@deprecated(
    reason='This function will be removed in v0.27.0.',
    version='v0.22.0',
)
def setup_matplotlib(
        x_signal: np.ndarray,
        y_signal: np.ndarray,
        figsize: tuple[int, int],
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
) -> MatplotlibSetupType:
    """Configure cmap.

    .. deprecated:: v0.22.0
       This function will be removed in v0.27.0.

    Parameters
    ----------
    x_signal: np.ndarray
        Time-step array.
    y_signal: np.ndarray
        Time-step array.
    figsize: tuple[int, int]
        Figure size.
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cmap_segmentdata: LinearSegmentedColormapType | None
        Color map segmentation to build color map. (default: None)
    cval: np.ndarray | None
        Line color values. (default: None)
    show_cbar: bool
        Shows color bar if True. (default: False)
    add_stimulus: bool
        Boolean value indicationg whether to plot the scanpath on the stimuls. (default: False)
    path_to_image_stimulus: str | None
        Path of the stimulus to be shown. (default: None)
    stimulus_origin: str
        Origin of stimuls to plot on the stimulus. (default: 'upper')
    padding: float | None
        Absolute padding value. If None it is inferred from pad_factor and limits. (default: None)
    pad_factor: float | None
        Relative padding factor to construct padding from value. (default: 0.05)

    Returns
    -------
    MatplotlibSetupType
        Configures fig, ax, cmap, cmap_norm, cmap_segmentdata, cval, and show_cbar.
    """
    return _setup_axes_and_colormap(
        x_signal=x_signal,
        y_signal=y_signal,
        figsize=figsize,
        cmap=cmap,
        cmap_norm=cmap_norm,
        cmap_segmentdata=cmap_segmentdata,
        cval=cval,
        show_cbar=show_cbar,
        add_stimulus=add_stimulus,
        path_to_image_stimulus=path_to_image_stimulus,
        stimulus_origin=stimulus_origin,
        padding=padding,
        pad_factor=pad_factor,
    )


@deprecated(
    reason='Please use ImageStimulus.show() instead. '
           'This function will be removed in v0.27.0.',
    version='v0.22.0',
)
def draw_image_stimulus(
        image_stimulus: str | Path,
        origin: str = 'upper',
        show: bool = False,
        figsize: tuple[float, float] = (15, 10),
        extent: list[float] | None = None,
        fig: matplotlib.pyplot.figure | None = None,
        ax: matplotlib.pyplot.Axes | None = None,
) -> tuple[matplotlib.pyplot.figure, matplotlib.pyplot.Axes]:
    """Draw stimulus.

    .. deprecated:: v0.22.0
       Please use :py:meth:`~pymovements.ImageStimulus.show()` instead.
       This function will be removed in v0.27.0.

    Parameters
    ----------
    image_stimulus: str | Path
        Path to image stimulus.
    origin: str
        Origin how to draw the image.
    show: bool
        Boolean whether to show the image. (default: False)
    figsize: tuple[float, float]
        Size of the figure. (default: (15, 10))
    extent: list[float] | None
        Extent of image. (default: None)
    fig: matplotlib.pyplot.figure | None
        Matplotlib canvas. (default: None)
    ax: matplotlib.pyplot.Axes | None
        Matplotlib axes. (default: None)

    Returns
    -------
    fig: matplotlib.pyplot.figure
    ax: matplotlib.pyplot.Axes
    """
    return _draw_image_stimulus(
        image_stimulus=image_stimulus,
        origin=origin,
        show=show,
        figsize=figsize,
        extent=extent,
        fig=fig,
        ax=ax,
    )


@deprecated(
    reason='This function will be removed in v0.27.0.',
    version='v0.22.0',
)
def draw_line_data(
        x_signal: np.ndarray,
        y_signal: np.ndarray,
        ax: matplotlib.pyplot.Axes,
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cval: np.ndarray | None = None,
) -> matplotlib.pyplot.Axes:
    """Draw line data.

    .. deprecated:: v0.22.0
       This function will be removed in v0.27.0.

    Parameters
    ----------
    x_signal: np.ndarray
        Data to be plotted.
    y_signal: np.ndarray
        Data to be plotted.
    ax: matplotlib.pyplot.Axes
        Matplotlib axes.
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cval: np.ndarray | None
        Line color values. (default: None)

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes with added line data.
    """
    return _draw_line_data(
        x_signal=x_signal,
        y_signal=y_signal,
        ax=ax,
        cmap=cmap,
        cmap_norm=cmap_norm,
        cval=cval,
    )
