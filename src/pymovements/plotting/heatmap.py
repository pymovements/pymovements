# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Heatmap module."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from pymovements.gaze import Gaze
from pymovements.plotting._matplotlib import finalize_figure
from pymovements.plotting._matplotlib import prepare_figure
from pymovements.stimulus.image import _draw_image_stimulus


def heatmap(
        gaze: Gaze,
        position_column: str = 'pixel',
        gridsize: tuple[int, int] = (10, 10),
        cmap: colors.Colormap | str = 'jet',
        interpolation: str = 'gaussian',
        origin: str = 'upper',
        figsize: tuple[float, float] = (15, 10),
        cbar_label: str | None = None,
        show_cbar: bool = True,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        show: bool = True,
        savepath: str | None = None,
        add_stimulus: bool = False,
        path_to_image_stimulus: str | Path | None = None,
        stimulus_origin: str = 'upper',
        alpha: float = 1.,
        *,
        ax: plt.Axes | None = None,
        closefig: bool | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of gaze data.

    The heatmap displays the distribution of gaze positions across the experiment screen,
    for a given Gaze object.
    The color values indicate the time spent at each position in seconds.

    Parameters
    ----------
    gaze: Gaze
        A Gaze object.
    position_column: str
        The column name of the x and y position data. (default: 'pixel')
    gridsize: tuple[int, int]
        The number of bins in the x and y dimensions. (default: (10, 10))
    cmap: colors.Colormap | str
        The colormap to use. (default: 'jet')
    interpolation: str
        The interpolation method to use for plotting the heatmap.
        See matplotlib.pyplot.imshow for more information on available methods
        for interpolation. (default: 'gaussian')
    origin: str
        Set origin of y-axis, valid values are 'lower' or 'upper'. (default: 'upper')
    figsize: tuple[float, float]
        Figure size. (default: (15, 10))
    cbar_label: str | None
        Label for the colorbar. (default: None)
    show_cbar: bool
        Whether to show the colorbar. (default: True)
    title: str | None
        Figure title. (default: None)
    xlabel: str | None
        Set x-axis label. (default: None)
    ylabel: str | None
        Set y-axis label. (default: None)
    show: bool
        Whether to show the plot. (default: True)
    savepath: str | None
        If provided, the figure will be saved to this path. (default: None)
    add_stimulus: bool
        Define whether stimulus should be included. (default: False)
    path_to_image_stimulus: str | Path | None
        Path to image stimulus. (default: None)
    stimulus_origin: str
        Origin of stimulus. (default: 'upper')
    alpha: float
        Alpha value of heatmap. (default: 1.)
    ax: plt.Axes | None
        External axes to draw into. If provided, the function will not show or close
        the figure automatically. (default: None)
    closefig: bool | None
        Whether to close the figure. If None, close only when the function created
        the figure. (default: None)

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The created or provided figure and axes.

    Raises
    ------
    ValueError
        If the position columns are not in pixels or degrees
    ValueError
        If the experiment property of the Gaze is None
    """
    # Extract x and y positions from the gaze dataframe
    x = gaze.samples[position_column].list.get(0).to_numpy()
    y = gaze.samples[position_column].list.get(1).to_numpy()

    # Check if experiment properties are available
    if not gaze.experiment:
        raise ValueError(
            'Experiment property of Gaze is None. '
            'Gaze must be associated with an experiment.',
        )

    assert gaze.experiment.sampling_rate is not None

    # Get experiment screen properties
    screen = gaze.experiment.screen

    # Use screen properties to define the grid or degrees of visual angle
    if position_column == 'pixel':
        xmin, xmax = 0, screen.width_px
        ymin, ymax = 0, screen.height_px
    elif position_column == 'position':
        xmin, xmax = int(screen.x_min_dva), int(screen.x_max_dva)
        ymin, ymax = int(screen.y_min_dva), int(screen.y_max_dva)
    else:
        xmin, xmax = int(x.min()), int(x.max())
        ymin, ymax = int(y.min()), int(y.max())

    assert xmin is not None
    assert xmax is not None
    assert ymin is not None
    assert ymax is not None

    # Define the grid and bin the gaze data
    x_bins = np.linspace(xmin, xmax, num=gridsize[0]).astype(int)
    y_bins = np.linspace(ymin, ymax, num=gridsize[1]).astype(int)

    # Bin the gaze data
    heatmap_value, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Transpose to match the orientation of the screen
    heatmap_value = heatmap_value.T

    # Convert heatmap values from sample count to seconds
    heatmap_value /= gaze.experiment.sampling_rate

    if origin == 'upper':
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    else:
        extent = [x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]]

    # If add_stimulus is requested, we still reuse/create fig/ax via prepare_figure and then draw
    fig, ax, own_figure = prepare_figure(ax, figsize, func_name='heatmap')

    if add_stimulus:
        assert path_to_image_stimulus
        # draw the background stimulus onto the current axes
        _draw_image_stimulus(
            path_to_image_stimulus,
            origin=stimulus_origin,
            show=False,
            figsize=figsize,
            extent=extent,
            fig=fig,
            ax=ax,
        )

    # Plot the heatmap
    heatmap_plot = ax.imshow(
        heatmap_value,
        cmap=cmap,
        origin=origin,
        interpolation=interpolation,
        extent=extent,
        alpha=alpha,
    )

    # Set the plot title and axis labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add a color bar to the plot
    if show_cbar:
        cbar = fig.colorbar(heatmap_plot, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    # Finalize (save/show/close) with standardized behavior
    finalize_figure(
        fig,
        show=show,
        savepath=savepath,
        closefig=closefig,
        own_figure=own_figure,
        func_name='heatmap',
    )

    return fig, ax
