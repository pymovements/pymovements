# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Provides the time series plotting function."""
from __future__ import annotations

import math
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pymovements.gaze import Gaze
from pymovements.gaze._utils._column_nesting import get_nested_columns
from pymovements.gaze._utils._column_nesting import unnest_list_columns
from pymovements.plotting._matplotlib import prepare_figure


def tsplot(
        gaze: Gaze,
        channels: str | list[str] | None = None,
        *,
        xlabel: str | None = None,
        n_cols: int | None = None,
        n_rows: int | None = None,
        rotate_ylabels: bool = True,
        share_y: bool = False,
        zero_centered_yaxis: bool = False,
        line_color: tuple[int, int, int] | str = 'k',
        line_width: int = 1,
        show_grid: bool = True,
        show_yticks: bool = True,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot time series with each channel getting a separate subplot.

    Parameters
    ----------
    gaze: Gaze
        The Gaze to plot.
    channels: str | list[str] | None
        Name(s) of channels to plot. List columns are unnested into one channel per component,
        e.g. ``pixel`` becomes ``pixel_x`` and ``pixel_y``. If None, all numeric columns
        including list columns with numeric components will be plotted. (default: None)
    xlabel: str | None
        Set the x label. (default: None)
    n_cols: int | None
        Number of channel subplot columns. If None, it will be inferred. (default: None)
    n_rows: int | None
        Number of channel subplot rows. If None, it will be inferred. (default: None)
    rotate_ylabels: bool
        Set whether to rotate ylabels. (default: True)
    share_y: bool
        Set if y-axes should share a common axis. (default: False)
    zero_centered_yaxis: bool
        Set if y-axis should be zero-centered. (default: False)
    line_color: tuple[int, int, int] | str
        Set line color. (default: 'k')
    line_width: int
        Set line width. (default: 1)
    show_grid: bool
        Set whether to show the background grid. (default: True)
    show_yticks: bool
        Set whether to show yticks. (default: True)
    figsize: tuple[int, int]
        Figure size. (default: (15, 5))
    title: str | None
        Figure title. (default: None)
    savepath: str | None
        If given, figure will be saved to this path. (default: None)
    ax: plt.Axes | None
        External axes to draw into when plotting a single channel. Ignored when
        ``n_channels > 1``. (default: None)

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The created or provided figure and the primary axes (the first subplot).

    Raises
    ------
    ValueError
        If array has more than two dimensions.
    """
    if channels is None:
        # Select all numeric (and nested numeric) channels
        channels = [
            c
            for c in gaze.samples.columns
            if gaze.samples[c].dtype.is_numeric() or (
                gaze.samples[c].dtype == pl.List and gaze.samples[c].dtype.inner.is_numeric()
            )
        ]

    df = gaze.samples.select(channels)
    nested_columns = get_nested_columns(df)
    if nested_columns:
        df = unnest_list_columns(df, nested_columns)
    channels = df.columns
    arr = df.to_numpy().transpose()

    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    channel_axis = 0
    sample_axis = 1

    n_channels = arr.shape[channel_axis]
    n_samples = arr.shape[sample_axis]

    if n_cols is None:
        if n_channels % 2 == 0:
            n_cols = 2
        else:
            n_cols = 1

    if n_rows is None:
        n_rows = math.ceil(n_channels / n_cols)

    # determine number of subplots and height ratios for events
    height_ratios = [1] * n_rows

    external_ax = ax is not None

    if n_channels == 1:
        fig, ax = prepare_figure(ax, figsize, func_name='tsplot')
        axs = [ax]
    else:
        if external_ax:
            warn(
                'tsplot: "ax" is ignored when plotting multiple channels.',
                UserWarning,
                stacklevel=2,
            )
        fig, axs_grid = plt.subplots(
            ncols=n_cols,
            nrows=n_rows,
            sharex=True,
            sharey=share_y,
            squeeze=False,
            figsize=figsize,
            gridspec_kw={
                'hspace': 0,
                'height_ratios': height_ratios,
            },
        )
        axs = axs_grid.flatten()

    t = np.arange(n_samples)
    xlims = t.min(), t.max()

    # set ylims to have zero centered y-axis (for all axes)
    # will be overwritten if share_y is False
    ylims = _compute_ylims(arr, zero_centered_yaxis=zero_centered_yaxis)

    for channel_id in range(n_channels):
        ax = axs[channel_id]

        x_channel = arr[channel_id, :]
        ax.plot(t, x_channel, color=line_color, linewidth=line_width)

        if not share_y:
            ylims = _compute_ylims(arr[channel_id], zero_centered_yaxis=zero_centered_yaxis)

        if xlims[0] != xlims[1]:
            ax.set_xlim(xlims)
        if ylims is not None and ylims[0] != ylims[1]:
            ax.set_ylim(ylims)

        ax.grid(show_grid, which='major')
        ax.grid(show_grid, which='minor')

        ax.tick_params(
            which='both', direction='out',
            length=4, width=1, colors='k',
            grid_color='#999999', grid_alpha=0.5,
        )

        if show_yticks:
            # from matplotlib.ticker import AutoMinorLocator
            # ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.setp(ax.get_yticklabels(), visible=True)
        else:
            ax.set_yticks([])
            plt.setp(ax.get_yticklabels(), visible=False)

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        # set channel names as y-axis labels
        if rotate_ylabels:
            ax.set_ylabel(
                channels[channel_id],
                rotation='horizontal',
                ha='right', va='center',
            )
        else:
            ax.set_ylabel(channels[channel_id])

        # set x label on all axes
        # share_y=True will automatically hide those that are not on the bottom
        ax.set_xlabel(xlabel)

    if title:
        axs[0].set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    return fig, axs[0]


def _compute_ylims(
        values: np.ndarray,
        *,
        zero_centered_yaxis: bool,
        y_pad_factor: float = 1.1,
) -> tuple[float, float] | None:
    """Compute padded y-axis limits, or None if there are no finite values to infer them from."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    if zero_centered_yaxis:
        ylim_abs = np.max(np.abs(finite_values))
        return -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor
    ylim_max = np.max(finite_values)
    ylim_min = np.min(finite_values)
    y_pad = (ylim_max - ylim_min) * (y_pad_factor - 1)
    return ylim_min - y_pad, ylim_max + y_pad
