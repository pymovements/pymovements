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
"""Provides the main sequence plotting function."""
from __future__ import annotations

from typing import Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import Collection
from sklearn.metrics import r2_score

from pymovements._utils._checks import check_is_mutual_exclusive
from pymovements.events.events import Events
from pymovements.events.frame import EventDataFrame
from pymovements.plotting._matplotlib import finalize_figure
from pymovements.plotting._matplotlib import prepare_figure


def main_sequence_plot(
        events: Events | EventDataFrame | None = None,
        *,
        marker: str = 'o',
        marker_size: float = 25,
        marker_color: str = 'purple',
        marker_alpha: float = 0.5,
        fit: bool = True,
        fit_measure: bool | Literal['r2', 's'] = True,
        fit_color: str = 'red',
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
        event_df: Events | EventDataFrame | None = None,
        event_name: str = 'saccade',
        ax: plt.Axes | None = None,
        closefig: bool | None = None,
        **kwargs: Collection,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the saccade main sequence.

    Parameters
    ----------
    events: Events | EventDataFrame | None
        It must contain columns "peak_velocity" and "amplitude".
    marker: str
        Marker symbol. Possible values defined by matplotlib.markers. (default: 'o')
    marker_size: float
        Size of the marker symbol. (default: 25)
    marker_color: str
        Color of the marker symbol. (default: 'purple')
    marker_alpha: float
        Alpha value (=transparency) of the marker symbol. Between 0 and 1. (default: 0.5)
    fit: bool
        Draw a linear fit line if True. If False, no line is drawn.
    fit_measure: bool | Literal['r2', 's']
        Annotate a goodness-of-fit statistic:
        - ``True`` or ``'r2'``: coefficient of determination (R²)
        - ``'s'``: standard error of the regression (S)
        - ``False``: no annotation
    fit_color: str
        Color of the linear fit line (default: 'red')
    figsize: tuple[int, int]
        Figure size. (default: (15, 5))
    title: str | None
        Figure title. (default: None)
    savepath: str | None
        If given, figure will be saved to this path. (default: None)
    show: bool
        If True, figure will be shown. (default: True)
    event_df: Events | EventDataFrame | None
        It must contain columns "peak_velocity" and "amplitude". (default: None)
        .. deprecated:: v0.22.0
        Please use the ``events`` argument instead. This argument will be removed in v0.27.0.
    event_name: str
        Filters events for a particular value in "name" column. (default: 'saccade')
    ax: plt.Axes | None
        External axes to draw into. If provided, the function will not show or close the figure.
    closefig: bool | None
        Close figure after plotting. If None, defaults to closing only when the figure
        was created by this function.
    **kwargs: Collection
        Additional keyword arguments passed to matplotlib.axes.Axes.scatter.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The created or provided figure and axes.

    Raises
    ------
    KeyError
        If the input dataframe has no 'amplitude' and/or 'peak_velocity' column.
        Those are needed to create the plot.
    ValueError
        If the event dataframe does not contain any saccades.
    """
    if event_df is not None:
        warn(
            'The argument event_df has been renamed to events '
            'in v0.22.0 and will be removed in v0.27.0.',
            DeprecationWarning,
        )
        check_is_mutual_exclusive(events=events, event_df=event_df)
        events = event_df

    event_col_name = 'name'

    if events is None or events.frame.is_empty():
        raise ValueError(
            'Events object is empty. '
            'Please make sure you ran a saccade detection algorithm. '
            f'The event name should be stored in a colum called "{event_col_name}".',
        )

    saccades = events.frame.filter(pl.col(event_col_name) == event_name)

    if saccades.is_empty():
        raise ValueError(
            'There are no saccades in the event dataframe. '
            'Please make sure you ran a saccade detection algorithm. '
            f'The event name should be stored in a colum called "{event_col_name}".',
        )

    try:
        peak_velocities = saccades['peak_velocity'].to_list()
    except pl.exceptions.ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade peak velocities which are needed to create '
            'the main sequence plot. ',
        ) from exc

    try:
        amplitudes = saccades['amplitude'].to_list()
    except pl.exceptions.ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade amplitudes which are needed to create '
            'the main sequence plot. ',
        ) from exc

    fig, ax, own = prepare_figure(ax, figsize, func_name='main_sequence_plot')

    # Use plt.scatter when we own the figure to preserve legacy test expectations.
    if own:
        plt.scatter(
            amplitudes,
            peak_velocities,
            color=marker_color,
            alpha=marker_alpha,
            s=marker_size,
            marker=marker,
            label=event_name,
            **kwargs,
        )
        plt.legend()
    else:
        ax.scatter(
            amplitudes,
            peak_velocities,
            color=marker_color,
            alpha=marker_alpha,
            s=marker_size,
            marker=marker,
            label=event_name,
            **kwargs,
        )
        ax.legend()

    # --- Linear fit (only if requested) ---
    if fit:
        # Compute linear fit
        a, b = np.polyfit(amplitudes, peak_velocities, 1)

        min_ampl, max_ampl = min(amplitudes), max(amplitudes)
        line_x = [min_ampl, max_ampl]
        line_y = [a * min_ampl + b, a * max_ampl + b]

        line_axes = plt.gca() if own else ax

        fit_label = None

        # Compute fit measure if requested
        if fit_measure:
            y_pred = np.array(amplitudes) * a + b
            residuals = np.array(peak_velocities) - y_pred

            if fit_measure is True or fit_measure == 'r2':
                val = np.round(r2_score(peak_velocities, y_pred), 3)
                fit_label = f"R² = {val}"

            elif fit_measure == 's':
                s = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))
                val = np.round(s, 3)
                fit_label = f"S = {val}"

            else:
                raise ValueError("measure must be one of: True, False, 'r2', 's'")

        # add fit label to the legend
        if fit_label is not None:
            line_axes.plot(line_x, line_y, c=fit_color, label=fit_label)
        else:
            line_axes.plot(line_x, line_y, c=fit_color)
        line_axes.legend()

    if title:
        ax.set_title(title)
    ax.set_xlabel('Amplitude [dva]')
    ax.set_ylabel('Peak Velocity [dva/s]')

    finalize_figure(
        fig,
        show=show,
        savepath=savepath,
        closefig=closefig,
        own_figure=own,
        func_name='main_sequence_plot',
    )

    return fig, ax
