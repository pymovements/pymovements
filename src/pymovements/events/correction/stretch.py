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
"""Provides the stretch drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import polars as pl
from scipy.optimize import minimize

from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import map_per_trial
from pymovements.events.correction._utils import nearest_index
from pymovements.events.correction._utils import to_line_values


def stretch(
    line_ys: pl.Series | Sequence[float],
    *,
    scale_bounds: tuple[float, float] = (0.9, 1.1),
    offset_bounds: tuple[float, float] = (-50, 50),
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Fit scale and offset bounds to stretch/compress fixations onto line centers.

    Reference: :cite:p:`Lohmeier2015,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    scale_bounds: tuple[float, float]
        Scaling factor bounds. (default: (0.9, 1.1))
    offset_bounds: tuple[float, float]
        Vertical offset bounds. (default: (-50, 50))
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. The returned expression
        operates on the full fixation sequence of a single trial, so it must be evaluated
        per trial. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = to_line_values(line_ys)
    core = partial(
        _stretch_core,
        line_values=line_values,
        scale_bounds=scale_bounds,
        offset_bounds=offset_bounds,
    )
    return map_per_trial(location, core, 'y_stretch')


def _stretch_core(
    locations: pl.Series,
    *,
    line_values: list[float],
    scale_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
) -> pl.Series:
    """Fit scale and offset for the y-values of a single trial and snap them to lines."""
    _, y_values = locations_to_lists(locations)
    objective = partial(_snapping_error, y_values=y_values, line_values=line_values)
    best_fit = minimize(objective, [1, 0], bounds=[scale_bounds, offset_bounds])
    return pl.Series(_snap_to_lines(best_fit.x, y_values=y_values, line_values=line_values))


def _snap_to_lines(
    params: Sequence[float],
    *,
    y_values: list[float],
    line_values: list[float],
) -> list[float]:
    """Scale and offset the y-values, then snap them to the nearest lines."""
    return [
        line_values[nearest_index(line_values, y * params[0] + params[1])]
        for y in y_values
    ]


def _snapping_error(
    params: Sequence[float],
    *,
    y_values: list[float],
    line_values: list[float],
) -> float:
    """Total snapping distance of the scaled and offset y-values to their nearest lines."""
    corrected = _snap_to_lines(params, y_values=y_values, line_values=line_values)
    return sum(
        abs(y * params[0] + params[1] - line_y)
        for y, line_y in zip(y_values, corrected)
    )
