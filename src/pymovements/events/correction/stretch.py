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

import importlib
from collections.abc import Sequence
from functools import partial

import numpy as np
import polars as pl

from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import map_locations
from pymovements.events.correction._utils import to_line_values


def stretch(
    line_ys: pl.Series | Sequence[float],
    *,
    scale_bounds: tuple[float, float] = (0.9, 1.1),
    offset_bounds: tuple[float, float] = (-50, 50),
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Fit scale and offset bounds to stretch/compress fixations onto line centers.

    The fitted parameters stretch or compress the fixation sequence onto the line
    centers. The transformed y-coordinates are then snapped to their nearest text
    lines.

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
    return map_locations(location, core, 'y_stretch')


def _stretch_core(
    locations: pl.Series,
    *,
    line_values: list[float],
    scale_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
) -> pl.Series:
    """Fit scale and offset for the y-values of a single trial and snap them to lines."""
    scipy_optimize = importlib.import_module('scipy.optimize')

    _, y_values = locations_to_lists(locations)
    y_array = np.asarray(y_values, dtype=np.float64)
    line_array = np.asarray(line_values, dtype=np.float64)
    objective = partial(_snapping_error, y_values=y_array, line_values=line_array)
    best_fit = scipy_optimize.minimize(objective, [1, 0], bounds=[scale_bounds, offset_bounds])
    return pl.Series(_snap_to_lines(best_fit.x, y_values=y_array, line_values=line_array))


def _transform(y_values: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """Scale and offset the y-values."""
    return y_values * params[0] + params[1]


def _nearest_lines(transformed: np.ndarray, line_values: np.ndarray) -> np.ndarray:
    """Return the nearest line value for each y-value, ties favoring the first line."""
    if transformed.size == 0:
        return np.empty(0, dtype=line_values.dtype)
    nearest_indices = np.abs(transformed[:, None] - line_values[None, :]).argmin(axis=1)
    return line_values[nearest_indices]


def _snap_to_lines(
    params: Sequence[float],
    *,
    y_values: np.ndarray,
    line_values: np.ndarray,
) -> np.ndarray:
    """Scale and offset the y-values, then snap them to the nearest lines."""
    return _nearest_lines(_transform(y_values, params), line_values)


def _snapping_error(
    params: Sequence[float],
    *,
    y_values: np.ndarray,
    line_values: np.ndarray,
) -> float:
    """Total snapping distance of the scaled and offset y-values to their nearest lines."""
    transformed = _transform(y_values, params)
    nearest = _nearest_lines(transformed, line_values)
    return float(np.abs(transformed - nearest).sum())
