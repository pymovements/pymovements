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
"""Provides shared helpers for the drift correction algorithms."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def location_expr(location: str | pl.Expr) -> pl.Expr:
    """Resolve a location argument to an expression of [x, y] lists."""
    expr = pl.col(location) if isinstance(location, str) else location
    return expr.cast(pl.List(pl.Float64))


def location_x(location: str | pl.Expr) -> pl.Expr:
    """Extract the x-coordinate from [x, y] locations."""
    return location_expr(location).list.get(0)


def location_y(location: str | pl.Expr) -> pl.Expr:
    """Extract the y-coordinate from [x, y] locations."""
    return location_expr(location).list.get(1)


def to_line_values(line_ys: pl.Series | Sequence[float]) -> list[float]:
    """Normalize line y-coordinates to a list of floats."""
    if isinstance(line_ys, pl.Series):
        return [float(line_y) for line_y in line_ys.to_list()]
    return [float(line_y) for line_y in line_ys]


def nearest_line_index(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the index of the nearest text line for each y-value."""
    distances = pl.concat_list([(y_expr - line_y).abs() for line_y in line_values])
    return distances.list.arg_min()


def nearest_line_y(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the y-coordinate of the nearest text line."""
    return nearest_line_index(y_expr, line_values).replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def line_index_to_y(index_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression mapping line indices to line y-coordinates."""
    return index_expr.replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def locations_to_lists(locations: pl.Series) -> tuple[list[float], list[float]]:
    """Split a series of [x, y] locations into lists of x and y values."""
    points = locations.cast(pl.List(pl.Float64)).to_list()
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return x_values, y_values


def nearest_index(values: Sequence[float], target: float) -> int:
    """Return the index of the value closest to target, ties favoring the first."""
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def is_right_to_left(directionality: str) -> bool:
    """Validate a directionality value and resolve it to a right-to-left flag.

    Parameters
    ----------
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left'.

    Returns
    -------
    bool
        True if the directionality is 'right-to-left', False if 'left-to-right'.

    Raises
    ------
    ValueError
        If the directionality is 'top-to-bottom' or not a known value.
    """
    if directionality == 'top-to-bottom':
        raise ValueError(
            "directionality 'top-to-bottom' is not supported by the drift correction "
            'algorithms, which assume horizontal lines of text.',
        )
    if directionality not in ('left-to-right', 'right-to-left'):
        raise ValueError(
            f"Unknown directionality '{directionality}'. "
            "Valid values are: 'left-to-right', 'right-to-left'.",
        )
    return directionality == 'right-to-left'
