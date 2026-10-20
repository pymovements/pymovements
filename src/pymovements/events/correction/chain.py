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
"""Provides the chain drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pymovements.events.correction._utils import location_x
from pymovements.events.correction._utils import location_y
from pymovements.events.correction._utils import nearest_line_y
from pymovements.events.correction._utils import to_line_values


def chain(
    line_ys: pl.Series | Sequence[float],
    *,
    x_thresh: float = 192,
    y_thresh: float = 32,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Group fixations into reading chains based on distance thresholds and align to lines.

    Reference: :cite:p:`Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    x_thresh: float
        Horizontal distance threshold to break a chain. (default: 192)
    y_thresh: float
        Vertical distance threshold to break a chain. (default: 32)
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
    x_expr = location_x(location)
    y_expr = location_y(location)
    chain_break = (
        (x_expr.diff().abs() > x_thresh) | (y_expr.diff().abs() > y_thresh)
    ).fill_null(value=False)
    chain_index = chain_break.cum_sum()
    chain_mean_y = y_expr.mean().over(chain_index)
    return nearest_line_y(chain_mean_y, line_values).alias('y_chain')
