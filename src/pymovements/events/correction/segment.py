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
"""Provides the segment drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pymovements.events.correction._utils import is_right_to_left
from pymovements.events.correction._utils import line_index_to_y
from pymovements.events.correction._utils import location_x
from pymovements.events.correction._utils import to_line_values


def segment(
    line_ys: pl.Series | Sequence[float],
    *,
    directionality: str = 'left-to-right',
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Segment fixations into m line subsequences using return sweeps.

    Reference: :cite:p:`Abdulin2015,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left',
        mirroring the directionality of a text stimulus writing system. For
        'right-to-left' the return sweeps are identified accordingly.
        (default: 'left-to-right')
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. The returned expression
        operates on the full fixation sequence of a single trial, so it must be evaluated
        per trial. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.

    Raises
    ------
    ValueError
        If the directionality is 'top-to-bottom' or not a known value.
    """
    right_to_left = is_right_to_left(directionality)
    line_values = to_line_values(line_ys)
    m = len(line_values)
    x_diff = location_x(location).diff()

    # The m - 1 largest return sweep candidates mark line changes: the most negative
    # x-differences for left-to-right reading, the most positive ones for right-to-left
    # reading. With a single line no ordinal rank is <= 0, so no line changes occur.
    sweep_rank = x_diff.rank(method='ordinal', descending=right_to_left)
    line_change = (sweep_rank <= m - 1).fill_null(value=False)
    line_index = line_change.cum_sum()
    return line_index_to_y(line_index, line_values).alias('y_segment')
