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
"""Provides the warp drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pymovements.events.correction._dynamic_time_warping import dynamic_time_warping_points
from pymovements.events.correction._utils import location_expr


def warp(
    word_locations: pl.Series,
    *,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Dynamic Time Warping alignment between fixation sequence and word positions.

    Reference: :cite:p:`Carr2022`.

    Parameters
    ----------
    word_locations: pl.Series
        Series of [x, y] word center coordinates, where y is the line position of the
        word's line.
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. The returned expression
        operates on the full fixation sequence of a single trial, so it must be evaluated
        per trial. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    word_points = word_locations.cast(pl.List(pl.Float64)).to_list()
    word_y_values = [point[1] for point in word_points]

    def _warp_core(locations: pl.Series) -> pl.Series:
        fixation_points = locations.cast(pl.List(pl.Float64)).to_list()
        _, dtw_path = dynamic_time_warping_points(fixation_points, word_points)
        corrected_y = [
            _mode([word_y_values[word_index] for word_index in mapped_words])
            for mapped_words in dtw_path
        ]
        return pl.Series(corrected_y, dtype=pl.Float64)

    return (
        location_expr(location)
        .map_batches(_warp_core, return_dtype=pl.Float64)
        .alias('y_warp')
    )


def _mode(values: Sequence[float]) -> float:
    """Calculate statistical mode of a sequence."""
    values_list = list(values)
    return float(max(set(values_list), key=values_list.count))
