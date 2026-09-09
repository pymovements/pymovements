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
"""Provides the compare drift correction algorithm."""
from __future__ import annotations

import math
from typing import cast

import polars as pl

from pymovements.events.correction._dynamic_time_warping import dynamic_time_warping_points
from pymovements.events.correction._utils import location_expr
from pymovements.events.correction._utils import locations_to_lists


def compare(
    word_locations: pl.Series,
    *,
    x_thresh: float = 512,
    n_nearest_lines: int = 3,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Match fixation lines to candidate text lines using Dynamic Time Warping (DTW).

    Reference: :cite:p:`LimaSanches2015,Carr2022`.

    Parameters
    ----------
    word_locations: pl.Series
        Series of [x, y] word center coordinates, where y is the line position of the
        word's line.
    x_thresh: float
        Threshold for detecting line breaks. (default: 512)
    n_nearest_lines: int
        Number of candidate nearest lines to evaluate with DTW. Values larger than the
        number of text lines are clamped. (default: 3)
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
        If n_nearest_lines is smaller than 1.
    """
    if n_nearest_lines < 1:
        raise ValueError('n_nearest_lines must be at least 1.')
    word_x_values, word_y_values = locations_to_lists(word_locations)
    line_values = sorted(set(word_y_values))
    word_x_per_line = {
        line_y: [
            word_x for word_x, word_y in zip(word_x_values, word_y_values)
            if word_y == line_y
        ]
        for line_y in line_values
    }
    # Clamping only extends behavior to inputs on which the reference implementation of
    # Carr et al. raises an IndexError; outputs are unchanged otherwise.
    n_candidates = min(n_nearest_lines, len(line_values))

    def _compare_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = locations_to_lists(locations)
        frame = pl.DataFrame({'x': x_values, 'y': y_values}).with_row_index()
        frame = frame.with_columns(
            (pl.col('x').diff() < -x_thresh)
            .fill_null(value=False)
            .cum_sum()
            .alias('gaze_line'),
        )

        # Assign each gaze line to the candidate text line with the lowest DTW cost.
        gaze_line_ys = {}
        for (gaze_line_value,), gaze_line in frame.group_by('gaze_line'):
            mean_y = cast(float, gaze_line['y'].mean())
            distances = sorted(
                (abs(line_y - mean_y), index) for index, line_y in enumerate(line_values)
            )
            candidate_lines = [line_values[index] for _, index in distances[:n_candidates]]

            gaze_x = [[x] for x in gaze_line['x'].to_list()]
            best_line = candidate_lines[0]
            best_cost = math.inf
            for line_y in candidate_lines:
                text_x = [[x] for x in word_x_per_line[line_y]]
                cost, _ = dynamic_time_warping_points(gaze_x, text_x)
                if cost < best_cost:
                    best_cost = cost
                    best_line = line_y
            gaze_line_ys[gaze_line_value] = best_line

        return (
            frame.with_columns(
                pl.col('gaze_line')
                .replace_strict(gaze_line_ys, return_dtype=pl.Float64)
                .alias('y_corrected'),
            )
            .sort('index')['y_corrected']
        )

    return (
        location_expr(location)
        .map_batches(_compare_core, return_dtype=pl.Float64)
        .alias('y_compare')
    )
