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
"""Provides the split drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import polars as pl
from sklearn.cluster import KMeans

from pymovements.events.correction._utils import is_right_to_left
from pymovements.events.correction._utils import location_expr
from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import nearest_line_y
from pymovements.events.correction._utils import to_line_values


def split(
    line_ys: pl.Series | Sequence[float],
    *,
    directionality: str = 'left-to-right',
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Split fixation sequence into line subsequences using K-Means return sweep identification.

    Reference: :cite:p:`Carr2022`.

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

    def _split_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = locations_to_lists(locations)
        x_diffs = [next_x - x for x, next_x in zip(x_values, x_values[1:])]

        # Split the saccades into two clusters; the cluster of largest leftward (rightward
        # for RTL scripts) saccades marks the return sweeps.
        cluster_labels = KMeans(2, n_init=10, max_iter=300).fit_predict(
            [[x_diff] for x_diff in x_diffs],
        )
        centers = [
            fmean(x_diff for x_diff, label in zip(x_diffs, cluster_labels) if label == 0),
            fmean(x_diff for x_diff, label in zip(x_diffs, cluster_labels) if label == 1),
        ]
        sweep_marker = centers.index(max(centers) if right_to_left else min(centers))

        is_sweep = [False] + [label == sweep_marker for label in cluster_labels]
        frame = pl.DataFrame({'y': y_values, 'is_sweep': is_sweep}).with_row_index()
        frame = frame.with_columns(pl.col('is_sweep').cum_sum().alias('segment'))
        corrected = frame.with_columns(
            nearest_line_y(pl.col('y').mean().over('segment'), line_values)
            .alias('y_corrected'),
        )
        return corrected.sort('index')['y_corrected']

    return (
        location_expr(location)
        .map_batches(_split_core, return_dtype=pl.Float64)
        .alias('y_split')
    )
