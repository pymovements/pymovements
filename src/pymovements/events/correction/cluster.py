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
"""Provides the cluster drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import polars as pl
from sklearn.cluster import KMeans

from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import map_per_trial
from pymovements.events.correction._utils import to_line_values


def cluster(
    line_ys: pl.Series | Sequence[float],
    *,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Cluster Y-coordinates into clusters matching text lines using K-Means.

    Reference: :cite:p:`Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
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
    core = partial(_cluster_core, line_values=line_values)
    return map_per_trial(location, core, 'y_cluster')


def _cluster_core(locations: pl.Series, *, line_values: list[float]) -> pl.Series:
    """Cluster the y-values of a single trial into one KMeans cluster per text line."""
    _, y_values = locations_to_lists(locations)
    cluster_labels = KMeans(len(line_values), n_init=100, max_iter=300).fit_predict(
        [[y] for y in y_values],
    )
    # Clusters ordered by their mean y-coordinate map to the text lines top to bottom.
    frame = pl.DataFrame({'y': y_values, 'cluster': cluster_labels}).with_row_index()
    cluster_ranks = (
        frame.group_by('cluster')
        .agg(pl.col('y').mean().alias('center'))
        .sort('center')
        .with_columns(pl.Series('y_corrected', line_values))
    )
    return (
        frame.join(cluster_ranks.select(['cluster', 'y_corrected']), on='cluster')
        .sort('index')['y_corrected']
    )
