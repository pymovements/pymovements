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
"""Algorithms for vertical drift correction in gaze data recorded during reading tasks.

Each algorithm function returns a :py:class:`polars.Expr` that computes the corrected
y-coordinates from a column of ``[x, y]`` fixation locations. The expressions operate on the
full fixation sequence of a trial, so they must be evaluated per trial.

The implementations follow the reference implementation of Carr et al. :cite:p:`Carr2022`.
Algorithms relying on k-means clustering, numerical optimization or dynamic time warping
('cluster', 'compare', 'merge', 'regress', 'slice', 'split', 'stretch', 'warp') materialize
the fixation sequence inside the expression via ``map_batches``, as their numeric cores
build on scikit-learn, scipy and dynamic programming.

References & Citations:
- :cite:p:`Abdulin2015`
- :cite:p:`AlMadi2025`
- :cite:p:`Carr2022`
- :cite:p:`Cohen2013`
- :cite:p:`Glandorf2021`
- :cite:p:`LimaSanches2015`
- :cite:p:`Lohmeier2015`
- :cite:p:`Mercier2024a`
- :cite:p:`Mercier2024b`
- :cite:p:`Spakov2019`
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.cluster import KMeans

_MERGE_PHASES = [
    {'min_i': 3, 'min_j': 3, 'no_constraints': False},  # Phase 1
    {'min_i': 1, 'min_j': 3, 'no_constraints': False},  # Phase 2
    {'min_i': 1, 'min_j': 1, 'no_constraints': False},  # Phase 3
    {'min_i': 1, 'min_j': 1, 'no_constraints': True},   # Phase 4
]


def _location_expr(location: str | pl.Expr) -> pl.Expr:
    """Resolve a location argument to an expression of [x, y] lists."""
    location_expr = pl.col(location) if isinstance(location, str) else location
    return location_expr.cast(pl.List(pl.Float64))


def _location_x(location: str | pl.Expr) -> pl.Expr:
    """Extract the x-coordinate from [x, y] locations."""
    return _location_expr(location).list.get(0)


def _location_y(location: str | pl.Expr) -> pl.Expr:
    """Extract the y-coordinate from [x, y] locations."""
    return _location_expr(location).list.get(1)


def _line_values(line_ys: pl.Series | Sequence[float]) -> list[float]:
    """Normalize line y-coordinates to a list of floats."""
    if isinstance(line_ys, pl.Series):
        return [float(line_y) for line_y in line_ys.to_list()]
    return [float(line_y) for line_y in line_ys]


def _nearest_line_index(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the index of the nearest text line for each y-value."""
    distances = pl.concat_list([(y_expr - line_y).abs() for line_y in line_values])
    return distances.list.arg_min()


def _nearest_line_y(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the y-coordinate of the nearest text line."""
    return _nearest_line_index(y_expr, line_values).replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def _line_index_to_y(index_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression mapping line indices to line y-coordinates."""
    return index_expr.replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def _locations_to_arrays(locations: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split a series of [x, y] locations into x and y arrays."""
    frame = locations.cast(pl.List(pl.Float64)).rename('location').to_frame()
    x_values = frame.select(pl.col('location').list.get(0))['location'].to_numpy()
    y_values = frame.select(pl.col('location').list.get(1))['location'].to_numpy()
    return x_values, y_values


######################################################################
# ATTACH
######################################################################


def attach(
    line_ys: pl.Series | Sequence[float],
    *,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Attach each fixation to the vertically closest text line center.

    Reference: :cite:p:`Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)
    return _nearest_line_y(_location_y(location), line_values).alias('y_attach')


######################################################################
# CHAIN
######################################################################


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
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)
    x_expr = _location_x(location)
    y_expr = _location_y(location)
    chain_break = (
        (x_expr.diff().abs() > x_thresh) | (y_expr.diff().abs() > y_thresh)
    ).fill_null(value=False)
    chain_index = chain_break.cum_sum()
    chain_mean_y = y_expr.mean().over(chain_index)
    return _nearest_line_y(chain_mean_y, line_values).alias('y_chain')


######################################################################
# CLUSTER
######################################################################


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
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)

    def _cluster_core(locations: pl.Series) -> pl.Series:
        _, y_values = _locations_to_arrays(locations)
        cluster_labels = KMeans(len(line_values), n_init=100, max_iter=300).fit_predict(
            y_values.reshape(-1, 1),
        )
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

    return (
        _location_expr(location)
        .map_batches(_cluster_core, return_dtype=pl.Float64)
        .alias('y_cluster')
    )


######################################################################
# COMPARE
######################################################################


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
        Column name or expression of [x, y] fixation locations. (default: 'location')

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
    word_x, word_y = _locations_to_arrays(word_locations)
    line_values = np.unique(word_y)
    # Clamping only extends behavior to inputs on which the reference implementation of
    # Carr et al. raises an IndexError; outputs are unchanged otherwise.
    n_candidates = min(n_nearest_lines, len(line_values))

    def _compare_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        frame = pl.DataFrame({'x': x_values, 'y': y_values}).with_row_index()
        frame = frame.with_columns(
            (pl.col('x').diff() < -x_thresh)
            .fill_null(value=False)
            .cum_sum()
            .alias('gaze_line'),
        )

        gaze_line_ys = {}
        for (gaze_line_value,), gaze_line in frame.group_by('gaze_line'):
            mean_y = gaze_line['y'].mean()
            candidate_order = np.argsort(np.abs(line_values - mean_y))[:n_candidates]
            candidate_costs = []
            for candidate_line in candidate_order:
                text_line_x = word_x[word_y == line_values[candidate_line]]
                cost, _ = _dynamic_time_warping(
                    gaze_line['x'].to_numpy().reshape(-1, 1),
                    text_line_x.reshape(-1, 1),
                )
                candidate_costs.append(cost)
            best_line = candidate_order[int(np.argmin(candidate_costs))]
            gaze_line_ys[gaze_line_value] = float(line_values[best_line])

        return (
            frame.with_columns(
                pl.col('gaze_line')
                .replace_strict(gaze_line_ys, return_dtype=pl.Float64)
                .alias('y_corrected'),
            )
            .sort('index')['y_corrected']
        )

    return (
        _location_expr(location)
        .map_batches(_compare_core, return_dtype=pl.Float64)
        .alias('y_compare')
    )


######################################################################
# MERGE
######################################################################


def merge(
    line_ys: pl.Series | Sequence[float],
    *,
    y_thresh: float = 32,
    g_thresh: float = 0.1,
    e_thresh: float = 20,
    text_right_to_left: bool = False,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Form progressive sequences and iteratively merge sequences belonging to the same line.

    Reference: :cite:p:`Spakov2019,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    y_thresh: float
        Vertical distance threshold for sequence splitting. (default: 32)
    g_thresh: float
        Gradient constraint for sequence merging. (default: 0.1)
    e_thresh: float
        Error constraint for sequence merging. (default: 20)
    text_right_to_left: bool
        If True, adjusts return sweep detection for Right-to-Left reading scripts.
        (default: False)
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)

    # pylint: disable=too-many-nested-blocks
    def _merge_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        n = len(x_values)
        m = len(line_values)
        diff_x = np.diff(x_values)
        dist_y = np.abs(np.diff(y_values))

        if text_right_to_left:
            boundaries = np.where((diff_x > 0) | (dist_y > y_thresh))[0] + 1
        else:
            boundaries = np.where((diff_x < 0) | (dist_y > y_thresh))[0] + 1

        sequence_starts = [0] + boundaries.tolist()
        sequence_ends = boundaries.tolist() + [n]
        sequences = [
            list(range(start, end)) for start, end in zip(sequence_starts, sequence_ends)
        ]

        for phase in _MERGE_PHASES:
            while len(sequences) > m:
                best_merger = None
                best_error = np.inf
                for i in range(len(sequences) - 1):
                    if len(sequences[i]) < phase['min_i']:
                        continue
                    for j in range(i + 1, len(sequences)):
                        if len(sequences[j]) < phase['min_j']:
                            continue
                        candidate_indices = sequences[i] + sequences[j]
                        gradient, intercept = np.polyfit(
                            x_values[candidate_indices], y_values[candidate_indices], 1,
                        )
                        residuals = y_values[candidate_indices] - (
                            gradient * x_values[candidate_indices] + intercept
                        )
                        error = np.sqrt(sum(residuals**2) / len(candidate_indices))
                        if phase['no_constraints'] or (
                            abs(gradient) < g_thresh and error < e_thresh
                        ):
                            if error < best_error:
                                best_merger = (i, j)
                                best_error = error
                if best_merger is None:
                    break
                merge_i, merge_j = best_merger
                merged_sequence = sequences[merge_i] + sequences[merge_j]
                sequences.append(merged_sequence)
                del sequences[merge_j], sequences[merge_i]

        corrected_y = np.zeros(n)
        sequence_mean_ys = [y_values[sequence].mean() for sequence in sequences]
        for line_i, sequence_i in enumerate(np.argsort(sequence_mean_ys)):
            corrected_y[sequences[sequence_i]] = line_values[line_i]
        return pl.Series(corrected_y)

    return (
        _location_expr(location)
        .map_batches(_merge_core, return_dtype=pl.Float64)
        .alias('y_merge')
    )


######################################################################
# REGRESS
######################################################################


def regress(
    line_ys: pl.Series | Sequence[float],
    *,
    k_bounds: tuple[float, float] = (-0.1, 0.1),
    o_bounds: tuple[float, float] = (-50, 50),
    s_bounds: tuple[float, float] = (1, 20),
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Fit linear regression parameters (slope, offset, std) to align fixations to lines.

    Reference: :cite:p:`Cohen2013,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    k_bounds: tuple[float, float]
        Slope bounds. (default: (-0.1, 0.1))
    o_bounds: tuple[float, float]
        Offset bounds. (default: (-50, 50))
    s_bounds: tuple[float, float]
        Standard deviation bounds. (default: (1, 20))
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = np.array(_line_values(line_ys))

    def _regress_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        n = len(x_values)
        m = len(line_values)

        def fit_lines(params: np.ndarray) -> np.ndarray:
            k = k_bounds[0] + (k_bounds[1] - k_bounds[0]) * norm.cdf(params[0])
            o = o_bounds[0] + (o_bounds[1] - o_bounds[0]) * norm.cdf(params[1])
            s = s_bounds[0] + (s_bounds[1] - s_bounds[0]) * norm.cdf(params[2])
            predicted_y_from_slope = x_values * k
            line_ys_plus_offset = line_values + o
            density = np.zeros((n, m))
            for line_i in range(m):
                fit_y = predicted_y_from_slope + line_ys_plus_offset[line_i]
                density[:, line_i] = norm.logpdf(y_values, fit_y, s)
            return density

        best_fit = minimize(lambda params: -sum(fit_lines(params).max(axis=1)), [0, 0, 0])
        line_assignments = fit_lines(best_fit.x).argmax(axis=1)
        return pl.Series(line_values[line_assignments])

    return (
        _location_expr(location)
        .map_batches(_regress_core, return_dtype=pl.Float64)
        .alias('y_regress')
    )


######################################################################
# SEGMENT
######################################################################


def segment(
    line_ys: pl.Series | Sequence[float],
    *,
    text_right_to_left: bool = False,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Segment fixations into m line subsequences using return sweeps.

    Reference: :cite:p:`Abdulin2015,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    text_right_to_left: bool
        If True, identifies return sweeps for Right-to-Left reading scripts.
        (default: False)
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)
    m = len(line_values)
    x_diff = _location_x(location).diff()

    # The m - 1 largest return sweep candidates mark line changes: the most negative
    # x-differences for left-to-right reading, the most positive ones for right-to-left
    # reading. With a single line no ordinal rank is <= 0, so no line changes occur.
    sweep_rank = x_diff.rank(method='ordinal', descending=text_right_to_left)
    line_change = (sweep_rank <= m - 1).fill_null(value=False)
    line_index = line_change.cum_sum()
    return _line_index_to_y(line_index, line_values).alias('y_segment')


######################################################################
# SPLIT
######################################################################


def split(
    line_ys: pl.Series | Sequence[float],
    *,
    text_right_to_left: bool = False,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Split fixation sequence into line subsequences using K-Means return sweep identification.

    Reference: :cite:p:`Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    text_right_to_left: bool
        If True, identifies return sweeps for Right-to-Left reading scripts.
        (default: False)
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = _line_values(line_ys)

    def _split_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        x_diff = np.diff(x_values)
        cluster_labels = KMeans(2, n_init=10, max_iter=300).fit_predict(
            x_diff.reshape(-1, 1),
        )
        centers = [x_diff[cluster_labels == 0].mean(), x_diff[cluster_labels == 1].mean()]
        sweep_marker = np.argmax(centers) if text_right_to_left else np.argmin(centers)

        is_sweep = np.concatenate([[False], cluster_labels == sweep_marker])
        frame = pl.DataFrame({'y': y_values, 'is_sweep': is_sweep}).with_row_index()
        frame = frame.with_columns(pl.col('is_sweep').cum_sum().alias('segment'))
        corrected = frame.with_columns(
            _nearest_line_y(pl.col('y').mean().over('segment'), line_values)
            .alias('y_corrected'),
        )
        return corrected.sort('index')['y_corrected']

    return (
        _location_expr(location)
        .map_batches(_split_core, return_dtype=pl.Float64)
        .alias('y_split')
    )


######################################################################
# STRETCH
######################################################################


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
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = np.array(_line_values(line_ys))

    def _stretch_core(locations: pl.Series) -> pl.Series:
        _, y_values = _locations_to_arrays(locations)
        n = len(y_values)

        def fit_lines(
            params: np.ndarray, return_correction: bool = False,
        ) -> np.ndarray | float:
            candidate_y = y_values * params[0] + params[1]
            corrected_y = np.zeros(n)
            for fixation_i in range(n):
                line_i = np.argmin(np.abs(line_values - candidate_y[fixation_i]))
                corrected_y[fixation_i] = line_values[line_i]
            if return_correction:
                return corrected_y
            return float(sum(np.abs(candidate_y - corrected_y)))

        best_fit = minimize(fit_lines, [1, 0], bounds=[scale_bounds, offset_bounds])
        corrected_y = cast(np.ndarray, fit_lines(best_fit.x, return_correction=True))
        return pl.Series(corrected_y)

    return (
        _location_expr(location)
        .map_batches(_stretch_core, return_dtype=pl.Float64)
        .alias('y_stretch')
    )


######################################################################
# WARP
######################################################################


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
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    word_x, word_y = _locations_to_arrays(word_locations)
    word_xy = np.column_stack([word_x, word_y])

    def _warp_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        fixation_xy = np.column_stack([x_values, y_values])
        _, dtw_path = _dynamic_time_warping(fixation_xy, word_xy)
        corrected_y = [
            _mode(word_y[words_mapped_to_fixation])
            for words_mapped_to_fixation in dtw_path
        ]
        return pl.Series(corrected_y, dtype=pl.Float64)

    return (
        _location_expr(location)
        .map_batches(_warp_core, return_dtype=pl.Float64)
        .alias('y_warp')
    )


def _mode(values: Sequence[float] | np.ndarray) -> float:
    """Calculate statistical mode of a sequence."""
    values_list = list(values)
    return float(max(set(values_list), key=values_list.count))


######################################################################
# Dynamic Time Warping
######################################################################


def _dynamic_time_warping(
    sequence1: np.ndarray,
    sequence2: np.ndarray,
) -> tuple[float, list[list[int]]]:
    """Calculate Dynamic Time Warping (DTW) cost and alignment path between two arrays."""
    n1 = len(sequence1)
    n2 = len(sequence2)
    dtw_cost = np.zeros((n1 + 1, n2 + 1))
    dtw_cost[0, :] = np.inf
    dtw_cost[:, 0] = np.inf
    dtw_cost[0, 0] = 0
    for i in range(n1):
        for j in range(n2):
            this_cost = np.sqrt(sum((sequence1[i] - sequence2[j]) ** 2))
            dtw_cost[i + 1, j + 1] = this_cost + min(
                dtw_cost[i, j + 1], dtw_cost[i + 1, j], dtw_cost[i, j],
            )
    dtw_cost = dtw_cost[1:, 1:]
    dtw_path: list[list[int]] = [[] for _ in range(n1)]
    i, j = n1 - 1, n2 - 1
    while i > 0 or j > 0:
        dtw_path[i].append(j)
        possible_moves = [np.inf, np.inf, np.inf]
        if i > 0 and j > 0:
            possible_moves[0] = dtw_cost[i - 1, j - 1]
        if i > 0:
            possible_moves[1] = dtw_cost[i - 1, j]
        if j > 0:
            possible_moves[2] = dtw_cost[i, j - 1]
        best_move = np.argmin(possible_moves)
        if best_move == 0:
            i -= 1
            j -= 1
        elif best_move == 1:
            i -= 1
        else:
            j -= 1
    dtw_path[0].append(0)
    return float(dtw_cost[-1, -1]), dtw_path


def dynamic_time_warping(
    sequence1: pl.Series,
    sequence2: pl.Series,
) -> tuple[float, list[list[int]]]:
    """Calculate Dynamic Time Warping (DTW) cost and alignment path between two sequences.

    Parameters
    ----------
    sequence1: pl.Series
        First sequence, either numeric or a series of [x, y] locations.
    sequence2: pl.Series
        Second sequence, either numeric or a series of [x, y] locations.

    Returns
    -------
    tuple[float, list[list[int]]]
        DTW cost and alignment path list mapping sequence1 elements to sequence2 elements.
    """
    return _dynamic_time_warping(
        _sequence_to_array(sequence1), _sequence_to_array(sequence2),
    )


def _sequence_to_array(sequence: pl.Series) -> np.ndarray:
    """Convert a numeric or [x, y] location series to a two-dimensional array."""
    if isinstance(sequence.dtype, (pl.List, pl.Array)):
        x_values, y_values = _locations_to_arrays(sequence)
        return np.column_stack([x_values, y_values])
    return sequence.to_numpy().reshape(-1, 1)


######################################################################
# SLICE
######################################################################


# pylint: disable=redefined-builtin,consider-using-tuple,consider-using-dict-items
def slice(
    line_ys: pl.Series | Sequence[float],
    *,
    x_thresh: float = 192,
    y_thresh: float = 32,
    w_thresh: float = 32,
    n_thresh: float = 90,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Slice algorithm to assign fixations in multi-line reading tasks.

    Reference: :cite:p:`Glandorf2021`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    x_thresh: float
        Horizontal run segmentation threshold. (default: 192)
    y_thresh: float
        Vertical run segmentation threshold. (default: 32)
    w_thresh: float
        Proto-line merger threshold. (default: 32)
    n_thresh: float
        Adjacent proto-line merger threshold. (default: 90)
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = np.array(_line_values(line_ys))

    def _slice_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_arrays(locations)
        fixation_xy = np.column_stack([x_values, y_values])
        n = len(fixation_xy)
        line_height = (
            float(np.mean(np.diff(line_values))) if len(line_values) > 1 else 32.0
        )
        proto_lines: dict[int, list[int]] = {}
        phantom_proto_lines: dict[int, np.ndarray] = {}

        # 1. Segment runs
        dist_x = np.abs(np.diff(fixation_xy[:, 0]))
        dist_y = np.abs(np.diff(fixation_xy[:, 1]))
        end_run_indices = list(
            (np.where(np.logical_or(dist_x > x_thresh, dist_y > y_thresh))[0] + 1).tolist(),
        )
        run_starts = [0] + end_run_indices
        run_ends = end_run_indices + [n]
        runs = [list(range(start, end)) for start, end in zip(run_starts, run_ends)]

        # 2. Determine starting run
        longest_run_i = int(
            np.argmax(
                [fixation_xy[run[-1], 0] - fixation_xy[run[0], 0] for run in runs],
            ),
        )
        proto_lines[0] = runs.pop(longest_run_i)

        # 3. Group runs into proto lines
        while runs:
            merger_on_this_iteration = False
            for proto_line_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
                proto_lines[proto_line_i + direction] = []
                if proto_lines[proto_line_i]:
                    proto_line_xy = fixation_xy[proto_lines[proto_line_i]]
                else:
                    proto_line_xy = phantom_proto_lines[proto_line_i]

                run_differences = np.zeros(len(runs))
                for run_i, run in enumerate(runs):
                    y_diffs = [
                        y - proto_line_xy[np.argmin(np.abs(proto_line_xy[:, 0] - x)), 1]
                        for x, y in fixation_xy[run]
                    ]
                    run_differences[run_i] = np.mean(y_diffs)

                merge_into_current = list(np.where(np.abs(run_differences) < w_thresh)[0])
                merge_into_adjacent = list(
                    np.where(
                        np.logical_and(
                            run_differences * direction >= w_thresh,
                            run_differences * direction < n_thresh,
                        ),
                    )[0],
                )

                for index in merge_into_current:
                    proto_lines[proto_line_i].extend(runs[index])
                for index in merge_into_adjacent:
                    proto_lines[proto_line_i + direction].extend(runs[index])

                if not merge_into_adjacent:
                    average_x, average_y = np.mean(proto_line_xy, axis=0)
                    adjacent_y = average_y + line_height * direction
                    phantom_proto_lines[proto_line_i + direction] = np.array(
                        [[average_x, adjacent_y]],
                    )

                for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                    del runs[index]
                    merger_on_this_iteration = True

            if not merger_on_this_iteration:
                break

        # 4. Assign leftover runs
        for run in runs:
            best_pl_distance = np.inf
            best_pl_assignment = next(iter(proto_lines))
            for proto_line_i in proto_lines:
                if proto_lines[proto_line_i]:
                    proto_line_xy = fixation_xy[proto_lines[proto_line_i]]
                else:
                    proto_line_xy = phantom_proto_lines[proto_line_i]
                y_diffs = [
                    y - proto_line_xy[np.argmin(np.abs(proto_line_xy[:, 0] - x)), 1]
                    for x, y in fixation_xy[run]
                ]
                pl_distance = float(np.abs(np.mean(y_diffs)))
                if pl_distance < best_pl_distance:
                    best_pl_distance = pl_distance
                    best_pl_assignment = proto_line_i
            proto_lines[best_pl_assignment].extend(run)

        # 5. Prune proto lines
        while len(proto_lines) > len(line_values):
            top, bot = min(proto_lines), max(proto_lines)
            if len(proto_lines[top]) < len(proto_lines[bot]):
                proto_lines[top + 1].extend(proto_lines[top])
                del proto_lines[top]
            else:
                proto_lines[bot - 1].extend(proto_lines[bot])
                del proto_lines[bot]

        # 6. Map proto lines to text lines
        corrected_y = np.zeros(n)
        for line_i, proto_line_i in enumerate(sorted(proto_lines)):
            corrected_y[proto_lines[proto_line_i]] = line_values[line_i]
        return pl.Series(corrected_y)

    return (
        _location_expr(location)
        .map_batches(_slice_core, return_dtype=pl.Float64)
        .alias('y_slice')
    )


######################################################################
# WISDOM OF THE CROWD (Ensemble Method)
######################################################################


def wisdom_of_the_crowd(assignment_columns: Sequence[str]) -> pl.Expr:
    """Ensemble correction choosing line assignment with most votes across algorithms.

    In the event of a tie, the left-most column in ``assignment_columns`` is given
    priority, following the reference implementation.

    Reference: :cite:p:`Mercier2024b`.

    Parameters
    ----------
    assignment_columns: Sequence[str]
        Names of the columns holding the corrected y-coordinates or line assignments of
        the candidate algorithms, in order of tie-breaking priority.

    Returns
    -------
    pl.Expr
        Expression computing the ensemble-corrected values.
    """
    column_priority = {column: priority for priority, column in enumerate(assignment_columns)}

    def _woc_core(votes: pl.Series) -> pl.Series:
        counted = (
            votes.rename('vote').to_frame()
            .unnest('vote')
            .with_row_index('fixation_index')
            .unpivot(index='fixation_index', variable_name='algorithm', value_name='y')
            .with_columns(
                pl.col('algorithm')
                .replace_strict(column_priority, return_dtype=pl.UInt32)
                .alias('priority'),
                pl.len().over(['fixation_index', 'y']).alias('votes'),
            )
        )
        return (
            counted
            .filter(pl.col('votes') == pl.col('votes').max().over('fixation_index'))
            .group_by('fixation_index', maintain_order=False)
            .agg(pl.col('y').sort_by('priority').first())
            .sort('fixation_index')['y']
        )

    return (
        pl.struct(list(assignment_columns))
        .map_batches(_woc_core)
        .alias('y_wisdom_of_the_crowd')
    )
