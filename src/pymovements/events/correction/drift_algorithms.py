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
Algorithms that build on k-means clustering, numerical optimization or line fitting
('cluster', 'compare', 'merge', 'regress', 'slice', 'split', 'stretch', 'warp') materialize
the fixation sequence inside the expression via ``map_batches``; their numeric cores use
scikit-learn, scipy and numpy's polyfit.

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

import math
import warnings
from collections.abc import Sequence
from statistics import fmean
from typing import cast

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.cluster import KMeans

_RANK_WARNING: type[Warning] = getattr(
    getattr(np, 'exceptions', np), 'RankWarning', RuntimeWarning,
)

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


def _locations_to_lists(locations: pl.Series) -> tuple[list[float], list[float]]:
    """Split a series of [x, y] locations into lists of x and y values."""
    points = locations.cast(pl.List(pl.Float64)).to_list()
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return x_values, y_values


def _nearest_index(values: Sequence[float], target: float) -> int:
    """Return the index of the value closest to target, ties favoring the first."""
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


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
        _, y_values = _locations_to_lists(locations)
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
    word_x_values, word_y_values = _locations_to_lists(word_locations)
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
        x_values, y_values = _locations_to_lists(locations)
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
                cost, _ = _dynamic_time_warping(gaze_x, text_x)
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

    def _fit_line_error(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
        """Fit a line through the points and return its gradient and root mean square error."""
        # Fitting a line through two-fixation candidates is expected in the unconstrained
        # merging phase and may be poorly conditioned; the resulting RankWarnings carry no
        # information for the user.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _RANK_WARNING)
            gradient, intercept = np.polyfit(x_values, y_values, 1)
        residuals = [
            y - (gradient * x + intercept) for x, y in zip(x_values, y_values)
        ]
        error = math.sqrt(sum(residual**2 for residual in residuals) / len(residuals))
        return float(gradient), error

    # pylint: disable=too-many-nested-blocks
    def _merge_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_lists(locations)
        n = len(x_values)
        m = len(line_values)

        # A new sequence starts at every regressive saccade (progressive for RTL scripts)
        # and at every large vertical jump.
        def _is_boundary(index: int) -> bool:
            x_diff = x_values[index + 1] - x_values[index]
            regressive = x_diff > 0 if text_right_to_left else x_diff < 0
            return regressive or abs(y_values[index + 1] - y_values[index]) > y_thresh

        boundaries = [index + 1 for index in range(n - 1) if _is_boundary(index)]
        sequences = [
            list(range(start, end))
            for start, end in zip([0] + boundaries, boundaries + [n])
        ]

        # Iteratively merge the pair of sequences with the best line fit, relaxing the
        # sequence length and fit quality constraints phase by phase.
        for phase in _MERGE_PHASES:
            while len(sequences) > m:
                best_merger = None
                best_error = math.inf
                for i in range(len(sequences) - 1):
                    if len(sequences[i]) < phase['min_i']:
                        continue
                    for j in range(i + 1, len(sequences)):
                        if len(sequences[j]) < phase['min_j']:
                            continue
                        candidate = sequences[i] + sequences[j]
                        gradient, error = _fit_line_error(
                            [x_values[index] for index in candidate],
                            [y_values[index] for index in candidate],
                        )
                        if phase['no_constraints'] or (
                            abs(gradient) < g_thresh and error < e_thresh
                        ):
                            if error < best_error:
                                best_merger = (i, j)
                                best_error = error
                if best_merger is None:
                    break
                merge_i, merge_j = best_merger
                sequences.append(sequences[merge_i] + sequences[merge_j])
                del sequences[merge_j], sequences[merge_i]

        # Sequences ordered by their mean y-coordinate map to the text lines top to bottom.
        corrected_y = [0.0] * n
        sequence_order = sorted(
            range(len(sequences)),
            key=lambda index: fmean(y_values[i] for i in sequences[index]),
        )
        for line_index, sequence_index in enumerate(sequence_order):
            for fixation_index in sequences[sequence_index]:
                corrected_y[fixation_index] = line_values[line_index]
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
    line_values = _line_values(line_ys)

    def _regress_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_lists(locations)

        def line_log_densities(params: Sequence[float]) -> list[np.ndarray]:
            """Per-line log-densities of observing the fixation y-values."""
            slope = k_bounds[0] + (k_bounds[1] - k_bounds[0]) * norm.cdf(params[0])
            offset = o_bounds[0] + (o_bounds[1] - o_bounds[0]) * norm.cdf(params[1])
            deviation = s_bounds[0] + (s_bounds[1] - s_bounds[0]) * norm.cdf(params[2])
            predicted_y = [x * slope for x in x_values]
            return [
                norm.logpdf(
                    y_values,
                    [predicted + line_y + offset for predicted in predicted_y],
                    deviation,
                )
                for line_y in line_values
            ]

        def negative_log_likelihood(params: Sequence[float]) -> float:
            densities = line_log_densities(params)
            return -sum(max(fixation) for fixation in zip(*densities))

        best_fit = minimize(negative_log_likelihood, [0, 0, 0])
        densities = line_log_densities(best_fit.x)
        corrected_y = [
            line_values[max(range(len(line_values)), key=list(fixation).__getitem__)]
            for fixation in zip(*densities)
        ]
        return pl.Series(corrected_y)

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
        x_values, y_values = _locations_to_lists(locations)
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
        sweep_marker = centers.index(max(centers) if text_right_to_left else min(centers))

        is_sweep = [False] + [label == sweep_marker for label in cluster_labels]
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
    line_values = _line_values(line_ys)

    def _stretch_core(locations: pl.Series) -> pl.Series:
        _, y_values = _locations_to_lists(locations)

        def snap_to_lines(params: Sequence[float]) -> list[float]:
            """Scale and offset the y-values, then snap them to the nearest lines."""
            return [
                line_values[_nearest_index(line_values, y * params[0] + params[1])]
                for y in y_values
            ]

        def snapping_error(params: Sequence[float]) -> float:
            corrected = snap_to_lines(params)
            return sum(
                abs(y * params[0] + params[1] - line_y)
                for y, line_y in zip(y_values, corrected)
            )

        best_fit = minimize(snapping_error, [1, 0], bounds=[scale_bounds, offset_bounds])
        return pl.Series(snap_to_lines(best_fit.x))

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
    word_points = word_locations.cast(pl.List(pl.Float64)).to_list()
    word_y_values = [point[1] for point in word_points]

    def _warp_core(locations: pl.Series) -> pl.Series:
        fixation_points = locations.cast(pl.List(pl.Float64)).to_list()
        _, dtw_path = _dynamic_time_warping(fixation_points, word_points)
        corrected_y = [
            _mode([word_y_values[word_index] for word_index in mapped_words])
            for mapped_words in dtw_path
        ]
        return pl.Series(corrected_y, dtype=pl.Float64)

    return (
        _location_expr(location)
        .map_batches(_warp_core, return_dtype=pl.Float64)
        .alias('y_warp')
    )


def _mode(values: Sequence[float]) -> float:
    """Calculate statistical mode of a sequence."""
    values_list = list(values)
    return float(max(set(values_list), key=values_list.count))


######################################################################
# Dynamic Time Warping
######################################################################


def _dynamic_time_warping(
    sequence1: list[list[float]],
    sequence2: list[list[float]],
) -> tuple[float, list[list[int]]]:
    """Calculate Dynamic Time Warping (DTW) cost and alignment path between two point lists."""
    n1 = len(sequence1)
    n2 = len(sequence2)
    cost = [[math.inf] * (n2 + 1) for _ in range(n1 + 1)]
    cost[0][0] = 0.0
    for i in range(n1):
        for j in range(n2):
            step_cost = math.sqrt(
                sum(
                    (p - q) ** 2 for p, q in zip(sequence1[i], sequence2[j])
                ),
            )
            cost[i + 1][j + 1] = step_cost + min(
                cost[i][j + 1], cost[i + 1][j], cost[i][j],
            )

    dtw_path: list[list[int]] = [[] for _ in range(n1)]
    i, j = n1 - 1, n2 - 1
    while i > 0 or j > 0:
        dtw_path[i].append(j)
        possible_moves = [
            cost[i][j] if i > 0 and j > 0 else math.inf,
            cost[i][j + 1] if i > 0 else math.inf,
            cost[i + 1][j] if j > 0 else math.inf,
        ]
        best_move = possible_moves.index(min(possible_moves))
        if best_move == 0:
            i -= 1
            j -= 1
        elif best_move == 1:
            i -= 1
        else:
            j -= 1
    dtw_path[0].append(0)
    return cost[n1][n2], dtw_path


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
        _sequence_to_points(sequence1), _sequence_to_points(sequence2),
    )


def _sequence_to_points(sequence: pl.Series) -> list[list[float]]:
    """Convert a numeric or [x, y] location series to a list of points."""
    if isinstance(sequence.dtype, (pl.List, pl.Array)):
        return sequence.cast(pl.List(pl.Float64)).to_list()
    return [[value] for value in sequence.to_list()]


######################################################################
# SLICE
######################################################################


# pylint: disable=redefined-builtin
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
    line_values = _line_values(line_ys)
    if len(line_values) > 1:
        line_height = fmean(
            next_line - line for line, next_line in zip(line_values, line_values[1:])
        )
    else:
        line_height = 32.0

    def _slice_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = _locations_to_lists(locations)
        n = len(x_values)

        def run_offset(run: list[int], proto_line: list[tuple[float, float]]) -> float:
            """Mean vertical offset of a run to the horizontally closest proto line points."""
            proto_x = [point[0] for point in proto_line]
            return fmean(
                y_values[index] - proto_line[_nearest_index(proto_x, x_values[index])][1]
                for index in run
            )

        # 1. Segment runs of horizontally and vertically close fixations.
        boundaries = [
            index + 1 for index in range(n - 1)
            if abs(x_values[index + 1] - x_values[index]) > x_thresh
            or abs(y_values[index + 1] - y_values[index]) > y_thresh
        ]
        runs = [
            list(range(start, end))
            for start, end in zip([0] + boundaries, boundaries + [n])
        ]

        # 2. The horizontally longest run starts the first proto line.
        longest_run = max(
            range(len(runs)),
            key=lambda index: x_values[runs[index][-1]] - x_values[runs[index][0]],
        )
        proto_lines: dict[int, list[int]] = {0: runs.pop(longest_run)}
        phantom_proto_lines: dict[int, list[tuple[float, float]]] = {}

        def proto_line_points(proto_line_index: int) -> list[tuple[float, float]]:
            if proto_lines[proto_line_index]:
                return [
                    (x_values[index], y_values[index])
                    for index in proto_lines[proto_line_index]
                ]
            return phantom_proto_lines[proto_line_index]

        # 3. Grow proto lines above and below by merging runs within the thresholds; where
        # nothing merges, a phantom proto line one line height away keeps the search going.
        while runs:
            merged_on_this_iteration = False
            for proto_line_index, direction in (
                (min(proto_lines), -1), (max(proto_lines), 1),
            ):
                proto_lines[proto_line_index + direction] = []
                points = proto_line_points(proto_line_index)

                offsets = [run_offset(run, points) for run in runs]
                merge_into_current = [
                    index for index, offset in enumerate(offsets)
                    if abs(offset) < w_thresh
                ]
                merge_into_adjacent = [
                    index for index, offset in enumerate(offsets)
                    if w_thresh <= offset * direction < n_thresh
                ]

                for index in merge_into_current:
                    proto_lines[proto_line_index].extend(runs[index])
                for index in merge_into_adjacent:
                    proto_lines[proto_line_index + direction].extend(runs[index])

                if not merge_into_adjacent:
                    average_x = fmean(point[0] for point in points)
                    average_y = fmean(point[1] for point in points)
                    phantom_proto_lines[proto_line_index + direction] = [
                        (average_x, average_y + line_height * direction),
                    ]

                for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                    del runs[index]
                    merged_on_this_iteration = True

            if not merged_on_this_iteration:
                break

        # 4. Assign leftover runs to the vertically closest proto line.
        for run in runs:
            best_distance = math.inf
            best_proto_line = next(iter(proto_lines))
            for proto_line_index in proto_lines:
                distance = abs(run_offset(run, proto_line_points(proto_line_index)))
                if distance < best_distance:
                    best_distance = distance
                    best_proto_line = proto_line_index
            proto_lines[best_proto_line].extend(run)

        # 5. Merge the smaller of the outermost proto lines inwards until the number of
        # proto lines matches the number of text lines.
        while len(proto_lines) > len(line_values):
            top, bottom = min(proto_lines), max(proto_lines)
            if len(proto_lines[top]) < len(proto_lines[bottom]):
                proto_lines[top + 1].extend(proto_lines.pop(top))
            else:
                proto_lines[bottom - 1].extend(proto_lines.pop(bottom))

        # 6. Proto lines map to the text lines top to bottom.
        corrected_y = [0.0] * n
        for line_index, proto_line_index in enumerate(sorted(proto_lines)):
            for fixation_index in proto_lines[proto_line_index]:
                corrected_y[fixation_index] = line_values[line_index]
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
