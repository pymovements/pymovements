# Copyright (c) 2026 The pymovements Project Authors
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
"""Metrics for evaluating drift correction algorithms."""
from __future__ import annotations

import math
from collections import Counter
from numbers import Real

import polars as pl


def _validate_paired_series(first: pl.Series, second: pl.Series) -> None:
    """Validate two equally sized, non-null metric inputs."""
    if len(first) != len(second):
        raise ValueError(
            f'Metric inputs must have the same length, got {len(first)} and {len(second)}.',
        )
    if first.null_count() or second.null_count():
        raise ValueError('Metric inputs must not contain null values.')


def reassignment_rate(line_ys: pl.Series, corrected_ys: pl.Series) -> float:
    """Compute the fraction of fixations assigned to a different line.

    ``line_ys`` and ``corrected_ys`` must contain comparable line assignments, such
    as the integer line indices produced by the ensemble correction. They must have
    equal length and contain no null values.

    Parameters
    ----------
    line_ys: pl.Series
        Baseline line assignments or indices.
    corrected_ys: pl.Series
        Corrected line assignments or indices.

    Returns
    -------
    float
        The fraction of fixations that were reassigned to a different line.
    """
    _validate_paired_series(line_ys, corrected_ys)
    if len(line_ys) == 0:
        return 0.0
    return float((line_ys != corrected_ys).sum() / len(line_ys))


def mean_vertical_distance(line_ys: pl.Series, corrected_ys: pl.Series) -> float:
    """Compute mean absolute vertical distance from baseline line centers.

    Inputs are physical y-coordinates, not line indices. Equal-length, non-null
    inputs are required; empty inputs return ``0.0``.
    """
    _validate_paired_series(line_ys, corrected_ys)
    if len(line_ys) == 0:
        return 0.0
    vertical_distances = (line_ys - corrected_ys).abs()
    return float(vertical_distances.mean())


def imp_transitions(corrected_ys: pl.Series, *, max_line_distance: float) -> int:
    """Count adjacent corrected y-values that exceed a distance threshold.

    Parameters
    ----------
    corrected_ys: pl.Series
        Corrected physical y-coordinates in fixation order.
    max_line_distance: float
        Largest plausible distance between adjacent fixations, in coordinate units.
        A transition is counted only when its absolute distance is greater than this
        value.

    Returns
    -------
    int
        Number of implausibly large adjacent transitions. Empty input returns ``0``.

    Raises
    ------
    ValueError
        If the threshold is negative, non-finite, or the series contains nulls.
    """
    if not isinstance(max_line_distance, Real) or not math.isfinite(max_line_distance):
        raise ValueError('max_line_distance must be a finite, non-negative number.')
    if max_line_distance < 0:
        raise ValueError('max_line_distance must be a finite, non-negative number.')
    if corrected_ys.null_count():
        raise ValueError('Metric inputs must not contain null values.')
    if len(corrected_ys) < 2:
        return 0
    return int((corrected_ys.diff().abs() > max_line_distance).sum())


def ensemble_agreement(votes: pl.DataFrame) -> pl.Series:
    """Compute per-fixation agreement from a WoC vote table.

    ``votes`` must have one column per candidate algorithm and one row per fixation.
    Each result is the frequency of the most common vote divided by the number of
    algorithm columns. Ties receive the same score regardless of WoC's
    tie-breaking rule.

    Parameters
    ----------
    votes: pl.DataFrame
        DataFrame of fixation votes given by each fixation algorithm, with one row
        per algorithm and one row per fixation

    Returns
    -------
    pl.Series
        Float64 agreement values between 0.0 and 1.0, preserving row order.

    Raises
    ------
    ValueError
        If there are no algorithm columns in votes
        If votes contains null values
    """
    if not votes.columns:
        raise ValueError('votes must contain at least one algorithm column.')
    if votes.null_count().to_numpy().sum():
        raise ValueError('votes must not contain null values.')

    algorithm_count = len(votes.columns)
    agreements = [
        max(Counter(row).values()) / algorithm_count
        for row in votes.iter_rows()
    ]
    return pl.Series('ensemble_agreement', agreements, dtype=pl.Float64)
