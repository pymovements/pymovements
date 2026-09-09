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
"""Provides the merge drift correction algorithm."""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from functools import partial
from statistics import fmean

import numpy as np
import polars as pl

from pymovements.events.correction._utils import is_right_to_left
from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import map_locations
from pymovements.events.correction._utils import to_line_values

_MERGE_PHASES = [
    {'min_i': 3, 'min_j': 3, 'no_constraints': False},  # Phase 1
    {'min_i': 1, 'min_j': 3, 'no_constraints': False},  # Phase 2
    {'min_i': 1, 'min_j': 1, 'no_constraints': False},  # Phase 3
    {'min_i': 1, 'min_j': 1, 'no_constraints': True},   # Phase 4
]


def merge(
    line_ys: pl.Series | Sequence[float],
    *,
    y_thresh: float = 32,
    g_thresh: float = 0.1,
    e_thresh: float = 20,
    directionality: str = 'left-to-right',
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
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left',
        mirroring the directionality of a text stimulus writing system. For
        'right-to-left' the return sweep detection is adjusted accordingly.
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
    core = partial(
        _merge_core,
        line_values=line_values,
        right_to_left=right_to_left,
        y_thresh=y_thresh,
        g_thresh=g_thresh,
        e_thresh=e_thresh,
    )
    return map_locations(location, core, 'y_merge')


def _fit_line_error(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Fit a line through the points and return its gradient and root mean square error."""
    # Fitting a line through two-fixation candidates is expected in the unconstrained
    # merging phase and may be poorly conditioned; the resulting RankWarnings carry no
    # information for the user.
    rank_warning: type[Warning] = getattr(
        getattr(np, 'exceptions', np), 'RankWarning', RuntimeWarning,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rank_warning)
        gradient, intercept = np.polyfit(x_values, y_values, 1)
    residuals = [
        y - (gradient * x + intercept) for x, y in zip(x_values, y_values)
    ]
    error = math.sqrt(sum(residual**2 for residual in residuals) / len(residuals))
    return float(gradient), error


def _sequence_boundaries(
    x_values: list[float],
    y_values: list[float],
    *,
    right_to_left: bool,
    y_thresh: float,
) -> list[int]:
    """Find the indices starting a new sequence: regressive saccades and vertical jumps.

    For RTL scripts progressive saccades start a new sequence instead.
    """
    boundaries = []
    for index in range(len(x_values) - 1):
        x_diff = x_values[index + 1] - x_values[index]
        regressive = x_diff > 0 if right_to_left else x_diff < 0
        if regressive or abs(y_values[index + 1] - y_values[index]) > y_thresh:
            boundaries.append(index + 1)
    return boundaries


# pylint: disable=too-many-nested-blocks
def _merge_core(
    locations: pl.Series,
    *,
    line_values: list[float],
    right_to_left: bool,
    y_thresh: float,
    g_thresh: float,
    e_thresh: float,
) -> pl.Series:
    """Merge the progressive sequences of a single trial into one sequence per line."""
    x_values, y_values = locations_to_lists(locations)
    n = len(x_values)
    m = len(line_values)

    boundaries = _sequence_boundaries(
        x_values, y_values, right_to_left=right_to_left, y_thresh=y_thresh,
    )
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
