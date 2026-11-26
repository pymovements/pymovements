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
"""Tests for the dynamic time warping used by the drift correction algorithms."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from pymovements.events.correction._dynamic_time_warping import dynamic_time_warping_points
from pymovements.events.correction.compare import compare
from pymovements.events.correction.warp import warp


def reference_dynamic_time_warping_points(
    sequence1: list[list[float]],
    sequence2: list[list[float]],
) -> tuple[float, list[list[int]]]:
    """Return the original pure Python DTW cost and path as a reference."""
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


def _mode(values: list[float]) -> float:
    """Calculate the statistical mode exactly like the warp implementation."""
    return float(max(set(values), key=values.count))


def _make_word_points() -> list[list[float]]:
    """Return a three line text layout with eight words per line."""
    return [
        [float(x), float(line_y)]
        for line_y in (100, 200, 300)
        for x in np.linspace(100, 700, 8)
    ]


def _make_fixation_points() -> list[list[float]]:
    """Return a drifting fixation sequence with one gaze line per text line."""
    locations = []
    for line_y in (100, 200, 300):
        for x in (100, 180, 260, 340, 420, 500, 580, 660):
            locations.append([float(x + 3), float(line_y + 4)])
    return locations


@pytest.mark.parametrize(
    ('length1', 'length2'),
    [
        (1, 1),
        (1, 4),
        (4, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (3, 7),
        (7, 3),
        (1, 10),
        (10, 1),
        (8, 13),
        (13, 8),
    ],
)
@pytest.mark.parametrize('seed', range(5))
def test_dynamic_time_warping_points_matches_reference(length1, length2, seed):
    rng = np.random.default_rng(seed)
    sequence1 = rng.uniform(0, 500, size=(length1, 2)).tolist()
    sequence2 = rng.uniform(0, 500, size=(length2, 2)).tolist()

    expected_cost, expected_path = reference_dynamic_time_warping_points(sequence1, sequence2)
    cost, path = dynamic_time_warping_points(sequence1, sequence2)

    assert cost == expected_cost
    assert path == expected_path


def test_dynamic_time_warping_points_matches_reference_on_ties():
    sequence1 = [[10.0, 20.0]] * 5
    sequence2 = [[10.0, 20.0], [10.0, 20.0], [30.0, 40.0], [30.0, 40.0]]

    expected_cost, expected_path = reference_dynamic_time_warping_points(sequence1, sequence2)
    cost, path = dynamic_time_warping_points(sequence1, sequence2)

    assert cost == expected_cost
    assert path == expected_path


def test_warp_matches_reference_path():
    word_points = _make_word_points()
    fixation_points = _make_fixation_points()
    word_locations = pl.Series(
        'word_location', word_points, dtype=pl.List(pl.Float64),
    )

    frame = pl.DataFrame({'location': fixation_points})
    corrected_y = frame.select(
        warp(word_locations).alias('y_warp'),
    )['y_warp'].to_list()

    _, reference_path = reference_dynamic_time_warping_points(fixation_points, word_points)
    word_y_values = [point[1] for point in word_points]
    expected_y = [
        _mode([word_y_values[word_index] for word_index in mapped_words])
        for mapped_words in reference_path
    ]

    assert corrected_y == expected_y


def _compare_assignments(
    locations: list[list[float]],
    line_values: list[float],
    word_x_per_line: dict[float, list[float]],
    n_candidates: int,
    x_thresh: float,
    dtw: Callable[
        [list[list[float]], list[list[float]]],
        tuple[float, list[list[int]]],
    ],
) -> tuple[dict[int, float], list[float]]:
    """Replicate the compare core and return gaze line assignments and costs."""
    x_values = [point[0] for point in locations]
    y_values = [point[1] for point in locations]
    frame = pl.DataFrame({'x': x_values, 'y': y_values}).with_row_index()
    frame = frame.with_columns(
        (pl.col('x').diff() < -x_thresh)
        .fill_null(value=False)
        .cum_sum()
        .alias('gaze_line'),
    )

    assignments = {}
    costs = []
    for (gaze_line_value,), gaze_line in frame.group_by('gaze_line'):
        mean_y = gaze_line['y'].mean()
        distances = sorted(
            (abs(line_y - mean_y), index) for index, line_y in enumerate(line_values)
        )
        candidate_lines = [line_values[index] for _, index in distances[:n_candidates]]

        gaze_x = [[x] for x in gaze_line['x'].to_list()]
        best_line = candidate_lines[0]
        best_cost = math.inf
        for line_y in candidate_lines:
            text_x = [[x] for x in word_x_per_line[line_y]]
            cost, _ = dtw(gaze_x, text_x)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_line = line_y
        assignments[gaze_line_value] = best_line
    return assignments, costs


def test_compare_costs_and_assignments_match_reference():
    word_points = _make_word_points()
    fixation_points = _make_fixation_points()
    x_thresh = 512.0
    n_candidates = 3

    word_x_values = [point[0] for point in word_points]
    word_y_values = [point[1] for point in word_points]
    line_values = sorted(set(word_y_values))
    word_x_per_line = {
        line_y: [
            word_x for word_x, word_y in zip(word_x_values, word_y_values)
            if word_y == line_y
        ]
        for line_y in line_values
    }

    reference_assignments, reference_costs = _compare_assignments(
        fixation_points, line_values, word_x_per_line, n_candidates, x_thresh,
        reference_dynamic_time_warping_points,
    )
    optimized_assignments, optimized_costs = _compare_assignments(
        fixation_points, line_values, word_x_per_line, n_candidates, x_thresh,
        dynamic_time_warping_points,
    )

    assert optimized_costs == reference_costs
    assert optimized_assignments == reference_assignments

    word_locations = pl.Series(
        'word_location', word_points, dtype=pl.List(pl.Float64),
    )
    frame = pl.DataFrame({'location': fixation_points})
    corrected_y = frame.select(
        compare(word_locations).alias('y_compare'),
    )['y_compare'].to_list()

    gaze_lines = frame.with_row_index().with_columns(
        (pl.col('location').list.get(0).diff() < -x_thresh)
        .fill_null(value=False)
        .cum_sum()
        .alias('gaze_line'),
    )['gaze_line'].to_list()
    expected_y = [reference_assignments[gaze_line] for gaze_line in gaze_lines]

    assert corrected_y == expected_y
