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
"""Tests for the merge drift correction algorithm."""
from __future__ import annotations

import importlib
import math
from statistics import fmean

import numpy as np
import polars as pl
import pytest

from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction.merge import _best_merger
from pymovements.events.correction.merge import _fit_line_error
from pymovements.events.correction.merge import _fit_line_from_statistics
from pymovements.events.correction.merge import _merge_core
from pymovements.events.correction.merge import _sequence_boundaries
from pymovements.events.correction.merge import _sequence_statistics

merge_module = importlib.import_module('pymovements.events.correction.merge')


@pytest.mark.parametrize('n_points', [2, 20, 250])
def test_fit_line_from_statistics_matches_polyfit(n_points):
    rng = np.random.default_rng(n_points)
    x_values = rng.normal(loc=500.0, scale=200.0, size=n_points).tolist()
    y_values = (
        0.03 * np.asarray(x_values) + 120.0 + rng.normal(scale=5.0, size=n_points)
    ).tolist()

    fit = _fit_line_from_statistics(
        _sequence_statistics(x_values, y_values, range(n_points)),
    )
    assert fit is not None
    slope, intercept, error = fit

    expected_slope, expected_intercept = np.polyfit(x_values, y_values, 1)
    residuals = [
        y - (expected_slope * x + expected_intercept)
        for x, y in zip(x_values, y_values)
    ]
    expected_error = math.sqrt(sum(residual**2 for residual in residuals) / n_points)

    assert slope == pytest.approx(expected_slope, rel=1e-12, abs=1e-12)
    assert intercept == pytest.approx(expected_intercept, rel=1e-12, abs=1e-12)
    assert error == pytest.approx(expected_error, rel=1e-12, abs=1e-12)


def test_fit_line_from_statistics_degenerate_x_returns_none():
    x_values = [100.0] * 5
    y_values = [101.0, 99.0, 100.0, 102.0, 98.0]

    fit = _fit_line_from_statistics(
        _sequence_statistics(x_values, y_values, range(len(x_values))),
    )

    assert fit is None
    gradient, error = _fit_line_error(x_values, y_values)
    assert math.isfinite(gradient)
    assert math.isfinite(error)


def _brute_force_best_merger(
    sequences,
    x_values,
    y_values,
    *,
    min_i,
    min_j,
    no_constraints,
    g_thresh,
    e_thresh,
):
    """Find the best merger with the original np.polyfit-based line fit."""
    best_merger = None
    best_error = math.inf
    for i in range(len(sequences) - 1):
        if len(sequences[i]) < min_i:
            continue
        for j in range(i + 1, len(sequences)):
            if len(sequences[j]) < min_j:
                continue
            candidate = sequences[i] + sequences[j]
            gradient, error = _fit_line_error(
                [x_values[index] for index in candidate],
                [y_values[index] for index in candidate],
            )
            if no_constraints or (abs(gradient) < g_thresh and error < e_thresh):
                if error < best_error:
                    best_merger = (i, j)
                    best_error = error
    return best_merger


def _brute_force_merge_core(
    locations,
    *,
    line_values,
    right_to_left,
    y_thresh,
    g_thresh,
    e_thresh,
):
    """Return a reference merge core that keeps using np.polyfit for every candidate."""
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

    merge_phases = [(3, 3, False), (1, 3, False), (1, 1, False), (1, 1, True)]
    for min_i, min_j, no_constraints in merge_phases:
        while len(sequences) > m:
            best_merger = _brute_force_best_merger(
                sequences, x_values, y_values,
                min_i=min_i,
                min_j=min_j,
                no_constraints=no_constraints,
                g_thresh=g_thresh,
                e_thresh=e_thresh,
            )
            if best_merger is None:
                break
            merge_i, merge_j = best_merger
            sequences.append(sequences[merge_i] + sequences[merge_j])
            del sequences[merge_j], sequences[merge_i]

    corrected_y = [0.0] * n
    sequence_order = sorted(
        range(len(sequences)),
        key=lambda index: fmean(y_values[i] for i in sequences[index]),
    )
    for line_index, sequence_index in enumerate(sequence_order):
        for fixation_index in sequences[sequence_index]:
            corrected_y[fixation_index] = line_values[line_index]
    return pl.Series(corrected_y)


def test_merge_core_matches_brute_force_reference():
    x_values = [
        100.0, 140.0, 180.0, 220.0,
        200.0, 240.0, 280.0,
        100.0, 140.0, 180.0, 220.0, 260.0, 300.0,
        250.0, 290.0,
        100.0, 140.0, 180.0,
        130.0, 170.0, 210.0, 250.0, 290.0,
    ]
    y_values = [
        102.0, 98.0, 101.0, 99.0,
        101.0, 99.0, 102.0,
        201.0, 198.0, 202.0, 199.0, 201.0, 200.0,
        202.0, 198.0,
        301.0, 299.0, 302.0,
        298.0, 301.0, 299.0, 300.0, 302.0,
    ]
    locations = pl.DataFrame({'x': x_values, 'y': y_values}).select(
        pl.concat_list(['x', 'y']).alias('location'),
    ).to_series()
    line_values = [100.0, 200.0, 300.0]
    kwargs = {
        'line_values': line_values,
        'right_to_left': False,
        'y_thresh': 32.0,
        'g_thresh': 0.1,
        'e_thresh': 20.0,
    }

    expected = _brute_force_merge_core(locations, **kwargs)
    result = _merge_core(locations, **kwargs)

    assert len(set(expected.to_list())) > 1
    assert result.to_list() == expected.to_list()


def test_best_merger_falls_back_to_polyfit_for_degenerate_x(monkeypatch):
    calls = []
    original = merge_module._fit_line_error

    def recording_fit_line_error(x_values, y_values):
        calls.append((x_values, y_values))
        return original(x_values, y_values)

    monkeypatch.setattr(merge_module, '_fit_line_error', recording_fit_line_error)

    x_values = [100.0] * 4
    y_values = [100.0, 100.0, 200.0, 200.0]
    sequences = [[0, 1], [2, 3]]
    statistics = [
        _sequence_statistics(x_values, y_values, sequence) for sequence in sequences
    ]

    best_merger = _best_merger(
        sequences, statistics, x_values, y_values,
        min_i=1,
        min_j=1,
        no_constraints=True,
        g_thresh=0.1,
        e_thresh=20.0,
    )

    assert best_merger == (0, 1)
    assert len(calls) == 1
