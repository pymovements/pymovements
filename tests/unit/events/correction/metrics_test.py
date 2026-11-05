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
"""Tests for fixation correction metrics."""
import math

import polars as pl
import pytest

from pymovements.events.correction.metrics import ensemble_agreement
from pymovements.events.correction.metrics import imp_transitions
from pymovements.events.correction.metrics import mean_vertical_distance
from pymovements.events.correction.metrics import reassignment_rate


def test_reassignment_rate_compares_line_assignments():
    assert reassignment_rate(pl.Series([0, 1, 2]), pl.Series([0, 2, 2])) == pytest.approx(1 / 3)


def test_reassignment_rate_empty_inputs():
    assert reassignment_rate(pl.Series([], dtype=pl.Int64), pl.Series([], dtype=pl.Int64)) == 0.0


@pytest.mark.parametrize(
    'first, second',
    [
        (pl.Series([0, 1]), pl.Series([0])),
        (pl.Series([0, None]), pl.Series([0, 1])),
    ],
)
def test_reassignment_rate_rejects_invalid_inputs(first, second):
    with pytest.raises(ValueError):
        reassignment_rate(first, second)


def test_mean_vertical_distance_uses_physical_coordinates():
    distance = mean_vertical_distance(
        pl.Series([100.0, 200.0]),
        pl.Series([105.0, 198.0]),
    )
    assert distance == pytest.approx(3.5)


def test_mean_vertical_distance_empty_inputs():
    assert mean_vertical_distance(
        pl.Series([], dtype=pl.Float64),
        pl.Series([], dtype=pl.Float64),
    ) == 0.0


def test_mean_vertical_distance_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        mean_vertical_distance(pl.Series([100.0]), pl.Series([100.0, 200.0]))


def test_imp_transitions_uses_threshold_and_includes_backward_jumps():
    corrected_ys = pl.Series([100.0, 140.0, 240.0, 170.0])
    assert imp_transitions(corrected_ys, max_line_distance=50.0) == 2
    assert imp_transitions(corrected_ys, max_line_distance=100.0) == 0


def test_imp_transitions_empty_and_single_value_inputs():
    assert imp_transitions(pl.Series([], dtype=pl.Float64), max_line_distance=50.0) == 0
    assert imp_transitions(pl.Series([100.0]), max_line_distance=50.0) == 0


@pytest.mark.parametrize('threshold', [-1.0, math.inf, math.nan])
def test_imp_transitions_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        imp_transitions(pl.Series([100.0, 200.0]), max_line_distance=threshold)


def test_imp_transitions_rejects_nulls():
    with pytest.raises(ValueError):
        imp_transitions(pl.Series([100.0, None]), max_line_distance=50.0)


def test_ensemble_agreement_counts_votes_per_fixation():
    votes = pl.DataFrame({
        'attach': [0, 1, 2],
        'chain': [0, 2, 1],
        'cluster': [0, 2, 2],
    })
    result = ensemble_agreement(votes)
    assert result.dtype == pl.Float64
    assert result.to_list() == [1.0, 2 / 3, 2 / 3]


def test_ensemble_agreement_ties_are_not_broken_for_score():
    result = ensemble_agreement(pl.DataFrame({'first': [0], 'second': [1]}))
    assert result.to_list() == [0.5]


def test_ensemble_agreement_empty_rows_and_one_algorithm():
    empty = ensemble_agreement(pl.DataFrame(schema={'algorithm': pl.Int64}))
    assert empty.dtype == pl.Float64
    assert empty.len() == 0
    assert ensemble_agreement(pl.DataFrame({'algorithm': [1, 2]})).to_list() == [1.0, 1.0]


def test_ensemble_agreement_rejects_empty_columns_and_null_votes():
    with pytest.raises(ValueError):
        ensemble_agreement(pl.DataFrame())
    with pytest.raises(ValueError):
        ensemble_agreement(pl.DataFrame({'algorithm': [1, None]}))
