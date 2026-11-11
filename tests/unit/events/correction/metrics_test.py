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
import polars as pl
import pytest

from pymovements.events.correction.metrics import ensemble_agreement
from pymovements.events.correction.metrics import implausible_transitions
from pymovements.events.correction.metrics import mean_vertical_distance
from pymovements.events.correction.metrics import reassignment_rate


def test_reassignment_rate_compares_line_assignments():
    events = pl.DataFrame({
        'name': ['fixation'] * 3,
        'location_original': [(0.0, y) for y in (0.0, 1.0, 2.0)],
        'location': [(0.0, y) for y in (0.0, 2.0, 2.0)],
    })
    aois = pl.DataFrame({
        'start_y': [0.0, 1.0, 2.0],
        'height': [1.0, 1.0, 1.0],
    })
    assert reassignment_rate(events, aois) == pytest.approx(1 / 3)


def test_reassignment_rate_empty_inputs():
    events = pl.DataFrame({'name': ['saccade']})
    assert reassignment_rate(events, pl.DataFrame()) == 0.0


@pytest.mark.parametrize(
    'events', [
        pl.DataFrame({'name': ['fixation']}),
        pl.DataFrame({
            'name': ['fixation'],
            'location_original': [[0.0, 0.0]],
            'location': [None],
        }),
    ],
)
def test_reassignment_rate_rejects_invalid_inputs(events):
    with pytest.raises(ValueError, match=r'.+'):
        reassignment_rate(events, pl.DataFrame({'start_y': [0.0], 'height': [1.0]}))


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
    with pytest.raises(ValueError, match=r'.+'):
        mean_vertical_distance(pl.Series([100.0]), pl.Series([100.0, 200.0]))


def test_implausible_transitions_uses_threshold_and_includes_backward_jumps():
    events = pl.DataFrame({
        'trial': ['trial'] * 3,
        'location': [(0.0, y) for y in (100.0, 300.0, 100.0)],
    })
    lines = {('trial',): [100.0, 200.0, 300.0]}
    assert implausible_transitions(
        events, lines, trial_columns=('trial',),
    ) == 2
    assert implausible_transitions(
        events, lines, trial_columns=('trial',), max_forward_line_distance=2,
        max_backward_line_distance=2,
    ) == 0


def test_implausible_transitions_empty_and_single_value_inputs():
    schema = {
        'trial': pl.String,
        'location': pl.List(pl.Float64),
    }
    empty = pl.DataFrame(schema=schema)
    single = pl.DataFrame({'trial': ['trial'], 'location': [[0.0, 100.0]]})
    lines = {('trial',): [100.0]}
    assert implausible_transitions(empty, lines, trial_columns=('trial',)) == 0
    assert implausible_transitions(single, lines, trial_columns=('trial',)) == 0


@pytest.mark.parametrize('forward', [-1])
def test_implausible_transitions_rejects_invalid_threshold(forward):
    events = pl.DataFrame({'trial': ['trial'], 'location': [[0.0, 100.0]]})
    with pytest.raises(ValueError, match=r'.+'):
        implausible_transitions(
            events, {('trial',): [100.0]}, trial_columns=('trial',),
            max_forward_line_distance=forward,
        )


def test_implausible_transitions_rejects_nulls():
    events = pl.DataFrame({
        'trial': ['trial', 'trial'],
        'location': [[0.0, 100.0], [0.0, None]],
    })
    with pytest.raises(ValueError, match=r'.+'):
        implausible_transitions(
            events, {('trial',): [100.0]}, trial_columns=('trial',),
        )


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
    with pytest.raises(ValueError, match=r'.+'):
        ensemble_agreement(pl.DataFrame())
    with pytest.raises(ValueError, match=r'.+'):
        ensemble_agreement(pl.DataFrame({'algorithm': [1, None]}))
