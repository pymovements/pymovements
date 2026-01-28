# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Basic self-test for shared TextStimulus fixtures."""
from __future__ import annotations

from collections.abc import Sized

import polars
import pytest

from pymovements.stimulus.text import TextStimulus


@pytest.mark.parametrize(
    'fixture_name',
    [
        pytest.param('stimulus_both_columns', id='both'),
        pytest.param('stimulus_only_trial', id='trial'),
        pytest.param('stimulus_only_page', id='page'),
        pytest.param('simple_stimulus', id='simple'),
        pytest.param('stimulus_with_trial_page', id='trial_page'),
        pytest.param('stimulus_overlap', id='overlap'),
    ],
)
def test_fixtures_provide_textstimulus(request: pytest.FixtureRequest, fixture_name: str) -> None:
    stim = request.getfixturevalue(fixture_name)
    assert isinstance(stim, TextStimulus)
    # AOIs must contain the configured coordinate columns
    cols = set(stim.aois.columns)
    assert stim.start_x_column in cols
    assert stim.start_y_column in cols
    assert (stim.end_x_column in cols) or (stim.width_column in cols)


@pytest.mark.parametrize(
    ('fixture_name', 'row', 'expect', 'expect_multiple'),
    [
        ('stimulus_both_columns', {'x': 5, 'y': 5, 'trial': 1, 'page': 'X'}, 'A1', False),
        ('stimulus_both_columns', {'x': 5, 'y': 5, 'trial': 2, 'page': 'X'}, 'A2', False),
        ('stimulus_both_columns', {'x': 5, 'y': 5, 'trial': 2, 'page': 'Y'}, None, False),
        ('stimulus_both_columns', {'x': 15, 'y': 5, 'trial': 1, 'page': 'Y'}, 'B1', False),
        ('stimulus_both_columns', {'x': 15, 'y': 5, 'trial': 2, 'page': 'Y'}, None, False),
        ('stimulus_both_columns', {'x': 5, 'y': 5, 'trial': 1, 'page': 'Z'}, None, False),
        ('stimulus_only_trial', {'x': 5, 'y': 5, 'trial': 1}, 'T1', False),
        ('stimulus_only_trial', {'x': 5, 'y': 5, 'trial': 3}, None, False),
        ('stimulus_only_page', {'x': 5, 'y': 5, 'page': 'X'}, 'PX', False),
        ('stimulus_only_page', {'x': 5, 'y': 5, 'page': 'Z'}, None, False),
        ('stimulus_with_trials', {'x': 5, 'y': 5, 'trial': 1}, 'A', False),
        ('stimulus_with_trials', {'x': 5, 'y': 5, 'trial': 2}, None, False),
        ('stimulus_with_trials', {'x': 25, 'y': 25, 'trial': 1}, None, False),
        ('stimulus_with_trials', {'x': 25, 'y': 25, 'trial': 2}, 'B', False),
        ('stimulus_with_trials', {'x': 25, 'y': 25, 'trial': 2}, 'B', False),
        ('stimulus_with_trials', {'x': -19, 'y': -19, 'trial': 1}, ['C', 'D'], True),
        ('stimulus_with_trials', {'x': -17, 'y': -17, 'trial': 1}, 'D', False),
    ],
)
def test_fixtures_basic_get_aoi(
        request: pytest.FixtureRequest, fixture_name: str, row: dict,
        expect: Sized, expect_multiple: bool,
) -> None:
    stim: TextStimulus = request.getfixturevalue(fixture_name)
    if expect_multiple:
        with pytest.warns(UserWarning, match='Multiple AOIs matched this point'):
            aoi = stim.get_aoi(row=row, x_eye='x', y_eye='y')
        assert aoi.shape[0] == len(expect)
        labels = polars.Series(aoi.select(stim.aoi_column)).to_list()
        assert labels == expect
    else:
        aoi = stim.get_aoi(row=row, x_eye='x', y_eye='y')
        assert aoi.shape[0] == 1
        label = aoi.select(stim.aoi_column).item()
        assert label == expect


@pytest.mark.parametrize(
    ('x', 'y', 'expected_label'),
    [
        pytest.param(5, 5, 'A', id='inside'),
        pytest.param(10, 5, None, id='right-boundary-exclusive'),
        pytest.param(-1, 0, None, id='outside-left'),
    ],
)
def test_simple_stimulus_get_aoi(
    simple_stimulus: TextStimulus,
    x: int, y: int, expected_label: str | None,
) -> None:
    aoi = simple_stimulus.get_aoi(row={'x': x, 'y': y}, x_eye='x', y_eye='y')
    assert aoi.shape[0] == 1
    assert aoi.select(simple_stimulus.aoi_column).item() == expected_label


@pytest.mark.parametrize(
    ('trial', 'page', 'expected_label'),
    [
        pytest.param(1, 'X', 'TX', id='trial1-pageX'),
        pytest.param(1, 'Y', 'TY', id='trial1-pageY'),
        pytest.param(2, 'X', None, id='trial-miss'),
        pytest.param(1, 'Z', None, id='page-miss'),
    ],
)
def test_stimulus_with_trial_page_selection(
    stimulus_with_trial_page: TextStimulus, trial: int, page: str, expected_label: str | None,
) -> None:
    aoi = stimulus_with_trial_page.get_aoi(
        row={
            'x': 5,
            'y': 5,
            'trial': trial,
            'page': page,
        },
        x_eye='x',
        y_eye='y',
    )
    assert aoi.shape[0] == 1
    assert aoi.select(stimulus_with_trial_page.aoi_column).item() == expected_label


@pytest.mark.parametrize(
    ('row', 'expected_label', 'expect_len'),
    [
        pytest.param({'x': 5, 'y': 5}, 'L1', 2, id='overlap-warns-first-selected'),
        pytest.param({'x': 15, 'y': 5}, None, 1, id='outside-no-warn'),
    ],
)
def test_stimulus_overlap_behaviour(
    stimulus_overlap: TextStimulus,
        row: dict[str, int], expected_label: str | None, expect_len: int,
) -> None:
    if expect_len > 1:
        with pytest.warns(UserWarning, match='Multiple AOIs matched this point'):
            aoi = stimulus_overlap.get_aoi(row=row, x_eye='x', y_eye='y')
    else:
        aoi = stimulus_overlap.get_aoi(row=row, x_eye='x', y_eye='y')
    labels = aoi.get_column('label').to_list()
    assert len(labels) == expect_len
    assert labels[0] == expected_label
