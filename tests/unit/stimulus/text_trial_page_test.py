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
"""Tests for TextStimulus.get_aoi with trial/page filtering and boundaries.

These tests specifically cover the trial/page filtering that was buggy before and
regression cases around exclusive end boundaries and no-match behavior.
"""
from __future__ import annotations

import pytest

from pymovements.stimulus.text import TextStimulus


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        pytest.param({'x': 5, 'y': 5, 'trial': 1, 'page': 'X'}, 'A1', id='match-trial1-pageX'),
        pytest.param({'x': 5, 'y': 5, 'trial': 2, 'page': 'X'}, 'A2', id='match-trial2-pageX'),
        pytest.param({'x': 5, 'y': 5, 'trial': 1, 'page': 'Y'}, None, id='no-match-trial1-pageY'),
        pytest.param({'x': 15, 'y': 5, 'trial': 1, 'page': 'Y'}, 'B1', id='match-other-aoi'),
    ],
)
def test_get_aoi_filters_by_trial_and_page(
        stimulus_both_columns: TextStimulus, row: dict, expected: str | None,
) -> None:
    aoi = stimulus_both_columns.get_aoi(row=row, x_eye='x', y_eye='y')
    assert aoi.shape[0] == 1  # always one row (possibly filled with Nones)
    assert aoi.select('label').item() == expected


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        pytest.param({'x': 5, 'y': 5, 'trial': 1}, 'T1', id='trial-1'),
        pytest.param({'x': 5, 'y': 5, 'trial': 2}, 'T2', id='trial-2'),
        pytest.param({'x': 5, 'y': 5, 'trial': 3}, None, id='trial-no-match'),
    ],
)
def test_get_aoi_filters_by_trial_only(
        stimulus_only_trial: TextStimulus, row: dict, expected: str | None,
) -> None:
    aoi = stimulus_only_trial.get_aoi(row=row, x_eye='x', y_eye='y')
    assert aoi.shape[0] == 1
    assert aoi.select('label').item() == expected


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        pytest.param({'x': 5, 'y': 5, 'page': 'X'}, 'PX', id='page-X'),
        pytest.param({'x': 5, 'y': 5, 'page': 'Y'}, 'PY', id='page-Y'),
        pytest.param({'x': 5, 'y': 5, 'page': 'Z'}, None, id='page-no-match'),
    ],
)
def test_get_aoi_filters_by_page_only(
        stimulus_only_page: TextStimulus, row: dict, expected: str | None,
) -> None:
    aoi = stimulus_only_page.get_aoi(row=row, x_eye='x', y_eye='y')
    assert aoi.shape[0] == 1
    assert aoi.select('label').item() == expected


@pytest.mark.parametrize(
    'x',
    [
        pytest.param(10, id='end-exclusive-x'),
    ],
)
def test_get_aoi_end_is_exclusive(stimulus_only_trial: TextStimulus, x: int) -> None:
    # For trial=1 AOI is [0,10) x [0,10) - x==10 should be outside
    row = {'x': x, 'y': 5, 'trial': 1}
    aoi = stimulus_only_trial.get_aoi(row=row, x_eye='x', y_eye='y')
    assert aoi.shape[0] == 1
    # When outside, label should be None
    assert aoi.select('label').item() is None
