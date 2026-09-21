# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test pymovements filters."""
from __future__ import annotations

import numpy as np
import pytest

from pymovements.events._utils._filters import events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans


@pytest.mark.parametrize(
    ('candidates', 'values', 'expected'),
    [
        pytest.param(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            np.array([
                (np.nan, np.nan), (0, 0),
                (0, 0), (0, 0),
                (np.nan, np.nan),
                (np.nan, np.nan),
                (0, 0), (0, 0),
                (0, 0),
            ]),
            [np.array([1, 2, 3]), np.array([6, 7, 8])],
            id='leading_and_trailing_nans',
        ),
        pytest.param(
            [[]],
            np.array([(0, 0)]),
            [],
            id='no_candidates_in_array',
        ),
        pytest.param(
            [[0, 1, 2, 3, 4, 5, 6]],   # 7-sample window, all NaN
            np.array([
                (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan),
                (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan),
                (np.nan, np.nan),
            ]),
            [],   # all-NaN candidate must be dropped, not crash
            id='all_nan_candidate_skipped',
        ),
        pytest.param(
            [[0, 1, 2]],
            np.array([
                (np.nan, np.nan), (1.0, 2.0), (np.nan, np.nan),
            ]),
            [np.array([1])],   # single valid sample surrounded by NaNs
            id='single_valid_sample_candidate',
        ),
    ],
)
def test_filter_candidates_remove_nans(candidates, values, expected):
    results = filter_candidates_remove_nans(candidates, values)

    assert len(results) == len(expected)
    for result, expected_candidate in zip(results, expected):
        assert np.array_equal(result, expected_candidate)


@pytest.mark.parametrize(
    ('candidates', 'values', 'expected'),
    [
        pytest.param(
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            np.array([
                (0, 0),
                (0, 0), (0, 0),
                (np.nan, np.nan),
                (np.nan, np.nan),
                (0, 0), (0, 0),
                (0, 0),
            ]),
            [np.array([0, 1, 2]), np.array([5, 6, 7])],
            id='nans_in_middle',
        ),
        pytest.param(
            [[]],
            np.array([(np.nan, np.nan)]),
            [],
            id='no_candidates_in_array_nan',
        ),
        pytest.param(
            [[0, 1, 2]],
            np.array([
                (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan),
            ]),
            [],   # all-NaN candidate must be dropped, not crash
            id='all_nan_candidate_skipped',
        ),
    ],
)
def test_events_split_nans(candidates, values, expected):
    results = events_split_nans(candidates, values)

    assert len(results) == len(expected)
    for result, expected_candidate in zip(results, expected):
        assert np.array_equal(result, expected_candidate)
