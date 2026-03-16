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
    ('params', 'expected'),
    [
        pytest.param(
            {
                'candidates': [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                'values': np.array(
                    [
                        (np.nan, np.nan),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (np.nan, np.nan),
                        (np.nan, np.nan),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ]
                ),
            },
            {'values_filter': [np.array([1, 2, 3]), np.array([6, 7, 8])]},
            id='test_filters',
        ),
        pytest.param(
            {
                'candidates': [[0, 1, 2, 3, 4, 5, 6, 7]],
                'values': np.array(
                    [
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (np.nan, np.nan),
                        (np.nan, np.nan),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ]
                ),
            },
            {'values_split': [np.array([0, 1, 2]), np.array([5, 6, 7])]},
            id='test_events_split',
        ),
        pytest.param(
            {'candidates': [[]], 'values': np.array([(0, 0)])},
            {'values_filter': []},
            id='test_no_candidates_in_array',
        ),
        pytest.param(
            {'candidates': [[]], 'values': np.array([(np.nan, np.nan)])},
            {'values_split': []},
            id='test_no_candidates_in_array_nan',
        ),
    ],
)
def test_filters(params, expected):
    if 'values_filter' in expected:
        results = filter_candidates_remove_nans(
            params['candidates'],
            params['values'],
        )
        assert np.all(np.array(expected['values_filter']) == np.array(results))

    if 'values_split' in expected:
        results = events_split_nans(
            params['candidates'],
            params['values'],
        )
        assert np.all(np.array(expected['values_split']) == np.array(results))
