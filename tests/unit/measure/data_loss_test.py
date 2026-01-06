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
"""Test data_loss sample measure."""
from __future__ import annotations

from math import inf
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    (
        'times',
        'values',
        'sampling_rate',
        'start',
        'end',
        'expected_count',
        'expected_time',
        'expected_ratio',
    ),
    [
        # 1 Hz with one internal time gap and one invalid value (NaN) -> missing=1
        # (time) + 1 (invalid)
        pytest.param(
            [0.0, 1.0, 2.0, 4.0], [1.0, 1.0, nan, 1.0], 1.0, None, None,
            2, 2.0, 2 / 5,
            id='gap_plus_nan_invalid',
        ),

        # Explicit bounds expand range (2 expected missing) and one +inf value -> total 3
        pytest.param(
            [10.0, 11.0, 12.0], [1.0, inf, 1.0], 1.0, 9.0, 13.0,
            3, 3.0, 3 / 5,
            id='explicit_bounds_with_inf',
        ),

        # Irregular times, 1/3 Hz expectation over [0,12], plus one None value
        pytest.param(
            [0.0, 2.0, 6.0], [1.0, None, 1.0], 1 / 3, None, 12.0,
            3, 9.0, 3 / 5,
            id='irregular_with_none',
        ),

        # Single sample - include invalid value to ensure invalid path is counted when expected>=1
        pytest.param(
            [5.0], [None], 1.0, None, None, 1, 1.0, 1.0,
            id='single_sample_invalid_only',
        ),

        # Degenerate interval (end==start) -> no time-based missing - with all valid values -> 0
        pytest.param(
            [1.0, 2.0], [1.0, 1.0], 1.0, 5.0, 5.0, 0, 0.0, 0.0,
            id='degenerate_interval_zero',
        ),

        # Clip negative: observed > expected due to narrow end - invalid present but should still be
        # clipped by expected? No: invalid rows are counted regardless - however our function
        # sums both. To keep observed>expected scenario only, use no invalids -> 0.
        pytest.param(
            [0.0, 1.0, 2.0], [1.0, 1.0, 1.0], 1.0, 0.0, 1.5, 0, 0.0, 0.0,
            id='clip_negative_time_only',
        ),

        # Unsorted input with an invalid (-inf) value
        pytest.param(
            [3.0, 1.0, 0.0, 4.0], [1.0, 1.0, -inf, 1.0], 1.0, None, None,
            2, 2.0, 2 / 5,
            id='unsorted_with_neginf',
        ),

        # Larger internal gap and one invalid in values
        pytest.param(
            [0.0, 1.0, 2.0, 6.0, 7.0], [1.0, None, 1.0, 1.0, 1.0], 1.0, None, None,
            4, 4.0, 4 / 8,
            id='large_gap_with_invalid',
        ),

        # Perfect regular sampling at 2 Hz with all valid -> 0
        pytest.param(
            [0.0, 0.5, 1.0, 1.5], [1.0, 1.0, 1.0, 1.0], 2.0, None, None,
            0, 0.0, 0.0,
            id='no_missing_regular_all_valid',
        ),

        # List at 1 Hz with invalid values and missing samples
        pytest.param(
            [1, 2, 3, 4, 5, 9], [[1, 1], [1, 1], None, None, [1, 1], [1, None]],
            1.0, None, None,
            6, 6.0, 6 / 9,
            id='list_with_invalid_and_missing',
        ),
    ],
)
@pytest.mark.parametrize('unit', ['count', 'time', 'ratio'])
def test_data_loss(
        times, values, sampling_rate, start, end, expected_count, expected_time,
        expected_ratio, unit,
):
    df = pl.from_dict(
        data={'time': times, 'value': values},
    )

    expr = pm.measure.data_loss(
        'time', 'value', sampling_rate=sampling_rate, start_time=start, end_time=end, unit=unit,
    )
    result = df.select(expr)

    expected_column_by_unit = {
        'count': 'data_loss_count',
        'time': 'data_loss_time',
        'ratio': 'data_loss_ratio',
    }
    expected_value_by_unit = {
        'count': expected_count,
        'time': expected_time,
        'ratio': expected_ratio,
    }
    expected = pl.from_dict(
        data={expected_column_by_unit[unit]: [expected_value_by_unit[unit]]},
    )

    assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize('bad_unit', ['invalid', '', None, 'COUNT'])
def test_data_loss_invalid_unit_raises(bad_unit):
    df = pl.DataFrame({'time': [0.0, 1.0], 'value': [1.0, 1.0]})
    # We purposely pass an invalid unit to exercise the error branch
    with pytest.raises(ValueError) as excinfo:
        df.select(pm.measure.data_loss('time', 'value', sampling_rate=1.0, unit=bad_unit))

    (message,) = excinfo.value.args
    assert message == "unit must be one of {'count', 'time', 'ratio'} but got: " + repr(
        bad_unit,
    )


@pytest.mark.parametrize('bad_time_column', [123, None, {'col': 'time'}])
def test_data_loss_invalid_time_column_raises(bad_time_column):
    df = pl.DataFrame({'time': [0.0, 1.0], 'value': [1.0, 1.0]})
    with pytest.raises(TypeError) as excinfo:
        df.select(pm.measure.data_loss(bad_time_column, 'value', sampling_rate=1.0))

    (message,) = excinfo.value.args
    assert message == (
        f"invalid type for 'time_column'. Expected 'str' , got "
        f"'{type(bad_time_column).__name__}'"
    )


@pytest.mark.parametrize('bad_sampling_rate', [0, -1, 0.0, -10.5, '1Hz'])
def test_data_loss_invalid_sampling_rate_raises(bad_sampling_rate):
    df = pl.DataFrame({'time': [0.0, 1.0], 'value': [1.0, 1.0]})
    with pytest.raises(ValueError) as excinfo:
        df.select(pm.measure.data_loss('time', 'value', sampling_rate=bad_sampling_rate))

    (message,) = excinfo.value.args
    assert message == (
        f'sampling_rate must be a positive number, but got: {repr(bad_sampling_rate)}'
    )
