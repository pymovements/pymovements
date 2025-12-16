# Copyright (c) 2025 The pymovements Project Authors
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

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    (
        'times',
        'start',
        'end',
        'expected_count',
        'expected_time',
        'expected_ratio',
    ),
    [
        # Regular sampling, one internal gap: [0,1,2,4]
        # span=4, median ISI=1 -> expected=5, observed=4 -> missing=1
        pytest.param(
            [0.0, 1.0, 2.0, 4.0], None, None, 1, 1.0, 1 / 5,
            id='regular_internal_gap',
        ),

        # Explicit recording start/end expanding range by 2 samples
        # times=[10,11,12], start=9, end=13: span=4, ISI=1 -> expected=5, observed=3 -> missing=2
        pytest.param(
            [10.0, 11.0, 12.0], 9.0, 13.0, 2, 2.0, 2 / 5,
            id='explicit_bounds',
        ),

        # Irregular sampling, median ISI computed from [0,2,6] -> diffs [2,4] -> median=3
        # end=12 -> span=12 -> expected=floor(12/3)+1=5, observed=3 -> missing=2
        pytest.param(
            [0.0, 2.0, 6.0], None, 12.0, 2, 6.0, 2 / 5,
            id='irregular_sampling',
        ),

        # Single sample -> median ISI null, returns 0
        pytest.param([1.0], None, None, 0, 0.0, 0.0, id='single_sample'),

        # Degenerate interval (end == start) -> returns 0
        pytest.param([1.0, 2.0], 5.0, 5.0, 0, 0.0, 0.0, id='degenerate_interval'),

        # Clip negative missing to zero: observed > expected due to narrow end
        # times=[0,1,2], start=0, end=1.5 -> ISI=1, expected=2, observed=3 -> missing -> 0
        pytest.param([0.0, 1.0, 2.0], 0.0, 1.5, 0, 0.0, 0.0, id='clip_negative'),

        # Perfect regular sampling: no missing
        # times=[0,1,2,3], span=3, ISI=1 -> expected=4, observed=4 -> 0
        pytest.param(
            [0.0, 1.0, 2.0, 3.0], None, None, 0, 0.0, 0.0,
            id='no_missing_regular',
        ),

        # Non-integer ISI (0.5): end extends range -> missing occurs
        # times=[0,0.5,1.0], end=2.0, ISI=0.5, span=2 -> expected=5, observed=3 -> 2
        pytest.param(
            [0.0, 0.5, 1.0], None, 2.0, 2, 1.0, 2 / 5,
            id='half_second_isi_extend_end',
        ),

        # Start earlier than first timestamp adds expected samples at the beginning
        # times=[5,6,7], start=3, end=None -> span= (7-3)=4, ISI=1 -> expected=5, observed=3 -> 2
        pytest.param(
            [5.0, 6.0, 7.0], 3.0, None, 2, 2.0, 2 / 5,
            id='start_before_first',
        ),

        # End before last timestamp reduces expected below observed -> clipped to 0
        # times=[0,1,2,3], end=1.1 -> expected=floor((1.1-0)/1)+1=2, observed=4 -> 0
        pytest.param(
            [0.0, 1.0, 2.0, 3.0], None, 1.1, 0, 0.0, 0.0,
            id='end_before_last_clip',
        ),

        # Unsorted timestamps: sorting inside measure should handle it (one missing between 1 and 3)
        # times=[3,1,0,4] -> sorted [0,1,3,4], ISI median=1 -> expected=5, observed=4 -> 1
        pytest.param(
            [3.0, 1.0, 0.0, 4.0], None, None, 1, 1.0, 1 / 5,
            id='unsorted_input',
        ),

        # Larger gap inside
        # times=[0,1,2,6,7], ISI median=1, span=7 -> expected=8, observed=5 -> missing=3
        pytest.param(
            [0.0, 1.0, 2.0, 6.0, 7.0], None, None, 3, 3.0, 3 / 8,
            id='large_internal_gap',
        ),

        # Very short span with two samples -> expected=2, observed=2 -> 0
        pytest.param(
            [10.0, 10.1], None, None, 0, 0.0, 0.0,
            id='short_span_two_samples_no_missing',
        ),
    ],
)
@pytest.mark.parametrize('unit', ['count', 'time', 'ratio'])
def test_data_loss(
        times, start, end, expected_count, expected_time,
        expected_ratio, unit,
):
    df = pl.from_dict(data={'time': times}, schema={'time': pl.Float64})

    expr = pm.measure.data_loss('time', start_time=start, end_time=end, unit=unit)
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

    # Compare with tolerance to accommodate floating point representation for time/ratio
    assert_frame_equal(
        result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12,
    )


@pytest.mark.parametrize('bad_unit', ['invalid', '', None, 'COUNT'])
def test_data_loss_invalid_unit_raises(bad_unit):
    df = pl.DataFrame({'time': [0.0, 1.0]})
    # We purposely pass an invalid unit to exercise the error branch
    with pytest.raises(ValueError) as excinfo:
        df.select(pm.measure.data_loss('time', unit=bad_unit))

    (message,) = excinfo.value.args
    assert message == "unit must be one of {'count', 'time', 'ratio'} but got: " + repr(
        bad_unit,
    )


@pytest.mark.parametrize('bad_timestamp_col', [123, None, ['time'], {'col': 'time'}])
def test_data_loss_invalid_timestamp_col_raises(bad_timestamp_col):
    df = pl.DataFrame({'time': [0.0, 1.0]})
    with pytest.raises(TypeError) as excinfo:
        df.select(pm.measure.data_loss(bad_timestamp_col))

    (message,) = excinfo.value.args
    assert message == (f'timestamp_col must be a string, but got: '
                       f'{type(bad_timestamp_col).__name__}')
