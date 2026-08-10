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
"""Test pymovements _utils._time."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements._utils._time import durations_to_ms
from pymovements._utils._time import numeric_to_duration_us
from pymovements._utils._time import timesteps_to_numpy


@pytest.mark.parametrize(
    ('frame', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'onset': pl.Series([1000, 2000], dtype=pl.Duration('us'))}),
            pl.DataFrame({'onset': pl.Series([1, 2], dtype=pl.Int64)}),
            id='whole_milliseconds_narrowed_to_int64',
        ),
        pytest.param(
            pl.DataFrame({'onset': pl.Series([1500, 2000], dtype=pl.Duration('us'))}),
            pl.DataFrame({'onset': pl.Series([1.5, 2.0], dtype=pl.Float64)}),
            id='fractional_milliseconds_kept_as_float64',
        ),
        pytest.param(
            pl.DataFrame({'onset': pl.Series([-2500, 1000], dtype=pl.Duration('us'))}),
            pl.DataFrame({'onset': pl.Series([-2.5, 1.0], dtype=pl.Float64)}),
            id='negative_fractional_milliseconds',
        ),
        pytest.param(
            pl.DataFrame({'onset': pl.Series([1000, None], dtype=pl.Duration('us'))}),
            pl.DataFrame({'onset': pl.Series([1, None], dtype=pl.Int64)}),
            id='null_values_kept_in_whole_millisecond_column',
        ),
        pytest.param(
            pl.DataFrame({'onset': pl.Series([None, None], dtype=pl.Duration('us'))}),
            pl.DataFrame({'onset': pl.Series([None, None], dtype=pl.Int64)}),
            id='all_null_column_narrowed_to_int64',
        ),
        pytest.param(
            pl.DataFrame({'onset': pl.Series([5, 10], dtype=pl.Duration('ms'))}),
            pl.DataFrame({'onset': pl.Series([5, 10], dtype=pl.Int64)}),
            id='millisecond_unit_duration_column',
        ),
        pytest.param(
            pl.DataFrame({
                'time': pl.Series([500, 1000], dtype=pl.Duration('us')),
                'x': [0.1, 0.2],
            }),
            pl.DataFrame({
                'time': pl.Series([0.5, 1.0], dtype=pl.Float64),
                'x': [0.1, 0.2],
            }),
            id='non_duration_columns_untouched',
        ),
    ],
)
def test_durations_to_ms_converts_duration_columns(frame, expected):
    assert_frame_equal(durations_to_ms(frame), expected)


def test_durations_to_ms_returns_frame_without_duration_columns_unchanged():
    frame = pl.DataFrame({'time': [1, 2, 3], 'x': [0.1, 0.2, 0.3]})
    assert durations_to_ms(frame) is frame


@pytest.mark.parametrize(
    ('series', 'expected'),
    [
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Int64), [0.0, 1.0, 2.0],
            id='int_series_unchanged',
        ),
        pytest.param(
            pl.Series([0.0, 0.5, 1.0], dtype=pl.Float64), [0.0, 0.5, 1.0],
            id='float_series_unchanged',
        ),
        pytest.param(
            pl.Series([0, 1000, 2000], dtype=pl.Duration('us')), [0.0, 1.0, 2.0],
            id='duration_us_to_fractional_ms',
        ),
        pytest.param(
            pl.Series([0, 500, 1500], dtype=pl.Duration('us')), [0.0, 0.5, 1.5],
            id='duration_us_sub_millisecond',
        ),
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Duration('ms')), [0.0, 1.0, 2.0],
            id='duration_ms_to_ms',
        ),
    ],
)
def test_timesteps_to_numpy_returns_numeric_milliseconds(series, expected):
    result = timesteps_to_numpy(series)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == expected


def test_timesteps_to_numpy_raises_on_non_numeric_dtype():
    with pytest.raises(TypeError, match='timesteps dtype must be float or int'):
        timesteps_to_numpy(pl.Series(['a', 'b']))


@pytest.mark.parametrize(
    ('values', 'time_unit', 'expected_us'),
    [
        pytest.param([5, 10], 's', [5_000_000, 10_000_000], id='seconds'),
        pytest.param([5, 10], 'ms', [5_000, 10_000], id='milliseconds'),
        pytest.param([5, 10], 'us', [5, 10], id='microseconds'),
        pytest.param([0.5, 1.25], 'ms', [500, 1250], id='fractional_milliseconds'),
        pytest.param([1.5, 2.6], 'us', [2, 3], id='sub_microsecond_rounded'),
    ],
)
def test_numeric_to_duration_us_converts_by_unit(values, time_unit, expected_us):
    result = pl.DataFrame({'time': values}).with_columns(
        numeric_to_duration_us('time', time_unit),
    )
    assert result.schema['time'] == pl.Duration('us')
    assert result['time'].dt.total_microseconds().to_list() == expected_us


def test_numeric_to_duration_us_accepts_expression():
    result = pl.DataFrame({'time': [5]}).with_columns(
        numeric_to_duration_us(pl.col('time'), 'ms'),
    )
    assert result['time'].dt.total_microseconds().to_list() == [5_000]


@pytest.mark.parametrize('time_unit', ['step', 'sec', 'minutes', ''])
def test_numeric_to_duration_us_raises_on_unsupported_unit(time_unit):
    with pytest.raises(ValueError, match='unsupported time unit'):
        numeric_to_duration_us('time', time_unit)
