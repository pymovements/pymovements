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
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements._utils._time import durations_to_ms


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
