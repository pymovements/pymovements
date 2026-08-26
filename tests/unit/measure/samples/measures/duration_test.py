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
"""Test duration sample measure."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.samples import duration


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            {},
            pl.DataFrame({'time': [1000, 1001, 1004]}),
            pl.DataFrame({'duration': [4]}),
            id='integer_timestamps',
        ),
        pytest.param(
            {},
            pl.DataFrame({'time': [10.5, 4.0, 8.0]}),
            pl.DataFrame({'duration': [6.5]}),
            id='float_unsorted_timestamps',
        ),
        pytest.param(
            {},
            pl.DataFrame({'time': [42]}),
            pl.DataFrame({'duration': [0]}),
            id='single_sample',
        ),
        pytest.param(
            {},
            pl.DataFrame({
                'time': pl.Series([1000, 1250, 1750], dtype=pl.Duration('us')),
            }),
            pl.DataFrame({
                'duration': pl.Series([750], dtype=pl.Duration('us')),
            }),
            id='duration_timestamps',
        ),
        pytest.param(
            {'time_column': 'timestamp'},
            pl.DataFrame({'timestamp': [5, 9]}),
            pl.DataFrame({'duration': [4]}),
            id='custom_time_column',
        ),
    ],
)
def test_duration_has_expected_result(init_kwargs, input_df, expected_df):
    result_df = input_df.select(duration(**init_kwargs))

    assert_frame_equal(result_df, expected_df)


def test_duration_raises_for_missing_time_column():
    expression = duration(time_column='missing')

    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='missing'):
        pl.DataFrame({'time': [0, 1]}).select(expression)
