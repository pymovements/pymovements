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
"""Test pymovements.gaze.transforms.downsample."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('factor', 'exception', 'msg_substrings'),
    [
        pytest.param(
            -1,
            ValueError,
            ('factor', 'must not', 'negative', '-1'),
            id='negative_factor_raises_value_error',
        ),
        pytest.param(
            1.5,
            TypeError,
            ('factor', 'must', 'type', 'int', 'but', 'float'),
            id='float_factor_raises_value_error',
        ),
    ],
)
def test_downsample_init_raises_error(factor, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.downsample(factor=factor)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('factor', 'series', 'expected_df'),
    [
        pytest.param(
            1,
            pl.Series('pixel', [], pl.Float64),
            pl.Series('pixel', [], pl.Float64),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            1,
            pl.Series('pixel', [1, 2, 3], pl.Int64),
            pl.Series('pixel', [1, 2, 3], pl.Int64),
            id='factor_1_returns_same_series',
        ),
        pytest.param(
            2,
            pl.Series('pixel', [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], pl.Int64),
            pl.Series('pixel', [0, 1, 2, 3, 4, 5], pl.Int64),
            id='factor_2_returns_every_second_item',
        ),
        pytest.param(
            3,
            pl.Series('pixel', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], pl.Int64),
            pl.Series('pixel', [0, 1, 2, 3, 4, 5], pl.Int64),
            id='factor_3_returns_every_third_item',
        ),
    ],
)
def test_downsample_returns(factor, series, expected_df):
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.downsample(factor=factor),
    )
    assert_frame_equal(result_df, expected_df.to_frame())
