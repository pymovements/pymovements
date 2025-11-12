# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Test pymovements.gaze.transforms.center_origin."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'origin': 'center', 'pixel_column': 'pixel', 'n_components': 2},
            TypeError,
            ('screen_resolution', 'missing'),
            id='no_screen_resolution_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'lower left', 'pixel_column': 'pixel',
                'n_components': 2,
            },
            ValueError,
            ('origin string lower left was corrected to upper left. please update your definition'),
            id='lower_left_origin_raises_value_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'foobar', 'pixel_column': 'pixel',
                'n_components': 2,
            },
            ValueError,
            ('origin', 'invalid', 'foobar', 'valid', 'center', 'upper left'),
            id='invalid_origin_raises_value_error',
        ),
    ],
)
def test_center_origin_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.center_origin(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'center', 'pixel_column': 'pixel',
                'n_components': 2,
            },
            pl.Series('pixel', [], pl.List(pl.Float64)),
            pl.Series('pixel', [], pl.List(pl.Float64)),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'center',
                'pixel_column': 'pixel', 'output_column': 'centered',
                'n_components': 2,
            },
            pl.Series('pixel', [], pl.List(pl.Float64)),
            pl.Series('centered', [], pl.List(pl.Float64)),
            id='empty_series_returns_empty_series_with_output_column',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'center', 'pixel_column': 'pixel',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, (100 - 1) / 2]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 49.5]], pl.List(pl.Float64)),
            id='origin_center',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'origin': 'upper left', 'pixel_column': 'pixel',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, (100 - 1) / 2]], pl.List(pl.Float64)),
            pl.Series('pixel', [[-49.5, 0]], pl.List(pl.Float64)),
            id='origin_upperleft',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'pixel_column': 'pixel', 'n_components': 2,
            },
            pl.Series('pixel', [[0, (100 - 1) / 2]], pl.List(pl.Float64)),
            pl.Series('pixel', [[-49.5, 0]], pl.List(pl.Float64)),
            id='origin_default',
        ),
    ],
)
def test_center_origin_returns(kwargs, series, expected_df):
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.center_origin(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame())
