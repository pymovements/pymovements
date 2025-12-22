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
"""Test pymovements.gaze.transforms.smooth."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    'kwargs, series, expected_df',
    [
        # Method: moving_average
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 1,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            id='moving_average_window_length_1_returns_same_series',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': None,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [None, None], [1 / 2, 1 / 2],
                    [1 / 2, 1 / 2],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_2_no_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 0.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [1 / 2, 1 / 2], [1 / 2, 1 / 2]], pl.List(pl.Float64)),
            id='moving_average_window_length_2_constant_padding_0',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 1.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [[1 / 2, 1 / 2], [1 / 2, 1 / 2], [1 / 2, 1 / 2]], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_2_constant_padding_1',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [1 / 2, 1 / 2], [1 / 2, 1 / 2]], pl.List(pl.Float64)),
            id='moving_average_window_length_2_nearest_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 'mirror',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [1 / 2, 1 / 2], [1 / 2, 1 / 2],
                    [1 / 2, 1 / 2],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_2_mirror_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': None,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [None, None], [1 / 3, 1 / 3],
                    [None, None],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_3_no_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 0.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [1 / 3, 1 / 3], [1 / 3, 1 / 3],
                    [1 / 3, 1 / 3],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_3_constant_padding_0',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 1.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [[2 / 3, 2 / 3], [1 / 3, 1 / 3], [2 / 3, 2 / 3]], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_3_constant_padding_1',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [1 / 3, 1 / 3], [1 / 3, 1 / 3],
                    [1 / 3, 1 / 3],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_3_nearest_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'mirror',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [2 / 3, 2 / 3], [1 / 3, 1 / 3],
                    [2 / 3, 2 / 3],
                ], pl.List(pl.Float64),
            ),
            id='moving_average_window_length_3_mirror_padding',
        ),
        # Method: exponential_moving_average
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 1,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_1_returns_same_series',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': None,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [None, None], [2 / 3, 2 / 3],
                    [2 / 9, 2 / 9],
                ], pl.List(pl.Float64),
            ),
            id='exponential_moving_average_window_length_2_no_padding',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 0.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [2 / 3, 2 / 3], [2 / 9, 2 / 9]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_2_constant_padding_0',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 1.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [[1 / 3, 1 / 3], [7 / 9, 7 / 9], [7 / 27, 7 / 27]], pl.List(pl.Float64),
            ),
            id='exponential_moving_average_window_length_2_constant_padding_1',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [2 / 3, 2 / 3], [2 / 9, 2 / 9]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_2_nearest_padding',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 2,
                'padding': 'mirror',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [[1 / 3, 1 / 3], [7 / 9, 7 / 9], [7 / 27, 7 / 27]], pl.List(pl.Float64),
            ),
            id='exponential_moving_average_window_length_2_mirror_padding',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': None,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[None, None], [None, None], [0.25, 0.25]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_3_no_padding',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 0.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [0.5, 0.5], [0.25, 0.25]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_3_constant_padding_0',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 1.0,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0.5, 0.5], [0.75, 0.75], [0.375, 0.375]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_3_constant_padding_1',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'nearest',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [0.5, 0.5], [0.25, 0.25]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_3_nearest_padding',
        ),
        pytest.param(
            {
                'measure': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'mirror',
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [[0.25, 0.25], [0.625, 0.625], [0.3125, 0.3125]], pl.List(pl.Float64),
            ),
            id='exponential_moving_average_window_length_3_mirror_padding',
        ),
        # Method: savitzky_golay
        pytest.param(
            {
                'measure': 'savitzky_golay',
                'n_components': 2,
                'window_length': 2,
                'degree': 1,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0.5, 0.5], [0.5, 0.5], [0., 0.]], pl.List(pl.Float64)),
            id='savitzky_golay_window_length_2_degree_1_returns_mean_of_window',
        ),
        pytest.param(
            {
                'measure': 'savitzky_golay',
                'n_components': 2,
                'window_length': 3,
                'degree': 1,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series(
                'position', [
                    [1 / 3, 1 / 3], [1 / 3, 1 / 3],
                    [1 / 3, 1 / 3],
                ], pl.List(pl.Float64),
            ),
            id='savitzky_golay_window_length_3_degree_1_returns_mean_of_window',
        ),
        pytest.param(
            {
                'measure': 'savitzky_golay',
                'n_components': 2,
                'window_length': 3,
                'degree': 2,
            },
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            pl.Series('position', [[0., 0.], [1., 1.], [0., 0.]], pl.List(pl.Float64)),
            id='savitzky_golay_window_length_3_degree_2_returns',
        ),
    ],
)
def test_smooth_returns(kwargs, series, expected_df):
    """Test if smooth returns the expected dataframe."""
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.smooth(**kwargs),
    )

    assert_frame_equal(result_df, expected_df.to_frame())


@pytest.mark.parametrize(
    'kwargs, exception, msg_substrings',
    [
        pytest.param(
            {
                'measure': 'invalid_method',
                'n_components': 2,
                'window_length': 3,
            },
            ValueError,
            "Unkown measure 'invalid_method'. Supported measures are: ",
            id='invalid_method_raises_value_error',
        ),
        pytest.param(
            {
                'measure': 'savitzky_golay',
                'n_components': 2,
                'window_length': 3,
                'degree': None,
            },
            TypeError,
            "'degree' must not be none for measure 'savitzky_golay'",
            id='savitzky_golay_degree_none_raises_type_error',
        ),
    ],
)
def test_smooth_init_raises_error(kwargs, exception, msg_substrings):
    """Test if smooth init raises the expected error."""
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.smooth(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


def test_identity_returns_same_series():
    """Test if identity returns the same series."""
    series = pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64))
    series_identity = pm.gaze.transforms._identity(series)

    assert_frame_equal(series.to_frame(), series_identity.to_frame())
