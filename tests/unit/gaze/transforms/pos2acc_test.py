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
"""Test pymovements.gaze.transforms.pos2acc."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'window_length': 1, 'degree': 0, 'sampling_rate': 1, 'n_components': 2},
            ValueError,
            ('degree', 'must', 'greater than zero'),
            id='degree_zero_raises_type_error',
        ),
        pytest.param(
            {'window_length': 3, 'degree': 1.0, 'sampling_rate': 1, 'n_components': 2},
            TypeError,
            ('degree', "must be of type 'int'", "is of type 'float'"),
            id='degree_float_raises_type_error',
        ),
        pytest.param(
            {'window_length': 1, 'degree': 1, 'sampling_rate': 1, 'n_components': 2},
            ValueError,
            ("'degree' must be less than 'window_length'"),
            id='degree_equal_window_size_raises_value_error',
        ),
        pytest.param(
            {'window_length': 1, 'degree': 2, 'sampling_rate': 1, 'n_components': 2},
            ValueError,
            ("'degree' must be less than 'window_length'"),
            id='degree_greater_than_window_size_raises_value_error',
        ),
        pytest.param(
            {'window_length': 3, 'degree': 1, 'padding': [], 'sampling_rate': 1, 'n_components': 2},
            TypeError,
            (
                "'padding' must be of type 'str', 'int', 'float' or None",
                "but is of type 'list'",
            ),
            id='invalid_padding_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'foobar', 'sampling_rate': 1,
                'n_components': 2,
            },
            ValueError,
            (
                'padding', 'invalid', 'foobar',
                'valid', 'mirror', 'nearest', 'wrap', 'None', 'scalar',
            ),
            id='invalid_padding_raises_value_error',
        ),
    ],
)
def test_pos2acc_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.pos2acc(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': None, 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            ValueError,
            ('If mode is \'interp\', window_length must be less than or equal to the size of x',),
            id='no_padding_input_shorter_than_window_length_raises_valueerror',
        ),
    ],
)
def test_pos2acc_raises_error(kwargs, series, exception, msg_substrings):
    df = series.to_frame()
    expression = pm.gaze.transforms.pos2acc(**kwargs)

    with pytest.raises(exception) as excinfo:
        df.select(expression)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {'window_length': 3, 'degree': 1, 'sampling_rate': 1, 'n_components': 2},
            pl.Series('position', [], pl.List(pl.Float64)),
            pl.Series('acceleration', [], pl.List(pl.Float64)),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            {'window_length': 3, 'degree': 1, 'sampling_rate': 1, 'n_components': 2},
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0]], pl.List(pl.Float64)),
            id='single_element_results_zero',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'mirror', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0]], pl.List(pl.Float64)),
            id='single_element_results_zero_mirror_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'nearest', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0]], pl.List(pl.Float64)),
            id='single_element_results_zero_wrap_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'wrap', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0]], pl.List(pl.Float64)),
            id='single_element_results_zero_wrap_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 1, 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0]], pl.List(pl.Float64)),
            id='single_element_results_zero_equal_scalar_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'nearest', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1], [1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0], [0, 0]], pl.List(pl.Float64)),
            id='two_equal_elements_results_zero_nearest_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'nearest', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1], [1, 1]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0], [0, 0]], pl.List(pl.Float64)),
            id='two_equal_elements_differentation_nearest_padding_result_zero',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'padding': 'nearest', 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1], [2, 2]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[0, 0], [0, 0]], pl.List(pl.Float64)),
            id='two_elements_1_2_double_differentation_nearest_padding_result_zero',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 2, 'padding': None, 'sampling_rate': 1,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1], [4, 4], [9, 9]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[2, 2], [2, 2], [2, 2]], pl.List(pl.Float64)),
            id='three_elements_1_4_9_double_differentation_none_padding_result_two',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 2, 'padding': None, 'sampling_rate': 10,
                'n_components': 2,
            },
            pl.Series('position', [[1, 1], [4, 4], [9, 9]], pl.List(pl.Float64)),
            pl.Series('acceleration', [[200, 200], [200, 200], [200, 200]], pl.List(pl.Float64)),
            id='three_elements_1_4_9_double_differentation_sampling_rate_10_no_padding_result_200',
        ),
    ],
)
def test_pos2acc_returns(kwargs, series, expected_df):
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.pos2acc(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame())
