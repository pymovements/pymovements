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
"""Test pymovements.gaze.transforms.savitzky_golay."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'window_length': 1, 'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            TypeError,
            ('degree', 'missing'),
            id='no_degree_raises_type_error',
        ),
        pytest.param(
            {
                'degree': 1, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            TypeError,
            ('window_length', 'missing'),
            id='no_window_length_raises_type_error',
        ),
        pytest.param(
            {
                'window_length': 1, 'degree': 0, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            ValueError,
            ('degree', 'must', 'greater than zero'),
            id='degree_zero_raises_type_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1.0, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            TypeError,
            ('degree', "must be of type 'int'", "is of type 'float'"),
            id='degree_float_raises_type_error',
        ),
        pytest.param(
            {
                'window_length': 1, 'degree': 1, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            ValueError,
            ("'degree' must be less than 'window_length'"),
            id='degree_equal_window_size_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 1, 'degree': 2, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            ValueError,
            ("'degree' must be less than 'window_length'"),
            id='degree_greater_than_window_size_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': -1, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            ValueError,
            ("'derivative' must not be negative", '-1'),
            id='derivative_negative_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 1.0, 'sampling_rate': 1,
                'n_components': 2, 'input_column': 'A',
            },
            TypeError,
            ("'derivative' must be of type 'int'", "is of type 'float'"),
            id='derivative_float_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': [],
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            TypeError,
            (
                "'padding' must be of type 'str', 'int', 'float' or None",
                "but is of type 'list'",
            ),
            id='invalid_padding_raises_value_error',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': 'foobar',
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
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
def test_savitzky_golay_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.savitzky_golay(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': None,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            ValueError,
            ('If mode is \'interp\', window_length must be less than or equal to the size of x',),
            id='no_padding_input_shorter_than_window_length_raises_valueerror',
        ),
    ],
)
def test_savitzky_golay_raises_error(kwargs, series, exception, msg_substrings):
    df = series.to_frame()
    expression = pm.gaze.transforms.savitzky_golay(**kwargs)

    with pytest.raises(exception) as excinfo:
        df.select(expression)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {
                'window_length': 3, 'degree': 1,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [], pl.List(pl.Float64)),
            pl.Series('A', [], pl.List(pl.Float64)),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            id='single_element_stays_the_same',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': 'mirror',
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            id='single_element_stays_the_same_mirror_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': 'nearest',
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            id='single_element_stays_the_same_nearest_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': 'wrap',
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            id='single_element_stays_the_same_wrap_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': 1,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1]], pl.List(pl.Float64)),
            id='single_element_stays_the_same_scalar_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 0, 'padding': None,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1], [1, 1], [1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1], [1, 1], [1, 1]], pl.List(pl.Float64)),
            id='three_elements_stay_the_same_none_padding',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 1, 'padding': None,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1], [1, 1], [1, 1]], pl.List(pl.Float64)),
            pl.Series('A', [[0, 0], [0, 0], [0, 0]], pl.List(pl.Float64)),
            id='three_equal_elements_differentation_none_padding_result_zero',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 1, 'derivative': 1, 'padding': None,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1], [2, 2], [3, 3]], pl.List(pl.Float64)),
            pl.Series('A', [[1, 1], [1, 1], [1, 1]], pl.List(pl.Float64)),
            id='three_elements_1_2_3_differentation_none_padding_result_one',
        ),
        pytest.param(
            {
                'window_length': 3, 'degree': 2, 'derivative': 2, 'padding': None,
                'sampling_rate': 1, 'n_components': 2, 'input_column': 'A',
            },
            pl.Series('A', [[1, 1], [4, 4], [9, 9]], pl.List(pl.Float64)),
            pl.Series('A', [[2, 2], [2, 2], [2, 2]], pl.List(pl.Float64)),
            id='three_elements_1_4_9_double_differentation_none_padding_result_two',
        ),
    ],
)
def test_savitzky_golay_returns(kwargs, series, expected_df):
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.savitzky_golay(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame())
