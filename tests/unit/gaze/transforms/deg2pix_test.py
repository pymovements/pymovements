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
"""Test pymovements.gaze.transforms.deg2pix. Based on tests for pix2deg."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'screen_size': (100, 00),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_resolution', 'missing'),
            id='no_screen_resolution_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 00), 'distance': 100, 'pixel_origin': 'center',
                'pixel_column': 'pixel', 'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_size', 'missing'),
            id='no_screen_size_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100),
                'pixel_origin': 'center', 'pixel_column': 'pixel',
                'degree_column': 'degree', 'n_components': 2,
            },
            TypeError,
            ('distance', 'missing'),
            id='no_distance_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': None,
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_resolution must not be None'),
            id='none_screen_resolution_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': None,
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_size must not be None'),
            id='none_screen_size_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': None,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('distance', 'None', 'float', 'int'),
            id='none_distance_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (0, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_resolution', 'must be greater than zero', '0'),
            id='zero_screen_resolution_zero',
        ),
        pytest.param(
            {
                'screen_resolution': 1,
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_resolution', 'must be of type tuple[int, int]', 'type int'),
            id='screen_resolution_int_scalar',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_resolution must have length of 2, but is of length 3',),
            id='screen_resolution_3-tuple',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (0, 100), 'distance': 100,
                'pixel_origin': 'center', 'pixel_column': 'pixel', 'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_size', 'must be greater than zero', '0'),
            id='zero_screen_size_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': 1,
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            TypeError,
            ('screen_size', 'must be of type tuple[int, int]', 'type int'),
            id='screen_size_int_scalar',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_size must have length of 2, but is of length 3',),
            id='screen_size_3-tuple',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100), 'distance': 0,
                'pixel_origin': 'center', 'n_components': 2,
                'pixel_column': 'pixel', 'degree_column': 'degree',
            },
            ValueError,
            ('distance', 'must be greater than zero', '0'),
            id='zero_distance_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (-1, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_resolution', 'must be greater than zero', '-1'),
            id='negative_screen_resolution_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (-1, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('screen_size', 'must be greater than zero', '-1'),
            id='negative_screen_size_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': -1,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('distance', 'must be greater than zero', '-1'),
            id='negative_distance_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'foobar',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            ValueError,
            ('pixel_origin', 'invalid', 'foobar', 'valid', 'center', 'upper left'),
            id='invalid_origin_raises_value_error',
        ),
    ],
)
def test_deg2pix_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.deg2pix(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100), 'distance': 100,
                'pixel_origin': 'center', 'pixel_column': 'aaa', 'degree_column': 'bbb',
                'n_components': 2,
            },
            pl.Series('ccc', [0], pl.Float64),
            pl.exceptions.ColumnNotFoundError,
            ('bbb',),
            id='df_missing_column_raises_column_not_found_error',
        ),
    ],
)
def test_deg2pix_raises_error(kwargs, series, exception, msg_substrings):
    df = series.to_frame()

    with pytest.raises(exception) as excinfo:
        df.with_columns(
            pm.gaze.transforms.deg2pix(**kwargs),
        )

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100.,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [], pl.List(pl.Float64)),
            pl.Series('pixel', [], pl.List(pl.Float64)),
            id='empty_series',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[0, 0]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 0]], pl.List(pl.Float64)),
            id='zero_origin_center_returns_0',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-26.3354, 0]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, (100 - 1) / 2]], pl.List(pl.Float64)),
            id='center_pixel_origin_lowerleft_returns_49.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[0, 45]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            id='isosceles_triangle_origin_center_returns_50',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-44.712084, 45]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            id='isosceles_triangle_origin_lowerleft_returns_99.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-44.712084, -45]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, -0.5]], pl.List(pl.Float64)),
            id='isosceles_triangle_left_origin_lowerleft_returns_neg0.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[0, 26.5650]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            id='ankathet_half_origin_center_returns_50',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-26.3354, 26.5650]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            id='ankathet_half_origin_lowerleft_returns_99.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 / np.sqrt(3),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[0, 60]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            id='ankathet_sqrt3_origin_center_returns_50',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 / np.sqrt(3),
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-59.75, 60]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            id='ankathet_sqrt3_origin_lowerleft_returns_99.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 * np.sqrt(3),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[0, 30]], pl.List(pl.Float64)),
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            id='opposite_sqrt3_origin_center_returns_50',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 * np.sqrt(3),
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[-30, 30]], pl.List(pl.Float64)),
            pl.Series('pixel', [[-0.5, 100 - 0.5]], pl.List(pl.Float64)),
            id='opposite_sqrt3_origin_lowerleft_returns_99.5',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 200),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[45, 26.5650]], pl.List(pl.Float64)),
            pl.Series('pixel', [[100, 100]], pl.List(pl.Float64)),
            id='screen_resolution_different_values',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 200),
                'distance': 100,
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            pl.Series('degree', [[45, 63.4349]], pl.List(pl.Float64)),
            pl.Series('pixel', [[100, 100]], pl.List(pl.Float64)),
            id='screen_size_different_values',
        ),
    ],
)
@pytest.mark.parametrize(
    'distance_as_column',
    [
        pytest.param(True, id='column_distance'),
        pytest.param(False, id='scalar_distance'),
    ],
)
def test_deg2pix_returns(kwargs, series, expected_df, distance_as_column):
    df = series.to_frame().clone()
    kwargs = kwargs.copy()

    # Decide whether to pass distance as column or scalar
    if distance_as_column:

        # unit of distance values has to be in mm when passing as column
        distance_value = kwargs['distance'] * 10

        try:
            df = df.with_columns(pl.Series('distance', [distance_value], pl.Float64))
        except pl.exceptions.InvalidOperationError:
            df = df.with_columns(distance=distance_value)

        kwargs['distance'] = 'distance'

    result_df = df.select(
        pm.gaze.transforms.deg2pix(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame(), abs_tol=1e-4)


@pytest.mark.parametrize(
    ('kwargs', 'data', 'expected_df'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [100., 100., 100.],
            },
            pl.Series('pixel', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_center_constant_distance_and_degree_returns_constant_pixel',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [1000., 100., 10.],
            },
            pl.Series('pixel', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_center_constant_centered_degree_changing_distance_returns_constant_pixel',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [100., 100., 100.],
            },
            pl.Series('pixel', [[49.5, 49.5], [49.5, 49.5], [49.5, 49.5]], pl.List(pl.Float64)),
            id='origin_lower_left_constant_distance_and_degree_returns_constant_pixel',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'upper left',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [1000., 100., 10.],
            },
            pl.Series('pixel', [[49.5, 49.5], [49.5, 49.5], [49.5, 49.5]], pl.List(pl.Float64)),
            id='origin_lower_left_constant_centered_pos_changing_distance_returns_constant_pixel',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[0., 63.4349], [0., 45.], [0., 26.5650]],
                'distance': [250., 500., 1000.],
            },
            pl.Series(
                'pixel', [[0., 50.], [0., 50.], [0., 50.]], pl.List(pl.Float64),
            ),
            id='origin_center_constant_degree_centered_x_changing_distance_returns',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'pixel_origin': 'center',
                'pixel_column': 'pixel',
                'degree_column': 'degree',
                'n_components': 2,
            },
            {
                'degree': [[63.4349, 0.], [45., 0.], [26.5650, 0.]],
                'distance': [250., 500., 1000.],
            },
            pl.Series(
                'pixel', [[50., 0.], [50., 0.], [50., 0.]], pl.List(pl.Float64),
            ),
            id='origin_center_constant_degree_centered_y_changing_distance_returns',
        ),
    ],
)
def test_deg2pix_distance_as_colum_returns(kwargs, data, expected_df):
    df = pl.DataFrame(
        data, schema={'degree': pl.List(pl.Float64), 'distance': pl.Float64},
    )

    result_df = df.select(
        pm.gaze.transforms.deg2pix(**kwargs, distance='distance'),
    )

    assert_frame_equal(result_df, expected_df.to_frame())
