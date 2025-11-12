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
"""Test pymovements.gaze.transforms.pix2deg."""
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            TypeError,
            ('screen_resolution', 'missing'),
            id='no_screen_resolution_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 00), 'distance': 100, 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
                'n_components': 2,
            },
            TypeError,
            ('screen_size', 'missing'),
            id='no_screen_size_raises_type_error',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100), 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
                'n_components': 2,
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            ValueError,
            ('screen_resolution must have length of 2, but is of length 3',),
            id='screen_resolution_3-tuple',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (0, 100), 'distance': 100,
                'origin': 'center', 'pixel_column': 'pixel', 'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            ValueError,
            ('screen_size must have length of 2, but is of length 3',),
            id='screen_size_3-tuple',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100), 'distance': 0,
                'origin': 'center', 'n_components': 2,
                'pixel_column': 'pixel', 'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
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
                'origin': 'foobar',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            ValueError,
            ('origin', 'invalid', 'foobar', 'valid', 'center', 'upper left'),
            id='invalid_origin_raises_value_error',
        ),
    ],
)
def test_pix2deg_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.pix2deg(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100), 'screen_size': (100, 100), 'distance': 100,
                'origin': 'center', 'pixel_column': 'aaa', 'position_column': 'bbb',
                'n_components': 2,
            },
            pl.Series('ccc', [0], pl.Float64),
            pl.exceptions.ColumnNotFoundError,
            ('aaa',),
            id='df_missing_column_raises_column_not_found_error',
        ),
    ],
)
def test_pix2deg_raises_error(kwargs, series, exception, msg_substrings):
    df = series.to_frame()

    with pytest.raises(exception) as excinfo:
        df.with_columns(
            pm.gaze.transforms.pix2deg(**kwargs),
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
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [], pl.List(pl.Float64)),
            pl.Series('position', [], pl.List(pl.Float64)),
            id='empty_series',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 0]], pl.List(pl.Float64)),
            pl.Series('position', [[0, 0]], pl.List(pl.Float64)),
            id='zero_origin_center_returns_0',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, (100 - 1) / 2]], pl.List(pl.Float64)),
            pl.Series('position', [[-26.3354, 0]], pl.List(pl.Float64)),
            id='center_pixel_origin_upperleft_returns_0',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            pl.Series('position', [[0, 45]], pl.List(pl.Float64)),
            id='isosceles_triangle_origin_center_returns_45',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-44.7120, 45]], pl.List(pl.Float64)),
            id='isosceles_triangle_origin_upperleft_returns_45',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-44.7120, 45]], pl.List(pl.Float64)),
            id='isosceles_triangle_origin_default_returns_45',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, -0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-44.7120, -45]], pl.List(pl.Float64)),
            id='isosceles_triangle_left_origin_upperleft_returns_neg45',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50,
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, -0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-44.7120, -45]], pl.List(pl.Float64)),
            id='isosceles_triangle_left_origin_default_returns_neg45',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            pl.Series('position', [[0, 26.5650]], pl.List(pl.Float64)),
            id='ankathet_half_origin_center_returns_26.565',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-26.3354, 26.5650]], pl.List(pl.Float64)),
            id='ankathet_half_origin_upperleft_returns_26.565',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 100,
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-26.3354, 26.5650]], pl.List(pl.Float64)),
            id='ankathet_half_origin_default_returns_26.565',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 / np.sqrt(3),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            pl.Series('position', [[0, 60]], pl.List(pl.Float64)),
            id='ankathet_sqrt3_origin_center_returns_60',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 / np.sqrt(3),
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-59.75, 60]], pl.List(pl.Float64)),
            id='ankathet_sqrt3_origin_upperleft_returns_60',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 * np.sqrt(3),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[0, 50]], pl.List(pl.Float64)),
            pl.Series('position', [[0, 30]], pl.List(pl.Float64)),
            id='opposite_sqrt3_origin_center_returns_30',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'distance': 50 * np.sqrt(3),
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[-0.5, 100 - 0.5]], pl.List(pl.Float64)),
            pl.Series('position', [[-30, 30]], pl.List(pl.Float64)),
            id='opposite_sqrt3_origin_upperleft_returns_30',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 200),
                'screen_size': (100, 100),
                'distance': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[100, 100]], pl.List(pl.Float64)),
            pl.Series('position', [[45, 26.5650]], pl.List(pl.Float64)),
            id='screen_resolution_different_values',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 200),
                'distance': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            pl.Series('pixel', [[100, 100]], pl.List(pl.Float64)),
            pl.Series('position', [[45, 63.4349]], pl.List(pl.Float64)),
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
def test_pix2deg_returns(kwargs, series, expected_df, distance_as_column):
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
        pm.gaze.transforms.pix2deg(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame())


@pytest.mark.parametrize(
    ('kwargs', 'data', 'expected_df'),
    [
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [100., 100., 100.],
            },
            pl.Series('position', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_center_constant_distance_and_pixel_returns_constant_position',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[0., 0.], [0., 0.], [0., 0.]],
                'distance': [1000., 100., 10.],
            },
            pl.Series('position', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_center_constant_centered_pixel_changing_distance_returns_constant_position',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[49.5, 49.5], [49.5, 49.5], [49.5, 49.5]],
                'distance': [100., 100., 100.],
            },
            pl.Series('position', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_upper_left_constant_distance_and_pixel_'
               'returns_constant_position',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'upper left',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[49.5, 49.5], [49.5, 49.5], [49.5, 49.5]],
                'distance': [1000., 100., 10.],
            },
            pl.Series('position', [[0., 0.], [0., 0.], [0., 0.]], pl.List(pl.Float64)),
            id='origin_upper_left_constant_centered_pixel_changing_distance_'
               'returns_constant_position',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[0., 50.], [0., 50.], [0., 50.]],
                'distance': [250., 500., 1000.],
            },
            pl.Series(
                'position', [[0., 63.4349], [0., 45.], [0., 26.5650]], pl.List(pl.Float64),
            ),
            id='origin_center_constant_pixel_centered_x_changing_distance_returns',
        ),
        pytest.param(
            {
                'screen_resolution': (100, 100),
                'screen_size': (100, 100),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
                'n_components': 2,
            },
            {
                'pixel': [[50., 0.], [50., 0.], [50., 0.]],
                'distance': [250., 500., 1000.],
            },
            pl.Series(
                'position', [[63.4349, 0.], [45., 0.], [26.5650, 0.]], pl.List(pl.Float64),
            ),
            id='origin_center_constant_pixel_centered_y_changing_distance_returns',
        ),
    ],
)
def test_pix2deg_distance_as_colum_returns(kwargs, data, expected_df):
    df = pl.DataFrame(
        data, schema={'pixel': pl.List(pl.Float64), 'distance': pl.Float64},
    )

    result_df = df.select(
        pm.gaze.transforms.pix2deg(**kwargs, distance='distance'),
    )

    assert_frame_equal(result_df, expected_df.to_frame())
