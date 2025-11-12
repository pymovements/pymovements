# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test Gaze.unnest."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('init_data', 'unnest_kwargs', 'expected'),
    [
        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
            id='empty_df_with_schema_two_pixel_columns',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
            id='empty_df_with_three_column_schema_two_pixel_columns',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr']},
            pl.DataFrame(
                schema={
                    'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64,
                },
            ),
            id='empty_df_with_schema_four_pixel_columns',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr', 'xa', 'ya']},
            pl.DataFrame(
                schema={
                    'xl': pl.Float64, 'yl': pl.Float64,
                    'xr': pl.Float64, 'yr': pl.Float64,
                    'xa': pl.Float64, 'ya': pl.Float64,
                },
            ),
            id='empty_df_with_schema_six_pixel_columns',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame({'x': [1.23], 'y': [4.56]}),
            id='df_single_row_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'abc': [1], 'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame({'abc': [1], 'x': [1.23], 'y': [4.56]}),
            id='df_single_row_three_columns_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.2, 3.4, 5.6, 7.8]]}),
            {'input_columns': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr']},
            pl.DataFrame({'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]}),
            id='df_single_row_four_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}),
            {'input_columns': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr', 'xa', 'ya']},
            pl.DataFrame({
                'xl': [.1], 'yl': [.2], 'xr': [.3], 'yr': [.4], 'xa': [.5], 'ya': [.6],
            }),
            id='df_single_row_six_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y'], 'output_columns': None},
            pl.DataFrame(schema={'time': pl.Int64, 'pixel_x': pl.Float64, 'pixel_y': pl.Float64}),
            id='empty_df_with_schema_two_pixel_suffixes_columns_none',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y'], 'output_columns': None},
            pl.DataFrame(
                schema={
                    'abc': pl.Int64, 'time': pl.Int64, 'pixel_x': pl.Float64, 'pixel_y': pl.Float64,
                },
            ),
            id='empty_df_with_three_column_schema_two_pixel_suffixes_columns_none',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {
                'input_columns': 'pixel',
                'output_suffixes': ['_xl', '_yl', '_xr', '_yr'],
                'output_columns': None,
            },
            pl.DataFrame(
                schema={
                    'pixel_xl': pl.Float64, 'pixel_yl': pl.Float64,
                    'pixel_xr': pl.Float64, 'pixel_yr': pl.Float64,
                },
            ),
            id='empty_df_with_schema_four_pixel_suffixes_columns_none',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {
                'input_columns': 'pixel',
                'output_suffixes': ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya'],
            },
            pl.DataFrame(
                schema={
                    'pixel_xl': pl.Float64, 'pixel_yl': pl.Float64,
                    'pixel_xr': pl.Float64, 'pixel_yr': pl.Float64,
                    'pixel_xa': pl.Float64, 'pixel_ya': pl.Float64,
                },
            ),
            id='empty_df_with_schema_six_pixel_columns',
            marks=pytest.mark.xfail(reason='#522'),
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y']},
            pl.DataFrame({'pixel_x': [1.23], 'pixel_y': [4.56]}),
            id='df_single_row_two_pixel_suffixes',
        ),

        pytest.param(
            pl.DataFrame({'abc': [1], 'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y']},
            pl.DataFrame({'abc': [1], 'pixel_x': [1.23], 'pixel_y': [4.56]}),
            id='df_single_row_three_columns_two_pixel_suffixes',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.2, 3.4, 5.6, 7.8]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_xl', '_yl', '_xr', '_yr']},
            pl.DataFrame({
                'pixel_xl': [1.2], 'pixel_yl': [3.4],
                'pixel_xr': [5.6], 'pixel_yr': [7.8],
            }),
            id='df_single_row_four_pixel_suffixes',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}),
            {
                'input_columns': 'pixel',
                'output_suffixes': ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya'],
            },
            pl.DataFrame({
                'pixel_xl': [.1], 'pixel_yl': [.2],
                'pixel_xr': [.3], 'pixel_yr': [.4],
                'pixel_xa': [.5], 'pixel_ya': [.6],
            }),
            id='df_single_row_six_pixel_suffixes',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel'},
            pl.DataFrame({'pixel_x': [1.23], 'pixel_y': [4.56]}),
            id='df_single_row_two_pixel_suffixes_default_values',
        ),

        pytest.param(
            pl.DataFrame({'abc': [1], 'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel'},
            pl.DataFrame({'abc': [1], 'pixel_x': [1.23], 'pixel_y': [4.56]}),
            id='df_single_row_three_columns_two_pixel_suffixes_default_values',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.2, 3.4, 5.6, 7.8]]}),
            {'input_columns': 'pixel'},
            pl.DataFrame({
                'pixel_xl': [1.2], 'pixel_yl': [3.4],
                'pixel_xr': [5.6], 'pixel_yr': [7.8],
            }),
            id='df_single_row_four_pixel_suffixes_default_values',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}),
            {'input_columns': 'pixel'},
            pl.DataFrame({
                'pixel_xl': [.1], 'pixel_yl': [.2],
                'pixel_xr': [.3], 'pixel_yr': [.4],
                'pixel_xa': [.5], 'pixel_ya': [.6],
            }),
            id='df_single_row_six_pixel_suffixes_default_values',
        ),

    ],
)
def test_gaze_unnest_has_expected_frame(init_data, unnest_kwargs, expected):
    gaze = pm.Gaze(init_data)
    gaze.unnest(**unnest_kwargs)
    assert_frame_equal(gaze.samples, expected)


@pytest.mark.parametrize(
    ('init_data', 'unnest_kwargs', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {'input_columns': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame({
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
                'x': [1.23],
                'y': [4.56],
            }),
            id='df_single_row_two_components_unnest_one_multiple_columns_columns_specified',
        ),
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {},
            pl.DataFrame({
                'pixel_x': [1.23], 'pixel_y': [4.56],
                'position_x': [1.23], 'position_y': [4.56],
                'velocity_x': [1.23], 'velocity_y': [4.56],
                'acceleration_x': [1.23], 'acceleration_y': [4.56],
            }),
            id='df_single_row_two_components_unnest_all_default_values',
        ),
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56, 7.89, 10.11]],
                'position': [[1.23, 4.56, 7.89, 10.11]],
                'velocity': [[1.23, 4.56, 7.89, 10.11]],
                'acceleration': [[1.23, 4.56, 7.89, 10.11]],
            }),
            {},
            pl.DataFrame({
                'pixel_xl': [1.23], 'pixel_yl': [4.56],
                'pixel_xr': [7.89], 'pixel_yr': [10.11],
                'position_xl': [1.23], 'position_yl': [4.56],
                'position_xr': [7.89], 'position_yr': [10.11],
                'velocity_xl': [1.23], 'velocity_yl': [4.56],
                'velocity_xr': [7.89], 'velocity_yr': [10.11],
                'acceleration_xl': [1.23], 'acceleration_yl': [4.56],
                'acceleration_xr': [7.89], 'acceleration_yr': [10.11],
            }),
            id='df_single_row_four_components_unnest_all_default_values',
        ),
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56, 7.89, 10.11, 12.13, 14.15]],
                'position': [[1.23, 4.56, 7.89, 10.11, 12.13, 14.15]],
                'velocity': [[1.23, 4.56, 7.89, 10.11, 12.13, 14.15]],
                'acceleration': [[1.23, 4.56, 7.89, 10.11, 12.13, 14.15]],
            }),
            {},
            pl.DataFrame({
                'pixel_xl': [1.23], 'pixel_yl': [4.56],
                'pixel_xr': [7.89], 'pixel_yr': [10.11],
                'pixel_xa': [12.13], 'pixel_ya': [14.15],
                'position_xl': [1.23], 'position_yl': [4.56],
                'position_xr': [7.89], 'position_yr': [10.11],
                'position_xa': [12.13], 'position_ya': [14.15],
                'velocity_xl': [1.23], 'velocity_yl': [4.56],
                'velocity_xr': [7.89], 'velocity_yr': [10.11],
                'velocity_xa': [12.13], 'velocity_ya': [14.15],
                'acceleration_xl': [1.23], 'acceleration_yl': [4.56],
                'acceleration_xr': [7.89], 'acceleration_yr': [10.11],
                'acceleration_xa': [12.13], 'acceleration_ya': [14.15],
            }),
            id='df_single_row_six_components_unnest_all_default_values',
        ),

        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {'input_columns': ['position'], 'output_suffixes': ['_x', '_y']},
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
                'position_x': [1.23], 'position_y': [4.56],
            }),
            id='df_single_row_two_components_unnest_one_multiple_columns_suffix_specified',
        ),

        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {'input_columns': ['position', 'acceleration'], 'output_suffixes': ['_x', '_y']},
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'position_x': [1.23], 'position_y': [4.56],
                'acceleration_x': [1.23], 'acceleration_y': [4.56],
            }),
            id='df_single_row_two_components_unnest_two_multiple_columns_suffix_specified',
        ),
    ],
)
def test_gaze_unnest_has_expected_frame_multiple_unnest(
        init_data, unnest_kwargs, expected,
):
    gaze = pm.Gaze(init_data)
    gaze.unnest(**unnest_kwargs)
    assert_frame_equal(gaze.samples, expected)


@pytest.mark.parametrize(
    ('init_data', 'unnest_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y', '_z']},
            ValueError,
            'Number of output columns / suffixes (3) must match number of components (2)',
            id='df_single_row_two_pixel_components_three_output_suffixes',
        ),
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_columns': ['x']},
            ValueError,
            'Number of output columns / suffixes (1) must match number of components (2)',
            id='df_single_row_two_pixel_components_one_output_column',
        ),
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_x']},
            ValueError,
            'Output columns / suffixes must be unique',
            id='df_single_row_two_output_suffixes_non_unique',
        ),
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_columns': ['x', 'x']},
            ValueError,
            'Output columns / suffixes must be unique',
            id='df_single_row_two_output_columns_non_unique',
        ),
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {
                'input_columns': 'pixel', 'output_suffixes': [
                    '_x', '_y',
                ], 'output_columns': ['x', 'y'],
            },
            ValueError,
            'The arguments "output_columns" and "output_suffixes" are mutually exclusive.',
            id='df_single_row_two_output_columns_and_suffixes',
        ),
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {'input_columns': ['position', 'acceleration'], 'output_columns': ['_x', '_y']},
            ValueError,
            'You cannot specify output columns if you want to unnest more than one input column. '
            'Please specify output suffixes or use a single input column instead.',
            id='df_single_row_two_components_unnest_two_multiple_columns_columns_specified',
        ),
        pytest.param(
            pl.DataFrame({
                'pixel': [[1.23, 4.56]],
                'position': [[1.23, 4.56]],
                'velocity': [[1.23, 4.56]],
                'acceleration': [[1.23, 4.56]],
            }),
            {'output_columns': ['_x', '_y']},
            ValueError,
            'You cannot specify output columns if you want to unnest more than one input column. '
            'Please specify output suffixes or use a single input column instead.',
            id='df_single_row_two_components_unnest_all_default_values_multiple_columns_specified',
        ),
        pytest.param(
            pl.DataFrame({
                'wrong_pixel_name': [[1.23, 4.56]],
                'wrong_position_name': [[1.23, 4.56]],
                'wrong_velocity_name': [[1.23, 4.56]],
                'wrong_acceleration_name': [[1.23, 4.56]],
            }),
            {},
            Warning,
            'No columns to unnest. '
            'Please specify columns to unnest via the "input_columns" argument.',
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_two_components_unnest_all_default_values_no_cols_to_unnest',
        ),
    ],

)
def test_gaze_unnest_errors(init_data, unnest_kwargs, exception, exception_msg):
    with pytest.raises(exception) as exc_info:
        gaze = pm.Gaze(init_data)
        gaze.unnest(**unnest_kwargs)

    msg, = exc_info.value.args
    assert msg == exception_msg


@pytest.mark.parametrize(
    ('init_data', 'unnest_kwargs', 'n_components', 'exception', 'exception_msg'),
    [
        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'input_columns': 'pixel', 'output_suffixes': ['_x', '_y']},
            1,
            AttributeError,
            'Number of components required but no gaze components could be inferred.',
            id='df_single_row_two_pixel_component_invalid_number_of_components',
        ),
    ],

)
def test_gaze_unnest_invalid_number_of_components(
        init_data, unnest_kwargs, n_components, exception, exception_msg,
):
    with pytest.raises(exception) as exc_info:
        gaze = pm.Gaze(init_data)
        gaze.n_components = n_components
        gaze.unnest(**unnest_kwargs)

    msg, = exc_info.value.args
    assert msg.startswith(exception_msg)
