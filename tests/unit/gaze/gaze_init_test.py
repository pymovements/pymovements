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
"""Test Gaze initialization."""
# pylint: disable=too-many-lines
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import DatasetDefinition
from pymovements import Events
from pymovements import Experiment
from pymovements import Gaze


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_samples', 'expected_n_components'),
    [
        pytest.param(
            {
                'samples': pl.DataFrame(),
            },
            pl.DataFrame(schema={}),
            None,
            id='empty_df_no_schema',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'abc': pl.Int64}),
            },
            pl.DataFrame(schema={'abc': pl.Int64}),
            None,
            id='empty_df_with_schema_no_component_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'abc': pl.Int64}),
                'pixel_columns': [],
                'position_columns': [],
                'velocity_columns': [],
                'acceleration_columns': [],
            },
            pl.DataFrame(schema={'abc': pl.Int64}),
            None,
            id='empty_df_with_schema_all_component_columns_empty_lists',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            2,
            id='empty_df_with_schema_two_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)}),
            2,
            id='empty_df_with_three_column_schema_two_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl'],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            4,
            id='empty_df_with_schema_four_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            6,
            id='empty_df_with_schema_six_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'pixel': [[1.23, 4.56]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'pixel': [[1.23, 4.56]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            2,
            marks=pytest.mark.filterwarnings(
                'ignore:.*data.*samples.*:DeprecationWarning',
            ),
            id='deprecated_data_argument',
        ),


        pytest.param(
            {
                'samples': pl.from_dict(
                    {'abc': [1], 'x': [1.23], 'y': [4.56]},
                    schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'abc': [1], 'pixel': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_three_columns_two_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]},
                    schema={'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64},
                ),
                'pixel_columns': ['xl', 'yl', 'xr', 'yr'],
            },
            pl.from_dict(
                {'pixel': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            4,
            id='df_single_row_four_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_right': [0.1], 'y_right': [0.2],
                        'x_left': [0.3], 'y_left': [0.4],
                        'x_avg': [0.5], 'y_avg': [0.6],
                    },
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.from_dict(
                {'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            6,
            id='df_single_row_six_pixel_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            2,
            id='empty_df_with_schema_two_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.DataFrame(
                schema={
                    'abc': pl.Int64, 'position': pl.List(pl.Float64),
                },
            ),
            2,
            id='empty_df_with_three_column_schema_two_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl'],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            4,
            id='empty_df_with_schema_four_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'position_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            6,
            id='empty_df_with_schema_six_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'position': [[1.23, 4.56]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'abc': [1], 'x': [1.23], 'y': [4.56]},
                    schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'abc': [1], 'position': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_three_columns_two_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]},
                    schema={'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64},
                ),
                'position_columns': ['xl', 'yl', 'xr', 'yr'],
            },
            pl.from_dict(
                {'position': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            4,
            id='df_single_row_four_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_right': [0.1], 'y_right': [0.2],
                        'x_left': [0.3], 'y_left': [0.4],
                        'x_avg': [0.5], 'y_avg': [0.6],
                    },
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'position_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.from_dict(
                {'position': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            6,
            id='df_single_row_six_position_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            2,
            id='empty_df_with_schema_two_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.DataFrame(
                schema={
                    'abc': pl.Int64, 'velocity': pl.List(pl.Float64),
                },
            ),
            2,
            id='empty_df_with_three_column_schema_two_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'yr_vel', 'xl_vel', 'yl_vel'],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            4,
            id='empty_df_with_schema_four_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                        'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                        'x_avg_vel': pl.Float64, 'y_avg_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'x_right_vel', 'y_right_vel',
                    'x_left_vel', 'y_left_vel',
                    'x_avg_vel', 'y_avg_vel',
                ],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            6,
            id='empty_df_with_schema_six_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x_vel': [1.23], 'y_vel': [4.56]},
                    schema={'x_vel': pl.Float64, 'y_vel': pl.Float64},
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.from_dict(
                {'velocity': [[1.23, 4.56]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'abc': [1], 'x_vel': [1.23], 'y_vel': [4.56]},
                    schema={'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64},
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.from_dict(
                {'abc': [1], 'velocity': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_three_columns_two_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'xl_vel': [1.2], 'yl_vel': [3.4],
                        'xr_vel': [5.6], 'yr_vel': [7.8],
                    },
                    schema={
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xl_vel', 'yl_vel', 'xr_vel', 'yr_vel'],
            },
            pl.from_dict(
                {'velocity': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            4,
            id='df_single_row_four_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_right_vel': [0.1], 'y_right_vel': [0.2],
                        'x_left_vel': [0.3], 'y_left_vel': [0.4],
                        'x_avg_vel': [0.5], 'y_avg_vel': [0.6],
                    },
                    schema={
                        'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                        'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                        'x_avg_vel': pl.Float64, 'y_avg_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'x_right_vel', 'y_right_vel',
                    'x_left_vel', 'y_left_vel',
                    'x_avg_vel', 'y_avg_vel',
                ],
            },
            pl.from_dict(
                {'velocity': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            6,
            id='df_single_row_six_velocity_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            2,
            id='empty_df_with_schema_two_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64, 'x_acc': pl.Float64, 'y_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.DataFrame(
                schema={
                    'abc': pl.Int64, 'acceleration': pl.List(pl.Float64),
                },
            ),
            2,
            id='empty_df_with_three_column_schema_two_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'yr_acc', 'xl_acc', 'yl_acc'],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            4,
            id='empty_df_with_schema_four_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x_right_acc': pl.Float64, 'y_right_acc': pl.Float64,
                        'x_left_acc': pl.Float64, 'y_left_acc': pl.Float64,
                        'x_avg_acc': pl.Float64, 'y_avg_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'x_right_acc', 'y_right_acc',
                    'x_left_acc', 'y_left_acc',
                    'x_avg_acc', 'y_avg_acc',
                ],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            6,
            id='empty_df_with_schema_six_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x_acc': [1.23], 'y_acc': [4.56]},
                    schema={'x_acc': pl.Float64, 'y_acc': pl.Float64},
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {'acceleration': [[1.23, 4.56]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'abc': [1], 'x_acc': [1.23], 'y_acc': [4.56]},
                    schema={'abc': pl.Int64, 'x_acc': pl.Float64, 'y_acc': pl.Float64},
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {'abc': [1], 'acceleration': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'acceleration': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_three_columns_two_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'xl_acc': [1.2], 'yl_acc': [3.4],
                        'xr_acc': [5.6], 'yr_acc': [7.8],
                    },
                    schema={
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xl_acc', 'yl_acc', 'xr_acc', 'yr_acc'],
            },
            pl.from_dict(
                {'acceleration': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            4,
            id='df_single_row_four_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_right_acc': [0.1], 'y_right_acc': [0.2],
                        'x_left_acc': [0.3], 'y_left_acc': [0.4],
                        'x_avg_acc': [0.5], 'y_avg_acc': [0.6],
                    },
                    schema={
                        'x_right_acc': pl.Float64, 'y_right_acc': pl.Float64,
                        'x_left_acc': pl.Float64, 'y_left_acc': pl.Float64,
                        'x_avg_acc': pl.Float64, 'y_avg_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'x_right_acc', 'y_right_acc',
                    'x_left_acc', 'y_left_acc',
                    'x_avg_acc', 'y_avg_acc',
                ],
            },
            pl.from_dict(
                {'acceleration': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            6,
            id='df_single_row_six_acceleration_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_pix': [0.1], 'y_pix': [0.2],
                        'x_dva': [1.1], 'y_dva': [1.2],
                        'x_vel': [3.1], 'y_vel': [3.2],
                        'x_acc': [5.1], 'y_acc': [5.2],
                    },
                    schema={
                        'x_pix': pl.Float64, 'y_pix': pl.Float64,
                        'x_dva': pl.Float64, 'y_dva': pl.Float64,
                        'x_vel': pl.Float64, 'y_vel': pl.Float64,
                        'x_acc': pl.Float64, 'y_acc': pl.Float64,
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'position_columns': ['x_dva', 'y_dva'],
                'velocity_columns': ['x_vel', 'y_vel'],
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {
                    'pixel': [[0.1, 0.2]],
                    'position': [[1.1, 1.2]],
                    'velocity': [[3.1, 3.2]],
                    'acceleration': [[5.1, 5.2]],
                },
                schema={
                    'pixel': pl.List(pl.Float64),
                    'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64),
                    'acceleration': pl.List(pl.Float64),
                },
            ),
            2,
            id='df_single_row_all_types_two_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'x_right_pos_pix': [0.1], 'y_right_pos_pix': [0.2],
                        'x_left_pos_pix': [0.3], 'y_left_pos_pix': [0.4],
                        'x_avg_pos_pix': [0.5], 'y_avg_pos_pix': [0.6],
                        'x_right_pos_dva': [1.1], 'y_right_pos_dva': [1.2],
                        'x_left_pos_dva': [1.3], 'y_left_pos_dva': [1.4],
                        'x_avg_pos_dva': [1.5], 'y_avg_pos_dva': [1.6],
                        'x_right_vel_dva': [3.1], 'y_right_vel_dva': [3.2],
                        'x_left_vel_dva': [3.3], 'y_left_vel_dva': [3.4],
                        'x_avg_vel_dva': [3.5], 'y_avg_vel_dva': [3.6],
                        'x_right_acc_dva': [5.1], 'y_right_acc_dva': [5.2],
                        'x_left_acc_dva': [5.3], 'y_left_acc_dva': [5.4],
                        'x_avg_acc_dva': [5.5], 'y_avg_acc_dva': [5.6],
                    },
                    schema={
                        'x_right_pos_pix': pl.Float64, 'y_right_pos_pix': pl.Float64,
                        'x_left_pos_pix': pl.Float64, 'y_left_pos_pix': pl.Float64,
                        'x_avg_pos_pix': pl.Float64, 'y_avg_pos_pix': pl.Float64,
                        'x_right_pos_dva': pl.Float64, 'y_right_pos_dva': pl.Float64,
                        'x_left_pos_dva': pl.Float64, 'y_left_pos_dva': pl.Float64,
                        'x_avg_pos_dva': pl.Float64, 'y_avg_pos_dva': pl.Float64,
                        'x_right_vel_dva': pl.Float64, 'y_right_vel_dva': pl.Float64,
                        'x_left_vel_dva': pl.Float64, 'y_left_vel_dva': pl.Float64,
                        'x_avg_vel_dva': pl.Float64, 'y_avg_vel_dva': pl.Float64,
                        'x_right_acc_dva': pl.Float64, 'y_right_acc_dva': pl.Float64,
                        'x_left_acc_dva': pl.Float64, 'y_left_acc_dva': pl.Float64,
                        'x_avg_acc_dva': pl.Float64, 'y_avg_acc_dva': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right_pos_pix', 'y_right_pos_pix',
                    'x_left_pos_pix', 'y_left_pos_pix',
                    'x_avg_pos_pix', 'y_avg_pos_pix',
                ],
                'position_columns': [
                    'x_right_pos_dva', 'y_right_pos_dva',
                    'x_left_pos_dva', 'y_left_pos_dva',
                    'x_avg_pos_dva', 'y_avg_pos_dva',
                ],
                'velocity_columns': [
                    'x_right_vel_dva', 'y_right_vel_dva',
                    'x_left_vel_dva', 'y_left_vel_dva',
                    'x_avg_vel_dva', 'y_avg_vel_dva',
                ],
                'acceleration_columns': [
                    'x_right_acc_dva', 'y_right_acc_dva',
                    'x_left_acc_dva', 'y_left_acc_dva',
                    'x_avg_acc_dva', 'y_avg_acc_dva',
                ],
            },
            pl.from_dict(
                {
                    'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                    'position': [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
                    'velocity': [[3.1, 3.2, 3.3, 3.4, 3.5, 3.6]],
                    'acceleration': [[5.1, 5.2, 5.3, 5.4, 5.5, 5.6]],
                },
                schema={
                    'pixel': pl.List(pl.Float64),
                    'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64),
                    'acceleration': pl.List(pl.Float64),
                },
            ),
            6,
            id='df_single_row_all_types_six_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1.0, 1.5, 2.],
                        'x': [0., 1., 2.],
                        'y': [3., 4., 5.],
                    },
                    schema={'time': pl.Float64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
                'time_unit': 'ms',
                'time_column': 'time',
            },
            pl.from_dict(
                {
                    'time': [1.0, 1.5, 2.],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Float64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_float_millis_time_no_conversion',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'x': [0., 1., 2.],
                        'y': [3., 4., 5.],
                    },
                    schema={'time': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
                'time_unit': 'ms',
                'time_column': 'time',
            },
            pl.from_dict(
                {
                    'time': [1, 2, 3],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_int_millis_time_no_conversion',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1.0, 1.1, 1.2],
                        'x': [0., 1., 2.],
                        'y': [3., 4., 5.],
                    },
                    schema={'time': pl.Float64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
                'time_unit': 's',
                'time_column': 'time',
            },
            pl.from_dict(
                {
                    'time': [1000, 1100, 1200],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_float_seconds_time_converts_to_int_millis',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1.0005, 1.001, 1.0015],
                        'x': [0., 1., 2.],
                        'y': [3., 4., 5.],
                    },
                    schema={'time': pl.Float64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
                'time_unit': 's',
                'time_column': 'time',
            },
            pl.from_dict(
                {
                    'time': [1000.5, 1001., 1001.5],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Float64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_float_seconds_time_converts_to_float_millis',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'x': [0., 1., 2.],
                        'y': [3., 4., 5.],
                    },
                    schema={'time': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
                'time_unit': 'step',
                'time_column': 'time',
                'experiment': Experiment(
                    screen_width_px=1,
                    screen_width_cm=1,
                    screen_height_px=1,
                    screen_height_cm=1,
                    sampling_rate=200,
                ),
            },
            pl.from_dict(
                {
                    'time': [5, 10, 15],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_float_step_time_converts_to_int_millis',
        ),


        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23, 2.34, 3.45], 'y': [4.56, 5.67, 6.78]},
                    schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'position': [[1.23, 4.56], [2.34, 5.67], [3.45, 6.78]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_columns_no_time_no_experiment',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23, 2.34, 3.45], 'y': [4.56, 5.67, 6.78]},
                    schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
                'experiment': Experiment(1024, 768, 38, 30, None, 'center', 100),
            },
            pl.from_dict(
                {'time': [0, 10, 20], 'position': [[1.23, 4.56], [2.34, 5.67], [3.45, 6.78]]},
                schema={'time': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_columns_no_time_100_hz',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23, 2.34, 3.45], 'y': [4.56, 5.67, 6.78]},
                    schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
                'experiment': Experiment(1024, 768, 38, 30, None, 'center', 1000),
            },
            pl.from_dict(
                {'time': [0, 1, 2], 'position': [[1.23, 4.56], [2.34, 5.67], [3.45, 6.78]]},
                schema={'time': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_three_rows_two_position_columns_no_time_1000_hz',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'pixel_x': [0., 1., 2.],
                        'pixel_y': [3., 4., 5.],
                    },
                    schema={'time': pl.Int64, 'pixel_x': pl.Float64, 'pixel_y': pl.Float64},
                ),
                'auto_column_detect': True,
            },
            pl.from_dict(
                {
                    'time': [1, 2, 3],
                    'pixel': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_auto_columns_pixel',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'position_x': [0., 1., 2.],
                        'position_y': [3., 4., 5.],
                    },
                    schema={'time': pl.Int64, 'position_x': pl.Float64, 'position_y': pl.Float64},
                ),
                'auto_column_detect': True,
            },
            pl.from_dict(
                {
                    'time': [1, 2, 3],
                    'position': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_auto_columns_position',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'velocity_x': [0., 1., 2.],
                        'velocity_y': [3., 4., 5.],
                    },
                    schema={'time': pl.Int64, 'velocity_x': pl.Float64, 'velocity_y': pl.Float64},
                ),
                'auto_column_detect': True,
            },
            pl.from_dict(
                {
                    'time': [1, 2, 3],
                    'velocity': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            2,
            id='df_auto_columns_velocity',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {
                        'time': [1, 2, 3],
                        'acceleration_x': [0., 1., 2.],
                        'acceleration_y': [3., 4., 5.],
                    },
                    schema={
                        'time': pl.Int64,
                        'acceleration_x': pl.Float64,
                        'acceleration_y': pl.Float64,
                    },
                ),
                'auto_column_detect': True,
            },
            pl.from_dict(
                {
                    'time': [1, 2, 3],
                    'acceleration': [[0., 3.], [1., 4.], [2., 5.]],
                },
                schema={'time': pl.Int64, 'acceleration': pl.List(pl.Float64)},
            ),
            2,
            id='df_auto_columns_acceleration',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(pixel_columns=['x', 'y']),
            },
            pl.from_dict(
                {'pixel': [[1.23, 4.56]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_pixel_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(pixel_columns=['foo', 'bar']),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'pixel': [[1.23, 4.56]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_pixel_columns_overwrite_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'abc': [1], 'x': [1.23], 'y': [4.56]},
                    schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(pixel_columns=['x', 'y']),
            },
            pl.from_dict(
                {'abc': [1], 'pixel': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_three_columns_two_pixel_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]},
                    schema={'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64},
                ),
                'definition': DatasetDefinition(pixel_columns=['xl', 'yl', 'xr', 'yr']),
            },
            pl.from_dict(
                {'pixel': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            4,
            id='df_single_row_four_pixel_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(position_columns=['x', 'y']),
            },
            pl.from_dict(
                {'position': [[1.23, 4.56]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_position_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(position_columns=['foo', 'bar']),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'position': [[1.23, 4.56]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_position_columns_overwrite_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(velocity_columns=['x', 'y']),
            },
            pl.from_dict(
                {'velocity': [[1.23, 4.56]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_velocity_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(velocity_columns=['foo', 'bar']),
                'velocity_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'velocity': [[1.23, 4.56]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_velocity_columns_overwrite_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(acceleration_columns=['x', 'y']),
            },
            pl.from_dict(
                {'acceleration': [[1.23, 4.56]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_acceleration_columns_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'definition': DatasetDefinition(acceleration_columns=['foo', 'bar']),
                'acceleration_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'acceleration': [[1.23, 4.56]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            2,
            id='df_single_row_two_acceleration_columns_overwrite_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'t': [1.23]}, schema={'t': pl.Float64}),
                'definition': DatasetDefinition(time_column='t'),
            },
            pl.from_dict({'time': [1.23]}, schema={'time': pl.Float64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_time_column_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'t': [1.23]}, schema={'t': pl.Float64}),
                'definition': DatasetDefinition(time_column='foo'),
                'time_column': 't',
            },
            pl.from_dict({'time': [1.23]}, schema={'time': pl.Float64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_time_column_overwrites_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'time': [1.23]}, schema={'time': pl.Float64}),
                'definition': DatasetDefinition(time_unit='s'),
            },
            pl.from_dict({'time': [1230]}, schema={'time': pl.Int64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_time_unit_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'time': [4.56]}, schema={'time': pl.Float64}),
                'definition': DatasetDefinition(time_unit='ms'),
                'time_unit': 's',
            },
            pl.from_dict({'time': [4560]}, schema={'time': pl.Int64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_time_unit_overwrites_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'d': [1.23]}, schema={'d': pl.Float64}),
                'definition': DatasetDefinition(distance_column='d'),
            },
            pl.from_dict({'distance': [1.23]}, schema={'distance': pl.Float64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_distance_column_dataset_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict({'d': [1.23]}, schema={'d': pl.Float64}),
                'definition': DatasetDefinition(distance_column='foo'),
                'distance_column': 'd',
            },
            pl.from_dict({'distance': [1.23]}, schema={'distance': pl.Float64}),
            None,
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='df_single_row_distance_column_overwrites_dataset_definition',
        ),

    ],
)
def test_init_gaze_has_expected_attrs(init_kwargs, expected_samples, expected_n_components):
    gaze = Gaze(**init_kwargs)
    assert_frame_equal(gaze.samples, expected_samples)
    assert gaze.n_components == expected_n_components


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_experiment'),
    [
        pytest.param(
            {
                'experiment': Experiment(sampling_rate=1000),
            },
            Experiment(sampling_rate=1000),
            id='experiment',
        ),

        pytest.param(
            {
                'definition': DatasetDefinition(experiment=Experiment(sampling_rate=1234)),
            },
            Experiment(sampling_rate=1234),
            id='definition',
        ),

        pytest.param(
            {
                'experiment': Experiment(sampling_rate=5678),
                'definition': DatasetDefinition(experiment=Experiment(sampling_rate=1111)),
            },
            Experiment(sampling_rate=5678),
            id='experiment_overwrites_definition',
        ),

    ],
)
def test_init_gaze_has_expected_experiment(init_kwargs, expected_experiment):
    gaze = Gaze(**init_kwargs)
    assert gaze.experiment == expected_experiment


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
@pytest.mark.parametrize(
    ('init_kwargs', 'expected_trial_columns'),
    [
        pytest.param(
            {
                'samples': pl.from_dict(
                    data={'trial': [1]},
                    schema={'trial': pl.Int64},
                ),
                'trial_columns': [],
            },
            None,
            id='df_empty_trial_columns',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    data={'trial': [1]},
                    schema={'trial': pl.Int64},
                ),
                'trial_columns': 'trial',
            },
            ['trial'],
            id='df_single_trial_column_str',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    data={'trial': [1]},
                    schema={'trial': pl.Int64},
                ),
                'trial_columns': ['trial'],
            },
            ['trial'],
            id='df_single_trial_column_list',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    data={'trial': [1]},
                    schema={'trial': pl.Int64},
                ),
                'definition': DatasetDefinition(trial_columns=['trial']),
            },
            ['trial'],
            id='df_single_trial_column_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    data={'trial': [1]},
                    schema={'trial': pl.Int64},
                ),
                'definition': DatasetDefinition(trial_columns=['foobar']),
                'trial_columns': 'trial',
            },
            ['trial'],
            id='df_single_trial_column_overwrites_definition',
        ),

        pytest.param(
            {
                'samples': pl.from_dict(
                    {'group': [2], 'trial': [1]},
                    schema={'group': pl.Int64, 'trial': pl.Int64},
                ),
                'trial_columns': ['group', 'trial'],
            },
            ['group', 'trial'],
            id='df_two_trial_columns',
        ),

    ],
)
def test_init_gaze_has_expected_trial_columns(init_kwargs, expected_trial_columns):
    gaze = Gaze(**init_kwargs)
    assert gaze.trial_columns == expected_trial_columns


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': 1,
            },
            TypeError,
            'pixel_columns must be of type list, but is of type int',
            id='pixel_columns_int',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': 'x',
            },
            TypeError,
            'pixel_columns must be of type list, but is of type str',
            id='pixel_columns_str',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': [0, 1],
            },
            TypeError,
            'all elements in pixel_columns must be of type str,'
            ' but one of the elements is of type int',
            id='pixel_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 1',
            id='pixel_columns_list_of_one',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'xl', 'yl'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 3',
            id='pixel_columns_list_of_three',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl', 'xa'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 5',
            id='pixel_columns_list_of_five',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl', 'xa', 'ya', 'abc'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 7',
            id='pixel_columns_list_of_seven',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Int64}),
                'pixel_columns': ['x', 'y'],
            },
            ValueError,
            'all columns in pixel_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='pixel_columns_different_type',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y from pixel_columns is not available in samples dataframe',
            id='pixel_columns_missing_column',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': 1,
            },
            TypeError,
            'position_columns must be of type list, but is of type int',
            id='position_columns_int',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': 'x',
            },
            TypeError,
            'position_columns must be of type list, but is of type str',
            id='position_columns_str',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': [0, 1],
            },
            TypeError,
            'all elements in position_columns must be of type str,'
            ' but one of the elements is of type int',
            id='position_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 1',
            id='position_columns_list_of_one',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'xl', 'yl'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 3',
            id='position_columns_list_of_three',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl', 'xa'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 5',
            id='position_columns_list_of_five',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl', 'xa', 'ya', 'abc'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 7',
            id='position_columns_list_of_seven',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Int64}),
                'position_columns': ['x', 'y'],
            },
            ValueError,
            'all columns in position_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='position_columns_different_type',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y from position_columns is not available in samples dataframe',
            id='position_columns_missing_column',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': 1,
            },
            TypeError,
            'velocity_columns must be of type list, but is of type int',
            id='velocity_columns_int',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': 'x_vel',
            },
            TypeError,
            'velocity_columns must be of type list, but is of type str',
            id='velocity_columns_str',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': [0, 1],
            },
            TypeError,
            'all elements in velocity_columns must be of type str,'
            ' but one of the elements is of type int',
            id='velocity_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': ['x_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 1',
            id='velocity_columns_list_of_one',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'xl_vel', 'yl_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 3',
            id='velocity_columns_list_of_three',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xa_vel': pl.Float64, 'ya_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'yr_vel', 'xl_vel', 'yl_vel', 'xa_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 5',
            id='velocity_columns_list_of_five',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xa_vel': pl.Float64, 'ya_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'xr_vel', 'yr_vel', 'xl_vel', 'yl_vel', 'xa_vel', 'ya_vel', 'abc',
                ],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 7',
            id='velocity_columns_list_of_seven',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Int64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            ValueError,
            'all columns in velocity_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='velocity_columns_different_type',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_vel': pl.Float64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y_vel from velocity_columns is not available in samples dataframe',
            id='velocity_columns_missing_column',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': 1,
            },
            TypeError,
            'acceleration_columns must be of type list, but is of type int',
            id='acceleration_columns_int',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': 'x_acc',
            },
            TypeError,
            'acceleration_columns must be of type list, but is of type str',
            id='acceleration_columns_str',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': [0, 1],
            },
            TypeError,
            'all elements in acceleration_columns must be of type str,'
            ' but one of the elements is of type int',
            id='acceleration_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': ['x_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 1',
            id='acceleration_columns_list_of_one',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'xl_acc', 'yl_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 3',
            id='acceleration_columns_list_of_three',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xa_acc': pl.Float64, 'ya_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'yr_acc', 'xl_acc', 'yl_acc', 'xa_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 5',
            id='acceleration_columns_list_of_five',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xa_acc': pl.Float64, 'ya_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'xr_acc', 'yr_acc', 'xl_acc', 'yl_acc', 'xa_acc', 'ya_acc', 'abc',
                ],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 7',
            id='acceleration_columns_list_of_seven',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Int64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            ValueError,
            'all columns in acceleration_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='acceleration_columns_different_type',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(schema={'x_acc': pl.Float64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y_acc from acceleration_columns is not available in samples dataframe',
            id='acceleration_columns_missing_column',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x': pl.Float64, 'y': pl.Float64,
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'pixel_columns': ['x', 'y'],
                'position_columns': ['xl', 'yl', 'xr', 'yr'],
            },
            ValueError,
            'inconsistent number of components inferred: {2, 4}',
            id='inconsistent_number_of_components',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x': pl.Float64, 'y': pl.Float64,
                        'time': pl.Float64,
                    },
                ),
                'pixel_columns': ['x', 'y'],
                'time_column': 'time',
                'time_unit': 'step',
            },
            ValueError,
            "experiment with sampling rate must be specified if time_unit is 'step'",
            id='time_unit_step_no_experiment',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x': pl.Float64, 'y': pl.Float64,
                        'trial': pl.Int64, 'time': pl.Float64,
                    },
                ),
                'pixel_columns': ['x', 'y'],
                'trial_columns': ['trial', 'trial'],
            },
            ValueError,
            'duplicates in trial_columns: trial',
            id='duplicate_trial_columns',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x': pl.Float64, 'y': pl.Float64,
                        'time': pl.Float64, 'bar': pl.Int64,
                    },
                ),
                'pixel_columns': ['x', 'y'],
                'trial_columns': ['foo', 'bar'],
            },
            KeyError,
            'trial_columns missing in samples: foo',
            id='trial_columns_not_in_samples',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(
                    schema={
                        'x': pl.Float64, 'y': pl.Float64,
                        'time': pl.Float64,
                    },
                ),
                'pixel_columns': ['x', 'y'],
                'time_column': 'time',
                'time_unit': 'invalid',
            },
            ValueError,
            "unsupported time unit 'invalid'. "
            "Supported units are 's' for seconds, 'ms' for milliseconds and "
            "'step' for steps.",
            id='time_unit_unsupported',
        ),

        pytest.param(
            {
                'samples': pl.DataFrame(),
                'data': pl.DataFrame(),
            },
            ValueError,
            'The arguments "samples" and "data" are mutually exclusive.',
            marks=pytest.mark.filterwarnings('ignore:.*data.*samples.*:DeprecationWarning'),
            id='samples_data_mutually_exclusive',
        ),

    ],
)
def test_gaze_init_exceptions(init_kwargs, exception, exception_msg):
    with pytest.raises(exception) as excinfo:
        Gaze(**init_kwargs)

    msg, = excinfo.value.args
    assert msg == exception_msg


def test_gaze_copy_init_has_same_n_components():
    """Tests if gaze initialization with frame with nested columns has correct n_components.

    Refers to issue #514.
    """
    df_orig = pl.from_numpy(np.zeros((3, 1000)), orient='col', schema=['t', 'x', 'y'])
    gaze = Gaze(df_orig, position_columns=['x', 'y'], time_column='t')

    df_copy = gaze.samples.clone()
    gaze_copy = Gaze(df_copy)

    assert gaze.n_components == gaze_copy.n_components


@pytest.mark.parametrize(
    ('events', 'init_kwargs'),
    [
        pytest.param(
            None,
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            id='samples_with_no_events',
        ),

        pytest.param(
            Events(),
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            id='samples_empty_events',
        ),

        pytest.param(
            Events(),
            {},
            id='no_samples_empty_events',
        ),

        pytest.param(
            Events(name='saccade', onsets=[0], offsets=[10]),
            {},
            id='no_samples_with_saccades',
        ),

        pytest.param(
            Events(name='fixation', onsets=[100], offsets=[910]),
            {
                'samples': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            id='samples_with_fixations',
        ),
    ],
)
def test_gaze_init_events(events, init_kwargs):
    if events is None:
        expected_events = Events().frame
    else:
        expected_events = events.frame

    gaze = Gaze(events=events, **init_kwargs)

    assert_frame_equal(gaze.events.frame, expected_events)
    # We don't want the events point to the same reference.
    assert gaze.events.frame is not expected_events


def test_gaze_init_warnings():
    with pytest.warns(UserWarning) as record:
        Gaze(samples=pl.from_dict({'a': [1, 2, 3]}))

    expected_msg_prefix = 'Gaze contains samples but no components could be inferred.'

    assert len(record) == 1
    assert record[0].message.args[0].startswith(expected_msg_prefix)


@pytest.mark.parametrize(
    'init_kwargs',
    [
        pytest.param(
            {'data': pl.DataFrame()},
            id='data',
        ),
    ],
)
def test_gaze_init_parameter_is_deprecated(init_kwargs):
    with pytest.warns(DeprecationWarning):
        Gaze(**init_kwargs)


@pytest.mark.parametrize(
    'init_kwargs',
    [
        pytest.param(
            {'data': pl.DataFrame()},
            id='data',
        ),
    ],
)
def test_gaze_init_parameter_is_removed(init_kwargs, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        Gaze(**init_kwargs)
    function_name = f'Gaze init argument {list(init_kwargs.keys())[0]}'
    assert_deprecation_is_removed(function_name, info.value.args[0], __version__)
