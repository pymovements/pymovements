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
"""Test read from csv."""
import polars as pl
import pytest

from pymovements import DatasetLibrary
from pymovements.gaze import from_csv


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_shape', 'expected_schema'),
    [
        pytest.param(
            'monocular_example.csv',
            {
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
            },
            (10, 2),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape',
        ),

        pytest.param(
            'monocular_example.csv',
            {
                'column_map': {
                    'x_left_pix': 'pixel_xl',
                    'y_left_pix': 'pixel_yl',
                },
                'auto_column_detect': True,
            },
            (10, 2),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape_auto_column_detect',
        ),

        pytest.param(
            'monocular_example.csv',
            {
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
                'add_columns': {'test': 1},
                'column_schema_overrides': {'test': pl.Float64},
            },
            (10, 3),
            {'time': pl.Int64, 'test': pl.Float64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape_add_columns',
        ),

        pytest.param(
            'binocular_example.csv',
            {
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            },
            (10, 3),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64), 'position': pl.List(pl.Float64)},
            id='csv_bino_shape',
        ),

        pytest.param(
            'binocular_example.csv',
            {
                'column_map': {
                    'x_left_pix': 'pixel_xl',
                    'y_left_pix': 'pixel_yl',
                    'x_right_pix': 'pixel_xr',
                    'y_right_pix': 'pixel_yr',
                    'x_left_pos': 'position_xl',
                    'y_left_pos': 'position_yl',
                    'x_right_pos': 'position_xr',
                    'y_right_pos': 'position_yr',
                },
                'auto_column_detect': True,
            },
            (10, 3),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64), 'position': pl.List(pl.Float64)},
            id='csv_bino_shape_auto_column_detect',
        ),

        pytest.param(
            'missing_values_example.csv',
            {
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['pixel_x', 'pixel_y'],
                'position_columns': ['position_x', 'position_y'],
            },
            (103, 3),
            {'time': pl.Int64, 'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64)},
            id='csv_missing_values',
        ),

        pytest.param(
            'gaze_on_faces_example.csv',
            {
                'experiment': DatasetLibrary.get('GazeOnFaces').experiment,
                **DatasetLibrary.get('GazeOnFaces').resources[0].load_kwargs,
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='gaze_on_faces_example',
        ),

        pytest.param(
            'gazebase_example.csv',
            {
                'experiment': DatasetLibrary.get('GazeBase').experiment,
                **DatasetLibrary.get('GazeBase').resources[0].load_kwargs,
            },
            (10, 7),
            {
                'time': pl.Int64, 'validity': pl.Int64, 'dP': pl.Float32, 'lab': pl.Int64,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_example',
        ),

        pytest.param(
            'gazebase_vr_example.csv',
            {
                'experiment': DatasetLibrary.get('GazeBaseVR').experiment,
                **DatasetLibrary.get('GazeBaseVR').resources[0].load_kwargs,
            },
            (10, 11),
            {
                'time': pl.Float32,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32, 'z_target_pos': pl.Float32,
                'clx': pl.Float32, 'cly': pl.Float32, 'clz': pl.Float32,
                'crx': pl.Float32, 'cry': pl.Float32, 'crz': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_vr_example',
        ),

        pytest.param(
            'hbn_example.csv',
            {
                'experiment': DatasetLibrary.get('HBN').experiment,
                **DatasetLibrary.get('HBN').resources[0].load_kwargs,
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='hbn_example',
        ),

        pytest.param(
            'judo1000_example.csv',
            {
                'experiment': DatasetLibrary.get('JuDo1000').experiment,
                **DatasetLibrary.get('JuDo1000').resources[0].load_kwargs,
            },
            (10, 4),
            {
                'trial_id': pl.Int64, 'point_id': pl.Int64,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_example',
        ),

        pytest.param(
            'potec_example.tsv',
            {
                'experiment': DatasetLibrary.get('PoTeC').experiment,
                **DatasetLibrary.get('PoTeC').resources[0].load_kwargs,
            },
            (10, 3),
            {
                'time': pl.Int64, 'pupil_diameter': pl.Float32, 'pixel': pl.List(pl.Float32),
            },
            id='potec_example',
        ),

        pytest.param(
            'potec_example.tsv',
            {
                'experiment': DatasetLibrary.get('PoTeC').experiment,
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x', 'y'],
                'schema_overrides': {
                    'time': pl.Int64,
                    'x': pl.Float64,
                    'y': pl.Float64,
                    'pupil_diameter': pl.Float64,
                },
                'separator': '\t',
            },
            (10, 3),
            {
                'time': pl.Int64, 'pupil_diameter': pl.Float64, 'pixel': pl.List(pl.Float64),
            },
            marks=pytest.mark.filterwarnings('ignore:from_csv.*kwargs.*:DeprecationWarning'),
            id='potec_example_deprecated_kwargs',
        ),

        pytest.param(
            'sbsat_example.csv',
            {
                'experiment': DatasetLibrary.get('SBSAT').experiment,
                **DatasetLibrary.get('SBSAT').resources[0].load_kwargs,
            },
            (10, 5),
            {
                'book_name': pl.String, 'screen_id': pl.Int64, 'time': pl.Int64,
                'pupil_left': pl.Float32, 'pixel': pl.List(pl.Float32),
            },
            id='sbsat_example',
        ),
    ],
)
def test_from_csv_gaze_has_expected_shape_and_columns(
        filename, kwargs, expected_shape, expected_schema, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_csv(file=filepath, **kwargs)

    assert gaze.samples.shape == expected_shape
    assert gaze.samples.schema == expected_schema


@pytest.mark.parametrize(
    ('filename', 'kwargs'),
    [
        pytest.param(
            'monocular_example.csv',
            {
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
                'skip_lines': 0,
            },
            id='**kwargs',
        ),
    ],
)
def test_from_asc_parameter_is_deprecated(
        filename, kwargs, make_example_file, assert_deprecation_is_removed,
):
    filepath = make_example_file(filename)

    with pytest.raises(DeprecationWarning) as info:
        from_csv(filepath, **kwargs)

    assert_deprecation_is_removed(
        function_name=f'keyword argument {list(kwargs.keys())[0]}',
        warning_message=info.value.args[0],
        scheduled_version='0.29.0',

    )


pytest.mark.filterwarnings('ignore:Gaze contains samples but no components could be inferred.')


def test_from_csv_decimal_overrides_with_precision_and_scale(tmp_path):
    p = tmp_path / 'mini.csv'
    p.write_text('time,pupil\n0,1.23\n1,4.56\n')

    gaze = from_csv(
        file=str(p),
        time_column='time',
        time_unit='ms',
        column_schema_overrides={'pupil': pl.Decimal(38, 10)},
    )

    assert gaze.samples.schema['pupil'] == pl.Decimal(38, 10)
