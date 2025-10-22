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
import re

import polars as pl
import pytest

from pymovements import __version__
from pymovements import DatasetDefinition
from pymovements import datasets
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
                'definition': DatasetDefinition(
                    time_column='time',
                    time_unit='ms',
                    pixel_columns=['x_left_pix', 'y_left_pix'],
                ),
            },
            (10, 2),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape_definition',
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
                'definition': datasets.GazeOnFaces(),
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
        ),

        pytest.param(
            'gaze_on_faces_example.csv',
            {
                'definition': datasets.GazeOnFaces(),
                'pixel_columns': ['foo', 'bar'],
                'read_csv_kwargs': {
                    'separator': ',',
                    'has_header': False,
                    'new_columns': ['foo', 'bar'],
                    'schema_overrides': [pl.Float32, pl.Float32],
                },
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='gaze_on_faces_dataset_explicit_read_kwargs_and_columns',
        ),

        pytest.param(
            'gaze_on_faces_example.csv',
            {
                'definition': datasets.GazeOnFaces(),
                'pixel_columns': ['foo', 'bar'],
                **{
                    'separator': ',',
                    'has_header': False,
                    'new_columns': ['foo', 'bar'],
                    'schema_overrides': [pl.Float32, pl.Float32],
                },
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            marks=pytest.mark.filterwarnings('ignore:.*from_csv.*kwargs.*:DeprecationWarning'),
            id='gaze_on_faces_dataset_**kwargs_and_columns',
        ),

        pytest.param(
            'gazebase_example.csv',
            {
                'definition': datasets.GazeBase(),
            },
            (10, 7),
            {
                'time': pl.Int64, 'validity': pl.Int64, 'dP': pl.Float32, 'lab': pl.Int64,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_dataset_example',
        ),

        pytest.param(
            'gazebase_example.csv',
            {
                'definition': datasets.GazeBase(),
                'column_map': {'dP': 'test'},
            },
            (10, 7),
            {
                'time': pl.Int64, 'val': pl.Int64, 'test': pl.Float32, 'lab': pl.Int64,
                'xT': pl.Float32, 'yT': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_dataset_example_column_map_overrides_definition',
        ),

        pytest.param(
            'gazebase_vr_example.csv',
            {
                'definition': datasets.GazeBaseVR(),
            },
            (10, 11),
            {
                'time': pl.Float32,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32, 'z_target_pos': pl.Float32,
                'clx': pl.Float32, 'cly': pl.Float32, 'clz': pl.Float32,
                'crx': pl.Float32, 'cry': pl.Float32, 'crz': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_vr_dataset_example',
        ),

        pytest.param(
            'hbn_example.csv',
            {
                'definition': datasets.HBN(),
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='hbn_dataset_example',
        ),

        pytest.param(
            'hbn_example.csv',
            {
                'definition': datasets.HBN(),
                'pixel_columns': [],
                'position_columns': ['x_pix', 'y_pix'],
            },
            (10, 2),
            {'time': pl.Float64, 'position': pl.List(pl.Float32)},
            id='hbn_dataset_example_columns_override_definition',
        ),

        pytest.param(
            'judo1000_example.csv',
            {
                'definition': datasets.JuDo1000(),
            },
            (10, 4),
            {
                'trial_id': pl.Int64, 'point_id': pl.Int64,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_dataset_example',
        ),

        pytest.param(
            'judo1000_example.csv',
            {
                'definition': datasets.JuDo1000(),
                'column_schema_overrides': {'trial_id': pl.String},
            },
            (10, 4),
            {
                'trial_id': pl.String, 'point_id': pl.Int64,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_dataset_example_column_schema_overrides',
        ),

        pytest.param(
            'judo1000_example.csv',
            {
                'definition': datasets.JuDo1000(
                    custom_read_kwargs={
                        'gaze': {
                            'schema_overrides': {
                                'trialId': pl.String,
                                'pointId': pl.String,
                                'time': pl.Int64,
                                'x_left': pl.Float32,
                                'y_left': pl.Float32,
                                'x_right': pl.Float32,
                                'y_right': pl.Float32,
                            },
                            'separator': '\t',
                        },
                    },
                ),
            },
            (10, 4),
            {
                'trial_id': pl.String, 'point_id': pl.String,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_dataset_example_schema_overrides_from_definition',
        ),

        pytest.param(
            'judo1000_example.csv',
            {
                'definition': datasets.JuDo1000(
                    custom_read_kwargs={
                        'gaze': {
                            'schema_overrides': {
                                'trialId': pl.String,
                                'pointId': pl.String,
                                'time': pl.Int64,
                                'x_left': pl.Float32,
                                'y_left': pl.Float32,
                                'x_right': pl.Float32,
                                'y_right': pl.Float32,
                            },
                            'separator': '\t',
                        },
                    },
                ),
                'column_schema_overrides': {
                    'trial_id': pl.Int64,
                    'point_id': pl.Int64,
                },
            },
            (10, 4),
            {
                'trial_id': pl.Int64, 'point_id': pl.Int64,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_dataset_example_column_schema_overrides_overrides_definition',
        ),

        pytest.param(
            'sbsat_example.csv',
            {
                'definition': datasets.SBSAT(),
            },
            (10, 5),
            {
                'book_name': pl.String, 'screen_id': pl.Int64, 'time': pl.Int64,
                'pupil_left': pl.Float32, 'pixel': pl.List(pl.Float32),
            },
            id='sbsat_dataset_example',
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
def test_from_asc_parameter_is_deprecated(filename, kwargs, make_example_file):
    filepath = make_example_file(filename)

    with pytest.raises(DeprecationWarning) as info:
        from_csv(filepath, **kwargs)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    argument_name = list(kwargs.keys())[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'keyword argument {argument_name} was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )
