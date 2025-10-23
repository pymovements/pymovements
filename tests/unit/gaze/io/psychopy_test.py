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

from pymovements import DatasetDefinition
from pymovements.gaze import from_psychopy_csv


@pytest.mark.parametrize(
    ('kwargs', 'expected_shape', 'expected_schema'),
    [
        pytest.param(
            {
                'file': 'tests/files/psychopy_example.csv',
                'time_column': 'device_time',
                'time_unit': 'ms',
                'pixel_columns': ['left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y'],
            },
            (25, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Int64)},
            id='psychopy_csv_mono_shape',
        ),

        pytest.param(
            {
                'file': 'tests/files/psychopy_example.csv',
                'definition': DatasetDefinition(
                    time_column='device_time',
                    time_unit='ms',
                    pixel_columns=['left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y'],
                ),
            },
            (25, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Int64)},
            id='psychopy_csv_mono_shape_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/psychopy_example.csv',
                'time_column': 'device_time',
                'column_map': {
                    'left_gaze_x': 'pixel_xl',
                    'left_gaze_y': 'pixel_yl',
                    'right_gaze_x': 'pixel_xr',
                    'right_gaze_y': 'pixel_yr',
                },
                'auto_column_detect': True,
            },
            (25, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Int64)},
            id='psychopy_csv_mono_shape_auto_column_detect',
        ),
    ],
)
def test_from_psychopy_csv_gaze_has_expected_shape_and_columns(
        kwargs, expected_shape, expected_schema):
    gaze = from_psychopy_csv(**kwargs)

    assert gaze.samples.shape == expected_shape
    assert gaze.samples.schema == expected_schema
