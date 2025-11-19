# Copyright (c) 2025 The pymovements Project Authors
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
"""Test for `_metadata_to_cal_val_frames` for asc files."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.gaze.io import _metadata_to_cal_val_frames


def _cal_schema():
    return {
        'time': pl.Float64,
        'num_points': pl.Int64,
        'eye': pl.Utf8,
        'tracking_mode': pl.Utf8,
    }


def _val_schema():
    return {
        'time': pl.Float64,
        'num_points': pl.Int64,
        'eye': pl.Utf8,
        'accuracy_avg': pl.Float64,
        'accuracy_max': pl.Float64,
    }


@pytest.mark.parametrize(
    'metadata, expected_cal, expected_val',
    [
        pytest.param(
            {},
            pl.DataFrame(schema=_cal_schema()),
            pl.DataFrame(schema=_val_schema()),
            id='empty-metadata',
        ),
        pytest.param(
            {
                'calibrations': [
                    {'timestamp': '1.5', 'num_points': '9', 'tracked_eye': 'LEFT', 'type': 'P-CR'},
                    {'timestamp': 2, 'num_points': 5, 'tracked_eye': 'RIGHT', 'type': 'CR'},
                ],
            },
            pl.from_dicts([
                {'time': 1.5, 'num_points': 9, 'eye': 'left', 'tracking_mode': 'P-CR'},
                {'time': 2.0, 'num_points': 5, 'eye': 'right', 'tracking_mode': 'CR'},
            ]).with_columns([
                pl.col('time').cast(pl.Float64),
                pl.col('num_points').cast(pl.Int64),
                pl.col('eye').cast(pl.Utf8),
                pl.col('tracking_mode').cast(pl.Utf8),
            ]),
            pl.DataFrame(schema=_val_schema()),
            id='only-calibrations',
        ),
        pytest.param(
            {
                'validations': [
                    {
                        'timestamp': '3.0', 'num_points': '9', 'tracked_eye': 'LEFT',
                        'validation_score_avg': '0.25', 'validation_score_max': '0.80',
                    },
                    {
                        'timestamp': 4, 'num_points': 5, 'tracked_eye': 'RIGHT',
                        'validation_score_avg': 1.5, 'validation_score_max': 2.5,
                    },
                ],
            },
            pl.DataFrame(schema=_cal_schema()),
            pl.from_dicts([
                {
                    'time': 3.0, 'num_points': 9, 'eye': 'left',
                    'accuracy_avg': 0.25, 'accuracy_max': 0.80,
                },
                {
                    'time': 4.0, 'num_points': 5, 'eye': 'right',
                    'accuracy_avg': 1.5, 'accuracy_max': 2.5,
                },
            ]).with_columns([
                pl.col('time').cast(pl.Float64),
                pl.col('num_points').cast(pl.Int64),
                pl.col('eye').cast(pl.Utf8),
                pl.col('accuracy_avg').cast(pl.Float64),
                pl.col('accuracy_max').cast(pl.Float64),
            ]),
            id='only-validations',
        ),
        pytest.param(
            {
                'calibrations': [
                    {'timestamp': None, 'num_points': '', 'tracked_eye': 'UNKNOWN', 'type': ''},
                ],
                'validations': [
                    {
                        'timestamp': '', 'num_points': None, 'tracked_eye': None,
                        'validation_score_avg': '', 'validation_score_max': None,
                    },
                ],
            },
            pl.from_dicts([
                {'time': None, 'num_points': None, 'eye': None, 'tracking_mode': None},
            ]).with_columns([
                pl.col('time').cast(pl.Float64),
                pl.col('num_points').cast(pl.Int64),
                pl.col('eye').cast(pl.Utf8),
                pl.col('tracking_mode').cast(pl.Utf8),
            ]),
            pl.from_dicts([
                {
                    'time': None, 'num_points': None, 'eye': None,
                    'accuracy_avg': None, 'accuracy_max': None,
                },
            ]).with_columns([
                pl.col('time').cast(pl.Float64),
                pl.col('num_points').cast(pl.Int64),
                pl.col('eye').cast(pl.Utf8),
                pl.col('accuracy_avg').cast(pl.Float64),
                pl.col('accuracy_max').cast(pl.Float64),
            ]),
            id='invalid-values',
        ),
    ],
)
def test_metadata_to_cal_val_frames(metadata, expected_cal, expected_val):
    cal_df, val_df = _metadata_to_cal_val_frames(metadata)

    assert cal_df.schema == _cal_schema()
    assert val_df.schema == _val_schema()

    assert_frame_equal(cal_df, expected_cal)
    assert_frame_equal(val_df, expected_val)
