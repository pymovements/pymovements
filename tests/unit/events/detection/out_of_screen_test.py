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
"""Test functionality of the out_of_screen detection algorithm."""
from __future__ import annotations

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import out_of_screen


@pytest.mark.parametrize(
    ('kwargs', 'expected_error'),
    [
        pytest.param(
            {
                'pixels': None,
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='pixels_none_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones(100),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='pixels_not_2d_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones((100, 3)),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='pixels_not_2_columns_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones((100, 2)),
                'x_min': 100, 'x_max': 100,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='x_min_equal_x_max_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones((100, 2)),
                'x_min': 200, 'x_max': 100,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='x_min_greater_than_x_max_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones((100, 2)),
                'x_min': 0, 'x_max': 1920,
                'y_min': 1080, 'y_max': 0,
            },
            ValueError,
            id='y_min_greater_than_y_max_raises_value_error',
        ),
        pytest.param(
            {
                'pixels': np.ones((10, 2)),
                'timesteps': np.arange(20, dtype=int),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            ValueError,
            id='pixels_timesteps_length_mismatch_raises_value_error',
        ),
    ],
)
def test_out_of_screen_raise_error(kwargs, expected_error):
    """Test if out_of_screen raises expected error."""
    with pytest.raises(expected_error):
        out_of_screen(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'pixels': np.array([[960, 540], [960, 540], [960, 540]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(),
            id='all_within_screen_no_events',
        ),
        pytest.param(
            {
                'pixels': np.array([[0, 0], [1919.9, 1079.9], [960, 540]]),
                'x_max': 1920, 'y_max': 1080,
            },
            Events(),
            id='on_boundary_inclusive_min_exclusive_max_no_events',
        ),
        pytest.param(
            {
                'pixels': np.array([[0, 0], [1920, 1080], [960, 540]]),
                'x_max': 1920, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[1],
                offsets=[1],
            ),
            id='at_max_boundary_exclusive_detected',
        ),
        pytest.param(
            {
                'pixels': np.array([[-1, 540], [960, 540], [960, 540]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[0],
                offsets=[0],
            ),
            id='single_sample_below_x_min',
        ),
        pytest.param(
            {
                'pixels': np.array([[960, 540], [1921, 540], [960, 540]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[1],
                offsets=[1],
            ),
            id='single_sample_above_x_max',
        ),
        pytest.param(
            {
                'pixels': np.array([[960, -1], [960, 540], [960, 540]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[0],
                offsets=[0],
            ),
            id='single_sample_below_y_min',
        ),
        pytest.param(
            {
                'pixels': np.array([[960, 540], [960, 540], [960, 1081]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[2],
                offsets=[2],
            ),
            id='single_sample_above_y_max',
        ),
        pytest.param(
            {
                'pixels': np.array([
                    [-1, 540], [-1, 540], [-1, 540],
                    [960, 540],
                    [960, 1081], [960, 1081],
                ]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[0, 4],
                offsets=[2, 5],
            ),
            id='two_consecutive_out_of_screen_events',
        ),
        pytest.param(
            {
                'pixels': np.array([
                    [-1, 540], [960, 540], [960, 1081],
                ]),
                'timesteps': np.array([1000, 1001, 1002], dtype=int),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[1000, 1002],
                offsets=[1000, 1002],
            ),
            id='with_timesteps',
        ),
        pytest.param(
            {
                'pixels': np.array([[-1, 540], [960, 540]]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
                'name': 'noise',
            },
            Events(
                name='noise',
                onsets=[0],
                offsets=[0],
            ),
            id='custom_name',
        ),
        pytest.param(
            {
                'pixels': np.array([
                    [-1, -1], [2000, 2000], [-5, 2000],
                ]),
                'x_min': 0, 'x_max': 1920,
                'y_min': 0, 'y_max': 1080,
            },
            Events(
                name='out_of_screen',
                onsets=[0],
                offsets=[2],
            ),
            id='all_out_of_screen_single_event',
        ),
    ],
)
def test_out_of_screen_detects_events(kwargs, expected):
    """Test if out_of_screen correctly detects out-of-screen events."""
    events = out_of_screen(**kwargs)

    assert_frame_equal(events.frame, expected.frame)
