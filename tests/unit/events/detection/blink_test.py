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
"""Test functionality of the blink detection algorithm."""
from __future__ import annotations

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import blink
from pymovements.events.detection._library import EventDetectionLibrary


@pytest.mark.parametrize(
    ('kwargs', 'expected_error'),
    [
        pytest.param(
            {
                'pupil': np.ones((10, 2)),
            },
            ValueError,
            id='2d_pupil_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'timesteps': np.arange(20, dtype=int),
            },
            ValueError,
            id='pupil_timesteps_length_mismatch_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'delta': -1.0,
            },
            ValueError,
            id='negative_delta_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'delta': 0.0,
            },
            ValueError,
            id='zero_delta_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'minimum_duration': 0,
            },
            ValueError,
            id='zero_minimum_duration_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'minimum_duration': -1,
            },
            ValueError,
            id='negative_minimum_duration_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'maximum_duration': 0,
            },
            ValueError,
            id='zero_maximum_duration_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'maximum_duration': -1,
            },
            ValueError,
            id='negative_maximum_duration_raises_value_error',
        ),
        pytest.param(
            {
                'pupil': np.ones(10),
                'minimum_duration': 100,
                'maximum_duration': 50,
            },
            ValueError,
            id='maximum_less_than_minimum_raises_value_error',
        ),
    ],
)
def test_blink_raise_error(kwargs, expected_error):
    """Test if blink raises expected error."""
    with pytest.raises(expected_error):
        blink(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'pupil': np.full(200, 500.0),
                'timesteps': np.arange(200, dtype=int),
            },
            Events(),
            id='constant_pupil_no_blinks',
        ),
        pytest.param(
            # Zero-pupil blink: 80 samples. Auto-delta also flags the transitions
            # (500->0 and 0->500), expanding the blink by 1 sample on each side.
            # Flagged region: samples 9..90, duration = 81.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, 0.0),
                    np.full(110, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
            },
            Events(name='blink', onsets=[9], offsets=[90]),
            id='zero_pupil_detected_as_blink',
        ),
        pytest.param(
            # NaN blink: 80 samples. NaN diffs are excluded from delta flagging,
            # so only the NaN samples themselves are flagged.
            # Flagged region: samples 10..89, duration = 79.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(110, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
            },
            Events(name='blink', onsets=[10], offsets=[89]),
            id='nan_pupil_detected_as_blink',
        ),
        pytest.param(
            # NaN blink with explicit timesteps starting at 1000.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(110, 500.0),
                ]),
                'timesteps': np.arange(1000, 1200, dtype=int),
            },
            Events(name='blink', onsets=[1010], offsets=[1089]),
            id='with_explicit_timesteps',
        ),
        pytest.param(
            # Two NaN blinks of 80 ms each, separated by 100 good samples.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(100, 500.0),
                    np.full(80, np.nan),
                    np.full(30, 500.0),
                ]),
                'timesteps': np.arange(300, dtype=int),
                'max_value_run': 0,
            },
            Events(name='blink', onsets=[10, 190], offsets=[89, 269]),
            id='two_separate_blinks',
        ),
        pytest.param(
            # Two NaN blinks with a 3-sample gap — absorbed when max_value_run=3.
            # Region 1: 10..89, gap: 90,91,92, Region 2: 93..172.
            # After absorption: single event 10..172, duration = 162.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(3, 500.0),
                    np.full(80, np.nan),
                    np.full(27, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
                'max_value_run': 3,
                'nas_around_run': 2,
            },
            Events(name='blink', onsets=[10], offsets=[172]),
            id='island_absorption_merges_nearby_events',
        ),
        pytest.param(
            # Same layout but absorption disabled — two separate blink events.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(3, 500.0),
                    np.full(80, np.nan),
                    np.full(27, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
                'max_value_run': 0,
            },
            Events(name='blink', onsets=[10, 93], offsets=[89, 172]),
            id='max_value_run_0_disables_absorption',
        ),
        pytest.param(
            # 30 ms NaN blink — below default minimum_duration=50, filtered out.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(31, np.nan),
                    np.full(159, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
                'max_value_run': 0,
            },
            Events(),
            id='minimum_duration_filters_short_events',
        ),
        pytest.param(
            # 600 ms NaN blink — above default maximum_duration=500, filtered out.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(601, np.nan),
                    np.full(89, 500.0),
                ]),
                'timesteps': np.arange(700, dtype=int),
                'max_value_run': 0,
            },
            Events(),
            id='maximum_duration_filters_long_events',
        ),
        pytest.param(
            # maximum_duration=None disables upper bound — 600 ms blink passes.
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(601, np.nan),
                    np.full(89, 500.0),
                ]),
                'timesteps': np.arange(700, dtype=int),
                'max_value_run': 0,
                'maximum_duration': None,
                'minimum_duration': 1,
            },
            Events(name='blink', onsets=[10], offsets=[610]),
            id='maximum_duration_none_disables_upper_bound',
        ),
        pytest.param(
            {
                'pupil': np.array([], dtype=float),
            },
            Events(),
            id='empty_input_no_events',
        ),
        pytest.param(
            # All NaN, 100 samples = 99 ms duration.
            {
                'pupil': np.full(100, np.nan),
                'timesteps': np.arange(100, dtype=int),
                'minimum_duration': 1,
                'maximum_duration': None,
            },
            Events(name='blink', onsets=[0], offsets=[99]),
            id='all_nan_single_blink_event',
        ),
        pytest.param(
            {
                'pupil': np.concatenate([
                    np.full(10, 500.0),
                    np.full(80, np.nan),
                    np.full(110, 500.0),
                ]),
                'timesteps': np.arange(200, dtype=int),
                'name': 'my_blink',
            },
            Events(name='my_blink', onsets=[10], offsets=[89]),
            id='custom_name_parameter',
        ),
    ],
)
def test_blink_detects_events(kwargs, expected):
    """Test if blink correctly detects blink events."""
    events = blink(**kwargs)

    assert_frame_equal(events.frame, expected.frame)


def test_blink_registered_in_library():
    """Test that blink is registered in EventDetectionLibrary."""
    assert 'blink' in EventDetectionLibrary.methods
