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
"""Test all Gaze functionality."""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Gaze
from pymovements import Screen


@pytest.fixture(name='make_gaze_with_events', scope='function')
def fixture_make_gaze_with_events():
    """Make a fixture function to create simple Gaze objects with event data."""

    def _make_gaze_with_events(names: list[str], properties: list[str] | None = None) -> Gaze:
        data = {
            'name': names,
            'onset': range(0, 2 * len(names), 2),
            'offset': range(1, 2 * len(names) + 1, 2),
        }
        events = Events(pl.from_dict(data))
        gaze = Gaze(events=events)

        # adding columns afterward to not count them as non-property additional_columns
        if properties is not None:
            gaze.events.frame = gaze.events.frame.select(
                [pl.all()] + [
                    pl.int_ranges(0, 100 * len(names), 100).alias(property)
                    for property in properties
                ],
            )
        return gaze

    return _make_gaze_with_events


@pytest.mark.parametrize(
    'init_arg',
    [
        pytest.param(
            None,
            id='None',
        ),
        pytest.param(
            pl.DataFrame(),
            id='no_eye_velocity_columns',
        ),
    ],
)
def test_gaze_init(init_arg):
    gaze = Gaze(init_arg)
    assert isinstance(gaze.samples, pl.DataFrame)


@pytest.mark.parametrize(
    ('init_df', 'velocity_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64}),
            ['x_right_vel', 'y_right_vel'],
            id='right_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64}),
            ['x_left_vel', 'y_left_vel'],
            id='left_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                    'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                },
            ),
            ['x_left_vel', 'y_left_vel', 'x_right_vel', 'y_right_vel'],
            id='both_eyes_velocity_columns',
        ),
    ],
)
def test_gaze_velocity_columns(init_df, velocity_columns):
    gaze = Gaze(init_df, velocity_columns=velocity_columns)

    assert 'velocity' in gaze.columns


@pytest.mark.parametrize(
    ('init_df', 'pixel_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64}),
            ['x_right_pix', 'y_right_pix'],
            id='right_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64}),
            ['x_left_pix', 'y_left_pix'],
            id='left_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64,
                },
            ),
            ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            id='both_eyes_pix_pos_columns',
        ),
    ],
)
def test_gaze_pixel_position_columns(init_df, pixel_columns):
    gaze = Gaze(init_df, pixel_columns=pixel_columns)

    assert 'pixel' in gaze.columns


@pytest.mark.parametrize(
    ('init_df', 'position_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64}),
            ['x_right_pos', 'y_right_pos'],
            id='right_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64}),
            ['x_left_pos', 'y_left_pos'],
            id='left_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64,
                    'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64,
                },
            ),
            ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            id='both_eyes_pos_columns',
        ),
    ],
)
def test_gaze_position_columns(init_df, position_columns):
    gaze = Gaze(init_df, position_columns=position_columns)

    assert 'position' in gaze.columns


@pytest.mark.parametrize(
    ('gaze_left', 'gaze_right', 'expected'),
    [
        pytest.param(
            Gaze(),
            Gaze(),
            True,
            id='empty_gaze',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
            ),
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
            ),
            True,
            id='same_samples',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34]},
                    schema={'time': pl.Int64, 'x': pl.Int64, 'y': pl.Int64},
                ),
                pixel_columns=['x', 'y'],
            ),
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34]},
                    schema={'time': pl.Float64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                pixel_columns=['x', 'y'],
            ),
            True,
            id='same_samples_int_float',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
            ),
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [10, 14]}),
                pixel_columns=['x', 'y'],
            ),
            False,
            id='different_samples',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34], 'trial': [1, 2]},
                ),
                pixel_columns=['x', 'y'],
                trial_columns='trial',
            ),
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34], 'trial': [1, 2]},
                ),
                pixel_columns=['x', 'y'],
                trial_columns='trial',
            ),
            True,
            id='same_samples_same_trial_columns',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34], 'trial': [1, 2]},
                ),
                pixel_columns=['x', 'y'],
                trial_columns='trial',
            ),
            Gaze(
                samples=pl.from_dict(
                    {'time': [0, 1], 'x': [20, 21], 'y': [30, 34], 'trial': [1, 2]},
                ),
                pixel_columns=['x', 'y'],
            ),
            False,
            id='same_samples_different_trial_columns',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
                events=Events(name=['saccade'], onsets=[0], offsets=[1]),
            ),
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
                events=Events(name=['saccade'], onsets=[0], offsets=[1]),
            ),
            True,
            id='same_samples_same_events',
        ),
        pytest.param(
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
                events=Events(name=['saccade'], onsets=[0], offsets=[1]),
            ),
            Gaze(
                samples=pl.from_dict({'time': [0, 1], 'x': [20, 21], 'y': [30, 34]}),
                pixel_columns=['x', 'y'],
                events=Events(name=['fixation'], onsets=[0], offsets=[1]),
            ),
            False,
            id='same_samples_different_events',
        ),
        pytest.param(
            Gaze(experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000)),
            Gaze(experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000)),
            True,
            id='same_experiment',
        ),
        pytest.param(
            Gaze(experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000)),
            Gaze(experiment=Experiment(1280, 1024, 38, 30, 60, 'center', 1000)),
            False,
            id='different_experiment',
        ),
    ],
)
def test_gaze_equals(gaze_left, gaze_right, expected):
    assert (gaze_left == gaze_right) == expected


def test_gaze_copy_with_experiment():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    # We want to have separate dataframes but with the exact same data.
    assert gaze.samples is not gaze_copy.samples
    assert_frame_equal(gaze.samples, gaze_copy.samples)

    # We want to have separate experiment instances but the same values.
    assert gaze.experiment is not gaze_copy.experiment
    assert gaze.experiment == gaze_copy.experiment


def test_gaze_copy_no_experiment():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    # We want to have separate dataframes but with the exact same data.
    assert gaze.samples is not gaze_copy.samples
    assert_frame_equal(gaze.samples, gaze_copy.samples)

    # We want to have separate experiment instances but the same values.
    assert gaze.experiment is gaze_copy.experiment


def test_gaze_is_copy():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    assert gaze_copy is not gaze
    assert_frame_equal(gaze.samples, gaze_copy.samples)


def test_gaze_copy_events():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            name='saccade',
            onsets=[0],
            offsets=[123],
        ),
    )

    gaze_copy = gaze.clone()

    assert gaze_copy.events is not gaze.events
    assert_frame_equal(gaze.events.frame, gaze_copy.events.frame)


def test_gaze_split_by_str():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
    )

    split_gaze = gaze.split('trial_id')
    assert all(gaze.samples.n_unique('trial_id') == 1 for gaze in split_gaze)
    assert len(split_gaze) == 3
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 0), split_gaze[0].samples)
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 1), split_gaze[1].samples)
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 2), split_gaze[2].samples)


def test_gaze_split_example():
    samples = pl.from_dict(
        {'x': range(100), 'y': range(100), 'trial': np.repeat([1, 2, 3, 4, 5], 20)},
    )
    gaze = Gaze(samples=samples, pixel_columns=['x', 'y'], trial_columns='trial')
    gazes = gaze.split(by='trial')
    assert len(gazes) == 5


@pytest.mark.parametrize(
    ('gaze', 'by', 'expected_splits'),
    [
        pytest.param(
            Gaze(),
            'trial',
            {},
            id='empty_gaze',
        ),

        pytest.param(
            Gaze(
                samples=pl.DataFrame(schema={'x': pl.Int64, 'y': pl.Int64, 'trial': pl.Int64}),
                events=None,
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            'trial',
            {},
            id='empty_samples_no_events_none_with_trial_columns_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='one_sample_no_events_one_trial_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                    pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
                ),
            },
            id='one_sample_no_events_with_experiment_one_trial_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            None,
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
            },
            id='one_sample_no_events_one_trial_by_none',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1], 'task': ['A']}),
                pixel_columns=['x', 'y'],
            ),
            ['task', 'trial'],
            {
                ('A', 1): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1], 'task': ['A']}),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='one_sample_no_events_one_trial_by_two_columns',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    pixel_columns=['x', 'y'],
                ),
                (2,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [2]}),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_no_events_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, None]}),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    pixel_columns=['x', 'y'],
                ),
                (None,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [None]}),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_no_events_one_trial_int_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': ['A', None]}),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                ('A',): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': ['A']}),
                    pixel_columns=['x', 'y'],
                ),
                (None,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [None]}),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_no_events_one_trial_str_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
                ),
                (2,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [2]}),
                    pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
                ),
            },
            id='two_samples_no_events_with_experiment_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            None,
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
                (2,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [2]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
            },
            id='two_samples_no_events_with_experiment_two_trials_by_default',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': range(5, 10), 'y': range(5), 'trial': [8, 5, 3, 4, 1]}),
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            'trial',
            {
                (8,): Gaze(
                    samples=pl.from_dict({'x': [5], 'y': [0], 'trial': [8]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
                (5,): Gaze(
                    samples=pl.from_dict({'x': [6], 'y': [1], 'trial': [5]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
                (3,): Gaze(
                    samples=pl.from_dict({'x': [7], 'y': [2], 'trial': [3]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
                (4,): Gaze(
                    samples=pl.from_dict({'x': [8], 'y': [3], 'trial': [4]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
                (1,): Gaze(
                    samples=pl.from_dict({'x': [9], 'y': [4], 'trial': [1]}),
                    pixel_columns=['x', 'y'], trial_columns='trial',
                ),
            },
            id='five_samples_no_events_five_trials_single_column_trials',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                events=Events(onsets=[0], offsets=[10], trials=[1]),
                pixel_columns=['x', 'y'],
                trial_columns=['trial'],
            ),
            None,
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                    events=Events(onsets=[0], offsets=[10], trials=[1]),
                    pixel_columns=['x', 'y'],
                    trial_columns=['trial'],
                ),
            },
            id='one_sample_one_event_same_trial_by_default',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                events=Events(onsets=[10], offsets=[100], trials=[2]),
                pixel_columns=['x', 'y'],
                trial_columns=['trial'],
            ),
            None,
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                    events=None,
                    pixel_columns=['x', 'y'],
                    trial_columns=['trial'],
                ),
                (2,): Gaze(
                    samples=pl.DataFrame(schema={'x': pl.Int64, 'y': pl.Int64, 'trial': pl.Int64}),
                    events=Events(onsets=[10], offsets=[100], trials=[2]),
                    pixel_columns=['x', 'y'],
                    trial_columns=['trial'],
                ),
            },
            id='one_sample_one_event_different_trial_by_default',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                events=Events(
                    pl.DataFrame({'trial': [2], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    pixel_columns=['x', 'y'],
                ),
                (2,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [2]}),
                    events=Events(
                        pl.DataFrame(
                            {'trial': [2], 'name': ['saccade'], 'onset': [0], 'offset': [1]},
                        ),
                    ),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_one_event_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                events=Events(
                    pl.DataFrame(
                        {
                            'trial': [1, 2], 'name': ['fixation', 'saccade'],
                            'onset': [0, 100], 'offset': [1, 200],
                        },
                    ),
                ),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    events=Events(
                        pl.DataFrame(
                            {'trial': [1], 'name': ['fixation'], 'onset': [0], 'offset': [1]},
                        ),
                    ),
                    pixel_columns=['x', 'y'],
                ),
                (2,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [2]}),
                    events=Events(
                        pl.DataFrame(
                            {'trial': [2], 'name': ['saccade'], 'onset': [100], 'offset': [200]},
                        ),
                    ),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_two_events_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, None]}),
                events=Events(
                    pl.DataFrame(
                        {
                            'trial': [1, None], 'name': ['fixation', 'saccade'],
                            'onset': [0, 100], 'offset': [1, 200],
                        },
                    ),
                ),
                pixel_columns=['x', 'y'],
            ),
            'trial',
            {
                (1,): Gaze(
                    samples=pl.from_dict({'x': [0], 'y': [2], 'trial': [1]}),
                    events=Events(
                        pl.DataFrame(
                            {'trial': [1], 'name': ['fixation'], 'onset': [0], 'offset': [1]},
                        ),
                    ),
                    pixel_columns=['x', 'y'],
                ),
                (None,): Gaze(
                    samples=pl.from_dict({'x': [1], 'y': [3], 'trial': [None]}),
                    events=Events(
                        pl.DataFrame(
                            {'trial': [None], 'name': ['saccade'], 'onset': [100], 'offset': [200]},
                        ),
                    ),
                    pixel_columns=['x', 'y'],
                ),
            },
            id='two_samples_two_events_one_trial_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                events=Events(
                    pl.DataFrame({'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            'trial',
            {
                (1,): Gaze(
                    events=Events(
                        pl.DataFrame(
                            {'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]},
                        ),
                    ),
                ),
            },
            id='no_samples_one_event_one_trial_by_single_column',
        ),

        pytest.param(
            Gaze(
                events=Events(
                    pl.DataFrame(
                        {
                            'trial': [1, 2], 'name': ['fixation', 'saccade'],
                            'onset': [0, 100], 'offset': [1, 200],
                        },
                    ),
                ),
            ),
            'trial',
            {
                (1,): Gaze(
                    events=Events(
                        pl.DataFrame(
                            {'trial': [1], 'name': ['fixation'], 'onset': [0], 'offset': [1]},
                        ),
                    ),
                ),
                (2,): Gaze(
                    events=Events(
                        pl.DataFrame(
                            {'trial': [2], 'name': ['saccade'], 'onset': [100], 'offset': [200]},
                        ),
                    ),
                ),
            },
            id='no_samples_two_events_two_trials_by_single_column',
        ),
    ],
)
def test_gaze_split_as_dict(gaze, by, expected_splits):
    gaze_splits = gaze.split(by=by, as_dict=True)
    assert gaze_splits == expected_splits


@pytest.mark.parametrize(
    ('gaze', 'by', 'expected_exception', 'expected_message'),
    [
        pytest.param(
            Gaze(),
            None,
            TypeError,
            "Either 'by' or 'Gaze.trial_columns' must be specified",
            id='empty_gaze_by_none',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'],
            ),
            'task',
            pl.exceptions.ColumnNotFoundError,
            '"task" not found',
            id='columns_missing_from_samples',
        ),

        pytest.param(
            Gaze(
                events=Events(
                    pl.DataFrame({'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            'task',
            pl.exceptions.ColumnNotFoundError,
            '"task" not found',
            id='columns_missing_from_events_no_samples',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'task': ['A']}),
                pixel_columns=['x', 'y'],
                events=Events(
                    pl.DataFrame({'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            'task',
            pl.exceptions.ColumnNotFoundError,
            '"task" not found',
            id='columns_missing_from_events_has_samples',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict(
                    {'x': [0, 2], 'y': [1, 3], 'task': [b'\x00\x10', None]},
                ),
                pixel_columns=['x', 'y'],
            ),
            'task',
            TypeError,
            'dtype bytes not supported .* in split.* supported dtypes are',
            id='none_trial_values_with_unsupported_split_column_dtype',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'task': ['A']}),
                pixel_columns=['x', 'y'],
                events=Events(
                    pl.DataFrame({'task': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            'task',
            TypeError,
            '"by" column dtypes do not match between samples and events.*str.*!=.*int',
            id='by_column_dtypes_do_not_match',
        ),

    ],
)
def test_gaze_split_as_dict_raises_exception(gaze, by, expected_exception, expected_message):
    with pytest.raises(expected_exception, match=expected_message):
        gaze.split(by=by, as_dict=True)


def test_gaze_split_by_list():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_ida': [0, 1, 1, 2],
                'trial_idb': ['a', 'b', 'c', 'c'],
            },
            schema={
                'x': pl.Float64,
                'y': pl.Float64,
                'trial_ida': pl.Int8,
                'trial_idb': pl.Utf8,
            },
        ),
        experiment=None,
        position_columns=['x', 'y'],
    )

    split_gaze = gaze.split(['trial_ida', 'trial_idb'])
    assert all(gaze.samples.n_unique(['trial_ida', 'trial_idb']) == 1 for gaze in split_gaze)
    assert len(split_gaze) == 4


def test_gaze_split_events_by_str():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )

    by = 'trial_id'
    split_gaze = gaze.split(by)
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 0), split_gaze[0].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 1), split_gaze[1].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 2), split_gaze[2].events.frame)


def test_gaze_dataframe_split_events_by_list():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_ida': [0, 1, 1, 2],
                'trial_idb': [0, 1, 2, 2],
            },
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_ida': [0, 1, 1, 2],
                    'trial_idb': [0, 1, 2, 2],
                },
            ),
        ),
    )

    by = ['trial_ida', 'trial_idb']
    split_gaze = gaze.split(by)
    assert len(split_gaze) == 4
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)


def test_gaze_dataframe_split_default():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
        trial_columns=['trial_id'],
    )

    by = 'trial_id'
    split_gaze = gaze.split()
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 0), split_gaze[0].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 1), split_gaze[1].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 2), split_gaze[2].events.frame)


def test_gaze_dataframe_split_default_no_trial_columns():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )

    with pytest.raises(TypeError):
        gaze.split()


def test_gaze_drop_event_properties(make_gaze_with_events):
    gaze = make_gaze_with_events(names=['fixation', 'saccade'], properties=['test1', 'test2'])
    gaze.drop_event_properties('test1')
    assert set(gaze.events.event_property_columns) == {'test2'}


def test_gaze_compute_event_properties_no_events():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8}),
        position_columns=['x', 'y'],
        trial_columns=['trial_id'],
    )

    with pytest.warns(
        UserWarning,
        match='No events available to compute event properties. Did you forget to use detect()?',
    ):
        gaze.compute_event_properties('amplitude')


@pytest.mark.parametrize(
    ('gaze', 'attribute'),
    [
        pytest.param(
            Gaze(),
            'frame',
            id='frame',
        ),
    ],
)
def test_dataset_definition_get_attribute_is_deprecated(gaze, attribute):
    with pytest.warns(DeprecationWarning):
        getattr(gaze, attribute)


@pytest.mark.parametrize(
    ('gaze', 'attribute', 'value'),
    [
        pytest.param(
            Gaze(),
            'frame',
            pl.DataFrame(),
            id='frame',
        ),
    ],
)
def test_gaze_set_attribute_is_deprecated(gaze, attribute, value):
    with pytest.warns(DeprecationWarning):
        setattr(gaze, attribute, value)


@pytest.mark.parametrize(
    'attribute',
    [
        'frame',
    ],
)
def test_gaze_get_attribute_is_removed(attribute, assert_deprecation_is_removed):
    definition = Gaze()
    with pytest.raises(DeprecationWarning) as info:
        getattr(definition, attribute)
    assert_deprecation_is_removed(f'Gaze.{attribute}', info.value.args[0], __version__)


def _create_gaze():
    # Creating a Gaze object
    return Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'pixel': [[260, 150], [270, 120], [271, 122], [240, 22]],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'pixel': list, 'trial_id': pl.Int8},
        ),
        experiment=Experiment(
            screen=Screen(
                width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0,
                distance_cm=68.0, origin='upper left',
            ), eyetracker=EyeTracker(
                sampling_rate=1000.0, left=None,
                right=None, model='MyModel', version=None, vendor=None, mount=None,
            ),
        ),
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )


def test_gaze_save_csv(tmp_path):

    gaze = _create_gaze()
    # Saving Gaze to tmp_path
    gaze.save(
        dirpath=tmp_path,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_feather(tmp_path):
    gaze = _create_gaze()
    # Saving Gaze to tmp_path
    gaze.save(
        dirpath=tmp_path,
        verbose=2,
        extension='feather',
    )
    assert os.path.exists(tmp_path / 'samples.feather')
    assert os.path.exists(tmp_path / 'events.feather')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_events(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirpath=tmp_path,
        save_events=False,
        verbose=2,
        extension='csv',
    )
    assert not os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_samples(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirpath=tmp_path,
        save_samples=False,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert not os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_experiment(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirpath=tmp_path,
        save_experiment=False,
        verbose=1,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_with_empty_events(tmp_path):

    gaze = _create_gaze()
    gaze.events = None

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_events=True,
            verbose=2,
            extension='csv',
        )


def test_gaze_save_wrong_extension_events(tmp_path):
    gaze = _create_gaze()

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            verbose=0,
            extension='blabla',
        )


def test_gaze_save_wrong_extension_samples(tmp_path):
    gaze = _create_gaze()

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_events=False,
            verbose=1,
            extension='blabla',
        )


def test_gaze_save_empty_experiment(tmp_path):
    gaze = _create_gaze()
    gaze.experiment = None

    gaze.save(
        dirpath=tmp_path,
        verbose=1,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_empty_experiment_true_save(tmp_path):
    gaze = _create_gaze()
    gaze.experiment = None

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_experiment=True,
            verbose=1,
            extension='csv',
        )
