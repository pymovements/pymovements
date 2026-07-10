# Copyright (c) 2023-2026 The pymovements Project Authors
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

from copy import deepcopy

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements import Experiment
from pymovements import Gaze


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


@pytest.mark.parametrize(
    'gaze',
    [
        pytest.param(
            Gaze(
                pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                experiment=None,
                position_columns=['x', 'y'],
                events=Events(
                    name='saccade',
                    onsets=[0],
                    offsets=[123],
                ),
            ),
            id='simple_events_no_trials',
        ),
        pytest.param(
            Gaze(
                pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                experiment=None,
                position_columns=['x', 'y'],
                events=Events(
                    data=pl.from_dict(
                        {
                            'trial_id': [1],
                            'name': ['saccade'],
                            'onset': [0],
                            'offset': [123],
                            'custom_property': [42],
                        },
                    ),
                    trial_columns='trial_id',
                ),
            ),
            id='events_with_trial_columns_and_custom_property',  # regression test for #1349
        ),
    ],
)
def test_gaze_copy_events(gaze):
    gaze_copy = gaze.clone()

    assert gaze_copy.events is not gaze.events
    assert_frame_equal(gaze.events.frame, gaze_copy.events.frame)


def test_gaze_copy_metadata():
    gaze_obj = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        metadata={'key': 'value', 'nested': {'inner': 42}},
    )
    gaze_copy = gaze_obj.clone()

    assert gaze_copy.metadata is not gaze_obj.metadata
    assert gaze_copy.metadata == gaze_obj.metadata

    gaze_copy.metadata['key'] = 'modified'
    assert gaze_obj.metadata['key'] == 'value'


def test_gaze_copy_metadata_default():
    gaze_obj = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )
    gaze_copy = gaze_obj.clone()

    assert gaze_copy.metadata is not gaze_obj.metadata
    assert gaze_copy.metadata == gaze_obj.metadata


def test_gaze_copy_messages():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        messages=pl.DataFrame({'time': [0, 1], 'content': ['msg1', 'msg2']}),
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.messages is not gaze.messages
    assert_frame_equal(gaze.messages, gaze_copy.messages)


def test_gaze_copy_messages_none():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        messages=None,
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.messages is None


def test_gaze_copy_trial_columns():
    gaze = Gaze(
        pl.DataFrame(
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial': pl.Int64},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        trial_columns='trial',
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.trial_columns is not gaze.trial_columns
    assert gaze_copy.trial_columns == gaze.trial_columns


def test_gaze_copy_trial_columns_none():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        trial_columns=None,
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.trial_columns is None


@pytest.mark.parametrize(
    'gaze',
    [
        pytest.param(
            Gaze(
                pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                experiment=None,
                position_columns=['x', 'y'],
            ),
            id='without_calibrations',
        ),
    ],
)
def test_gaze_copy_calibrations(gaze):
    gaze.calibrations = pl.DataFrame({'timestamp': [0], 'num_points': [9]})
    gaze_copy = gaze.clone()

    assert gaze_copy.calibrations is not gaze.calibrations
    assert_frame_equal(gaze.calibrations, gaze_copy.calibrations)


def test_gaze_copy_calibrations_none():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.calibrations is None


@pytest.mark.parametrize(
    'gaze',
    [
        pytest.param(
            Gaze(
                pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                experiment=None,
                position_columns=['x', 'y'],
            ),
            id='without_validations',
        ),
    ],
)
def test_gaze_copy_validations(gaze):
    gaze.validations = pl.DataFrame({'timestamp': [0], 'accuracy_avg': [0.5]})
    gaze_copy = gaze.clone()

    assert gaze_copy.validations is not gaze.validations
    assert_frame_equal(gaze.validations, gaze_copy.validations)


def test_gaze_copy_validations_none():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )
    gaze_copy = gaze.clone()

    assert gaze_copy.validations is None


def test_gaze_copy_n_components():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )
    gaze.n_components = 2

    gaze_copy = gaze.clone()

    assert gaze_copy.n_components == 2


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
    samples = pl.DataFrame(
        {
            'x': list(range(100)),
            'y': list(range(100)),
            'trial': [1, 2, 3, 4, 5] * 20,
        },
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


@pytest.mark.parametrize(
    ('gaze', 'by', 'expected_metadata'),
    [
        pytest.param(
            Gaze(),
            ['trial'],
            [],
            id='empty_gaze',
        ),

        pytest.param(
            Gaze(
                samples=pl.DataFrame(schema={'x': pl.Int64, 'y': pl.Int64, 'trial': pl.Int64}),
                events=None,
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            ['trial'],
            [],
            id='empty_samples_no_events_none_with_trial_columns_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            [{'trial': 1}],
            id='one_sample_no_events_one_trial_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1], 'task': ['A']}),
                pixel_columns=['x', 'y'],
            ),
            ['task', 'trial'],
            [{'trial': 1, 'task': 'A'}],
            id='one_sample_no_events_one_trial_by_two_columns',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            [{'trial': 1}, {'trial': 2}],
            id='two_samples_no_events_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, None]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            [{'trial': None}, {'trial': 1}],
            id='two_samples_no_events_one_trial_int_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': ['A', None]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            [{'trial': None}, {'trial': 'A'}],
            id='two_samples_no_events_one_trial_str_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
            ),
            ['trial'],
            [{'trial': 1}, {'trial': 2}],
            id='two_samples_no_events_with_experiment_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': range(5, 10), 'y': range(5), 'trial': [8, 5, 3, 4, 1]}),
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            ['trial'],
            [{'trial': 1}, {'trial': 3}, {'trial': 4}, {'trial': 5}, {'trial': 8}],
            id='five_samples_no_events_five_trials_single_column_trials',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                events=Events(
                    pl.DataFrame({'trial': [2], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            [{'trial': 1}, {'trial': 2}],
            id='two_samples_one_event_two_trials_by_single_column',
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
            ['trial'],
            [{'trial': None}, {'trial': 1}],
            id='two_samples_two_events_one_trial_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                events=Events(
                    pl.DataFrame({'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            ['trial'],
            [{'trial': 1}],
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
            ['trial'],
            [{'trial': 1}, {'trial': 2}],
            id='no_samples_two_events_two_trials_by_single_column',
        ),
    ],
)
def test_gaze_split_extend_metadata_correct(gaze, by, expected_metadata):
    gaze_splits = gaze.split(by=by, as_dict=True, extend_metadata=True)

    assert len(gaze_splits) == len(expected_metadata)
    for split, expected_metadata_split in zip(gaze_splits.items(), expected_metadata):
        split_key, gaze_split = split
        assert gaze_split.metadata == expected_metadata_split
        for column_name, split_key_value in zip(by, split_key):
            assert gaze_split.metadata[column_name] == split_key_value


@pytest.mark.parametrize(
    ('gaze', 'by'),
    [
        pytest.param(
            Gaze(),
            ['trial'],
            id='empty_gaze',
        ),

        pytest.param(
            Gaze(
                samples=pl.DataFrame(schema={'x': pl.Int64, 'y': pl.Int64, 'trial': pl.Int64}),
                events=None,
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            ['trial'],
            id='empty_samples_no_events_none_with_trial_columns_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            id='one_sample_no_events_one_trial_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0], 'y': [1], 'trial': [1], 'task': ['A']}),
                pixel_columns=['x', 'y'],
            ),
            ['task', 'trial'],
            id='one_sample_no_events_one_trial_by_two_columns',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            id='two_samples_no_events_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, None]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            id='two_samples_no_events_one_trial_int_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': ['A', None]}),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            id='two_samples_no_events_one_trial_str_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                pixel_columns=['x', 'y'], experiment=Experiment(1024, 768, 30, 31, 1000),
            ),
            ['trial'],
            id='two_samples_no_events_with_experiment_two_trials_by_single_column',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': range(5, 10), 'y': range(5), 'trial': [8, 5, 3, 4, 1]}),
                pixel_columns=['x', 'y'], trial_columns='trial',
            ),
            ['trial'],
            id='five_samples_no_events_five_trials_single_column_trials',
        ),

        pytest.param(
            Gaze(
                samples=pl.from_dict({'x': [0, 1], 'y': [2, 3], 'trial': [1, 2]}),
                events=Events(
                    pl.DataFrame({'trial': [2], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
                pixel_columns=['x', 'y'],
            ),
            ['trial'],
            id='two_samples_one_event_two_trials_by_single_column',
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
            ['trial'],
            id='two_samples_two_events_one_trial_one_none_by_single_column',
        ),

        pytest.param(
            Gaze(
                events=Events(
                    pl.DataFrame({'trial': [1], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                ),
            ),
            ['trial'],
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
            ['trial'],
            id='no_samples_two_events_two_trials_by_single_column',
        ),
    ],
)
def test_gaze_split_extend_metadata_false_unchanged_metadata(gaze, by):
    metadata_prior = deepcopy(gaze.metadata)
    gaze_splits = gaze.split(by=by, extend_metadata=False)

    for gaze_split in gaze_splits:
        assert gaze_split.metadata == metadata_prior


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


@pytest.mark.parametrize(
    ('gaze', 'by'),
    [
        pytest.param(
            Gaze(
                samples=pl.DataFrame(
                    {'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3], 'trial': [1, 1, 2, 2]},
                ),
                experiment=None,
                pixel_columns=['x', 'y'],
                trial_columns='trial',
                messages=pl.DataFrame({'time': [0], 'content': ['msg']}),
            ),
            'trial',
            id='with_messages',
        ),
        pytest.param(
            Gaze(
                samples=pl.DataFrame(
                    {'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3], 'trial': [1, 1, 2, 2]},
                ),
                experiment=None,
                pixel_columns=['x', 'y'],
                trial_columns='trial',
                calibrations=pl.DataFrame({'timestamp': [0], 'num_points': [9]}),
            ),
            'trial',
            id='with_calibrations',
        ),
        pytest.param(
            Gaze(
                samples=pl.DataFrame(
                    {'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3], 'trial': [1, 1, 2, 2]},
                ),
                experiment=None,
                pixel_columns=['x', 'y'],
                trial_columns='trial',
                validations=pl.DataFrame({'timestamp': [0], 'accuracy_avg': [0.5]}),
            ),
            'trial',
            id='with_validations',
        ),
    ],
)
def test_gaze_split_preserves_attributes(gaze, by):
    split_gazes = gaze.split(by=by)

    for split_gaze in split_gazes:
        assert split_gaze.messages is not None or gaze.messages is None
        assert split_gaze.calibrations is not None or gaze.calibrations is None
        assert split_gaze.validations is not None or gaze.validations is None
        assert split_gaze.trial_columns == gaze.trial_columns
        assert split_gaze.n_components == gaze.n_components
        assert split_gaze.experiment == gaze.experiment


def test_gaze_split_preserves_n_components():
    gaze = Gaze(
        samples=pl.DataFrame(
            {'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3], 'trial': [1, 1, 2, 2]},
        ),
        experiment=None,
        pixel_columns=['x', 'y'],
        trial_columns='trial',
    )
    gaze.n_components = 2

    split_gazes = gaze.split(by='trial')

    for split_gaze in split_gazes:
        assert split_gaze.n_components == 2


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


@pytest.mark.filterwarnings('ignore:No events available for processing.*:UserWarning')
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
    ('existing_amplitude', 'expected_amplitude'),
    [
        pytest.param(0.0, np.sqrt(32), id='overwrite_zero'),
        pytest.param(123.0, np.sqrt(32), id='overwrite_nonzero'),
    ],
)
def test_gaze_compute_event_properties_overwrites_column(existing_amplitude, expected_amplitude):
    gaze = Gaze(
        samples=pl.DataFrame({
            'time': [0, 1, 2, 3, 4],
            'position': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        }),
        events=Events(
            pl.DataFrame({
                'name': ['fixation'],
                'onset': [0],
                'offset': [4],
                'amplitude': [existing_amplitude],
            }),
        ),
    )

    expected_events = gaze.events.frame.with_columns(pl.lit(expected_amplitude).alias('amplitude'))

    with pytest.warns(
            UserWarning,
            match='The following columns already exist in event and will be overwritten: '
                  r'\[\'amplitude\'\]',
    ):
        gaze.compute_event_properties('amplitude')

    assert_frame_equal(gaze.events.frame, expected_events, check_column_order=False)


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

    assert_deprecation_is_removed(
        function_name=f'Gaze.{attribute}',
        warning_message=info.value.args[0],
        scheduled_version='0.28.0',
    )
