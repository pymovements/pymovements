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
"""Test from gaze.from_numpy."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events
from pymovements import Experiment
from pymovements.gaze import from_numpy


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
def test_from_numpy():
    array = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )

    schema = ['x_pix', 'y_pix', 'x_pos', 'y_pos']

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = from_numpy(
        samples=array,
        schema=schema,
        experiment=experiment,
    )

    assert gaze.samples.shape == (4, 5)
    assert gaze.columns == schema + ['time']  # expected schema includes additional time column


def test_from_numpy_with_schema():
    array = np.array(
        [
            [101, 102, 103, 104],
            [100, 100, 100, 100],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [9, 8, 7, 6],
            [5, 4, 3, 2],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=np.float64,
    )

    schema = ['t', 'd', 'x_pix', 'y_pix', 'x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=None,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = from_numpy(
        samples=array,
        schema=schema,
        experiment=experiment,
        time_column='t',
        time_unit='ms',
        distance_column='d',
        pixel_columns=['x_pix', 'y_pix'],
        position_columns=['x_pos', 'y_pos'],
        velocity_columns=['x_vel', 'y_vel'],
        acceleration_columns=['x_acc', 'y_acc'],
    )

    expected = pl.DataFrame(
        {
            'time': [101, 102, 103, 104],
            'distance': [100, 100, 100, 100],
            'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
            'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
            'velocity': [[1, 5], [2, 6], [3, 7], [4, 8]],
            'acceleration': [[2, 6], [3, 7], [4, 8], [5, 9]],
        },
        schema={
            'time': pl.Int64,
            'distance': pl.Float64,
            'pixel': pl.List(pl.Float64),
            'position': pl.List(pl.Float64),
            'velocity': pl.List(pl.Float64),
            'acceleration': pl.List(pl.Float64),
        },
    )

    assert_frame_equal(gaze.samples, expected)
    assert gaze.n_components == 2


def test_from_numpy_with_trial_id():
    array = np.array(
        [
            [1, 1, 2, 2],
            [101, 102, 103, 104],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=np.float64,
    )

    schema = ['trial_id', 't', 'x_pix', 'y_pix']

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=None,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = from_numpy(
        samples=array,
        schema=schema,
        experiment=experiment,
        trial_columns='trial_id',
        time_column='t',
        pixel_columns=['x_pix', 'y_pix'],
    )

    expected = pl.DataFrame(
        {
            'trial_id': [1, 1, 2, 2],
            'time': [101, 102, 103, 104],
            'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
        },
        schema={
            'trial_id': pl.Float64,
            'time': pl.Int64,
            'pixel': pl.List(pl.Float64),
        },
    )

    assert_frame_equal(gaze.samples, expected)
    assert gaze.n_components == 2
    assert gaze.trial_columns == ['trial_id']


def test_from_numpy_explicit_columns():
    time = np.array([101, 102, 103, 104], dtype=np.int64)
    distance = np.array([100, 100, 100, 100], dtype=np.float64)
    pixel = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    position = np.array([[9, 8, 7, 6], [5, 4, 3, 2]], dtype=np.float64)
    velocity = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)
    acceleration = np.array([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=np.float64)

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=None,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = from_numpy(
        time=time,
        time_unit='ms',
        distance=distance,
        pixel=pixel,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        experiment=experiment,
    )

    expected = pl.DataFrame(
        {
            'time': [101, 102, 103, 104],
            'distance': [100, 100, 100, 100],
            'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
            'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
            'velocity': [[1, 5], [2, 6], [3, 7], [4, 8]],
            'acceleration': [[2, 6], [3, 7], [4, 8], [5, 9]],
        },
        schema={
            'time': pl.Int64,
            'distance': pl.Float64,
            'pixel': pl.List(pl.Int64),
            'position': pl.List(pl.Float64),
            'velocity': pl.List(pl.Float64),
            'acceleration': pl.List(pl.Float64),
        },
    )

    assert_frame_equal(gaze.samples, expected)
    assert gaze.n_components == 2


def test_from_numpy_explicit_columns_with_trial():
    trial = np.array([1, 1, 2, 2], dtype=np.int64)
    time = np.array([101, 102, 103, 104], dtype=np.int64)
    pixel = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    gaze = from_numpy(
        trial=trial,
        time=time,
        pixel=pixel,
    )

    expected = pl.DataFrame(
        {
            'trial': [1, 1, 2, 2],
            'time': [101, 102, 103, 104],
            'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
        },
        schema={
            'trial': pl.Int64,
            'time': pl.Int64,
            'pixel': pl.List(pl.Int64),
        },
    )

    assert_frame_equal(gaze.samples, expected)
    assert gaze.n_components == 2
    assert gaze.trial_columns == ['trial']


def test_from_numpy_all_none():
    gaze = from_numpy(
        samples=None,
        schema=None,
        experiment=None,
        time=None,
        pixel=None,
        position=None,
        velocity=None,
        acceleration=None,
        time_column=None,
        pixel_columns=None,
        position_columns=None,
        velocity_columns=None,
        acceleration_columns=None,
    )

    expected = pl.DataFrame()

    assert_frame_equal(gaze.samples, expected)
    assert gaze.n_components is None


@pytest.mark.parametrize(
    'events',
    [
        pytest.param(
            None,
            id='events_none',
        ),

        pytest.param(
            Events(),
            id='events_empty',
        ),

        pytest.param(
            Events(name='fixation', onsets=[123], offsets=[345]),
            id='fixation',
        ),

        pytest.param(
            Events(name='saccade', onsets=[34123], offsets=[67345]),
            id='saccade',
        ),

    ],
)
def test_from_numpy_events(events):
    if events is None:
        expected_events = Events().frame
    else:
        expected_events = events.frame

    gaze = from_numpy(events=events)

    assert_frame_equal(gaze.events.frame, expected_events)
    # We don't want the events point to the same reference.
    assert gaze.events.frame is not expected_events


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
def test_from_numpy_data_argument_is_deprecated():
    array = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    schema = ['x_pix', 'y_pix', 'x_pos', 'y_pos']

    with pytest.warns(DeprecationWarning):
        gaze = from_numpy(data=array, schema=schema)

    assert gaze.samples.shape == (4, 4)


def test_from_numpy_data_argument_is_removed(assert_deprecation_is_removed):
    array = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    schema = ['x_pix', 'y_pix', 'x_pos', 'y_pos']

    with pytest.raises(DeprecationWarning) as info:
        from_numpy(data=array, schema=schema)
    assert_deprecation_is_removed(
        'from_numpy() keyword argument "data"', info.value.args[0], __version__,
    )
