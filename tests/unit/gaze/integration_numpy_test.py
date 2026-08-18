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
"""Test from gaze.from_numpy."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


@pytest.mark.parametrize('argument', ['time', 'trial', 'distance'])
@pytest.mark.parametrize(
    'shape',
    [
        pytest.param((4,), id='one_dimension'),
        pytest.param((1, 4), id='leading_singleton'),
        pytest.param((4, 1), id='trailing_singleton'),
        pytest.param((1, 4, 1), id='surrounding_singletons'),
        pytest.param((4, 1, 1), id='multiple_trailing_singletons'),
    ],
)
def test_from_numpy_flattens_single_column_array_singleton_dimensions(argument, shape):
    array = np.array([101, 102, 103, 104], dtype=np.int64).reshape(shape)
    pixel = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    gaze = from_numpy(pixel=pixel, **{argument: array})

    assert gaze.samples[argument].to_list() == [101, 102, 103, 104]
    assert gaze.samples.height == 4


@pytest.mark.parametrize('argument', ['time', 'trial', 'distance'])
@pytest.mark.parametrize(
    'shape',
    [
        pytest.param((), id='zero_dimensions'),
        pytest.param((2, 4), id='two_dimensions'),
        pytest.param((2, 1, 4), id='two_dimensions_with_singleton'),
    ],
)
def test_from_numpy_single_column_array_unsupported_shape_raises(argument, shape):
    array = np.zeros(shape)

    with pytest.raises(ValueError) as error:
        from_numpy(**{argument: array})

    assert str(error.value) == (
        f'{argument} array must be at least one-dimensional and have at most one non-singleton '
        f'dimension, but got shape {shape}'
    )


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


@pytest.mark.parametrize(
    (
        'array', 'schema', 'kwargs', 'expected', 'expected_trial_columns',
    ),
    [
        pytest.param(
            np.array(
                [
                    [1, 2], [101, 102], [100, 100],
                    [0, 1], [4, 5], [9, 8], [5, 4],
                    [1, 2], [5, 6], [2, 3], [6, 7],
                ], dtype=np.float64,
            ),
            None,
            {
                'trial_columns': [0], 'time_column': 1, 'distance_column': 2,
                'pixel_columns': [3, 4], 'position_columns': [5, 6],
                'velocity_columns': [7, 8], 'acceleration_columns': [9, 10],
            },
            pl.DataFrame(
                {
                    'column_0': [1, 2], 'time': [101, 102], 'distance': [100, 100],
                    'pixel': [[0, 4], [1, 5]], 'position': [[9, 5], [8, 4]],
                    'velocity': [[1, 5], [2, 6]], 'acceleration': [[2, 6], [3, 7]],
                },
                schema={
                    'column_0': pl.Float64, 'time': pl.Int64, 'distance': pl.Float64,
                    'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64), 'acceleration': pl.List(pl.Float64),
                },
            ),
            ['column_0'],
            id='no_schema_all_indices',
        ),
        pytest.param(
            np.array(
                [
                    [1, 2], [101, 102], [100, 100],
                    [0, 1], [4, 5], [9, 8], [5, 4],
                    [1, 2], [5, 6], [2, 3], [6, 7],
                ], dtype=np.float64,
            ),
            [
                'trial_id', 't', 'd', 'x_pix', 'y_pix', 'x_pos',
                'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc',
            ],
            {
                'trial_columns': [0], 'time_column': 1, 'distance_column': 2,
                'pixel_columns': [3, 4], 'position_columns': [5, 6],
                'velocity_columns': [7, 8], 'acceleration_columns': [9, 10],
            },
            pl.DataFrame(
                {
                    'trial_id': [1, 2], 'time': [101, 102], 'distance': [100, 100],
                    'pixel': [[0, 4], [1, 5]], 'position': [[9, 5], [8, 4]],
                    'velocity': [[1, 5], [2, 6]], 'acceleration': [[2, 6], [3, 7]],
                },
                schema={
                    'trial_id': pl.Float64, 'time': pl.Int64, 'distance': pl.Float64,
                    'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64), 'acceleration': pl.List(pl.Float64),
                },
            ),
            ['trial_id'],
            id='schema_all_indices',
        ),
        pytest.param(
            np.array(
                [
                    [1, 2], [101, 102], [100, 100],
                    [0, 1], [4, 5], [9, 8], [5, 4],
                    [1, 2], [5, 6], [2, 3], [6, 7],
                ], dtype=np.float64,
            ),
            [
                'trial_id', 't', 'd', 'x_pix', 'y_pix', 'x_pos',
                'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc',
            ],
            {
                'trial_columns': 0, 'time_column': 1, 'distance_column': 2,
                'pixel_columns': [3, 4], 'position_columns': [5, 6],
                'velocity_columns': [7, 8], 'acceleration_columns': [9, 10],
            },
            pl.DataFrame(
                {
                    'trial_id': [1, 2], 'time': [101, 102], 'distance': [100, 100],
                    'pixel': [[0, 4], [1, 5]], 'position': [[9, 5], [8, 4]],
                    'velocity': [[1, 5], [2, 6]], 'acceleration': [[2, 6], [3, 7]],
                },
                schema={
                    'trial_id': pl.Float64, 'time': pl.Int64, 'distance': pl.Float64,
                    'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64), 'acceleration': pl.List(pl.Float64),
                },
            ),
            ['trial_id'],
            id='schema_single_trial_index',
        ),
        pytest.param(
            np.array(
                [
                    [0, 1], [1, 2], [101, 102], [100, 100],
                    [0, 1], [4, 5], [9, 8], [5, 4],
                    [1, 2], [5, 6], [2, 3], [6, 7],
                ], dtype=np.float64,
            ),
            [
                'trial_id_1', 'trial_id_2', 't', 'd', 'x_pix', 'y_pix', 'x_pos',
                'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc',
            ],
            {
                'trial_columns': [0, 1], 'time_column': 2, 'distance_column': 3,
                'pixel_columns': [4, 5], 'position_columns': [6, 7],
                'velocity_columns': [8, 9], 'acceleration_columns': [10, 11],
            },
            pl.DataFrame(
                {
                    'trial_id_1': [0, 1], 'trial_id_2': [1, 2], 'time': [101, 102],
                    'distance': [100, 100], 'pixel': [[0, 4], [1, 5]],
                    'position': [[9, 5], [8, 4]], 'velocity': [[1, 5], [2, 6]],
                    'acceleration': [[2, 6], [3, 7]],
                },
                schema={
                    'trial_id_1': pl.Float64, 'trial_id_2': pl.Float64, 'time': pl.Int64,
                    'distance': pl.Float64, 'pixel': pl.List(pl.Float64),
                    'position': pl.List(pl.Float64), 'velocity': pl.List(pl.Float64),
                    'acceleration': pl.List(pl.Float64),
                },
            ),
            ['trial_id_1', 'trial_id_2'],
            id='schema_multiple_trial_indices',
        ),
        pytest.param(
            np.array(
                [
                    [1, 2], [101, 102], [100, 100],
                    [0, 1], [4, 5], [9, 8], [5, 4],
                    [1, 2], [5, 6], [2, 3], [6, 7],
                ], dtype=np.float64,
            ),
            [
                'trial_id', 't', 'd', 'x_pix', 'y_pix', 'x_pos',
                'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc',
            ],
            {
                'trial_columns': 'trial_id', 'time_column': 't', 'distance_column': 'd',
                'pixel_columns': [3, 4], 'position_columns': [5, 6],
                'velocity_columns': [7, 8], 'acceleration_columns': [9, 10],
            },
            pl.DataFrame(
                {
                    'trial_id': [1, 2], 'time': [101, 102], 'distance': [100, 100],
                    'pixel': [[0, 4], [1, 5]], 'position': [[9, 5], [8, 4]],
                    'velocity': [[1, 5], [2, 6]], 'acceleration': [[2, 6], [3, 7]],
                },
                schema={
                    'trial_id': pl.Float64, 'time': pl.Int64, 'distance': pl.Float64,
                    'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64), 'acceleration': pl.List(pl.Float64),
                },
            ),
            ['trial_id'],
            id='schema_mixed_indices_and_names',
        ),
    ],
)
def test_from_numpy_with_column_indices(
        array, schema, kwargs, expected, expected_trial_columns,
):
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
        time_unit='ms',
        **kwargs,
    )

    assert_frame_equal(gaze.samples, expected.select(gaze.samples.columns))
    assert gaze.n_components == 2
    assert gaze.trial_columns == expected_trial_columns


def test_from_numpy_mixed_indices_and_names():
    array = np.array(
        [
            [101, 102],
            [1, 2],
            [3, 4],
        ], dtype=np.float64,
    )
    schema = ['time', 'x', 'y']

    gaze = from_numpy(
        samples=array,
        schema=schema,
        time_column='time',
        position_columns=['x', 2],  # 2 refers to 'y'
    )

    assert gaze.columns == ['time', 'position']
    assert gaze.samples['position'][0].to_list() == [1, 3]


def test_from_numpy_out_of_range_index_raises():
    array = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError, match='column index 5 is out of bounds for 2 columns'):
        from_numpy(samples=array, time_column=5)


def test_from_numpy_bool_index_raises():
    array = np.array([[10, 20], [30, 40]])
    schema = ['col0', 'col1']

    expected_msg = 'column specifiers must be of type int or str but got bool'
    with pytest.raises(TypeError, match=expected_msg):
        from_numpy(samples=array, schema=schema, time_column=True)


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
def test_from_numpy_negative_index():
    array = np.array([[10, 40], [20, 50], [30, 60]])  # 3 columns if orient='col'
    schema = ['col0', 'col1', 'col2']

    # -1 should be col2
    gaze = from_numpy(samples=array, schema=schema, time_column=-1, orient='col')
    assert 'time' in gaze.samples.columns
    assert gaze.samples['time'][0] == 30


@pytest.mark.parametrize(
    ('kwargs', 'expected_msg'),
    [
        pytest.param(
            {'time_column': 0},
            'time_column can only be used when samples is provided',
            id='time_column',
        ),
        pytest.param(
            {'pixel_columns': [0, 1]},
            'pixel_columns can only be used when samples is provided',
            id='pixel_columns',
        ),
        pytest.param(
            {'position_columns': [0, 1]},
            'position_columns can only be used when samples is provided',
            id='position_columns',
        ),
        pytest.param(
            {'velocity_columns': [0, 1]},
            'velocity_columns can only be used when samples is provided',
            id='velocity_columns',
        ),
        pytest.param(
            {'acceleration_columns': [0, 1]},
            'acceleration_columns can only be used when samples is provided',
            id='acceleration_columns',
        ),
        pytest.param(
            {'distance_column': 0},
            'distance_column can only be used when samples is provided',
            id='distance_column',
        ),
        pytest.param(
            {'trial_columns': 0},
            'trial_columns can only be used when samples is provided',
            id='trial_columns',
        ),
    ],
)
def test_from_numpy_columns_provided_without_samples_raises(kwargs, expected_msg):
    time = np.array([1, 2, 3])
    with pytest.raises(ValueError, match=expected_msg):
        from_numpy(time=time, **kwargs)


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
