# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Test all functions in pymovements.events.microsaccades."""
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import microsaccades
from pymovements.events.detection._microsaccades import compute_threshold
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'velocities': np.zeros(shape=(10, 2)),
                'threshold': (1, 1, 1),
            },
            ValueError,
            id='non_2d_tuple_threshold_raise_value_error',
        ),
        pytest.param(
            {
                'velocities': np.random.uniform(size=(10, 2)),
                'threshold': (0, 100),
            },
            ValueError,
            id='low_variance_yaw_threshold_raise_runtime_error',
        ),
        pytest.param(
            {
                'velocities': np.random.uniform(size=(10, 2)),
                'threshold': (100, 0),
            },
            ValueError,
            id='low_variance_pitch_threshold_raise_runtime_error',
        ),
    ],
)
def test_microsaccades_raises_error(kwargs, expected):
    with pytest.raises(expected):
        microsaccades(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 10,
            },
            Events(),
            id='two_steps_one_saccade_high_threshold_no_events',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1e-5,
            },
            Events(
                name='saccade',
                onsets=[40],
                offsets=[49],
            ),
            id='two_steps_one_saccade',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1e-5,
                'name': 'custom_saccade',
            },
            Events(
                name='custom_saccade',
                onsets=[40],
                offsets=[49],
            ),
            id='two_steps_one_saccade_custom_name',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[20, 30, 70, 80],
                    values=[(9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1e-5,
            },
            Events(
                name='saccade',
                onsets=[20, 70],
                offsets=[29, 79],
            ),
            id='four_steps_two_saccades',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[20, 25, 28, 30, 70, 80],
                    values=[(9, 9), (np.nan, np.nan), (9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1,
                'include_nan': True,
            },
            Events(
                name='saccade',
                onsets=[20, 70],
                offsets=[29, 79],
            ),
            id='two_saccades_nan_delete_ending_leading_nan',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[20, 25, 28, 30, 70, 80],
                    values=[(9, 9), (np.nan, np.nan), (9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='saccade',
                onsets=[20, 28, 70],
                offsets=[24, 29, 79],
            ),
            id='three_saccades_nan_delete_ending_leading_nan',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 'std',
            },
            Events(),
            id='string_based_threshold',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'timesteps': np.arange(1000, 1100, dtype=int),
                'threshold': 1e-5,
            },
            Events(
                name='saccade',
                onsets=[1040],
                offsets=[1049],
            ),
            id='two_steps_one_saccade_timesteps',
        ),
    ],
)
def test_microsaccades_detects_saccades(kwargs, expected):
    events = microsaccades(**kwargs)

    assert_frame_equal(events.frame, expected.frame)


@pytest.mark.parametrize(
    ('params', 'expected'),
    [
        pytest.param(
            {'method': 'invalid'},
            {'exception': ValueError},
            id='invalid_method_raises_value_error',
        ),
        pytest.param(
            {'method': 'std'},
            {'value': (1.16619038, 1.16619038)},
            id='std',
        ),
        pytest.param(
            {'method': 'mad'},
            {'value': (1, 1)},
            id='mad',
        ),
        pytest.param(
            {'method': 'engbert2003'},
            {'value': np.array((1., 1.))},
            id='engbert2003',
        ),
        pytest.param(
            {'method': 'engbert2015'},
            {'value': np.array((1., 1.))},
            id='engbert2015',
        ),
    ],
)
def test_compute_threshold(params, expected):
    v = np.empty((101, 2))
    v[:, 0] = np.linspace(-2, 2, 101)
    v[:, 1] = np.linspace(-2, 2, 101)

    if 'exception' in expected:
        with pytest.raises(expected['exception']):
            compute_threshold(arr=v, **params)
    else:
        result = compute_threshold(arr=v, **params)
        assert np.allclose(result, expected['value'])
