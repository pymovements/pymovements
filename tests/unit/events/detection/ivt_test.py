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
"""Test functionality of the IVT algorithm."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import ivt
from pymovements.gaze.transforms import pos2vel
from pymovements.gaze.transforms_numpy import pos2vel as pos2vel_numpy
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('kwargs', 'expected_error', 'expected_message'),
    [
        pytest.param(
            {
                'velocities': None,
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must have shape \(N, 2\) but have shape \(\)',
            id='velocities_none_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': 1,
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must have shape \(N, 2\) but have shape \(\)',
            id='velocities_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': pl.ones(100, dtype=pl.Int64, eager=True),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            TypeError,
            'velocities dtype must be List but is Int64',
            id='velocities_1d_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': np.ones(100),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must have shape \(N, 2\) but have shape \(100,\)',
            id='velocities_1d_raises_value_error_numpy',
        ),
        pytest.param(
            {
                'velocities': pl.repeat((1, 2, 3), 100, eager=True),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must be 2D list but list lengths are: \[3\]',
            id='velocities_3d_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': np.ones((100, 3)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must have shape \(N, 2\) but have shape \(100, 3\)',
            id='velocities_3d_raises_value_error_numpy',
        ),
        pytest.param(
            {
                'velocities': pl.Series([[1], [1, 2], [1, 2, 3]]),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'velocities must be 2D list but list lengths are: \[1, 2, 3\]',
            id='pixels_varying_dimensions_columns_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': np.ones((100, 2)),
                'velocity_threshold': None,
                'minimum_duration': 1,
            },
            ValueError,
            'velocity threshold must not be None',
            id='velocity_threshold_none_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': np.ones((100, 2)),
                'velocity_threshold': '1.',
                'minimum_duration': 1,
            },
            TypeError,
            "'<=' not supported between instances of 'str' and 'int'",
            id='velocity_threshold_not_float_raises_type_error',
        ),
        pytest.param(
            {
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 0.,
                'minimum_duration': 1,
            },
            ValueError,
            'velocity threshold must be greater than 0',
            id='velocity_threshold_not_greater_than_0_raises_value_error',
        ),
        pytest.param(
            {
                'velocities': pl.repeat((1, 2), 100, eager=True),
                'velocity_threshold': 1.0,
                'timesteps': pl.repeat('b', 10, eager=True),
            },
            TypeError,
            r'timesteps dtype must be float or int but is String',
            id='timesteps_str_raises_type_error',
        ),
    ],
)
def test_ivt_raise_error(kwargs, expected_error, expected_message):
    """Test if ivt raises expected error."""
    with pytest.raises(expected_error, match=expected_message):
        ivt(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'positions': np.stack([np.arange(0, 200, 2), np.arange(0, 200, 2)], axis=1),
                'velocity_threshold': 1,
                'minimum_duration': 10,
            },
            Events(),
            id='constant_velocity_no_fixation',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='constant_position_single_fixation',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'name': 'custom_fixation',
            },
            Events(
                name='custom_fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='constant_position_single_fixation_custom_name',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100,
                    steps=[49, 50],
                    values=[(9, 9), (1, 1)],
                    start_value=(0, 0),
                ),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[0, 51],
                offsets=[48, 99],
            ),
            id='three_steps_two_fixations',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100, steps=[10, 20, 90],
                    values=[
                        (np.nan, np.nan),
                        (0, 0), (np.nan, np.nan),
                    ],
                ),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[0, 21],
                offsets=[9, 89],
            ),
            id='two_fixations_nan_remove_leading_ending',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100, steps=[10, 20, 90],
                    values=[
                        (np.nan, np.nan),
                        (0, 0), (np.nan, np.nan),
                    ],
                ),
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'include_nan': True,
            },
            Events(
                name='fixation',
                onsets=[0],
                offsets=[89],
            ),
            id='one_fixation_nan_remove_leading_ending',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.arange(1000, 1100, dtype=int),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps',
        ),
    ],
)
def test_ivt_detects_fixations_numpy(kwargs, expected):
    velocities = pos2vel_numpy(
        kwargs['positions'], sampling_rate=10, method='preceding',
    )

    # Just use positions argument for velocity calculation
    kwargs.pop('positions')

    events = ivt(velocities=velocities, **kwargs)

    assert_frame_equal(events.frame, expected.frame)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'positions': pl.select(
                    pl.concat_list(
                        [pl.arange(0, 200, 2, eager=True), pl.arange(0, 200, 2, eager=True)],
                    ).alias('position'),
                ),
                'velocity_threshold': 1,
                'minimum_duration': 10,
            },
            Events(),
            id='constant_velocity_no_fixation',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[1],  # polars pos2vel set undefined samples to null, numpy to 0
                offsets=[99],
            ),
            id='constant_position_single_fixation',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'name': 'custom_fixation',
            },
            Events(
                name='custom_fixation',
                onsets=[1],
                offsets=[99],
            ),
            id='constant_position_single_fixation_custom_name',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(
                        length=100,
                        steps=[49, 50],
                        values=[(9, 9), (1, 1)],
                        start_value=(0, 0),
                    ),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[1, 51],
                offsets=[48, 99],
            ),
            id='three_steps_two_fixations',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(
                        length=100,
                        steps=[10, 20, 90],
                        values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                    ),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[1, 21],
                offsets=[9, 89],
            ),
            id='two_fixations_nan_remove_leading_ending',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(
                        length=100,
                        steps=[10, 20, 90],
                        values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                    ),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'include_nan': True,
            },
            Events(
                name='fixation',
                onsets=[1],
                offsets=[89],
            ),
            id='one_fixation_nan_remove_leading_ending',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position')),
                'timesteps': pl.arange(1000, 1100, 1, eager=True),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            Events(
                name='fixation',
                onsets=[1001],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps',
        ),
    ],
)
def test_ivt_detects_fixations_polars(kwargs, expected):
    positions = kwargs.pop('positions')
    velocities = positions.select(
        pos2vel(sampling_rate=10, method='preceding', n_components=2),
    )['velocity']

    events = ivt(velocities=velocities, **kwargs)

    assert_frame_equal(events.frame, expected.frame)
