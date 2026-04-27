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
"""Tests functionality of the IDT algorithm."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import idt
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('kwargs', 'expected_error', 'expected_message'),
    [
        pytest.param(
            {
                'positions': None,
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must have shape \(N, 2\) but have shape \(\)',
            id='positions_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': None,
                'minimum_duration': 1,
            },
            TypeError,
            "'<=' not supported between instances of 'NoneType' and 'int'",
            id='dispersion_threshold_none_raises_type_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': None,
            },
            TypeError,
            "'<=' not supported between instances of 'NoneType' and 'int'",
            id='duration_threshold_none_raises_type_error',
        ),
        pytest.param(
            {
                'positions': 1,
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must have shape \(N, 2\) but have shape \(\)',
            id='positions_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': pl.arange(10, eager=True),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            TypeError,
            'positions dtype must be List but is Int64',
            id='positions_1d_raises_type_error',
        ),
        pytest.param(
            {
                'positions': [1, 2, 3],
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must have shape \(N, 2\) but have shape \(3,\)',
            id='positions_1d_raises_value_error_list',
        ),
        pytest.param(
            {
                'positions': np.array([1, 2, 3]),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must have shape \(N, 2\) but have shape \(3,\)',
            id='positions_1d_raises_value_error_numpy',
        ),
        pytest.param(
            {
                'positions': pl.repeat((1, 2, 3), 100, eager=True),
                'dispersion_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must be 2D list but list lengths are: \[3\]',
            id='positions_3d_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2, 3], [1, 2, 3]],
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must have shape \(N, 2\) but have shape \(2, 3\)',
            id='positions_3d_raises_value_error_list',
        ),
        pytest.param(
            {
                'positions': pl.Series([[1], [1, 2], [1, 2, 3]]),
                'dispersion_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            r'positions must be 2D list but list lengths are: \[1, 2, 3\]',
            id='positions_varying_dimensions_columns_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 0,
                'minimum_duration': 1,
            },
            ValueError,
            'dispersion_threshold must be greater than 0',
            id='dispersion_threshold_not_greater_than_0_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': 0,
            },
            ValueError,
            'minimum_duration must be greater than 0',
            id='duration_threshold_not_greater_than_0_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': 1.1,
            },
            TypeError,
            'minimum_duration must be of type int but is of type',
            id='duration_threshold_not_integer_raises_type_error',
        ),
    ],
)
def test_idt_raises_error(kwargs, expected_error, expected_message):
    """Test if idt raises expected error."""
    with pytest.raises(expected_error, match=expected_message):
        idt(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'positions': pl.select(
                    pl.concat_list(
                        [pl.arange(0, 200, 2, eager=True), pl.arange(0, 200, 2, eager=True)],
                    ).alias('position'),
                )['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 10,
            },
            Events(),
            id='constant_velocity_no_fixation',
        ),
        pytest.param(
            {
                'positions': np.stack([np.arange(0, 200, 2), np.arange(0, 200, 2)], axis=1),
                'dispersion_threshold': 1,
                'minimum_duration': 10,
            },
            Events(),
            id='constant_velocity_no_fixation_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 2,
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
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='constant_position_single_fixation_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 2,
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
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
                'name': 'custom_fixation',
            },
            Events(
                name='custom_fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='constant_position_single_fixation_custom_name_numpy',
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
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[0, 50],
                offsets=[49, 99],
            ),
            id='three_steps_two_fixations',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100,
                    steps=[49, 50],
                    values=[(9, 9), (1, 1)],
                    start_value=(0, 0),
                ),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[0, 50],
                offsets=[49, 99],
            ),
            id='three_steps_two_fixations_numpy',
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
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[0, 20],
                offsets=[9, 89],
            ),
            id='two_fixations_nan_delete_leading_ending',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100, steps=[10, 20, 90],
                    values=[
                        (np.nan, np.nan), (0, 0),
                        (np.nan, np.nan),
                    ],
                ),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[0, 20],
                offsets=[9, 89],
            ),
            id='two_fixations_nan_delete_leading_ending_numpy',
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
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'dispersion_threshold': 1,
                'minimum_duration': 2,
                'include_nan': True,
            },
            Events(
                name='fixation',
                onsets=[0],
                offsets=[89],
            ),
            id='one_fixation_nan_delete_leading_ending',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100,
                    steps=[10, 20, 90],
                    values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                ),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
                'include_nan': True,
            },
            Events(
                name='fixation',
                onsets=[0],
                offsets=[89],
            ),
            id='one_fixation_nan_delete_leading_ending_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'timesteps': np.arange(1000, 1100, dtype=int),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.arange(1000, 1100, dtype=int),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'timesteps': np.reshape(np.arange(1000, 1100, dtype=int), (100, 1)),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_extra_dim',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.reshape(np.arange(1000, 1100, dtype=int), (100, 1)),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_extra_dim_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'timesteps': np.arange(1000, 1100, dtype=float),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_float_no_decimal',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.arange(1000, 1100, dtype=float),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_float_no_decimal_numpy',
        ),
        pytest.param(
            {
                'positions': pl.from_numpy(
                    step_function(length=100, steps=[0], values=[(0, 0)]),
                    schema=['x', 'y'],
                ).select(pl.concat_list(['x', 'y']).alias('position'))['position'],
                'timesteps': np.arange(1000, 1010, 0.1, dtype=float),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_float_with_decimal',
            marks=pytest.mark.xfail(reason='#532'),
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.arange(1000, 1010, 0.1, dtype=float),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            Events(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='constant_position_single_fixation_with_timesteps_float_with_decimal_numpy',
            marks=pytest.mark.xfail(reason='#532'),
        ),
    ],
)
def test_idt_detects_fixations(kwargs, expected):
    """Test if idt detects fixations."""
    events = idt(**kwargs)

    assert_frame_equal(events.frame, expected.frame)


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'exception_message'),
    [
        pytest.param(
            {
                'positions': step_function(length=10, steps=[0], values=[(0, 0)]),
                'timesteps': np.concatenate([
                    np.arange(0, 5, dtype=int), np.arange(7, 12, dtype=int),
                ]),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            'interval .* timesteps .* constant',
            id='non_constant_timesteps_interval',
        ),
        pytest.param(
            {
                'positions': step_function(length=10, steps=[0], values=[(0, 0)]),
                'timesteps': np.arange(0, 30, step=3, dtype=int),
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            ValueError,
            'minimum_duration must be divisible by the constant interval between timesteps',
            id='minimum_duration_not_divisible_by_timesteps_interval',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'timesteps': np.linspace(0, 1, 100),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            TypeError,
            'timesteps .* int',
            id='constant_position_single_fixation_with_timesteps_float_with_fractions',
        ),
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            'minimum_duration .* 2',
            id='minimum_duration_1_sample',
        ),
        pytest.param(
            {
                'positions': pl.repeat((1, 2), 100, eager=True),
                'dispersion_threshold': 1.0,
                'timesteps': pl.repeat('b', 10, eager=True),
            },
            TypeError,
            r'timesteps dtype must be float or int but is String',
            id='timesteps_str_raises_type_error',
        ),
    ],
)
def test_idt_timesteps_exceptions(kwargs, exception, exception_message):
    with pytest.raises(exception, match=exception_message):
        idt(**kwargs)
