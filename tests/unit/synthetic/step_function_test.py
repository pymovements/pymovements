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
"""Tests functionality of the synthetic eye gaze step function."""
import numpy as np
import pytest

from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('params', 'expected'),
    [
        pytest.param(
            {'length': 0, 'steps': [0], 'values': [0]},
            {'value': np.array([])},
            id='length_0_returns_empty_array',
        ),
        pytest.param(
            {'length': 10, 'steps': [0], 'values': [1], 'start_value': 0},
            {'value': np.ones(10)},
            id='length_10_with_step_at_start',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [0], 'start_value': 1},
            {'value': np.concatenate([np.ones(5), np.zeros(5)])},
            id='length_10_start_value_1_step_5_to_0',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [0], 'start_value': 1, 'dtype': np.int64},
            {'value': np.concatenate([np.ones(5, dtype=np.int64), np.zeros(5, dtype=np.int64)])},
            id='length_10_start_value_1_step_5_to_0_int64',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 50, 90], 'values': [1, 0, 20], 'start_value': 0},
            {'value': np.concatenate([np.zeros(10), np.ones(40), np.zeros(40), np.ones(10) * 20])},
            id='length_100_3_steps',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': 10},
            {'value': np.concatenate([np.tile(10.0, (5, 2)), np.tile((1, 2), (5, 1))])},
            id='length_10_2_channel_single_step_with_single_start_value',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': (11, 12)},
            {'value': np.concatenate([np.tile((11.0, 12), (5, 1)), np.tile((1, 2), (5, 1))])},
            id='length_10_2_channel_single_step_with_2_channel_start_value',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2, 3, 4)], 'start_value': (11, 12, 13, 14)},
            {
                'value': np.concatenate([
                    np.tile((11.0, 12, 13, 14), (5, 1)),
                    np.tile((1, 2, 3, 4), (5, 1)),
                ]),
            },
            id='length_10_4_channel_single_step_with_start_value',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 50, 90], 'values': [1, 0], 'start_value': 0},
            {'exception': ValueError},
            id='steps_values_unequal_length_raises_value_error',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 90, 50], 'values': [1, 0, 1], 'start_value': 0},
            {'exception': ValueError},
            id='steps_not_sorted_raises_value_error',
        ),
        pytest.param(
            {'length': 10, 'steps': [3, 5], 'values': [1, (2, 3)]},
            {'exception': ValueError},
            id='varying_number_of_channels_1_2_raises_value_error',
        ),
        pytest.param(
            {'length': 10, 'steps': [3, 5], 'values': [(1, 2), (3, 5, 6)]},
            {'exception': ValueError},
            id='varying_number_of_channels_2_3_raises_value_error',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': (1, 2, 3)},
            {'exception': ValueError},
            id='number_of_channels_unequal_start_value_channels_raises_value_error',
        ),
        pytest.param(
            {'length': 0, 'steps': [0], 'values': [0], 'noise': -1},
            {'exception': ValueError},
            id='negative_noise_raises_value_error',
        ),
        pytest.param(
            {'length': 4, 'steps': [2], 'values': [(np.nan, np.nan)], 'start_value': (0, 0)},
            {'dimension': (4, 2)},
            id='length_5_2_nan_value',
        ),
    ],
)
def test_step_function(params, expected):
    if 'exception' in expected:
        with pytest.raises(expected['exception']):
            step_function(**params)
        return
    if 'dimension' in expected:
        arr = step_function(**params)
        assert expected['dimension'] == arr.shape
        return

    arr = step_function(**params)
    assert np.array_equal(arr, expected['value']), f"arr = {arr}, expected = {expected['value']}"
    assert arr.dtype == expected['value'].dtype


@pytest.mark.parametrize(
    'params',
    [
        pytest.param(
            {'length': 10, 'steps': [0], 'values': [1], 'start_value': 0, 'noise': 0.1},
            id='length_10_with_step_at_start',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [0], 'start_value': 1, 'noise': 0.1},
            id='length_10_start_value_1_step_5_to_0',
        ),
        pytest.param(
            {
                'length': 100,
                'steps': [10, 50, 90],
                'values': [1, 0, 20],
                'start_value': 0,
                'noise': 0.1,
            },
            id='length_100_3_steps',
        ),
    ],
)
def test_step_function_with_noise(params):
    params_clean = {key: value for key, value in params.items() if key != 'noise'}

    arr_clean = step_function(**params_clean)
    arr_noise = step_function(**params)

    # First assert that arr is not exactly as the non-noisy output.
    assert not np.array_equal(arr_clean, arr_noise), (
        f'arr_clean = {arr_clean} must not be equal to arr_noise = {arr_noise}'
    )

    # Next check that all noisy values are still close to the clean equivalent.
    assert np.allclose(arr_clean, arr_noise, atol=params['noise'] * 5)
