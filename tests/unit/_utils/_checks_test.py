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
"""Test pymovements _checks."""
import numpy as np
import pytest

from pymovements._utils import _checks


@pytest.mark.parametrize(
    ('variable', 'expected_error', 'expected_err_msg'),
    [
        pytest.param(5, None, '', id='non_zero_single_variable_raises_no_error'),
        pytest.param(
            0,
            ValueError,
            'variable must not be zero',
            id='zero_single_variable_raises_value_error',
        ),
        pytest.param([1, 2, 3], None, '', id='non_zero_list_raises_no_error'),
        pytest.param(
            [1, 0, 3],
            ValueError,
            'each component in variable must not be zero',
            id='zero_list_raises_value_error',
        ),
        pytest.param(np.array([1, 2, 3]), None, '', id='non_zero_np_array_raises_no_error'),
        pytest.param(
            np.array([1, 0, 3]),
            ValueError,
            'each component in variable must not be zero',
            id='zero_np_array_raises_value_error',
        ),
    ],
)
def test_check_no_zeros_raises_error(variable, expected_error, expected_err_msg):
    """Test that check_no_zeros() only raises an Exception if there are zeros in the input array."""
    if expected_error is None:
        _checks.check_no_zeros(variable)
    else:
        with pytest.raises(expected_error) as excinfo:
            _checks.check_no_zeros(variable)
        msg, = excinfo.value.args
        assert msg == expected_err_msg


# Test check_shapes_positions_velocities
@pytest.mark.parametrize(
    ('kwargs', 'expected_error', 'expected_err_msg'),
    [
        # Test for one array
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
            },
            None,
            '',
            id='one_array_with_shape_N_2_raises_no_error',
        ),
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([[1, 2], [3, 4]]),
            },
            None,
            '',
            id='two_arrays_with_shape_N_2_raises_no_error',
        ),
        # Test for one wrong array
        pytest.param(
            {
                'positions': np.array([1, 2, 3, 4]),
            },
            ValueError,
            'positions must have shape (N, 2) but have shape (4,)',
            id='array_not_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([1, 2, 3, 4]),
            },
            ValueError,
            'velocities must have shape (N, 2) but have shape (4,)',
            id='positions_shape_N_2_velocities_not_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([1, 2, 3, 4]),
                'velocities': np.array([[1, 2], [3, 4]]),
            },
            ValueError,
            'positions must have shape (N, 2) but have shape (4,)',
            id='positions_not_shape_N_2_velocities_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([1, 2, 3, 4]),
                'velocities': np.array([1, 2, 3, 4]),
            },
            ValueError,
            'positions must have shape (N, 2) but have shape (4,)',
            id='positions_and_velocities_not_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([[1, 2], [3, 4], [5, 6]]),
            },
            ValueError,
            'positions, velocities'
            ' must have the same shape, but shapes are '
            '(2, 2), (3, 2)',
            id='positions_and_velocities_N_2_but_different_lengths_raises_value_error',
        ),
    ],
)
def test_check_shapes_raises_error(kwargs, expected_error, expected_err_msg):
    """Test that check_shapes() raises an Exception.

    Only if the shapes of the positions and velocities are not (N, 2) or if the lengths of the
    positions and velocities are not equal.
    """
    if expected_error is None:
        _checks.check_shapes(**kwargs)
    else:
        with pytest.raises(expected_error) as excinfo:
            _checks.check_shapes(**kwargs)
        msg, = excinfo.value.args
        assert msg == expected_err_msg


def test_check_two_kwargs_with_three_kwargs_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _checks.check_two_kwargs(a=1, b=2, c=3)


def test_check_is_not_none_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _checks.check_is_not_none(a=1, b=None)


def test_check_is_scalar_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _checks.check_is_scalar(a=1, b=None)
