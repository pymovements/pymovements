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
"""Provides basic checks to be reused in other modules."""
from __future__ import annotations

from collections.abc import Sized
from typing import Any


def check_no_zeros(variable: Any, name: str = 'variable') -> None:
    """Check if variable, or if it is iterable, any of its components are zero.

    Parameters
    ----------
    variable: Any
        Variable to check for zeros.

    name: str
        Name of variable to be used in error message. (default: 'variable')
    """
    # construct error message first
    error_message = f'{name} must not be zero'

    # ducktyping check if variable is iterable
    try:
        _ = iter(variable)

    # variable is not iterable, simple check for zero
    except TypeError as exception:
        if variable == 0:
            raise ValueError(error_message) from exception

    # variable is iterable, check each element for zero
    else:
        error_message = 'each component in ' + error_message

        for variable_component in variable:
            if variable_component == 0:
                raise ValueError(error_message)


def check_shapes(**kwargs: Any) -> None:
    """Check if all provided arrays are of shape ``(N, 2)`` and shape is equal for all arrays.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary with 2 keyword arguments.

    Raises
    ------
    ValueError
        If any of the arrays is not of shape ``(N, 2)`` or if the shapes are not equal.
    """
    for key, array in kwargs.items():
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f'{key} must have shape (N, 2) but have shape {array.shape}')

    # Check if shapes are equal, printing the key of the first array that is not equal
    if not all(array.shape == next(iter(kwargs.values())).shape for array in kwargs.values()):
        raise ValueError(
            f'{", ".join(key for key in kwargs)}'
            f' must have the same shape, but shapes are '
            f'{", ".join(str(array.shape) for array in kwargs.values())}',
        )


def check_two_kwargs(**kwargs: Any) -> None:
    """Check if exactly two keyword arguments are given.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary.

    Raises
    ------
    ValueError
        If number of keyword arguments is not 2.
    """
    if len(kwargs) != 2:
        raise ValueError('there must be exactly two keyword arguments in kwargs')


def check_is_mutual_exclusive(**kwargs: Any) -> None:
    """Check if at most one of two values is not None.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary with 2 keyword arguments.

    Raises
    ------
    ValueError
        If more than one value is not None, or if number of keyword arguments is not 2.

    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if (value_1 is not None) and (value_2 is not None):
        raise ValueError(
            f'The arguments "{key_1}" and "{key_2}" are mutually exclusive.',
        )


def check_is_none_is_mutual(**kwargs: Any) -> None:
    """Check if two values are either both None or both have a value.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary with 2 keyword arguments.

    Raises
    ------
    ValueError
        If exclusively one of the keyword argument values is None, or if number of keyword arguments
        is not 2.

    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if not (value_1 is None) == (value_2 is None):
        raise ValueError(
            f'The arguments "{key_1}" and "{key_2}" must be either both None or both not None.',
        )


def check_is_length_matching(**kwargs: Sized) -> None:
    """Check if two sequences are of equal length.

    Parameters
    ----------
    **kwargs: Sized
        Keyword argument dictionary with 2 keyword arguments. Both values must be sequences.

    Raises
    ------
    ValueError
        If both sequences are of equal length , or if number of keyword arguments is not 2.
    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if not len(value_1) == len(value_2):
        raise ValueError(f'The sequences "{key_1}" and "{key_2}" must be of equal length.')


def check_is_not_none(**kwargs: Any) -> None:
    """Check if all passed values are not None.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary.

    Raises
    ------
    TypeError
        If any of the passed arguments are not None.
    """
    for key, value in kwargs.items():
        if value is None:
            raise TypeError(f"'{key}' must not be None")


def check_is_int(**kwargs: Any) -> None:
    """Check if all passed values are of type `int`.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary.

    Raises
    ------
    TypeError
        If any of the passed arguments are not of type `int`.
    """
    for key, value in kwargs.items():
        if not isinstance(value, int):
            raise TypeError(
                f"'{key}' must be of type 'int' but is of type '{type(value).__name__}'",
            )


def check_is_scalar(**kwargs: Any) -> None:
    """Check if all passed values are of type `int` or `float`.

    Parameters
    ----------
    **kwargs: Any
        Keyword argument dictionary.

    Raises
    ------
    TypeError
        If any of the passed arguments are neither of type `int` nor `float`.
    """
    for key, value in kwargs.items():
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"'{key}' must be of type 'int' or 'float' but is of type '{type(value).__name__}'",
            )


def check_is_greater_than_zero(**kwargs: float | int) -> None:
    """Check if all passed values are greater than zero.

    Parameters
    ----------
    **kwargs: float | int
        Keyword argument dictionary.

    Raises
    ------
    ValueError
        If any of the passed arguments are not greater than zero.
    """
    for key, value in kwargs.items():
        if value < 1:
            raise ValueError(f"'{key}' must be greater than zero but is {value}")


def check_is_positive_value(**kwargs: float | int) -> None:
    """Check if all passed values are have a value greater or equal to zero.

    Parameters
    ----------
    **kwargs: float | int
        Keyword argument dictionary.

    Raises
    ------
    ValueError
        If any of the passed arguments are not greater than or equal to zero.
    """
    for key, value in kwargs.items():
        if value < 0:
            raise ValueError(f"'{key}' must not be negative but is {value}")
