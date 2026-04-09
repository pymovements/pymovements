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
"""Module for py:func:`pymovements.gaze.transforms."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from pymovements._utils import _checks


def _apply_on_columns(
        frame: pl.DataFrame,
        columns: list[str],
        transformation: Callable,
        n_components: int | None = None,
        return_dtype: pl.DataType | None = None,
) -> pl.DataFrame:
    """Apply a function on nested and normal columns of a DataFrame.

    Parameters
    ----------
    frame: pl.DataFrame
        The DataFrame to apply the function on.
    columns: list[str]
        The columns to apply the function on. Must be numeric columns.
    transformation: Callable
        The function to apply on the specified columns.
    n_components: int | None
        Number of components of nested columns in columns. (default: None)
    return_dtype: pl.DataType | None
        The data type to return for the transformed columns. (default: None)

    Returns
    -------
    pl.DataFrame
        The DataFrame with the function applied on the specified columns.

    Raises
    ------
    ValueError
        If n_components is not specified when nested columns are present.
    """
    for column in columns:
        # Determine if the column is nested based on its data type
        if isinstance(frame.schema[column], pl.List):

            # Raise an error if n_components is not specified for nested columns
            if n_components is None:
                raise ValueError(
                    f'n_components must be specified when processing nested column {column}',
                )

            # Apply the function on the nested components separately
            frame = frame.with_columns(
                pl.concat_list(
                    [
                        pl.col(column)
                        .list.get(component)
                        .map_batches(
                            transformation,
                            # If an override is provided, prefer it. For a list column we expect
                            # the override to describe the inner element type.
                            return_dtype=(
                                return_dtype
                                if return_dtype is not None
                                else (
                                    frame.schema[column].inner
                                    if hasattr(frame.schema[column], 'inner')
                                    else pl.Float64
                                )
                            ),
                        )
                        for component in range(n_components)
                    ],
                ).alias(column),
            )
        else:
            frame = frame.with_columns(
                pl.col(column).map_batches(
                    transformation,
                    return_dtype=(return_dtype or frame.schema[column]),
                ).alias(column),
            )

    return frame


def _identity(x: Any) -> Any:
    """Identity function as placeholder for None as padding.

    Parameters
    ----------
    x: Any
        The value to return.

    Returns
    -------
    Any
        The value passed to the function.
    """
    return x


def _check_window_length(window_length: Any) -> None:
    """Check that window length is an integer and greater than zero.

    Parameters
    ----------
    window_length: Any
        The window length to check.
    """
    _checks.check_is_not_none(window_length=window_length)
    _checks.check_is_int(window_length=window_length)
    _checks.check_is_greater_than_zero(degree=window_length)


def _check_degree(degree: Any, window_length: int) -> None:
    """Check that polynomial degree is an integer, greater than zero and less than window_length.

    Parameters
    ----------
    degree: Any
        The degree to check.

    window_length: int
        The window length to check against.
    """
    _checks.check_is_not_none(degree=degree)
    _checks.check_is_int(degree=degree)
    _checks.check_is_greater_than_zero(degree=degree)

    if degree >= window_length:
        raise ValueError("'degree' must be less than 'window_length'")


def _check_padding(padding: Any) -> None:
    """Check if padding argument is valid.

    Parameters
    ----------
    padding: Any
        The padding to check.

    """
    if not isinstance(padding, (float, int, str)) and padding is not None:
        raise TypeError(
            f"'padding' must be of type 'str', 'int', 'float' or None"
            f"' but is of type '{type(padding).__name__}'",
        )

    if isinstance(padding, str):
        supported_padding_modes = ['nearest', 'mirror', 'wrap']
        if padding not in supported_padding_modes:
            raise ValueError(
                f"Invalid 'padding' value '{padding}'."
                'Choose a valid padding string, a scalar, or None.'
                f' Valid padding strings are: {supported_padding_modes}',
            )


def _check_derivative(derivative: Any) -> None:
    """Check that derivative has a positive integer value.

    Parameters
    ----------
    derivative: Any
        The derivative to check.
    """
    _checks.check_is_int(derivative=derivative)
    _checks.check_is_positive_value(derivative=derivative)


def _check_distance(distance: float) -> None:
    """Check if all screen values are scalars and are greater than zero.

    Parameters
    ----------
    distance: float
        The distance to check.
    """
    _checks.check_is_scalar(distance=distance)
    _checks.check_is_greater_than_zero(distance=distance)


def _check_screen_resolution(screen_resolution: tuple[int, int]) -> None:
    """Check screen resolution value.

    Parameters
    ----------
    screen_resolution: tuple[int, int]
        The screen resolution to check.
    """
    if screen_resolution is None:
        raise TypeError('screen_resolution must not be None')

    if not isinstance(screen_resolution, (tuple, list)):
        raise TypeError(
            'screen_resolution must be of type tuple[int, int],'
            f' but is of type {type(screen_resolution).__name__}',
        )

    if len(screen_resolution) != 2:
        raise ValueError(
            f'screen_resolution must have length of 2, but is of length {len(screen_resolution)}',
        )

    for element in screen_resolution:
        _checks.check_is_scalar(screen_resolution=element)
        _checks.check_is_greater_than_zero(screen_resolution=element)


def _check_screen_size(screen_size: tuple[float, float]) -> None:
    """Check screen size value.

    Parameters
    ----------
    screen_size: tuple[float, float]
        The screen size to check.
    """
    if screen_size is None:
        raise TypeError('screen_size must not be None')

    if not isinstance(screen_size, (tuple, list)):
        raise TypeError(
            'screen_size must be of type tuple[int, int],'
            f' but is of type {type(screen_size).__name__}',
        )

    if len(screen_size) != 2:
        raise ValueError(f'screen_size must have length of 2, but is of length {len(screen_size)}')

    for element in screen_size:
        _checks.check_is_scalar(screen_size=element)
        _checks.check_is_greater_than_zero(screen_size=element)
