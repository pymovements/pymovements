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

from typing import Any

import numpy as np
import polars as pl

from pymovements.transforms._utils import _check_padding
from pymovements.transforms._utils import _check_window_length
from pymovements.transforms.library import register_transform
from pymovements.transforms.savitzky_golay import savitzky_golay


@register_transform
def smooth(
        *,
        method: str,
        window_length: int,
        n_components: int,
        degree: int | None = None,
        column: str = 'position',
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """Smooth data in a column.

    Parameters
    ----------
    method: str
        The method to use for smoothing. See Notes for more details.
    window_length: int
        For ``moving_average`` this is the window size to calculate the mean of the subsequent
        samples. For ``savitzky_golay`` this is the window size to use for the polynomial fit.
        For ``exponential_moving_average`` this is the span parameter.
    n_components: int
        Number of components in the input column.
    degree: int | None
        The degree of the polynomial to use. This has only an effect if using ``savitzky_golay`` as
        smoothing method. `degree` must be less than `window_length`. (default: None)
    column: str
        The input column name to which the smoothing is applied. (default: 'position')
    padding: str | float | int | None
        Must be either ``None``, a scalar or one of the strings ``mirror``, ``nearest`` or ``wrap``.
        This determines the type of extension to use for the padded signal to
        which the filter is applied.
        When passing ``None``, no extension padding is used.
        When passing a scalar value, data will be padded using the passed value.
        See the Notes for more details on the padding methods.
        (default: 'nearest')

    Returns
    -------
    pl.Expr
        The respective polars expression.

    Notes
    -----
    The following methods are available for smoothing:

    * ``savitzky_golay``: Smooth data by applying a Savitzky-Golay filter.
    See :py:func:`~pymovements.gaze.transforms.savitzky_golay` for further details.
    * ``moving_average``: Smooth data by calculating the mean of the subsequent samples.
    Each smoothed sample is calculated by the mean of the samples in the window around the sample.
    * ``exponential_moving_average``: Smooth data by exponentially weighted moving average.

    Details on the `padding` options:

    * ``None``: No padding extension is used.
    * scalar value (int or float): The padding extension contains the specified scalar value.
    * ``mirror``: Repeats the values at the edges in reverse order. The value closest to the edge is
      not included.
    * ``nearest``: The padding extension contains the nearest input value.
    * ``wrap``: The padding extension contains the values from the other end of the array.

    Given the input is ``[1, 2, 3, 4, 5, 6, 7, 8]``, and
    `window_length` is 7, the following table shows the padded data for
    the various ``padding`` options:

    +-------------+-------------+----------------------------+-------------+
    | mode        |   padding   |           input            |   padding   |
    +=============+=============+============================+=============+
    | ``None``    | ``-  -  -`` | ``1  2  3  4  5  6  7  8`` | ``-  -  -`` |
    +-------------+-------------+----------------------------+-------------+
    | ``0``       | ``0  0  0`` | ``1  2  3  4  5  6  7  8`` | ``0  0  0`` |
    +-------------+-------------+----------------------------+-------------+
    | ``1``       | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``1  1  1`` |
    +-------------+-------------+----------------------------+-------------+
    | ``nearest`` | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``8  8  8`` |
    +-------------+-------------+----------------------------+-------------+
    | ``mirror``  | ``4  3  2`` | ``1  2  3  4  5  6  7  8`` | ``7  6  5`` |
    +-------------+-------------+----------------------------+-------------+
    | ``wrap``    | ``6  7  8`` | ``1  2  3  4  5  6  7  8`` | ``1  2  3`` |
    +-------------+-------------+----------------------------+-------------+

    """
    _check_window_length(window_length=window_length)
    _check_padding(padding=padding)

    if method in {'moving_average', 'exponential_moving_average'}:
        pad_kwargs: dict[str, Any] = {'pad_width': 0}
        pad_func = _identity

        if isinstance(padding, (int, float)):
            pad_kwargs['constant_values'] = padding
            padding = 'constant'
        elif padding == 'nearest':
            # option 'nearest' is called 'edge' for np.pad
            padding = 'edge'
        elif padding == 'mirror':
            # option 'mirror' is called 'reflect' for np.pad
            padding = 'reflect'

        if padding is not None:
            pad_kwargs['mode'] = padding
            pad_kwargs['pad_width'] = int(np.ceil(window_length / 2))
            # Create a callable that applies numpy padding and ignores any extra kwargs

            def pad_callable(x: np.ndarray, **_: Any) -> np.ndarray:
                return np.pad(x, **pad_kwargs)  # pragma: no cover
            pad_func = pad_callable
        else:
            # No padding: identity callable that ignores any extra kwargs
            def identity_callable(x: np.ndarray, **_: Any) -> np.ndarray:
                return x  # pragma: no cover
            pad_func = identity_callable

        if method == 'moving_average':

            return pl.concat_list(
                [
                    pl.col(column)
                    .list.get(component)
                    .map_batches(pad_func, return_dtype=pl.Float64)
                    .list.explode()
                    .rolling_mean(window_size=window_length, center=True)
                    .shift(n=pad_kwargs['pad_width'])
                    .slice(pad_kwargs['pad_width'] * 2)
                    for component in range(n_components)
                ],
            ).alias(column)

        return pl.concat_list(
            [
                pl.col(column)
                .list.get(component)
                .map_batches(pad_func, return_dtype=pl.Float64)
                .list.explode()
                .ewm_mean(
                    span=window_length,
                    adjust=False,
                    min_samples=window_length,
                ).shift(n=pad_kwargs['pad_width'])
                .slice(pad_kwargs['pad_width'] * 2)
                for component in range(n_components)
            ],
        ).alias(column)

    if method == 'savitzky_golay':
        if degree is None:
            raise TypeError("'degree' must not be none for method 'savitzky_golay'")

        return savitzky_golay(
            window_length=window_length,
            degree=degree,
            sampling_rate=1,
            padding=padding,
            derivative=0,
            n_components=n_components,
            input_column=column,
            output_column=None,
        )

    supported_methods = ['moving_average', 'exponential_moving_average', 'savitzky_golay']

    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: {supported_methods}",
    )


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


