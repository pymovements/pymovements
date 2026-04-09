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

from functools import partial
from typing import Any

import polars as pl
import scipy

from pymovements._utils import _checks
from pymovements.transforms._utils import _check_padding
from pymovements.transforms._utils import _check_window_length
from pymovements.transforms.library import register_transform


@register_transform
def savitzky_golay(
        *,
        window_length: int,
        degree: int,
        sampling_rate: float,
        n_components: int,
        input_column: str,
        output_column: str | None = None,
        derivative: int = 0,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """Apply a 1-D Savitzky-Golay filter to a column|_|:cite:p:`SavitzkyGolay1964`.

    Parameters
    ----------
    window_length: int
        The length of the filter window (i.e., the number of coefficients).
        If `padding` is ``None``, `window_length` must be less than or equal
        to the length of the input.
    degree: int
        The degree of the polynomial used to fit the samples.
        `degree` must be less than `window_length`.
    sampling_rate: float
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    n_components: int
        Number of components in input column.
    input_column: str
        The input column name.
    output_column: str | None
        The output column name. (default: None)
    derivative: int
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating. (default: 0)
    padding: str | float | int | None
        Must be either ``None``, a scalar or one of the strings ``mirror``, ``nearest`` or ``wrap``.
        This determines the type of extension to use for the padded signal to
        which the filter is applied.
        When passing ``None``, no extension padding is used. Instead, a degree `degree` polynomial
        is fit to the last ``window_length`` values of the edges, and this polynomial is used to
        evaluate the last ``window_length // 2`` output values.
        When passing a scalar value, data will be padded using the passed value.
        See the Notes for more details on the padding methods ``mirror``, ``nearest`` or ``wrap``.
        (default: 'nearest')

    Returns
    -------
    pl.Expr
        The respective polars expression

    Notes
    -----
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
    _check_degree(degree=degree, window_length=window_length)
    _check_derivative(derivative=derivative)
    _check_padding(padding=padding)

    if output_column is None:
        output_column = input_column

    delta = 1 / sampling_rate

    constant_value = 0.0
    if isinstance(padding, (int, float)):
        constant_value = padding
        padding = 'constant'
    elif padding is None:
        padding = 'interp'

    func = partial(
        scipy.signal.savgol_filter,
        window_length=window_length,
        polyorder=degree,
        deriv=derivative,
        delta=delta,
        axis=0,
        mode=padding,
        cval=constant_value,
    )

    return pl.concat_list(
        [
            pl.col(input_column)
            .list.get(component)
            .map_batches(func, return_dtype=pl.Float64)
            .list.explode()
            for component in range(n_components)
        ],
    ).alias(output_column)


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


def _check_derivative(derivative: Any) -> None:
    """Check that derivative has a positive integer value.

    Parameters
    ----------
    derivative: Any
        The derivative to check.
    """
    _checks.check_is_int(derivative=derivative)
    _checks.check_is_positive_value(derivative=derivative)
