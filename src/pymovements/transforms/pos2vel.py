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

import polars as pl

from pymovements.transforms.library import register_transform
from pymovements.transforms.savitzky_golay import savitzky_golay

@register_transform
def pos2vel(
        *,
        sampling_rate: float,
        method: str,
        n_components: int,
        degree: int | None = None,
        window_length: int | None = None,
        padding: str | float | int | None = 'nearest',
        position_column: str = 'position',
        velocity_column: str = 'velocity',
) -> pl.Expr:
    """Compute velocitiy data from positional data.

    Parameters
    ----------
    sampling_rate: float
        Sampling rate of input time series.
    method: str
        The method to use for velocity calculation.
    n_components: int
        Number of components in the input column.
    degree: int | None
        The degree of the polynomial to use. This has only an effect if using ``savitzky_golay`` as
        calculation method. (default: None)
    window_length: int | None
        The window size to use. This has only an effect if using ``savitzky_golay`` as calculation
        method. (default: None)
    padding: str | float | int | None
        The padding to use.  This has only an effect if using ``savitzky_golay`` as calculation
        method. (default: 'nearest')
    position_column: str
        The input position column name. (default: 'position')
    velocity_column: str
        The output velocity column name. (default: 'velocity')

    Returns
    -------
    pl.Expr
        The respective polars expression.

    Notes
    -----
    There are three methods available for velocity calculation:

    * ``savitzky_golay``: velocity is calculated by a polynomial of fixed degree and window length.
      See :py:func:`~pymovements.gaze.transforms.savitzky_golay` for further details.
    * ``fivepoint``: velocity is calculated from the difference of the mean values
      of the subsequent two samples and the preceding two samples
    * ``neighbors``: velocity is calculated from the difference of the subsequent
      sample and the preceding sample
    * ``preceding``: velocity is calculated from the difference of the current
      sample to the preceding sample
    """
    if method == 'preceding':
        return pl.concat_list(
            [
                pl.col(position_column).list.get(component)
                .diff(n=1, null_behavior='ignore') * sampling_rate
                for component in range(n_components)
            ],
        ).alias(velocity_column)

    if method == 'neighbors':
        return pl.concat_list(
            [
                (
                    pl.col(position_column).shift(n=-1).list.get(component)
                    - pl.col(position_column).shift(n=1).list.get(component)
                ) * (sampling_rate / 2)
                for component in range(n_components)
            ],
        ).alias(velocity_column)

    if method in {'fivepoint', 'smooth'}:
        # Center of window is period 0 and will be filled.
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a division by three, all in all it's a division by 6
        return pl.concat_list(
            [
                (
                    pl.col(position_column).shift(n=-2).list.get(component)
                    + pl.col(position_column).shift(n=-1).list.get(component)
                    - pl.col(position_column).shift(n=1).list.get(component)
                    - pl.col(position_column).shift(n=2).list.get(component)
                ) * (sampling_rate / 6)
                for component in range(n_components)
            ],
        ).alias(velocity_column)

    if method == 'savitzky_golay':
        if window_length is None:
            raise TypeError("'window_length' must not be none for method 'savitzky_golay'")
        if degree is None:
            raise TypeError("'degree' must not be none for method 'savitzky_golay'")

        return savitzky_golay(
            window_length=window_length,
            degree=degree,
            sampling_rate=sampling_rate,
            padding=padding,
            derivative=1,
            n_components=n_components,
            input_column=position_column,
            output_column=velocity_column,
        )

    supported_methods = ['preceding', 'neighbors', 'fivepoint', 'smooth', 'savitzky_golay']
    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: {supported_methods}",
    )
