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

import numpy as np
import polars as pl
from polars.datatypes.classes import NumericType

from pymovements._utils import _checks
from pymovements.transforms.library import register_transform


@register_transform
def resample(
        samples: pl.DataFrame,
        resampling_rate: float,
        columns: str | list[str] = 'all',
        fill_null_strategy: str = 'interpolate_linear',
        n_components: int | None = None,
) -> pl.DataFrame:
    """Resample a DataFrame to a new sampling rate by timestamps in time column.

    The DataFrame is resampled by upsampling or downsampling the data to the new sampling rate.
    Can also be used to achieve a constant sampling rate for inconsistent data.

    Parameters
    ----------
    samples: pl.DataFrame
        The samples DataFrame to resample.
    resampling_rate: float
        The new sampling rate.
    columns: str | list[str]
        The columns to apply the fill null strategy. Specify a single column name or a list of
        column names. If 'all' is specified, the fill null strategy is applied to all columns.
        (default: 'all')
    fill_null_strategy: str
        The strategy to fill null values of the resampled DataFrame. Supported strategies
        are: 'forward', 'backward', 'interpolate_linear', 'interpolate_nearest'. Columns
        must be numeric when using interpolation.
        (default: 'interpolate_linear')
    n_components: int | None
        Number of components of nested columns in columns. (default: None)

    Returns
    -------
    pl.DataFrame
        The resampled DataFrame.

    Raises
    ------
    ValueError
        If the resampling rate is not a divisor of 1000000.

    Notes
    -----
    The following fill null strategies are available:

    * ``forward``: Fill null values with the previous non-null value.
    * ``backward``: Fill null values with the next non-null value.
    * ``interpolate_linear``: Fill null values by linear interpolation.
    * ``interpolate_nearest``: Fill null values by the nearest interpolation.

    """
    if columns == 'all':
        columns = [column for column in samples.columns if column != 'time']
    elif isinstance(columns, str):
        columns = [columns]

    _checks.check_is_greater_than_zero(resampling_rate=resampling_rate)

    # Return samples if empty
    if samples.is_empty():
        return samples

    # Calculate resampling time steps in microseconds
    resample_step_us = 1000000 / resampling_rate

    # Check if resample rate is supported when using microsecond precision
    # Allow for rounding error less than 1 microsecond
    if 1000000 % resample_step_us >= 1000:
        raise ValueError(
            f'Unsupported resampling rate: {resampling_rate}.'
            ' Sampling rate must result in rounding error less than 1 microsecond'
            ' for resampling time steps',
        )

    resample_step_us = int(resample_step_us)

    # Create microsecond precision datetime column from millisecond time column
    samples = samples.with_columns(
        pl.col('time').cast(pl.Float64).mul(1000).cast(pl.Datetime('us')).alias('datetime'),
    )

    # Sort columns by datetime
    samples = samples.sort('datetime')

    numeric_columns: list[str] | None = None
    if columns is not None:
        def _base_dtype(dt: pl.DataType) -> pl.DataType:
            # Follow nested dtypes (e.g., List(inner=...)) until reaching the base type
            while hasattr(dt, 'inner'):
                dt = dt.inner
            return dt

        numeric_columns = [
            c for c in columns
            if issubclass(
                (bd if isinstance(bd := _base_dtype(samples.schema[c]), type) else type(bd)),
                NumericType,
            )
        ]

    # Replace pre-existing null values with NaN only for numeric columns, as they should not be
    # interpolated. Ensure we cast to Float64 so the UDF output dtype matches the declaration.
    if numeric_columns:
        samples = _apply_on_columns(
            samples,
            columns=numeric_columns,
            transformation=lambda series: series.cast(pl.Float64).fill_null(np.nan),
            n_components=n_components,
            return_dtype=pl.Float64,
        )

    # Resample data by datetime column, create milliseconds time column and drop datetime column
    samples = samples.upsample(
        time_column='datetime',
        every=f'{resample_step_us}us',
    ).with_columns(
        pl.col('datetime').cast(pl.Float64).truediv(1000).alias('time'),
    ).drop('datetime')

    # Convert time column to integer if all values are integers
    all_decimals = samples.select(
        pl.col('time').round().eq(pl.col('time')).all(),
    ).item()

    if all_decimals:
        samples = samples.with_columns(
            pl.col('time').cast(pl.Int64),
        )

    # Fill null values with specified strategy
    if columns is not None and fill_null_strategy is not None:
        if fill_null_strategy in {'forward', 'backward'}:
            samples = samples.with_columns(
                pl.col(columns).fill_null(strategy=fill_null_strategy),
            )
        elif fill_null_strategy in {'interpolate_linear', 'interpolate_nearest'}:
            _, interpolate_method = fill_null_strategy.split('_')

            samples = _apply_on_columns(
                frame=samples,
                columns=numeric_columns,
                transformation=lambda series: series.cast(pl.Float64).interpolate(
                    method=interpolate_method,
                ),
                n_components=n_components,
                # Interpolation yields floats - ensure dtype is Float64.
                return_dtype=pl.Float64,
            )
        else:
            raise ValueError(
                f'Unknown fill_null_strategy: {fill_null_strategy}.'
                ' Supported strategies are: '
                'forward, backward, interpolate_linear, interpolate_nearest',
            )

        # Replace the pre-existing NaN values with Null
        samples = _apply_on_columns(
            samples,
            columns=[column for column in columns if samples[column].dtype != pl.String],
            transformation=lambda series: series.fill_nan(None),
            n_components=n_components,
        )

    return samples


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
