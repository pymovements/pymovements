# Copyright (c) 2026 The pymovements Project Authors
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
"""Provides helpers for handling time columns backed by ``polars.Duration``."""
from __future__ import annotations

import numpy as np
import polars as pl


def duration_to_ms(column: str | pl.Series | pl.Expr) -> pl.Series | pl.Expr:
    """Convert a ``polars.Duration`` column to fractional milliseconds.

    This is the single source of truth for the Duration to millisecond conversion. Passing a
    :py:class:`polars.Series` returns a ``Float64`` series; passing a column name or an
    expression returns a :py:class:`polars.Expr`.

    Parameters
    ----------
    column: str | pl.Series | pl.Expr
        A Duration column, given as a series, an expression, or a column name.

    Returns
    -------
    pl.Series | pl.Expr
        The column expressed as fractional milliseconds. A series in, a series out;
        an expression or column name in, an expression out.
    """
    expr = pl.col(column) if isinstance(column, str) else column
    return expr.dt.total_microseconds() / 1000


def numeric_to_duration_us(column: str | pl.Expr, time_unit: str) -> pl.Expr:
    """Return an expression converting a numeric time column to ``polars.Duration('us')``.

    Numeric values are interpreted according to ``time_unit`` and cast to microsecond
    ``polars.Duration``. Sub-microsecond fractions are rounded to the nearest microsecond
    rather than truncated, with exact half-microsecond ties rounded to the nearest even
    microsecond (polars ``round`` semantics). ``NaN`` values are mapped to null:
    ``polars.Duration`` has no NaN, so a missing time point is represented as null.

    Parameters
    ----------
    column: str | pl.Expr
        A numeric time column, given as an expression or a column name.
    time_unit: str
        The unit the numeric values are given in: ``'s'`` for seconds, ``'ms'`` for
        milliseconds or ``'us'`` for microseconds.

    Returns
    -------
    pl.Expr
        The column converted to ``polars.Duration('us')``, with ``NaN`` mapped to null.

    Raises
    ------
    ValueError
        If ``time_unit`` is not one of ``'s'``, ``'ms'`` or ``'us'``.
    """
    us_per_unit = {'s': 1_000_000, 'ms': 1_000, 'us': 1}
    if time_unit not in us_per_unit:
        raise ValueError(
            f"unsupported time unit '{time_unit}'. "
            "Supported units are 's' for seconds, 'ms' for milliseconds "
            "and 'us' for microseconds.",
        )
    expr = pl.col(column) if isinstance(column, str) else column
    # Map NaN to null before casting: Duration cannot hold NaN, and a NaN cast would raise.
    # Infinite values are left untouched so they still surface as an error rather than a null.
    return (
        expr.cast(pl.Float64).fill_nan(None) * us_per_unit[time_unit]
    ).round().cast(pl.Duration('us'))


def normalize_duration_to_us(column: str | pl.Expr) -> pl.Expr:
    """Return an expression normalizing a ``polars.Duration`` column to microseconds.

    Sub-microsecond input (e.g. ``polars.Duration('ns')``) is rounded to the nearest
    microsecond rather than truncated, with exact half-microsecond ties rounded to the
    nearest even microsecond (polars ``round`` semantics).

    Parameters
    ----------
    column: str | pl.Expr
        A Duration column, given as an expression or a column name.

    Returns
    -------
    pl.Expr
        The column cast to ``polars.Duration('us')`` with sub-microsecond values rounded.
    """
    expr = pl.col(column) if isinstance(column, str) else column
    # Work from whole microseconds (overflow-safe for any representable Duration) and add only
    # the sub-microsecond remainder. Computing ``total_nanoseconds`` on the whole duration would
    # overflow Int64 for durations beyond ~292 years; the remainder ``total_nanoseconds -
    # total_microseconds * 1000`` stays in (-1000, 1000) and is exact even when both operands
    # wrap, so the rounding matches the plain ``(total_nanoseconds / 1000).round()`` for every
    # representable input while degrading to float precision instead of wrapping past ~285 years.
    whole_us = expr.dt.total_microseconds()
    remainder_ns = expr.dt.total_nanoseconds() - whole_us * 1000
    return (whole_us + remainder_ns / 1000).round().cast(pl.Int64).cast(pl.Duration('us'))


def timesteps_to_numpy(timesteps: pl.Series) -> np.ndarray:
    """Convert a timesteps series to a numpy array of numeric time values.

    ``polars.Duration`` series are converted to fractional milliseconds, so detection
    functions receive the same millisecond-based timesteps as when called through
    :py:meth:`pymovements.Gaze.detect`. Numeric series are returned unchanged.

    Parameters
    ----------
    timesteps: pl.Series
        The timesteps series to convert.

    Returns
    -------
    np.ndarray
        The timesteps as a numpy array of numeric time values.

    Raises
    ------
    TypeError
        If the series is neither a Duration nor a numeric dtype.
    """
    if isinstance(timesteps.dtype, pl.Duration):
        return duration_to_ms(timesteps).to_numpy()

    numeric_dtypes = (pl.datatypes.FloatType, pl.datatypes.IntegerType)
    if not isinstance(timesteps.dtype, numeric_dtypes):
        raise TypeError(f'timesteps dtype must be float or int but is {timesteps.dtype}')
    return timesteps.to_numpy()


def durations_to_ms(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert all Duration columns of a dataframe to numeric milliseconds.

    Duration columns are converted to Float64 millisecond values. Columns
    holding only whole milliseconds are narrowed to Int64 afterwards.

    Parameters
    ----------
    frame: pl.DataFrame
        DataFrame whose Duration columns should be converted.

    Returns
    -------
    pl.DataFrame
        DataFrame with Duration columns converted to numeric milliseconds.
    """
    duration_columns = [
        column for column in frame.columns
        if isinstance(frame.schema[column], pl.Duration)
    ]
    if not duration_columns:
        return frame

    frame = frame.with_columns(
        [
            duration_to_ms(column).alias(column)
            for column in duration_columns
        ],
    )

    # Convert to int if possible.
    for column in duration_columns:
        all_whole = frame.select(
            pl.col(column).round().eq(pl.col(column)).all(),
        ).item()
        if all_whole:
            frame = frame.with_columns(pl.col(column).cast(pl.Int64))

    return frame
