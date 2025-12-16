# Copyright (c) 2024-2025 The pymovements Project Authors
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
"""Provides eye movement measure implementations."""
from __future__ import annotations

from typing import Literal

import polars as pl

from pymovements.measure.library import register_sample_measure


@register_sample_measure
def null_ratio(column: str, column_dtype: pl.DataType) -> pl.Expr:
    """Ratio of null values to overall values.

    In case of list columns, a null element in the list will count as overall null for the
    respective cell.

    Parameters
    ----------
    column: str
        Name of measured column.
    column_dtype: pl.DataType
        Data type of measured column.

    Returns
    -------
    pl.Expr
        Null ratio expression.
    """
    if column_dtype in {pl.Float64, pl.Int64}:
        value = 1 - pl.col(column).fill_nan(pl.lit(None)).count() / pl.col(column).len()
    elif column_dtype == pl.Utf8:
        value = 1 - pl.col(column).count() / pl.col(column).len()
    elif column_dtype == pl.List:
        non_null_lengths = pl.col(column).list.drop_nulls().drop_nans().list.len()
        value = (
            1 - (non_null_lengths == pl.col(column).list.len()).sum() /
            pl.col(column).len()
        )
    else:
        raise TypeError(
            'column_dtype must be of type {Float64, Int64, Utf8, List}'
            f' but is of type {column_dtype}',
        )

    return value.alias('null_ratio')


@register_sample_measure
def data_loss(
        timestamp_col: str = 'time',
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        unit: Literal['count', 'time', 'ratio'] = 'count',
) -> pl.Expr:
    """Measure gaze data loss within a group using only Polars expressions.

    Computes the number of missing samples ("count"), missing time ("time"),
    or the ratio of missing to expected samples ("ratio"). The expected
    sampling is derived from the group's median inter-sample interval (ISI)
    computed on the ``timestamp_col`` after sorting.

    If ``start_time``/``end_time`` are not provided, the group's first/last
    timestamps are used (i.e., ``min``/``max`` of ``timestamp_col``).

    Parameters
    ----------
    timestamp_col: str
        Name of the timestamp column. Should be numeric and (ideally) monotonic
        within the group.
    start_time: float | None
        Recording start time for the group. If ``None``, uses the group's first timestamp.
    end_time: float | None
        Recording end time for the group. If ``None``, uses the group's last timestamp.
    unit: Literal['count', 'time', 'ratio']
        - "count": number of missing samples (absolute)
        - "time": missing time in timestamp units (approx. ``missing_count * median_ISI``)
        - "ratio": ``missing_count / expected`` (in [0, 1])

    Returns
    -------
    pl.Expr
        A scalar (per-group) expression.

    Raises
    ------
    ValueError
        If ``unit`` is not one of 'count', 'time', or 'ratio'.
    TypeError
        If ``timestamp_col`` is not a string.
    """
    if not isinstance(timestamp_col, str):
        raise TypeError(
            f'timestamp_col must be a string, but got: {type(timestamp_col).__name__}',
        )

    timestamps = pl.col(timestamp_col)

    # Determine group anchors: provided or derived.
    start_expr = pl.lit(start_time) if start_time is not None else timestamps.min()
    end_expr = pl.lit(end_time) if end_time is not None else timestamps.max()

    # Observed samples in the group.
    observed = pl.len()

    # Median inter-sample interval (after sorting). Drop nulls before median.
    isi_median = timestamps.sort().diff().drop_nulls().median()

    # Expected sample count over [start, end] with inclusive endpoints.
    expected = ((end_expr - start_expr) / isi_median).floor().cast(pl.Int64) + 1

    # Missing samples, ensure non-negative using max with 0
    missing_count = pl.max_horizontal(expected - observed, pl.lit(0))

    # Guard against degenerate or undefined cases:
    # - fewer than 2 samples -> null median ISI
    # - non-positive interval -> end <= start
    valid_range = isi_median.is_not_null() & (end_expr > start_expr)
    missing_count_safe = (
        pl.when(valid_range)
        .then(missing_count)
        .otherwise(pl.lit(0))
    )

    if unit == 'count':
        return missing_count_safe.alias('data_loss_count')

    if unit == 'time':
        # Approximate missing time by missing_count * median ISI
        missing_time = (missing_count_safe.cast(pl.Float64) * isi_median).fill_null(0.0)
        return missing_time.alias('data_loss_time')

    if unit == 'ratio':
        ratio = (
            (missing_count_safe.cast(pl.Float64) / expected.cast(pl.Float64))
            .fill_null(0.0)
        )
        return ratio.alias('data_loss_ratio')

    raise ValueError(
        "unit must be one of {'count', 'time', 'ratio'} but got: " f"{unit!r}",
    )
