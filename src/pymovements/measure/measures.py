# Copyright (c) 2024-2026 The pymovements Project Authors
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

from math import isfinite
from typing import Any
from typing import Literal

import polars as pl

from pymovements.measure.library import register_sample_measure


@register_sample_measure
def null_ratio(column: str, column_dtype: pl.DataType) -> pl.Expr:
    """Ratio of null values to overall values.

    In the case of list columns,
    a null element in the list will count as overall null for the respective cell.

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
        time_column: str,
        data_column: str,
        *,
        sampling_rate: float,
        start_time: float | None = None,
        end_time: float | None = None,
        unit: Literal['count', 'time', 'ratio'] = 'count',
) -> pl.Expr:
    """Measure data loss using an expected, evenly sampled time base.

    The measure computes missing samples in three categories and returns either:

    - "count": total number of lost samples (integer)
    - "time": lost time in the units of ``time_column`` (``count / sampling_rate``)
    - "ratio": fraction of lost to expected samples in [0, 1]

    Lost samples are the sum of:

    1. Missing rows implied by gaps in the time axis, given ``sampling_rate``.
    2. Invalid rows in ``data_column``, where a row is invalid if it is ``null`` or
       contains any ``null``/``NaN``/``inf`` element
       (for list columns, any invalid element marks the row invalid).

    If ``start_time``/``end_time`` are not provided, the group's first/last timestamps
    (min/max of ``time_column``) are used as bounds.

    Parameters
    ----------
    time_column: str
        Name of the timestamp column.
    data_column: str
        Name of a data column used to count invalid samples due to null/NaN/inf values.
        For list columns, any null/NaN/inf element marks the whole row as invalid.
    sampling_rate: float
        Expected sampling rate in Hz (must be > 0).
    start_time: float | None
        Recording start time. If ``None``, uses the group's first timestamp.
    end_time: float | None
        Recording end time. If ``None``, uses the group's last timestamp.
    unit: Literal['count', 'time', 'ratio']
        Aggregation unit for the result.

    Returns
    -------
    pl.Expr
        A scalar (per-group) expression with alias ``data_loss_{unit}``.

    Raises
    ------
    ValueError
        If ``unit`` is not one of {'count','time','ratio'} or ``sampling_rate`` <= 0.
    TypeError
        If ``time_column`` is not a string.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import measure as m
    >>> df = pl.DataFrame({'time': [0.0, 1.0, 2.0, 4.0]})
    >>> df.select(m.data_loss('time', 'time', sampling_rate=1.0, unit='count'))
    shape: (1, 1)
    ┌─────────────────┐
    │ data_loss_count │
    │ ---             │
    │ i64             │
    ╞═════════════════╡
    │ 1               │
    └─────────────────┘
    >>> # Include invalid rows in a data column
    >>> df = pl.DataFrame({
    ...     'time': [1, 2, 3, 4, 5, 9],
    ...     'pixel':  [[1, 1], [1, 1], None, None, [1, 1], [1, None]],
    ... })
    >>> df.select(m.data_loss('time', 'pixel', sampling_rate=1.0, unit='count'))
    shape: (1, 1)
    ┌─────────────────┐
    │ data_loss_count │
    │ ---             │
    │ i64             │
    ╞═════════════════╡
    │ 6               │
    └─────────────────┘
    """
    try:
        timestamps = pl.col(time_column)
    except TypeError as e:
        raise TypeError(
            f"invalid type for 'time_column'. Expected 'str' , got '{type(time_column).__name__}'",
        ) from e

    # Validate sampling_rate
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        raise ValueError(
            f'sampling_rate must be a positive number, but got: {sampling_rate!r}',
        )

    # Group anchors: provided or derived
    start_expr = pl.lit(start_time) if start_time is not None else timestamps.min()
    end_expr = pl.lit(end_time) if end_time is not None else timestamps.max()

    # Observed rows in the group
    observed = pl.len()

    # Expected sample count over [start, end] with inclusive endpoints for a fixed rate.
    span = end_expr - start_expr
    expected = (span * pl.lit(sampling_rate)).floor().cast(pl.Int64) + 1

    # Missing rows due to time gaps, ensure non-negative and valid range
    valid_range = end_expr > start_expr
    time_missing = pl.when(valid_range).then(
        pl.max_horizontal(expected - observed, pl.lit(0)),
    ).otherwise(pl.lit(0))

    def _is_invalid_value(v: Any) -> bool:  # pragma: no cover
        """Check whether ``v`` (scalar or sequence) contains an invalid value.

        For scalar values, the function directly evaluates their validity. For
        sequences (list-like objects), the function iterates through each element
        and returns ``True`` if any contained element is invalid.

        Invalid values include:
        - ``None``
        - ``NaN``
        - Positive or negative infinity (``inf``)

        Parameters
        ----------
        v : Any
            The value or sequence of values to be checked for validity.

        Returns
        -------
        bool
            ``True`` if the input ``v`` is invalid, ``False`` otherwise.
        """
        # scalar None, NaN, or +/-inf
        if v is None:
            return True
        # sequence (e.g., list-like): any invalid element marks the row
        if isinstance(v, (list, tuple)) or (
                hasattr(v, '__iter__') and not isinstance(v, (str, bytes))
        ):
            for e in v:
                if e is None:
                    return True
                if isinstance(e, float) and (not isfinite(e)):
                    return True
            return False
        # scalar float: NaN/inf
        if isinstance(v, float):
            return not isfinite(v)
        return False

    invalid_missing = (
        pl.col(data_column)
        .map_elements(_is_invalid_value, return_dtype=pl.Boolean)
        .fill_null(True)
        .sum()
        .cast(pl.Int64)
    )

    total_missing = (time_missing + invalid_missing).alias('data_loss_count')

    if unit == 'count':
        return total_missing

    if unit == 'time':
        missing_time = (total_missing.cast(pl.Float64) / pl.lit(float(sampling_rate)))
        return missing_time.alias('data_loss_time')

    if unit == 'ratio':
        # Prevent division by zero when range invalid - expected will be >= 1 when valid.
        expected_safe = pl.when(valid_range).then(expected).otherwise(pl.lit(1))
        ratio = (total_missing.cast(pl.Float64) / expected_safe.cast(pl.Float64)).fill_null(0.0)
        return ratio.alias('data_loss_ratio')

    raise ValueError(
        "unit must be one of {'count', 'time', 'ratio'} but got: " f"{unit!r}",
    )
