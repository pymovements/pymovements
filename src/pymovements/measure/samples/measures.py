# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Provides sample measure implementations."""
from __future__ import annotations

from math import isfinite
from typing import Any
from typing import Literal

import polars as pl

from pymovements.measure.samples.library import register_sample_measure


def _is_invalid_value(v: Any) -> bool:
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


def _is_invalid(column: str | pl.Expr, dtype: pl.DataType | None = None) -> pl.Expr:
    """Check if any value in a column is invalid (null, NaN, or inf).

    Parameters
    ----------
    column: str | pl.Expr
        The column to check for invalid values.
    dtype: pl.DataType | None
        Data type of the column. If provided, more efficient expressions are used.

    Returns
    -------
    pl.Expr
        A boolean expression indicating whether each row is invalid.
    """
    if isinstance(column, str):
        column = pl.col(column)

    if dtype in {pl.Float32, pl.Float64}:
        return column.is_null() | column.is_nan() | column.is_infinite()

    if dtype == pl.List:
        # For list columns, any null/NaN/inf element marks the row invalid.
        # We use map_elements here for robust cross-type support.
        return column.map_elements(_is_invalid_value, return_dtype=pl.Boolean).fill_null(True)

    if dtype is not None:
        return column.is_null()

    return column.map_elements(_is_invalid_value, return_dtype=pl.Boolean).fill_null(True)


@register_sample_measure
def amplitude(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Amplitude of an event.

    The amplitude is calculated as:

    .. math::
        \text{Amplitude} = \sqrt{(x_{\text{max}} - x_{\text{min}})^2 +
        (y_{\text{max}} - y_{\text{min}})^2}

    where :math:`(x_{\text{min}},\; x_{\text{max}})` and
    :math:`(y_{\text{min}},\; y_{\text{max}})` are the minimum and maximum values of the
    :math:`x` and :math:`y` components of the gaze positions during an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The amplitude of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    result = (
        (x_position.max() - x_position.min()).pow(2)
        + (y_position.max() - y_position.min()).pow(2)
    ).sqrt()

    return result.alias('amplitude')


@register_sample_measure
def dispersion(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Dispersion of an event.

    The dispersion is calculated as:

    .. math::
        \text{Dispersion} = x_{\text{max}} - x_{\text{min}} + y_{\text{max}} - y_{\text{min}}

    where :math:`(x_{\text{min}},\; x_{\text{max}})` and
    :math:`(y_{\text{min}},\; y_{\text{max}})` are the minimum and maximum values of the
    :math:`x` and :math:`y` components of the gaze positions during an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The dispersion of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    result = x_position.max() - x_position.min() + y_position.max() - y_position.min()

    return result.alias('dispersion')


@register_sample_measure
def disposition(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Disposition of an event.

    The disposition is calculated as:

    .. math::
        \text{Disposition} = \sqrt{(x_0 - x_n)^2 + (y_0 - y_n)^2}

    where :math:`x_0` and :math:`y_0` are the coordinates of the starting position and
    :math:`x_n` and :math:`y_n` are the coordinates of the ending position of an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The disposition of the event.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    result = (
        (x_position.head(n=1) - x_position.reverse().head(n=1)).pow(2)
        + (y_position.head(n=1) - y_position.reverse().head(n=1)).pow(2)
    ).sqrt()

    return result.alias('disposition')


@register_sample_measure
def location(
        method: str = 'mean',
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Location of an event.

    For method ``mean`` the location is calculated as:

    .. math::
        \text{Location} = \frac{1}{n} \sum_{i=1}^n \text{position}_i

    For method ``median`` the location is calculated as:

    .. math::
        \text{Location} = \text{median} \left(\text{position}_1, \ldots,
         \text{position}_n \right)


    Parameters
    ----------
    method: str
        The centroid method to be used for calculation. Supported methods are ``mean``, ``median``.
        (default: 'mean')
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The location of the event.

    Raises
    ------
    ValueError
        If method is not one of the supported methods.
    """
    if method not in {'mean', 'median'}:
        raise ValueError(
            f"Method '{method}' not supported. "
            f"Please choose one of the following: ['mean', 'median'].",
        )

    component_expressions = []
    for component in range(n_components):
        position_component = (
            pl.col(position_column)
            .list.slice(0, None)
            .list.get(component)
        )

        if method == 'mean':
            expression_component = position_component.mean()
        else:  # by exclusion this must be median
            expression_component = position_component.median()

        component_expressions.append(expression_component)

    # Not sure why first() is needed here, but an outer list is being created somehow.
    result = pl.concat_list(component_expressions).first()

    return result.alias('location')


@register_sample_measure
def null_ratio(column: str, column_dtype: pl.DataType) -> pl.Expr:
    """Ratio of null values to overall values.

    In the case of list columns, a null element in the list will count as overall null for the
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
    valid_dtypes = {pl.Float64, pl.Int64, pl.Utf8, pl.List}
    if not any(
            column_dtype == d or (isinstance(column_dtype, pl.List) and d == pl.List)
            for d in valid_dtypes
    ):
        raise TypeError(
            'column_dtype must be of type {Float64, Int64, Utf8, List}'
            f' but is of type {column_dtype}',
        )

    return _is_invalid(column, dtype=column_dtype).mean().alias('null_ratio')


@register_sample_measure
def peak_velocity(
        *,
        velocity_column: str = 'velocity',
        n_components: int = 2,
) -> pl.Expr:
    r"""Peak velocity of an event.

    The peak velocity is calculated as:

    .. math::
        \text{Peak Velocity} = \max \left(\sqrt{v_x^2 + v_y^2} \right)

    where :math:`v_x` and :math:`v_y` are the velocity components in :math:`x` and :math:`y`
    direction, respectively.

    Parameters
    ----------
    velocity_column: str
        The column name of the velocity tuples. (default: 'velocity')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The peak velocity of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_velocity = pl.col(velocity_column).list.get(0)
    y_velocity = pl.col(velocity_column).list.get(1)

    result = (x_velocity.pow(2) + y_velocity.pow(2)).sqrt().max()

    return result.alias('peak_velocity')


def _check_has_two_componenents(n_components: int) -> None:
    """Check that number of components is two.

    Parameters
    ----------
    n_components: int
        Number of components.
    """
    if n_components != 2:
        raise ValueError('data must have exactly two components')


@register_sample_measure
def data_loss(
        column: str,
        *,
        sampling_rate: float,
        time_column: str = 'time',
        start_time: float | None = None,
        end_time: float | None = None,
        unit: Literal['count', 'time', 'ratio'] = 'ratio',
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
    column: str
        Name of a data column used to count invalid samples due to null/NaN/inf values.
        For list columns, any null/NaN/inf element marks the whole row as invalid.
    sampling_rate: float
        Expected sampling rate in Hz (must be > 0).
    time_column: str
        Name of the timestamp column. (default:   'time'  )
    start_time: float | None
        Recording start time. If ``None``, uses the group's first timestamp. (default:   None  )
    end_time: float | None
        Recording end time. If ``None``, uses the group's last timestamp. (default:   None  )
    unit: Literal['count', 'time', 'ratio']
        Aggregation unit for the result. (default:   'ratio'  )

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
    >>> from pymovements.measure import data_loss
    >>> df = pl.DataFrame({'time': [0.0, 1.0, 2.0, 4.0]})
    >>> df.select(data_loss('time', 'time', sampling_rate=1.0, unit='count'))
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
    >>> df.select(data_loss('time', 'pixel', sampling_rate=1.0, unit='count'))
    shape: (1, 1)
    ┌─────────────────┐
    │ data_loss_count │
    │ ---             │
    │ i64             │
    ╞═════════════════╡
    │ 6               │
    └─────────────────┘
    """
    if not isinstance(time_column, str):
        raise TypeError(
            f"invalid type for 'time_column'. Expected 'str' , got '{type(time_column).__name__}'",
        )
    timestamps = pl.col(time_column)

    # Validate sampling_rate
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        raise ValueError(
            f'sampling_rate must be a positive number, but got: {sampling_rate!r}',
        )

    if start_time is not None and end_time is not None and end_time < start_time:
        raise ValueError(
            f'end_time ({end_time}) must be greater than or equal to '
            f'start_time ({start_time})',
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
    valid_range = end_expr >= start_expr
    time_missing = pl.when(valid_range).then(
        pl.max_horizontal(expected - observed, pl.lit(0)),
    ).otherwise(pl.lit(None))

    invalid_missing = (
        _is_invalid(column)
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
        ratio = (total_missing.cast(pl.Float64) / expected.cast(pl.Float64)).fill_null(0.0)
        return ratio.alias('data_loss_ratio')

    raise ValueError(
        f"unit must be one of {'count', 'time', 'ratio'} but got: {unit!r}",
    )
