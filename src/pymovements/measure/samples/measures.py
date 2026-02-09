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
from math import pi
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
def std_dev(
    *,
    position_column: str = 'position',
    n_components: int = 2,
) -> pl.Expr:  # noqa: D401 - imperative mood
    r"""Standard deviation of gaze positions during a fixation.

    The standard deviation (STD) measures the spatial spread of gaze positions
    around their centroid. It is computed as the root mean square of the
    squared standard deviations along the horizontal and vertical directions:

    .. math::
        \text{STD}_x = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2},\quad
        \text{STD}_y = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (y_i - \bar{y})^2},\quad
        \text{STD} = \sqrt{\text{STD}_x^2 + \text{STD}_y^2}

    where :math:`x_i` and :math:`y_i` are the gaze positions for the
    :math:`i`-th sample, and :math:`\bar{x}` and :math:`\bar{y}` are their
    respective means. The STD is a radial measure representing the overall
    spatial extent of the gaze positions around their centroid.

    STD is relatively insensitive compared to RMS-S2S to displacement between
    successive gaze positions, making it a good measure of spatial spread
    rather than signal velocity. This makes STD particularly suitable for
    quantifying the precision of eye trackers as it reflects the area over
    which gaze positions are distributed during fixations.

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
        The radial standard deviation of the gaze positions.

    Raises
    ------
    ValueError
        If number of components is not 2.

    Notes
    -----
    This implementation uses sample standard deviation (dividing by :math:`n-1`).
    This is implemented using ``ddof=1`` in the standard deviation calculation.

    For sequences with a single sample, this measure returns ``None`` since there
    is no variance to measure.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    std_x_sq = x_position.std(ddof=1).pow(2)
    std_y_sq = y_position.std(ddof=1).pow(2)

    result = (std_x_sq + std_y_sq).sqrt()

    return result.alias('std_dev')


@register_sample_measure
def s2s_rms(
    *,
    position_column: str = 'position',
    n_components: int = 2,
) -> pl.Expr:
    r"""Root-mean-square of sample-to-sample displacements during a fixation.

    The RMS-S2S (Root Mean Square - Sample to Sample) measures the magnitude
    of displacements between successive gaze position samples. It is computed
    as the square root of the mean squared Euclidean distance between all
    adjacent sample pairs:

    .. math::
        \begin{align}
        \text{RMS-S2S} &= \sqrt{\frac{1}{n-1} \sum_{i=1}^{n-1} \theta_i^2},\quad
        \theta_i = \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}\\
        \text{RMS-S2S} &= \sqrt{\frac{1}{n-1} \sum_{i=1}^{n-1}
        \left[ (x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 \right]}
        \end{align}

    where :math:`x_i` and :math:`y_i` are the gaze positions for the
    :math:`i`-th sample, :math:`x_{i+1}` and :math:`y_{i+1}` are the positions
    for the next sample, :math:`\theta_i` is the Euclidean distance between
    successive samples, and :math:`n` is the total number of samples.

    RMS-S2S is closely proportional to the average velocity of the signal
    during fixations, making it a good indicator of slowest detectable eye
    movements. Unlike STD, which measures spatial spread, RMS-S2S captures
    the velocity aspect of signal variability. This makes RMS-S2S particularly
    useful for assessing what threshold might differentiate eye movements
    from measurement noise :cite:p:`Niehorster2020`.

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
        The root mean square of sample-to-sample displacements.

    Raises
    ------
    ValueError
        If number of components is not 2.

    Notes
    -----
    For a single sample (n=1), there are no successive sample pairs, and
    this measure returns ``None`` since displacements cannot be computed.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    x_diff = x_position.diff()
    y_diff = y_position.diff()

    squared_distances = x_diff.pow(2) + y_diff.pow(2)

    result = squared_distances.mean().sqrt()

    return result.alias('s2s_rms')


@register_sample_measure
def bcea(
    *,
    position_column: str = 'position',
    n_components: int = 2,
    confidence: float = 68.27,
) -> pl.Expr:
    r"""Bivariate contour ellipse area (BCEA) of gaze positions during a fixation.

    The Bivariate Contour Ellipse Area (BCEA) :cite:p:`Crossland2002`
    quantifies the area covered by gaze positions during a fixation.
    It represents the area of an ellipse that encompasses
    a specified proportion of all gaze position samples.
    BCEA accounts for both the spread of gaze positions and their correlation between the horizontal
    and vertical axes:

    .. math::
        \begin{equation}
        \sigma_x = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2},\quad
        \sigma_y = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (y_i - \bar{y})^2}\\
        \rho = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
        {\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2}
        \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}\\
        k = -2 \ln(1 - P/100),\quad
        \text{BCEA} = k \pi \sigma_x \sigma_y \sqrt{1 - \rho^2}
        \end{equation}

    where :math:`x_i` and :math:`y_i` are the gaze positions, :math:`\bar{x}`
    and :math:`\bar{y}` are their respective means, :math:`\sigma_x` and
    :math:`\sigma_y` are the standard deviations, :math:`\rho` is the Pearson
    correlation coefficient between the horizontal and vertical components,
    :math:`P` is the desired confidence level (as a percentage), and :math:`k`
    is the scaling factor derived from the chi-square distribution with 2
    degrees of freedom :cite:p:`Niehorster2020`.

    The relationship between confidence level :math:`P` and scaling factor
    :math:`k` comes from the chi-square distribution. For a bivariate normal
    distribution, the contours of constant Mahalanobis distance form ellipses.
    The value of squared Mahalanobis distance corresponding to confidence :math:`P`
    is :math:`k = -2 \ln(1-P/100)`, which is the :math:`P/100` quantile of the
    chi-square distribution with 2 degrees of freedom (i.e., :math:`k = \chi^2_2(P/100)`).
    Common confidence levels and their corresponding :math:`k` values are:

    - 68.27%: :math:`k \approx 2.30`
    - 95.45%: :math:`k \approx 6.18`
    - 99.73%: :math:`k \approx 9.21`

    The confidence level of 68.27% is the default as it corresponds to
    the same probability mass as a ±1σ interval in 1D.
    To choose the ``confidence`` :math:`P` based on the familiar 1D “±σ” coverage, first compute
    :math:`P = \Pr(|Z| \le \sigma) = \chi^2_1(\sigma^2)` (since :math:`Z^2 \sim \chi^2_1`),
    then use the same probability mass to get :math:`k = \chi^2_2(P)`.

    >>> from scipy.stats import chi2
    >>> sigma = 1.0
    >>> confidence = chi2.cdf(sigma*sigma, df=1)
    >>> k = chi2.ppf(confidence, df=2)
    >>> confidence, k
    (np.float64(0.6826894921370859), np.float64(2.295748928898636))

    BCEA provides a more comprehensive measure of fixation stability than :py:func:`std_dev`
    or :py:func:`s2s_rms` alone because it accounts for potential correlation between
    horizontal and vertical eye movements.
    When gaze positions are correlated, the effective spread is reduced compared to an uncorrelated
    distribution with the same component-wise variances.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)
    confidence: float
        The confidence level as a percentage (0-100). This is the proportion
        of gaze position samples that should fall within the ellipse contour.
        Most commonly used values are 68.27 (±1σ), 95.45 (±2σ), and 99.73 (±3σ).
        (default: 68.27)

    Returns
    -------
    pl.Expr
        The bivariate contour ellipse area.

    Raises
    ------
    ValueError
        If number of components is not 2.
    ValueError
        If confidence is not between 0 and 100.

    Notes
    -----
    This implementation uses sample variance (dividing by :math:`n-1`) for
    computing the variances and correlation coefficient.

    For sequences with fewer than 2 samples, or when the variance of either
    component is zero, this measure returns ``None`` since statistics
    (variance, correlation) cannot be meaningfully computed.
    """
    _check_has_two_componenents(n_components)

    if confidence < 0 or confidence >= 100:
        raise ValueError(
            f"confidence must be between 0 and 100 (exclusive of 100), "
            f"but got: {confidence}",
        )

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    x_mean = x_position.mean()
    y_mean = y_position.mean()

    n_minus_one = pl.len() - 1

    x_centered = x_position - x_mean
    y_centered = y_position - y_mean

    variance_x = pl.when(n_minus_one <= 0).then(pl.lit(None)).otherwise(
        x_centered.pow(2).sum() / n_minus_one,
    )
    variance_y = pl.when(n_minus_one <= 0).then(pl.lit(None)).otherwise(
        y_centered.pow(2).sum() / n_minus_one,
    )

    sigma_x = variance_x.sqrt()
    sigma_y = variance_y.sqrt()

    covariance = pl.when(n_minus_one <= 0).then(pl.lit(None)).otherwise(
        (x_centered * y_centered).sum() / n_minus_one,
    )

    invalid_variance = (
        variance_x.is_null()
        | variance_y.is_null()
        | (variance_x <= 0)
        | (variance_y <= 0)
    )

    rho_sq = pl.when(invalid_variance).then(pl.lit(None)).otherwise(
        covariance.pow(2) / (variance_x * variance_y),
    )

    factor_k = pl.lit(-2.0) * (1.0 - pl.lit(confidence) / 100.0).log()  # default is base e -> ln

    result = pl.when(invalid_variance).then(pl.lit(None)).otherwise(
        factor_k
        * pl.lit(pi)
        * sigma_x
        * sigma_y
        * (1.0 - rho_sq).sqrt(),
    )

    return result.alias('bcea')


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
        Name of the timestamp column. (default: 'time')
    start_time: float | None
        Recording start time. If ``None``, uses the group's first timestamp. (default: ``None``)
    end_time: float | None
        Recording end time. If ``None``, uses the group's last timestamp. (default: ``None``)
    unit: Literal['count', 'time', 'ratio']
        Aggregation unit for the result. (default: ``'ratio'``)

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
    >>> df.select(data_loss('time', sampling_rate=1.0, unit='count'))
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
    >>> df.select(data_loss('pixel', sampling_rate=1.0, unit='count'))
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
