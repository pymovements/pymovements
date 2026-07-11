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
"""Validation checks for individual :py:class:`~pymovements.Gaze` objects."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pymovements.gaze.gaze import Gaze


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single validation check.

    Attributes
    ----------
    code : str
        Short identifier, e.g. ``'trial_columns_exist'``.
    severity : str
        One of ``'pass'``, ``'warning'``, or ``'error'``.
    message : str
        Human-readable description of the outcome.
    sources : list[str]
        File paths of source files that were checked.

    Examples
    --------
    >>> from pymovements import CheckResult
    >>> r = CheckResult('time_column_exists', 'pass', 'OK')
    >>> r.severity
    'pass'
    """

    code: str
    severity: str
    message: str
    sources: list[str] = field(default_factory=list)


def check_trial_columns_exist(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that every name declared in ``trial_columns`` exists in the sample schema.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'error'`` if any declared trial column is absent from the schema;
        ``'pass'`` otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_trial_columns_exist
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 1], 'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_trial_columns_exist(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if not gaze.trial_columns:
        return CheckResult(
            code='trial_columns_exist',
            severity='pass',
            message='No trial_columns declared; check skipped.',
            sources=sources,
        )

    missing = [col for col in gaze.trial_columns if col not in gaze.samples.columns]

    if missing:
        return CheckResult(
            code='trial_columns_exist',
            severity='error',
            message=(
                f'trial_columns {missing!r} not found in sample schema. '
                f'Available columns: {gaze.samples.columns!r}'
            ),
            sources=sources,
        )

    return CheckResult(
        code='trial_columns_exist',
        severity='pass',
        message='All declared trial_columns are present in the sample schema.',
        sources=sources,
    )


def check_trial_columns_dtype(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that trial-identifier columns have integer or string dtype.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if any trial column has a dtype that is neither
        integer nor string; ``'pass'`` otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_trial_columns_dtype
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame(
    ...         {'time': [0, 1], 'trial': [1, 2], 'x': [0.0, 1.0], 'y': [0.0, 1.0]}
    ...     ),
    ...     trial_columns=['trial'],
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_trial_columns_dtype(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if not gaze.trial_columns:
        return CheckResult(
            code='trial_columns_dtype',
            severity='pass',
            message='No trial_columns declared; check skipped.',
            sources=sources,
        )

    invalid_cols = [
        col for col in gaze.trial_columns
        if col in gaze.samples.columns
        and not (
            gaze.samples[col].dtype == pl.String
            or gaze.samples[col].dtype.is_integer()
        )
    ]

    if invalid_cols:
        return CheckResult(
            code='trial_columns_dtype',
            severity='warning',
            message=(
                f'trial_columns {invalid_cols!r} have non-integer, non-string dtype. '
                'Trial identifiers should be integer or string to avoid join ambiguity.'
            ),
            sources=sources,
        )

    return CheckResult(
        code='trial_columns_dtype',
        severity='pass',
        message='All trial_columns have appropriate (integer or string) dtype.',
        sources=sources,
    )


def check_time_column_exists(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that a ``time`` column is present and carries a numeric dtype.

    After initialisation, pymovements renames the user-specified time column to
    ``'time'``. This check therefore looks for the column named ``'time'``.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'error'`` if the column is absent or has a non-numeric dtype;
        ``'pass'`` otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_time_column_exists
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_time_column_exists(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if 'time' not in gaze.samples.columns:
        return CheckResult(
            code='time_column_exists',
            severity='error',
            message=(
                "No 'time' column found in the sample schema. "
                'Specify time_column during Gaze initialisation or provide an Experiment '
                'with a sampling_rate to auto-generate timestamps.'
            ),
            sources=sources,
        )

    if not gaze.samples['time'].dtype.is_numeric():
        return CheckResult(
            code='time_column_exists',
            severity='error',
            message=(
                f"'time' column has dtype {gaze.samples['time'].dtype!r} which is not numeric. "
                'Timestamps must be numeric (integer or float).'
            ),
            sources=sources,
        )

    return CheckResult(
        code='time_column_exists',
        severity='pass',
        message="'time' column is present and has a numeric dtype.",
        sources=sources,
    )


def check_gaze_components_defined(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that at least one gaze coordinate column is present.

    After initialisation, pymovements nests raw coordinate columns into
    ``'pixel'``, ``'position'``, or ``'velocity'``. This check verifies that at
    least one of these nested columns exists.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'error'`` if none of the expected coordinate columns is present;
        ``'pass'`` otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_gaze_components_defined
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0], 'x': [1.0], 'y': [2.0]}),
    ...     position_columns=['x', 'y'],
    ... )
    >>> check_gaze_components_defined(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []
    coordinate_cols = {'pixel', 'position', 'velocity', 'acceleration'}
    present = coordinate_cols & set(gaze.samples.columns)

    if not present:
        return CheckResult(
            code='gaze_components_defined',
            severity='error',
            message=(
                'No gaze coordinate columns found (expected at least one of '
                f'{sorted(coordinate_cols)!r}). '
                'Specify pixel_columns, position_columns, or velocity_columns during '
                'Gaze initialisation.'
            ),
            sources=sources,
        )

    return CheckResult(
        code='gaze_components_defined',
        severity='pass',
        message=f'Gaze coordinate columns present: {sorted(present)!r}.',
        sources=sources,
    )


def _get_trial_groups(gaze: Gaze) -> list[pl.DataFrame]:
    """Return per-trial sub-DataFrames, or the whole frame as a single group."""
    cols = gaze.trial_columns
    if not cols:
        return [gaze.samples.select(['time'])]
    missing = [c for c in cols if c not in gaze.samples.columns]
    if missing:
        return []
    return gaze.samples.select(cols + ['time']).partition_by(cols, maintain_order=True)


def check_time_monotone(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that timestamps are strictly monotone increasing within each trial.

    If no ``trial_columns`` are declared the whole dataframe is treated as a
    single trial. The only precondition is that a ``'time'`` column exists.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if any trial contains non-strictly-increasing
        timestamps; ``'pass'`` otherwise or when preconditions are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_time_monotone
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 10, 20], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_time_monotone(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if 'time' not in gaze.samples.columns:
        return CheckResult(
            code='time_monotone',
            severity='pass',
            message="No 'time' column available; check skipped.",
            sources=sources,
        )

    groups = _get_trial_groups(gaze)
    non_monotone: list[str] = []

    for grp in groups:
        times = grp['time'].to_list()
        if len(times) < 2:
            continue
        if any(times[i + 1] - times[i] <= 0 for i in range(len(times) - 1)):
            if gaze.trial_columns:
                key_vals = {c: grp[c][0] for c in gaze.trial_columns if c in grp.columns}
                non_monotone.append(str(key_vals))
            else:
                non_monotone.append('(single trial)')

    if non_monotone:
        return CheckResult(
            code='time_monotone',
            severity='warning',
            message=(
                f'Non-monotone timestamps in {len(non_monotone)} trial(s): '
                f"{non_monotone[:3]}{'...' if len(non_monotone) > 3 else ''}"
            ),
            sources=sources,
        )

    return CheckResult(
        code='time_monotone',
        severity='pass',
        message='Timestamps are strictly monotone increasing within all trials.',
        sources=sources,
    )


def check_max_gap(
        gaze: Gaze,
        source_path: str = '',
        max_gap_factor: float = 5.0,
) -> CheckResult:
    """Check that no inter-sample gap exceeds ``max_gap_factor`` times the expected ISI.

    If no ``trial_columns`` are declared the whole dataframe is treated as a
    single trial. Requires a declared ``sampling_rate`` on ``gaze.experiment``
    to compute the expected ISI.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.
    max_gap_factor : float
        Maximum allowed gap as a multiple of the expected inter-sample interval.
        (default: 5.0)

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if any gap exceeds ``max_gap_factor × ISI``;
        ``'pass'`` otherwise or when preconditions are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.experiment import Experiment
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_max_gap
    >>> exp = Experiment(1280, 1024, 38, 30, 68, 'upper left', sampling_rate=100.0)
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 10, 20], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
    ...     experiment=exp,
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_max_gap(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if 'time' not in gaze.samples.columns:
        return CheckResult(
            code='max_gap',
            severity='error',
            message="No 'time' column available; check could not be performed.",
            sources=sources,
        )

    if gaze.experiment is None or gaze.experiment.sampling_rate is None:
        return CheckResult(
            code='max_gap',
            severity='error',
            message='No declared sampling_rate available; check could not be performed.',
            sources=sources,
        )

    max_gap_ms = max_gap_factor * (1000.0 / gaze.experiment.sampling_rate)
    groups = _get_trial_groups(gaze)
    gap_trials: list[str] = []

    for grp in groups:
        times = grp['time'].to_list()
        if len(times) < 2:
            continue
        if any(times[i + 1] - times[i] > max_gap_ms for i in range(len(times) - 1)):
            if gaze.trial_columns:
                key_vals = {c: grp[c][0] for c in gaze.trial_columns if c in grp.columns}
                gap_trials.append(str(key_vals))
            else:
                gap_trials.append('(single trial)')

    if gap_trials:
        return CheckResult(
            code='max_gap',
            severity='warning',
            message=(
                f'Timestamp gap >{max_gap_factor}× ISI ({max_gap_ms:.1f} ms) '
                f'in {len(gap_trials)} trial(s): '
                f"{gap_trials[:3]}{'...' if len(gap_trials) > 3 else ''}"
            ),
            sources=sources,
        )

    return CheckResult(
        code='max_gap',
        severity='pass',
        message=(
            f'No inter-sample gap exceeds {max_gap_factor}× ISI ({max_gap_ms:.1f} ms).'
        ),
        sources=sources,
    )


def check_sampling_rate_consistency(
        gaze: Gaze,
        source_path: str = '',
        max_deviation: float = 0.05,
) -> CheckResult:
    """Check that the empirical median ISI matches the declared sampling rate.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.
    max_deviation : float
        Maximum allowed relative deviation between empirical and declared rate.
        (default: 0.05, i.e. 5%)

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if the empirical rate deviates by more than
        ``max_deviation`` from the declared rate; ``'pass'`` otherwise or when
        preconditions are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.experiment import Experiment
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_sampling_rate_consistency
    >>> exp = Experiment(1280, 1024, 38, 30, 68, 'upper left', sampling_rate=100.0)
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame(
    ...         {'time': [0, 10, 20, 30], 'x': [0.0, 1.0, 2.0, 3.0], 'y': [0.0, 1.0, 2.0, 3.0]}
    ...     ),
    ...     experiment=exp,
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_sampling_rate_consistency(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if gaze.experiment is None or gaze.experiment.sampling_rate is None:
        return CheckResult(
            code='sampling_rate_consistency',
            severity='pass',
            message='No declared sampling_rate available; check skipped.',
            sources=sources,
        )

    if 'time' not in gaze.samples.columns or len(gaze.samples) < 2:
        return CheckResult(
            code='sampling_rate_consistency',
            severity='pass',
            message='Insufficient samples to estimate empirical sampling rate; check skipped.',
            sources=sources,
        )

    declared_rate = gaze.experiment.sampling_rate
    diffs = gaze.samples['time'].cast(pl.Float64).diff().drop_nulls()
    positive_diffs = diffs.filter(diffs > 0)

    if len(positive_diffs) == 0:
        return CheckResult(
            code='sampling_rate_consistency',
            severity='pass',
            message='No positive time differences found; check skipped.',
            sources=sources,
        )

    median_isi = positive_diffs.median()
    if not isinstance(median_isi, (int, float)):  # pragma: no cover
        return CheckResult(
            code='sampling_rate_consistency',
            severity='pass',
            message='Could not compute median ISI; check skipped.',
            sources=sources,
        )

    empirical_isi = float(median_isi)
    empirical_rate = 1000.0 / empirical_isi
    deviation = abs(empirical_rate - declared_rate) / declared_rate

    if deviation > max_deviation:
        return CheckResult(
            code='sampling_rate_consistency',
            severity='warning',
            message=(
                f'Empirical sampling rate {empirical_rate:.1f} Hz deviates '
                f'{deviation * 100:.1f}% from declared {declared_rate:.1f} Hz '
                f'(tolerance: {max_deviation * 100:.0f}%).'
            ),
            sources=sources,
        )

    return CheckResult(
        code='sampling_rate_consistency',
        severity='pass',
        message=(
            f'Empirical sampling rate {empirical_rate:.1f} Hz is within '
            f'{max_deviation * 100:.0f}% of declared {declared_rate:.1f} Hz.'
        ),
        sources=sources,
    )


def check_gaze_range(
        gaze: Gaze,
        source_path: str = '',
        min_fraction: float = 0.95,
) -> CheckResult:
    """Check that at least ``min_fraction`` of gaze samples fall within screen bounds.

    Uses the ``'position'`` column (degrees of visual angle) if available, falling
    back to ``'pixel'``. Screen bounds are taken from ``gaze.experiment.screen``.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.
    min_fraction : float
        Minimum fraction of non-null samples that must lie within screen bounds.
        (default: 0.95, i.e. 95%)

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if fewer than ``min_fraction`` of non-null samples
        lie within screen bounds; ``'pass'`` otherwise or when preconditions are
        not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements import Gaze
    >>> from pymovements.gaze.validation import check_gaze_range
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0], 'x': [0.0], 'y': [0.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_gaze_range(gaze).severity
    'pass'
    """
    sources = [source_path] if source_path else []

    if gaze.experiment is None:
        return CheckResult(
            code='gaze_range',
            severity='pass',
            message='No experiment definition available; check skipped.',
            sources=sources,
        )

    if 'position' in gaze.samples.columns:
        coord_col = 'position'
        use_dva = True
    elif 'pixel' in gaze.samples.columns:
        coord_col = 'pixel'
        use_dva = False
    else:
        return CheckResult(
            code='gaze_range',
            severity='pass',
            message='No position or pixel column available; check skipped.',
            sources=sources,
        )

    screen = gaze.experiment.screen

    try:
        if use_dva:
            x_min = screen.x_min_dva
            x_max = screen.x_max_dva
            y_min = screen.y_min_dva
            y_max = screen.y_max_dva
        else:
            if screen.width_px is None or screen.height_px is None:
                return CheckResult(
                    code='gaze_range',
                    severity='pass',
                    message='Screen pixel dimensions not set; check skipped.',
                    sources=sources,
                )
            x_min, x_max = 0.0, float(screen.width_px - 1)
            y_min, y_max = 0.0, float(screen.height_px - 1)
    except (ValueError, TypeError):
        return CheckResult(
            code='gaze_range',
            severity='pass',
            message='Screen bounds could not be computed (missing attributes); check skipped.',
            sources=sources,
        )

    non_null = gaze.samples.filter(pl.col(coord_col).is_not_null())
    n_total = len(non_null)

    if n_total == 0:
        return CheckResult(
            code='gaze_range',
            severity='pass',
            message='No non-null coordinate samples to check; check skipped.',
            sources=sources,
        )

    x_vals = non_null[coord_col].list.get(0)
    y_vals = non_null[coord_col].list.get(1)
    in_range = (
        (x_vals >= x_min) & (x_vals <= x_max)
        & (y_vals >= y_min) & (y_vals <= y_max)
    )
    ratio = int(in_range.sum()) / n_total

    if ratio < min_fraction:
        return CheckResult(
            code='gaze_range',
            severity='warning',
            message=(
                f'Only {ratio * 100:.1f}% of samples lie within screen bounds '
                f'(x: [{x_min:.2f}, {x_max:.2f}], y: [{y_min:.2f}, {y_max:.2f}]). '
                f'Threshold: {min_fraction * 100:.0f}%.'
            ),
            sources=sources,
        )

    return CheckResult(
        code='gaze_range',
        severity='pass',
        message=(
            f'{ratio * 100:.1f}% of samples lie within screen bounds. '
            f'Threshold: {min_fraction * 100:.0f}%.'
        ),
        sources=sources,
    )


_ALL_CHECKS: dict[str, Callable[[Gaze, str], CheckResult]] = {
    'trial_columns_exist': check_trial_columns_exist,
    'trial_columns_dtype': check_trial_columns_dtype,
    'time_column_exists': check_time_column_exists,
    'gaze_components_defined': check_gaze_components_defined,
    'time_monotone': check_time_monotone,
    'max_gap': check_max_gap,
    'sampling_rate_consistency': check_sampling_rate_consistency,
    'gaze_range': check_gaze_range,
}
