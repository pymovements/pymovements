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
"""Validation checks for individual :py:class:`~pymovements.gaze.Gaze` objects."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pymovements.gaze.gaze import Gaze


_NUMERIC_DTYPES: frozenset[type] = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})

_FLOAT_DTYPES: frozenset[type] = frozenset({pl.Float32, pl.Float64})


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes
    ----------
    check_id : str
        Short identifier, e.g. ``'trial_columns_exist'``.
    severity : str
        One of ``'pass'``, ``'warning'``, or ``'error'``.
    message : str
        Human-readable description of the outcome.
    affected_files : list[str]
        File paths of Gaze objects that failed. Empty on pass.

    Examples
    --------
    >>> from pymovements.gaze.validation import CheckResult
    >>> r = CheckResult('time_column_exists', 'pass', 'OK')
    >>> r.severity
    'pass'
    """

    check_id: str
    severity: str
    message: str
    affected_files: list[str] = field(default_factory=list)


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
    >>> from pymovements.gaze.gaze import Gaze
    >>> from pymovements.gaze.validation import check_trial_columns_exist
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 1], 'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_trial_columns_exist(gaze).severity
    'pass'
    """
    if not gaze.trial_columns:
        return CheckResult(
            check_id='trial_columns_exist',
            severity='pass',
            message='No trial_columns declared; check skipped.',
        )

    schema = gaze.samples.schema
    missing = [col for col in gaze.trial_columns if col not in schema]

    if missing:
        return CheckResult(
            check_id='trial_columns_exist',
            severity='error',
            message=(
                f'trial_columns {missing!r} not found in sample schema. '
                f'Available columns: {list(schema.keys())!r}'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='trial_columns_exist',
        severity='pass',
        message='All declared trial_columns are present in the sample schema.',
    )


def check_trial_columns_dtype(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that trial-identifier columns have integer or string dtype, not float.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if any trial column has a float dtype;
        ``'pass'`` otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.gaze import Gaze
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
    if not gaze.trial_columns:
        return CheckResult(
            check_id='trial_columns_dtype',
            severity='pass',
            message='No trial_columns declared; check skipped.',
        )

    schema = gaze.samples.schema
    float_cols = [
        col for col in gaze.trial_columns
        if col in schema and type(schema[col]) in _FLOAT_DTYPES
    ]

    if float_cols:
        return CheckResult(
            check_id='trial_columns_dtype',
            severity='warning',
            message=(
                f'trial_columns {float_cols!r} have float dtype. '
                'Trial identifiers should be integer or string to avoid join ambiguity.'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='trial_columns_dtype',
        severity='pass',
        message='All trial_columns have appropriate (integer or string) dtype.',
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
    >>> from pymovements.gaze.gaze import Gaze
    >>> from pymovements.gaze.validation import check_time_column_exists
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_time_column_exists(gaze).severity
    'pass'
    """
    schema = gaze.samples.schema

    if 'time' not in schema:
        return CheckResult(
            check_id='time_column_exists',
            severity='error',
            message=(
                "No 'time' column found in the sample schema. "
                'Specify time_column during Gaze initialisation or provide an Experiment '
                'with a sampling_rate to auto-generate timestamps.'
            ),
            affected_files=[source_path] if source_path else [],
        )

    if type(schema['time']) not in _NUMERIC_DTYPES:
        return CheckResult(
            check_id='time_column_exists',
            severity='error',
            message=(
                f"'time' column has dtype {schema['time']!r} which is not numeric. "
                'Timestamps must be numeric (integer or float).'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='time_column_exists',
        severity='pass',
        message="'time' column is present and has a numeric dtype.",
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
    >>> from pymovements.gaze.gaze import Gaze
    >>> from pymovements.gaze.validation import check_gaze_components_defined
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0], 'x': [1.0], 'y': [2.0]}),
    ...     position_columns=['x', 'y'],
    ... )
    >>> check_gaze_components_defined(gaze).severity
    'pass'
    """
    coordinate_cols = {'pixel', 'position', 'velocity', 'acceleration'}
    present = coordinate_cols & set(gaze.samples.schema.keys())

    if not present:
        return CheckResult(
            check_id='gaze_components_defined',
            severity='error',
            message=(
                'No gaze coordinate columns found (expected at least one of '
                f'{sorted(coordinate_cols)!r}). '
                'Specify pixel_columns, position_columns, or velocity_columns during '
                'Gaze initialisation.'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='gaze_components_defined',
        severity='pass',
        message=f'Gaze coordinate columns present: {sorted(present)!r}.',
    )


def check_trial_continuity(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that timestamps within each trial are strictly monotone increasing.

    Also checks that no gap between consecutive timestamps exceeds five times the
    expected inter-sample interval (ISI), derived from
    ``gaze.experiment.sampling_rate``.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if timestamps are not strictly increasing within any
        trial, or if a gap exceeds 5× ISI; ``'pass'`` otherwise or when preconditions
        are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.gaze import Gaze
    >>> from pymovements.gaze.validation import check_trial_continuity
    >>> samples = pl.DataFrame(
    ...     {'time': [0, 10, 20], 'trial': [1, 1, 1], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}
    ... )
    >>> gaze = Gaze(samples=samples, trial_columns=['trial'], pixel_columns=['x', 'y'])
    >>> check_trial_continuity(gaze).severity
    'pass'
    """
    if not gaze.trial_columns or 'time' not in gaze.samples.schema:
        return CheckResult(
            check_id='trial_continuity',
            severity='pass',
            message='Preconditions not met (no trial_columns or no time column); check skipped.',
        )

    schema = gaze.samples.schema
    missing_tc = [c for c in gaze.trial_columns if c not in schema]
    if missing_tc:
        return CheckResult(
            check_id='trial_continuity',
            severity='pass',
            message=(
                f'trial_columns {missing_tc!r} absent from schema; '
                'continuity check skipped.'
            ),
        )

    df = gaze.samples.select(gaze.trial_columns + ['time'])
    groups = df.partition_by(gaze.trial_columns, maintain_order=True)

    non_monotone_trials: list[str] = []
    gap_trials: list[str] = []

    sampling_rate = (
        gaze.experiment.sampling_rate
        if gaze.experiment is not None
        else None
    )
    max_gap: float | None = 5.0 * (1000.0 / sampling_rate) if sampling_rate else None

    for grp in groups:
        times = grp['time'].to_list()
        if len(times) < 2:
            continue

        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        if any(d <= 0 for d in diffs):
            key_vals = {c: grp[c][0] for c in gaze.trial_columns}
            non_monotone_trials.append(str(key_vals))
            continue

        if max_gap is not None and any(d > max_gap for d in diffs):
            key_vals = {c: grp[c][0] for c in gaze.trial_columns}
            gap_trials.append(str(key_vals))

    issues: list[str] = []
    if non_monotone_trials:
        issues.append(
            f'Non-monotone timestamps in {len(non_monotone_trials)} trial(s): '
            f"{non_monotone_trials[:3]}{'...' if len(non_monotone_trials) > 3 else ''}",
        )
    if gap_trials:
        assert max_gap is not None
        issues.append(
            f'Timestamp gap >5× ISI ({max_gap:.1f} ms) in {len(gap_trials)} trial(s): '
            f"{gap_trials[:3]}{'...' if len(gap_trials) > 3 else ''}",
        )

    if issues:
        return CheckResult(
            check_id='trial_continuity',
            severity='warning',
            message=' | '.join(issues),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='trial_continuity',
        severity='pass',
        message='Timestamps are strictly monotone increasing within all trials.',
    )


def check_sampling_rate_consistency(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that the empirical median ISI matches the declared sampling rate within 5%.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if the empirical rate deviates by more than 5% from
        the declared rate; ``'pass'`` otherwise or when preconditions are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.experiment import Experiment
    >>> from pymovements.gaze.gaze import Gaze
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
    if gaze.experiment is None or gaze.experiment.sampling_rate is None:
        return CheckResult(
            check_id='sampling_rate_consistency',
            severity='pass',
            message='No declared sampling_rate available; check skipped.',
        )

    if 'time' not in gaze.samples.schema or len(gaze.samples) < 2:
        return CheckResult(
            check_id='sampling_rate_consistency',
            severity='pass',
            message='Insufficient samples to estimate empirical sampling rate; check skipped.',
        )

    declared_rate = gaze.experiment.sampling_rate

    diffs = gaze.samples['time'].cast(pl.Float64).diff().drop_nulls()
    positive_diffs = diffs.filter(diffs > 0)

    if len(positive_diffs) == 0:
        return CheckResult(
            check_id='sampling_rate_consistency',
            severity='pass',
            message='No positive time differences found; check skipped.',
        )

    empirical_isi = float(positive_diffs.median())  # type: ignore[arg-type]
    empirical_rate = 1000.0 / empirical_isi
    deviation = abs(empirical_rate - declared_rate) / declared_rate

    if deviation > 0.05:
        return CheckResult(
            check_id='sampling_rate_consistency',
            severity='warning',
            message=(
                f'Empirical sampling rate {empirical_rate:.1f} Hz deviates '
                f'{deviation * 100:.1f}% from declared {declared_rate:.1f} Hz '
                f'(tolerance: 5%).'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='sampling_rate_consistency',
        severity='pass',
        message=(
            f'Empirical sampling rate {empirical_rate:.1f} Hz is within 5% of '
            f'declared {declared_rate:.1f} Hz.'
        ),
    )


def check_gaze_range(gaze: Gaze, source_path: str = '') -> CheckResult:
    """Check that ≥95% of gaze samples fall within screen bounds.

    Uses the ``'position'`` column (degrees of visual angle) if available, falling
    back to ``'pixel'``. Screen bounds are taken from ``gaze.experiment.screen``.

    Parameters
    ----------
    gaze : Gaze
        The gaze object to inspect.
    source_path : str
        Identifier for this gaze object (e.g. a file path). Used in error reports.

    Returns
    -------
    CheckResult
        Severity ``'warning'`` if fewer than 95% of non-null samples lie within
        screen bounds; ``'pass'`` otherwise or when preconditions are not met.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.gaze.gaze import Gaze
    >>> from pymovements.gaze.validation import check_gaze_range
    >>> gaze = Gaze(
    ...     samples=pl.DataFrame({'time': [0], 'x': [0.0], 'y': [0.0]}),
    ...     pixel_columns=['x', 'y'],
    ... )
    >>> check_gaze_range(gaze).severity
    'pass'
    """
    if gaze.experiment is None:
        return CheckResult(
            check_id='gaze_range',
            severity='pass',
            message='No experiment definition available; check skipped.',
        )

    schema = gaze.samples.schema

    if 'position' in schema:
        coord_col = 'position'
        use_dva = True
    elif 'pixel' in schema:
        coord_col = 'pixel'
        use_dva = False
    else:
        return CheckResult(
            check_id='gaze_range',
            severity='pass',
            message='No position or pixel column available; check skipped.',
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
                    check_id='gaze_range',
                    severity='pass',
                    message='Screen pixel dimensions not set; check skipped.',
                )
            x_min, x_max = 0.0, float(screen.width_px - 1)
            y_min, y_max = 0.0, float(screen.height_px - 1)
    except (ValueError, TypeError):
        return CheckResult(
            check_id='gaze_range',
            severity='pass',
            message='Screen bounds could not be computed (missing attributes); check skipped.',
        )

    samples = gaze.samples
    non_null = samples.filter(pl.col(coord_col).is_not_null())
    n_total = len(non_null)

    if n_total == 0:
        return CheckResult(
            check_id='gaze_range',
            severity='pass',
            message='No non-null coordinate samples to check; check skipped.',
        )

    x_vals = non_null[coord_col].list.get(0)
    y_vals = non_null[coord_col].list.get(1)

    in_range = (
        (x_vals >= x_min) & (x_vals <= x_max)
        & (y_vals >= y_min) & (y_vals <= y_max)
    )
    n_in_range = int(in_range.sum())
    ratio = n_in_range / n_total

    if ratio < 0.95:
        return CheckResult(
            check_id='gaze_range',
            severity='warning',
            message=(
                f'Only {ratio * 100:.1f}% of samples lie within screen bounds '
                f'(x: [{x_min:.2f}, {x_max:.2f}], y: [{y_min:.2f}, {y_max:.2f}]). '
                'Threshold: 95%.'
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='gaze_range',
        severity='pass',
        message=(
            f'{ratio * 100:.1f}% of samples lie within screen bounds. '
            'Threshold: 95%.'
        ),
    )


_ALL_CHECKS: dict[str, Callable[[Gaze, str], CheckResult]] = {
    'trial_columns_exist': check_trial_columns_exist,
    'trial_columns_dtype': check_trial_columns_dtype,
    'time_column_exists': check_time_column_exists,
    'gaze_components_defined': check_gaze_components_defined,
    'trial_continuity': check_trial_continuity,
    'sampling_rate_consistency': check_sampling_rate_consistency,
    'gaze_range': check_gaze_range,
}
