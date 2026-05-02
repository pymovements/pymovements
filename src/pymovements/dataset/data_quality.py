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
"""Provides data quality checks and reporting for eye-tracking datasets."""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import polars as pl

from pymovements.gaze.gaze import Gaze


_NUMERIC_DTYPES: frozenset[type] = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})

_FLOAT_DTYPES: frozenset[type] = frozenset({pl.Float32, pl.Float64})

_INTEGER_OR_STRING_DTYPES: frozenset[type] = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.String, pl.Utf8, pl.Categorical,
})


class GazeDataValidationError(Exception):
    """Raised when a validation check produces an error-severity result.

    Parameters
    ----------
    check_id : str
        Identifier of the failing check.
    message : str
        Human-readable description of the failure.
    affected_files : list[str]
        File paths of Gaze objects that triggered the failure.
    """

    def __init__(
            self,
            check_id: str,
            message: str,
            affected_files: list[str],
    ) -> None:
        self.check_id = check_id
        self.affected_files = affected_files
        super().__init__(message)


@dataclass
class CheckResult:
    """Result of a single validation check.

    Parameters
    ----------
    check_id : str
        Short identifier, e.g. ``'trial_columns_exist'``.
    severity : str
        One of ``'pass'``, ``'warning'``, or ``'error'``.
    message : str
        Human-readable description of the outcome.
    affected_files : list[str]
        File paths of Gaze objects that failed. Empty on pass.
    """

    check_id: str
    severity: str
    message: str
    affected_files: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Aggregated output of :py:meth:`~pymovements.Dataset.report_data_quality`.

    Parameters
    ----------
    check_results : list[CheckResult]
        Results of all validation checks that were run.
    measures : dict[str, pl.DataFrame]
        Quality measures keyed by aggregation level:
        ``'dataset'``, ``'subject'``, ``'session'``, ``'trial'``.
    """

    check_results: list[CheckResult] = field(default_factory=list)
    measures: dict[str, pl.DataFrame] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True if no check result has severity ``'error'``."""
        return all(r.severity != 'error' for r in self.check_results)

    def summary(self) -> str:
        """Return a formatted summary table of all check results.

        Returns
        -------
        str
            A plain-text table with columns ``check_id``, ``severity``,
            ``n_affected``, and ``message``.

        Examples
        --------
        >>> from pymovements.dataset.data_quality import CheckResult, DataQualityReport
        >>> report = DataQualityReport(
        ...     check_results=[CheckResult('trial_columns_exist', 'pass', 'All OK')],
        ... )
        >>> isinstance(report.summary(), str)
        True
        """
        header = f"{'check_id':<35} {'severity':<10} {'n_aff':>5}  message"
        separator = '-' * len(header)
        lines = [header, separator]
        for r in self.check_results:
            n = len(r.affected_files)
            lines.append(f'{r.check_id:<35} {r.severity:<10} {n:>5}  {r.message}')
        return '\n'.join(lines)

    def save_bids_report(
            self,
            output_path: Path,
            pipeline_name: str = 'pymovements',
    ) -> None:
        """Write BIDS-conformant derivative report files.

        Writes the following files under
        ``output_path / 'derivatives' / pipeline_name /``:

        * ``dataset_description.json`` — required BIDS derivative descriptor.
        * ``data_quality_checks.tsv`` + ``.json`` sidecar — one row per
          :py:class:`CheckResult`.
        * ``data_quality_measures_{level}.tsv`` + ``.json`` sidecar — one file
          per aggregation level present in :py:attr:`measures`.
        * ``warnings.log`` — one line per Python warning captured during the run.

        Parameters
        ----------
        output_path : Path
            Root of the BIDS dataset. Derivative files are written under
            ``output_path / 'derivatives' / pipeline_name /``.
        pipeline_name : str
            Name of the pipeline sub-directory. (default: ``'pymovements'``)

        Examples
        --------
        >>> import tempfile, pathlib
        >>> from pymovements.dataset.data_quality import DataQualityReport, CheckResult
        >>> report = DataQualityReport(
        ...     check_results=[CheckResult('trial_columns_exist', 'pass', 'All OK')],
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     report.save_bids_report(pathlib.Path(tmp))
        """
        from pymovements._version import __version__  # pylint: disable=import-outside-toplevel

        deriv_dir = Path(output_path) / 'derivatives' / pipeline_name
        deriv_dir.mkdir(parents=True, exist_ok=True)

        # dataset_description.json
        description: dict[str, Any] = {
            'Name': 'pymovements data quality report',
            'BIDSVersion': '1.11.1',
            'DatasetType': 'derivative',
            'GeneratedBy': [
                {
                    'Name': 'pymovements',
                    'Version': __version__,
                    'CodeURL': 'https://github.com/pymovements/pymovements',
                },
            ],
        }
        _write_json(deriv_dir / 'dataset_description.json', description)

        # data_quality_checks.tsv + sidecar
        checks_rows = [
            {
                'check_id': r.check_id,
                'severity': r.severity,
                'message': r.message,
                'affected_files': ';'.join(r.affected_files),
            }
            for r in self.check_results
        ]
        if checks_rows:
            pl.DataFrame(checks_rows).write_csv(
                deriv_dir / 'data_quality_checks.tsv', separator='\t',
            )
        else:
            _write_empty_tsv(
                deriv_dir / 'data_quality_checks.tsv',
                ['check_id', 'severity', 'message', 'affected_files'],
            )

        checks_sidecar: dict[str, Any] = {
            'check_id': {'Description': 'Unique identifier of the validation check.'},
            'severity': {'Description': "Result severity: 'pass', 'warning', or 'error'."},
            'message': {'Description': 'Human-readable description of the check outcome.'},
            'affected_files': {
                'Description': 'Semicolon-separated list of affected file paths; empty on pass.',
            },
        }
        _write_json(deriv_dir / 'data_quality_checks.json', checks_sidecar)

        # per-level measure TSVs + sidecars
        _measures_sidecar: dict[str, Any] = {
            'data_loss': {
                'Description': (
                    'Fraction of missing or invalid samples (null/NaN/inf). '
                    'Computed as ratio of lost to expected samples.'
                ),
                'Units': 'ratio',
            },
            'std_rms': {
                'Description': (
                    'Root-mean-square standard deviation of gaze position, '
                    'a measure of fixation precision.'
                ),
                'Units': 'degrees of visual angle',
            },
            'rms_s2s': {
                'Description': (
                    'Root-mean-square of sample-to-sample displacements, '
                    'a measure of positional noise.'
                ),
                'Units': 'degrees of visual angle',
            },
            'bcea': {
                'Description': (
                    'Bivariate contour ellipse area at the 68.27% confidence level, '
                    'a measure of gaze dispersion.'
                ),
                'Units': 'degrees of visual angle squared',
            },
        }

        for level, df in self.measures.items():
            tsv_path = deriv_dir / f'data_quality_measures_{level}.tsv'
            df.write_csv(tsv_path, separator='\t')

            sidecar: dict[str, Any] = {}
            for col in df.columns:
                if col in _measures_sidecar:
                    sidecar[col] = _measures_sidecar[col]
                else:
                    sidecar[col] = {'Description': f'Grouping column: {col}'}
            _write_json(deriv_dir / f'data_quality_measures_{level}.json', sidecar)

        # warnings.log  (populated externally; write placeholder if not set)
        log_lines: list[str] = getattr(self, '_warning_log', [])
        (deriv_dir / 'warnings.log').write_text('\n'.join(log_lines), encoding='utf-8')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path*."""
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _write_empty_tsv(path: Path, columns: list[str]) -> None:
    """Write an empty TSV with header columns to *path*."""
    path.write_text('\t'.join(columns) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

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
                f"trial_columns {missing!r} not found in sample schema. "
                f"Available columns: {list(schema.keys())!r}"
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
                f"trial_columns {float_cols!r} have float dtype. "
                "Trial identifiers should be integer or string to avoid join ambiguity."
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

    After initialisation, ``pymovements`` renames the user-specified time column to
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
    """
    schema = gaze.samples.schema

    if 'time' not in schema:
        return CheckResult(
            check_id='time_column_exists',
            severity='error',
            message=(
                "No 'time' column found in the sample schema. "
                "Specify time_column during Gaze initialisation or provide an Experiment "
                "with a sampling_rate to auto-generate timestamps."
            ),
            affected_files=[source_path] if source_path else [],
        )

    if type(schema['time']) not in _NUMERIC_DTYPES:
        return CheckResult(
            check_id='time_column_exists',
            severity='error',
            message=(
                f"'time' column has dtype {schema['time']!r} which is not numeric. "
                "Timestamps must be numeric (integer or float)."
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

    After initialisation, ``pymovements`` nests raw coordinate columns into
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
    """
    coordinate_cols = {'pixel', 'position', 'velocity', 'acceleration'}
    present = coordinate_cols & set(gaze.samples.schema.keys())

    if not present:
        return CheckResult(
            check_id='gaze_components_defined',
            severity='error',
            message=(
                "No gaze coordinate columns found (expected at least one of "
                f"{sorted(coordinate_cols)!r}). "
                "Specify pixel_columns, position_columns, or velocity_columns during "
                "Gaze initialisation."
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='gaze_components_defined',
        severity='pass',
        message=f"Gaze coordinate columns present: {sorted(present)!r}.",
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
                f"trial_columns {missing_tc!r} absent from schema; "
                "continuity check skipped."
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
            f"Non-monotone timestamps in {len(non_monotone_trials)} trial(s): "
            f"{non_monotone_trials[:3]}{'...' if len(non_monotone_trials) > 3 else ''}",
        )
    if gap_trials:
        assert max_gap is not None
        issues.append(
            f"Timestamp gap >5× ISI ({max_gap:.1f} ms) in {len(gap_trials)} trial(s): "
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
    expected_isi_ms = 1000.0 / declared_rate

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
                f"Empirical sampling rate {empirical_rate:.1f} Hz deviates "
                f"{deviation * 100:.1f}% from declared {declared_rate:.1f} Hz "
                f"(tolerance: 5%)."
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='sampling_rate_consistency',
        severity='pass',
        message=(
            f"Empirical sampling rate {empirical_rate:.1f} Hz is within 5% of "
            f"declared {declared_rate:.1f} Hz."
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
                f"Only {ratio * 100:.1f}% of samples lie within screen bounds "
                f"(x: [{x_min:.2f}, {x_max:.2f}], y: [{y_min:.2f}, {y_max:.2f}]). "
                "Threshold: 95%."
            ),
            affected_files=[source_path] if source_path else [],
        )

    return CheckResult(
        check_id='gaze_range',
        severity='pass',
        message=(
            f"{ratio * 100:.1f}% of samples lie within screen bounds. "
            "Threshold: 95%."
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


# ---------------------------------------------------------------------------
# Measure computation
# ---------------------------------------------------------------------------

def _coord_column(gaze: Gaze) -> str | None:
    """Return the best available coordinate column name (``'position'`` > ``'pixel'``)."""
    for col in ('position', 'pixel'):
        if col in gaze.samples.schema:
            return col
    return None


def _compute_data_loss_simple(gaze: Gaze, coord_col: str) -> float:
    """Fraction of samples where the coordinate column is null or contains invalid values."""
    series = gaze.samples[coord_col]
    n_total = len(series)
    if n_total == 0:
        return 0.0
    n_null = int(series.is_null().sum())
    return n_null / n_total


def _build_precision_agg(coord_col: str) -> list[pl.Expr]:
    """Return Polars aggregation expressions for precision measures on *coord_col*."""
    from pymovements.measure.samples.measures import bcea  # pylint: disable=import-outside-toplevel
    from pymovements.measure.samples.measures import rms_s2s  # pylint: disable=import-outside-toplevel
    from pymovements.measure.samples.measures import std_rms  # pylint: disable=import-outside-toplevel
    return [
        std_rms(column=coord_col),
        rms_s2s(column=coord_col),
        bcea(column=coord_col),
    ]


def _compute_measures(
        gaze_list: list[Gaze],
        fileinfo: Any,
        levels: list[str],
        measures: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute data quality measures at the requested aggregation levels.

    Parameters
    ----------
    gaze_list : list[Gaze]
        All loaded gaze frames from the dataset.
    fileinfo : dict[str, pl.DataFrame] | None
        Dataset fileinfo for subject/session mapping.
    levels : list[str]
        Subset of ``['dataset', 'subject', 'session', 'trial']``.
    measures : list[str] | None
        Subset of ``['data_loss', 'std_rms', 'rms_s2s', 'bcea']``. None means all.

    Returns
    -------
    dict[str, pl.DataFrame]
        Keyed by level. Each DataFrame has grouping columns plus available measure
        columns (``data_loss``, ``std_rms``, ``rms_s2s``, ``bcea``).
    """
    all_measures = {'data_loss', 'std_rms', 'rms_s2s', 'bcea'}
    requested = set(measures) if measures is not None else all_measures

    # Extract fileinfo rows if available
    fileinfo_rows: list[dict[str, Any]] = []
    if isinstance(fileinfo, dict) and 'gaze' in fileinfo:
        try:
            fileinfo_rows = fileinfo['gaze'].to_dicts()
        except Exception:  # noqa: BLE001
            fileinfo_rows = []

    results: dict[str, pl.DataFrame] = {}

    # Build per-gaze-frame measure rows
    rows_all: list[dict[str, Any]] = []

    for i, gaze in enumerate(gaze_list):
        coord_col = _coord_column(gaze)
        fi_row = fileinfo_rows[i] if i < len(fileinfo_rows) else {}

        subject_id = fi_row.get('subject_id', str(i))
        session_id = fi_row.get('session_id', None)

        row_base: dict[str, Any] = {'subject_id': subject_id}
        if session_id is not None:
            row_base['session_id'] = session_id

        if 'data_loss' in requested and coord_col is not None:
            sampling_rate = (
                gaze.experiment.sampling_rate
                if gaze.experiment is not None
                else None
            )
            if (
                sampling_rate is not None
                and 'time' in gaze.samples.schema
                and len(gaze.samples) > 0
            ):
                try:
                    from pymovements.measure.samples.measures import (  # pylint: disable=import-outside-toplevel
                        data_loss,
                    )
                    result_df = gaze.samples.select(
                        data_loss(coord_col, sampling_rate=sampling_rate, unit='ratio'),
                    )
                    dl_val = result_df[0, 'data_loss_ratio']
                except Exception:  # noqa: BLE001
                    dl_val = _compute_data_loss_simple(gaze, coord_col)
            else:
                dl_val = _compute_data_loss_simple(gaze, coord_col) if coord_col else None
            row_base['data_loss'] = dl_val

        if coord_col is not None and requested & {'std_rms', 'rms_s2s', 'bcea'}:
            try:
                agg_exprs = _build_precision_agg(coord_col)
                prec_df = gaze.samples.select(agg_exprs)
                if 'std_rms' in requested:
                    row_base['std_rms'] = prec_df[0, 'std_rms']
                if 'rms_s2s' in requested:
                    row_base['rms_s2s'] = prec_df[0, 'rms_s2s']
                if 'bcea' in requested:
                    row_base['bcea'] = prec_df[0, 'bcea']
            except Exception:  # noqa: BLE001
                for m in ('std_rms', 'rms_s2s', 'bcea'):
                    if m in requested:
                        row_base[m] = None

        rows_all.append(row_base)

    if not rows_all:
        return results

    file_df = pl.DataFrame(rows_all)

    measure_cols = [c for c in ['data_loss', 'std_rms', 'rms_s2s', 'bcea'] if c in file_df.columns]

    if 'dataset' in levels and measure_cols:
        dataset_row: dict[str, Any] = {
            col: file_df[col].cast(pl.Float64).mean() for col in measure_cols
        }
        results['dataset'] = pl.DataFrame([dataset_row])

    if 'subject' in levels and 'subject_id' in file_df.columns and measure_cols:
        results['subject'] = file_df.group_by('subject_id').agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort('subject_id')

    if (
        'session' in levels
        and 'session_id' in file_df.columns
        and measure_cols
    ):
        group_cols = [c for c in ['subject_id', 'session_id'] if c in file_df.columns]
        results['session'] = file_df.group_by(group_cols).agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort(group_cols)

    if 'trial' in levels:
        trial_rows: list[pl.DataFrame] = []
        for i, gaze in enumerate(gaze_list):
            if not gaze.trial_columns:
                continue
            missing_tc = [c for c in gaze.trial_columns if c not in gaze.samples.schema]
            if missing_tc:
                continue
            coord_col = _coord_column(gaze)
            if coord_col is None:
                continue

            fi_row = fileinfo_rows[i] if i < len(fileinfo_rows) else {}
            subject_id = fi_row.get('subject_id', str(i))

            agg_exprs_t: list[pl.Expr] = []
            try:
                prec_exprs = _build_precision_agg(coord_col)
                if requested & {'std_rms', 'rms_s2s', 'bcea'}:
                    agg_exprs_t.extend(prec_exprs)

                if 'data_loss' in requested:
                    sampling_rate = (
                        gaze.experiment.sampling_rate
                        if gaze.experiment is not None
                        else None
                    )
                    if sampling_rate is not None and 'time' in gaze.samples.schema:
                        from pymovements.measure.samples.measures import (  # pylint: disable=import-outside-toplevel
                            data_loss,
                        )
                        agg_exprs_t.append(
                            data_loss(coord_col, sampling_rate=sampling_rate, unit='ratio'),
                        )

                if not agg_exprs_t:
                    continue

                trial_df = gaze.samples.group_by(gaze.trial_columns).agg(agg_exprs_t)
                if 'data_loss_ratio' in trial_df.columns and 'data_loss' not in trial_df.columns:
                    trial_df = trial_df.rename({'data_loss_ratio': 'data_loss'})
                trial_df = trial_df.with_columns(pl.lit(subject_id).alias('subject_id'))
                trial_rows.append(trial_df)
            except Exception:  # noqa: BLE001
                continue

        if trial_rows:
            results['trial'] = pl.concat(trial_rows, how='diagonal')

    return results
