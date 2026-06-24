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
"""Data quality report classes and measure computation for Gaze objects.

Placing these here (inside the ``gaze`` package) avoids a circular import:
``gaze.gaze`` cannot import from ``dataset.data_quality`` because
``dataset.__init__`` imports ``dataset.dataset`` which imports
``pymovements.gaze.Gaze``. Keeping the classes in the ``gaze`` package
breaks that cycle while letting :py:class:`~pymovements.Dataset` re-export
them from :py:mod:`~pymovements.dataset.data_quality` for backwards
compatibility.
"""
from __future__ import annotations

import json
import warnings as _warnings
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import polars as pl

from pymovements._version import __version__
from pymovements.gaze.validation import _ALL_CHECKS
from pymovements.gaze.validation import CheckResult
from pymovements.measure.samples.measures import bcea
from pymovements.measure.samples.measures import data_loss
from pymovements.measure.samples.measures import rms_s2s
from pymovements.measure.samples.measures import std_rms


__all__ = [
    'CheckResult',
    'DataQualityReport',
    'GazeDataValidationError',
]


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

    Examples
    --------
    >>> from pymovements.gaze.quality import GazeDataValidationError
    >>> try:
    ...     raise GazeDataValidationError('time_column_exists', 'missing', ['f.csv'])
    ... except GazeDataValidationError as exc:
    ...     print(exc.check_id)
    time_column_exists
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
class DataQualityReport:
    """Aggregated output of a data quality run.

    Produced by :py:meth:`~pymovements.gaze.gaze.Gaze.report_data_quality`
    and :py:meth:`~pymovements.Dataset.report_data_quality`.

    Attributes
    ----------
    check_results : list[CheckResult]
        Results of all validation checks that were run.
    measures : dict[str, pl.DataFrame]
        Quality measures keyed by aggregation level:
        ``'dataset'``, ``'subject'``, ``'session'``, ``'trial'``.
    passed : bool
        ``True`` if no check result has severity ``'error'``. Computed from
        ``check_results`` at construction time.
    warning_log : list[str]
        Python warnings captured during the run. Written to ``warnings.log``
        by :py:meth:`save_bids_report`.

    Examples
    --------
    >>> from pymovements.gaze.quality import CheckResult, DataQualityReport
    >>> report = DataQualityReport(
    ...     check_results=[CheckResult('time_column_exists', 'pass', 'OK')],
    ... )
    >>> report.passed
    True
    """

    check_results: list[CheckResult] = field(default_factory=list)
    measures: dict[str, pl.DataFrame] = field(default_factory=dict)
    passed: bool = field(init=False)
    warning_log: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise ``passed`` from ``check_results``."""
        self.passed = all(r.severity != 'error' for r in self.check_results)

    def summary(self) -> str:
        """Return a formatted summary table of all check results.

        Returns
        -------
        str
            A plain-text table with columns ``check_id``, ``severity``,
            ``n_affected``, and ``message``.

        Examples
        --------
        >>> from pymovements.gaze.quality import CheckResult, DataQualityReport
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
          :py:class:`~pymovements.CheckResult`.
        * ``data_quality_measures_{level}.tsv`` + ``.json`` sidecar — one file
          per aggregation level present in :py:attr:`measures`.
        * ``warnings.log`` — one line per Python warning captured during the
          run.

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
        >>> from pymovements.gaze.quality import DataQualityReport, CheckResult
        >>> report = DataQualityReport(
        ...     check_results=[CheckResult('trial_columns_exist', 'pass', 'All OK')],
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     report.save_bids_report(pathlib.Path(tmp))
        """
        deriv_dir = Path(output_path) / 'derivatives' / pipeline_name
        deriv_dir.mkdir(parents=True, exist_ok=True)

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

        (deriv_dir / 'warnings.log').write_text('\n'.join(self.warning_log), encoding='utf-8')


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
# Measure computation
# ---------------------------------------------------------------------------

def _coord_column(gaze: Any) -> str | None:
    """Return the best available coordinate column name (``'position'`` > ``'pixel'``)."""
    for col in ('position', 'pixel'):
        if col in gaze.samples.schema:
            return col
    return None


def _compute_data_loss_simple(gaze: Any, coord_col: str) -> float:
    """Fraction of samples where the coordinate column is null."""
    series = gaze.samples[coord_col]
    n_total = len(series)
    if n_total == 0:
        return 0.0
    n_null = int(series.is_null().sum())
    return n_null / n_total


def _build_precision_agg(coord_col: str) -> list[pl.Expr]:
    """Return Polars aggregation expressions for precision measures on *coord_col*."""
    return [
        std_rms(column=coord_col),
        rms_s2s(column=coord_col),
        bcea(column=coord_col),
    ]


def _compute_file_row(
        gaze: Any,
        subject_id: str,
        session_id: str | None,
        requested: set[str],
) -> dict[str, Any]:
    """Compute measure values for a single gaze file.

    Parameters
    ----------
    gaze : Any
        The gaze object for one file.
    subject_id : str
        Subject identifier for this file.
    session_id : str | None
        Session identifier, or None if not available.
    requested : set[str]
        Which measures to compute (subset of ``{'data_loss', 'std_rms', 'rms_s2s', 'bcea'}``).

    Returns
    -------
    dict[str, Any]
        Row dict with ``subject_id`` (and optionally ``session_id``) plus any
        computed measure values.
    """
    row: dict[str, Any] = {'subject_id': subject_id}
    if session_id is not None:
        row['session_id'] = session_id

    coord_col = _coord_column(gaze)

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
                result_df = gaze.samples.select(
                    data_loss(coord_col, sampling_rate=sampling_rate, unit='ratio'),
                )
                dl_val = result_df[0, 'data_loss_ratio']
            except (pl.exceptions.PolarsError, ValueError, ArithmeticError):
                dl_val = _compute_data_loss_simple(gaze, coord_col)
        else:
            dl_val = _compute_data_loss_simple(gaze, coord_col) if coord_col else None
        row['data_loss'] = dl_val

    if coord_col is not None and requested & {'std_rms', 'rms_s2s', 'bcea'}:
        try:
            agg_exprs = _build_precision_agg(coord_col)
            prec_df = gaze.samples.select(agg_exprs)
            if 'std_rms' in requested:
                row['std_rms'] = prec_df[0, 'std_rms']
            if 'rms_s2s' in requested:
                row['rms_s2s'] = prec_df[0, 'rms_s2s']
            if 'bcea' in requested:
                row['bcea'] = prec_df[0, 'bcea']
        except (pl.exceptions.PolarsError, ValueError, ArithmeticError):
            row.update({m: None for m in ('std_rms', 'rms_s2s', 'bcea') if m in requested})

    return row


def _compute_trial_rows(
        gaze_list: list[Any],
        fileinfo_rows: list[dict[str, Any]],
        requested: set[str],
) -> list[pl.DataFrame]:
    """Compute per-trial measure DataFrames for each gaze file.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames.
    fileinfo_rows : list[dict[str, Any]]
        Per-file metadata rows (subject_id, session_id, etc.).
    requested : set[str]
        Which measures to compute.

    Returns
    -------
    list[pl.DataFrame]
        One DataFrame per gaze file that has valid trial columns and a
        coordinate column.
    """
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
            if requested & {'std_rms', 'rms_s2s', 'bcea'}:
                agg_exprs_t.extend(_build_precision_agg(coord_col))

            if 'data_loss' in requested:
                sampling_rate = (
                    gaze.experiment.sampling_rate
                    if gaze.experiment is not None
                    else None
                )
                if sampling_rate is not None and 'time' in gaze.samples.schema:
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
        except (pl.exceptions.PolarsError, ValueError, ArithmeticError):
            continue

    return trial_rows


def _compute_measures(
        gaze_list: list[Any],
        fileinfo: Any,
        levels: list[str],
        measures: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute data quality measures at the requested aggregation levels.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames from the dataset.
    fileinfo : Any
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

    fileinfo_rows: list[dict[str, Any]] = []
    if isinstance(fileinfo, dict) and 'gaze' in fileinfo:
        try:
            fileinfo_rows = fileinfo['gaze'].to_dicts()
        except (AttributeError, pl.exceptions.PolarsError):
            fileinfo_rows = []

    results: dict[str, pl.DataFrame] = {}
    rows_all: list[dict[str, Any]] = []

    for i, gaze in enumerate(gaze_list):
        fi_row = fileinfo_rows[i] if i < len(fileinfo_rows) else {}
        subject_id = fi_row.get('subject_id', str(i))
        session_id = fi_row.get('session_id', None)
        rows_all.append(_compute_file_row(gaze, subject_id, session_id, requested))

    if not rows_all:
        return results

    file_df = pl.DataFrame(rows_all)
    measure_cols = [c for c in ('data_loss', 'std_rms', 'rms_s2s', 'bcea') if c in file_df.columns]

    if 'dataset' in levels and measure_cols:
        dataset_row: dict[str, Any] = {
            col: file_df[col].cast(pl.Float64).mean() for col in measure_cols
        }
        results['dataset'] = pl.DataFrame([dataset_row])

    if 'subject' in levels and 'subject_id' in file_df.columns and measure_cols:
        results['subject'] = file_df.group_by('subject_id').agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort('subject_id')

    if 'session' in levels and 'session_id' in file_df.columns and measure_cols:
        group_cols = [c for c in ('subject_id', 'session_id') if c in file_df.columns]
        results['session'] = file_df.group_by(group_cols).agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort(group_cols)

    if 'trial' in levels:
        trial_rows = _compute_trial_rows(gaze_list, fileinfo_rows, requested)
        if trial_rows:
            results['trial'] = pl.concat(trial_rows, how='diagonal')

    return results


def _run_report(
        gaze: Any,
        checks: list[str] | None,
        measures: list[str] | None,
        levels: list[str] | None,
        raise_on_error: bool,
        output_path: Path | str | None,
        source_path: str,
) -> DataQualityReport:
    """Core implementation of ``Gaze.report_data_quality()``.

    Parameters
    ----------
    gaze : Any
        The :py:class:`~pymovements.gaze.gaze.Gaze` object to report on.
    checks : list[str] | None
        Check identifiers to run; ``None`` runs all seven.
    measures : list[str] | None
        Measure identifiers to compute; ``None`` computes all four.
    levels : list[str] | None
        Aggregation levels; ``None`` defaults to ``['dataset', 'trial']``.
    raise_on_error : bool
        Raise :py:class:`GazeDataValidationError` on the first error result.
    output_path : Path | str | None
        If given, write BIDS derivative files here.
    source_path : str
        Identifier for the gaze object used in ``affected_files``.

    Returns
    -------
    DataQualityReport
        Populated report object.

    Raises
    ------
    GazeDataValidationError
        If *raise_on_error* is ``True`` and any check produces an error.
    ValueError
        If any name in *checks* is not a valid check identifier.
    """
    checks_to_run = set(checks) if checks is not None else set(_ALL_CHECKS.keys())
    if checks is not None:
        unknown = checks_to_run - set(_ALL_CHECKS.keys())
        if unknown:
            raise ValueError(
                f'Unknown check identifier(s) {sorted(unknown)!r}. '
                f'Valid identifiers: {list(_ALL_CHECKS.keys())!r}',
            )

    levels_to_run = levels if levels is not None else ['dataset', 'trial']

    report = DataQualityReport()
    captured: list[str] = []

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')

        results = gaze.validate(
            trial_columns_exist='trial_columns_exist' in checks_to_run,
            trial_columns_dtype='trial_columns_dtype' in checks_to_run,
            time_column_exists='time_column_exists' in checks_to_run,
            gaze_components_defined='gaze_components_defined' in checks_to_run,
            trial_continuity='trial_continuity' in checks_to_run,
            sampling_rate_consistency='sampling_rate_consistency' in checks_to_run,
            gaze_range='gaze_range' in checks_to_run,
            source_path=source_path,
        )
        for result in results:
            report.check_results.append(result)
            if raise_on_error and result.severity == 'error':
                raise GazeDataValidationError(
                    check_id=result.check_id,
                    message=str(result.message),
                    affected_files=result.affected_files,
                )

        report.measures = _compute_measures([gaze], None, levels_to_run, measures)
        captured = [str(w.message) for w in caught]

    report.passed = all(r.severity != 'error' for r in report.check_results)
    report.warning_log = captured

    if output_path is not None:
        report.save_bids_report(Path(output_path))

    return report
