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
"""Data quality report classes and measure computation for Gaze objects."""
from __future__ import annotations

import contextlib
import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import TextIO
from typing import TYPE_CHECKING

import polars as pl

from pymovements._utils._time import duration_to_ms
from pymovements._version import __version__
from pymovements.gaze.validation import _ALL_CHECKS
from pymovements.gaze.validation import CheckResult
from pymovements.measure.samples.measures import bcea
from pymovements.measure.samples.measures import data_loss
from pymovements.measure.samples.measures import rms_s2s
from pymovements.measure.samples.measures import std_rms

if TYPE_CHECKING:
    from pymovements.gaze.gaze import Gaze


__all__ = [
    'CheckResult',
    'DataQualityReport',
    'ValidationError',
    'DataQualityWarning',
]

_ALL_LEVELS = ('dataset', 'subject', 'session', 'trial')
_ALL_MEASURES = ('data_loss', 'std_rms', 'rms_s2s', 'bcea')


class ValidationError(Exception):
    """Raised when a validation check produces a ``'fail'`` or ``'error'``-severity result.

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
    >>> from pymovements.gaze.quality import ValidationError
    >>> try:
    ...     raise ValidationError('time_column_exists', 'missing', ['f.csv'])
    ... except ValidationError as exc:
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


class DataQualityWarning(UserWarning):
    """Warning emitted when a quality measure is degraded, skipped, or falls back.

    Raised (via :py:func:`warnings.warn`) when measure computation cannot proceed
    as intended, for example when a Polars computation fails and a simpler
    fallback is used, or when a requested aggregation level cannot be mapped to
    subject/session identifiers. These warnings are captured into
    :py:attr:`~pymovements.DataQualityReport.warning_log` in addition to being
    emitted, so the degradation is never silent.
    """


@dataclass(frozen=True)
class DataQualityReport:
    """Aggregated output of a data quality run.

    Produced by :py:meth:`~pymovements.Gaze.report_data_quality` and
    :py:meth:`~pymovements.Dataset.report_data_quality`.

    The read-only :attr:`~pymovements.DataQualityReport.passed` property returns
    ``True`` when no check result has severity ``'fail'`` or ``'error'``.

    Attributes
    ----------
    check_results : list[CheckResult]
        Results of all validation checks that were run.
    measures : dict[str, pl.DataFrame]
        Quality measures keyed by aggregation level:
        ``'dataset'``, ``'subject'``, ``'session'``, ``'trial'``.
    warning_log : list[str]
        Python warnings captured during the run. Written to ``warnings.log``
        by :py:meth:`save_bids_report`.
    passed : bool
        ``True`` when no check result has severity ``'fail'`` or ``'error'``.

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
    warning_log: list[str] = field(default_factory=list, repr=False)

    @property
    def passed(self) -> bool:
        """Return True if no check result has severity 'fail' or 'error'."""
        return all(r.severity in {'pass', 'warning'} for r in self.check_results)

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
        header = f"{'code':<35} {'severity':<10} {'n_aff':>5}  message"
        separator = '-' * len(header)
        lines = [header, separator]
        for r in self.check_results:
            n = len(r.sources)
            lines.append(f'{r.code:<35} {r.severity:<10} {n:>5}  {r.message}')
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
                'code': r.code,
                'severity': r.severity,
                'message': r.message,
                'sources': ';'.join(r.sources),
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
                ['code', 'severity', 'message', 'sources'],
            )

        checks_sidecar: dict[str, Any] = {
            'code': {'Description': 'Unique identifier of the validation check.'},
            'severity': {'Description': "Result severity: 'pass', 'warning', 'fail', or 'error'."},
            'message': {'Description': 'Human-readable description of the check outcome.'},
            'sources': {
                'Description': (
                    'Semicolon-separated list of source file paths that were checked; '
                    'present for all results.'
                ),
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


@contextlib.contextmanager
def _record_warnings() -> Iterator[list[str]]:
    """Collect warnings raised within the context while still emitting them.

    Yields a list that is filled with the string form of every warning raised
    inside the context. Each warning is also forwarded to the previously
    installed :py:func:`warnings.showwarning`, so warnings still reach the user
    instead of being swallowed.

    This temporarily replaces the process-global :py:func:`warnings.showwarning`
    and warning filters, so it is not thread-safe: concurrent warnings from other
    threads during the context are also captured and forwarded.

    Yields
    ------
    list[str]
        List populated with one message string per warning raised in the context.
    """
    captured: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        original_showwarning = warnings.showwarning

        def _record(
                message: Warning | str,
                category: type[Warning],
                filename: str,
                lineno: int,
                file: TextIO | None = None,
                line: str | None = None,
        ) -> None:
            captured.append(str(message))
            original_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _record
        yield captured


# ---------------------------------------------------------------------------
# Measure computation
# ---------------------------------------------------------------------------
# The warnings below point at the user's call of report_data_quality: each stacklevel
# counts the frames up to it through run_report and compute_measures.

def _coord_column(gaze: Any) -> str | None:
    """Return the best available coordinate column name (``'position'`` > ``'pixel'``)."""
    for col in ('position', 'pixel'):
        if col in gaze.samples.schema:
            return col
    return None


def _metadata_entry(gaze: Any, key: str) -> Any:
    """Return the *key* metadata entry of *gaze*, or ``None`` if absent."""
    return gaze.metadata.get(key)


def _for_subject(subject_id: Any) -> str:
    """Return the subject clause of a warning message, empty if the subject id is unknown."""
    return '' if subject_id is None else f' for subject {subject_id!r}'


def _metadata_source_path(gaze: Any) -> str:
    """Return the first ``Sources`` metadata entry of *gaze*, or ``''`` if there is none.

    Loaders record the file an object was read from under ``metadata['Sources']``.
    The entry identifies the gaze object in ``sources`` of each check result.

    Parameters
    ----------
    gaze : Any
        Gaze object whose metadata is read.

    Returns
    -------
    str
        First recorded source path, or ``''``.
    """
    sources = _metadata_entry(gaze, 'Sources') or []
    return str(sources[0]) if sources else ''


def _compute_data_loss_simple(gaze: Any, coord_col: str) -> float:
    """Fraction of samples where the coordinate column is null."""
    series = gaze.samples[coord_col]
    n_total = len(series)
    if n_total == 0:
        return 0.0
    n_null = int(series.is_null().sum())
    return n_null / n_total


def _fall_back_to_simple_data_loss(gaze: Any, coord_col: str, cause: str) -> float:
    """Warn that *cause* forces the simple null-ratio, then return it."""
    warnings.warn(
        f'{cause}; falling back to the simple null-ratio, which ignores time gaps and '
        'per-element NaN/inf.',
        DataQualityWarning,
        stacklevel=7,
    )
    return _compute_data_loss_simple(gaze, coord_col)


def _build_precision_agg(coord_col: str, requested: set[str]) -> list[pl.Expr]:
    """Return Polars aggregation expressions for the requested precision measures on *coord_col*.

    The measures read the first ``(x, y)`` pair of the column, which is the left
    eye for binocular data with four or six components.
    """
    precision_measures = {'std_rms': std_rms, 'rms_s2s': rms_s2s, 'bcea': bcea}
    return [
        measure(column=coord_col)
        for name, measure in precision_measures.items()
        if name in requested
    ]


def _compute_file_row(
        gaze: Any,
        subject_id: Any,
        session_id: Any,
        requested: set[str],
) -> dict[str, Any]:
    """Compute measure values for a single gaze file.

    Parameters
    ----------
    gaze : Any
        The gaze object for one file.
    subject_id : Any
        Subject identifier for this file, or None if not available.
    session_id : Any
        Session identifier, or None if not available.
    requested : set[str]
        Which measures to compute (subset of ``{'data_loss', 'std_rms', 'rms_s2s', 'bcea'}``).

    Returns
    -------
    dict[str, Any]
        Row dict with ``subject_id``, ``session_id`` and the four measure keys.
        A measure that was not requested or could not be computed is ``None``.
    """
    row: dict[str, Any] = {'subject_id': subject_id, 'session_id': session_id}
    row.update({measure: None for measure in _ALL_MEASURES})

    coord_col = _coord_column(gaze)

    if 'data_loss' in requested and coord_col is not None:
        reason = _data_loss_time_base_missing(gaze)
        if reason is not None:
            dl_val = _fall_back_to_simple_data_loss(
                gaze,
                coord_col,
                f'data_loss{_for_subject(subject_id)} cannot use the time base because {reason}',
            )
        else:
            sampling_rate = gaze.experiment.sampling_rate
            try:
                # data_loss operates on a numeric millisecond time base, so convert the
                # canonical Duration time column before computing it. A non-Duration time
                # column would raise here and fall back to the simple estimate below.
                samples = gaze.samples.with_columns(duration_to_ms('time').alias('time'))
                result_df = samples.select(
                    data_loss(coord_col, sampling_rate=sampling_rate, unit='ratio'),
                )
                dl_val = result_df[0, 'data_loss_ratio']
            except (pl.exceptions.PolarsError, ValueError, ArithmeticError) as exc:
                dl_val = _fall_back_to_simple_data_loss(
                    gaze,
                    coord_col,
                    f'data_loss computation failed{_for_subject(subject_id)} '
                    f'({type(exc).__name__}: {exc})',
                )
        row['data_loss'] = dl_val

    if coord_col is not None and requested & {'std_rms', 'rms_s2s', 'bcea'}:
        try:
            prec_df = gaze.samples.select(_build_precision_agg(coord_col, requested))
            row.update(prec_df.row(0, named=True))
        except (pl.exceptions.PolarsError, ValueError, ArithmeticError) as exc:
            warnings.warn(
                f'precision measures failed{_for_subject(subject_id)} '
                f'({type(exc).__name__}: {exc}); setting them to None.',
                DataQualityWarning,
                stacklevel=6,
            )

    return row


def _data_loss_time_base_missing(gaze: Any) -> str | None:
    """Return why ``data_loss`` cannot use the time base of *gaze*, or ``None`` if it can."""
    if gaze.experiment is None:
        return 'the gaze has no experiment'
    if gaze.experiment.sampling_rate is None:
        return 'the experiment defines no sampling rate'
    if 'time' not in gaze.samples.schema:
        return "there is no 'time' column"
    if len(gaze.samples) == 0:
        return 'there are no samples'
    return None


def _compute_trial_rows(
        gaze_list: list[Any],
        requested: set[str],
) -> list[pl.DataFrame]:
    """Compute per-trial measure DataFrames for each gaze file.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames. The subject id of each file is read from its
        ``metadata['subject_id']`` entry.
    requested : set[str]
        Which measures to compute.

    Returns
    -------
    list[pl.DataFrame]
        One DataFrame per gaze file that has valid trial columns and a
        coordinate column.
    """
    trial_rows: list[pl.DataFrame] = []
    for gaze in gaze_list:
        if not gaze.trial_columns:
            continue
        missing_tc = [c for c in gaze.trial_columns if c not in gaze.samples.schema]
        if missing_tc:
            continue
        coord_col = _coord_column(gaze)
        if coord_col is None:
            continue

        subject_id = _metadata_entry(gaze, 'subject_id')

        agg_exprs_t: list[pl.Expr] = []
        try:
            if requested & {'std_rms', 'rms_s2s', 'bcea'}:
                agg_exprs_t.extend(_build_precision_agg(coord_col, requested))

            if 'data_loss' in requested:
                reason = _data_loss_time_base_missing(gaze)
                if reason is None:
                    agg_exprs_t.append(
                        data_loss(
                            coord_col, sampling_rate=gaze.experiment.sampling_rate, unit='ratio',
                        ),
                    )
                else:
                    warnings.warn(
                        f'trial-level data_loss{_for_subject(subject_id)} cannot use the '
                        f'time base because {reason}; data_loss is None for its trials.',
                        DataQualityWarning,
                        stacklevel=5,
                    )

            if not agg_exprs_t:
                continue

            # Convert the canonical Duration time column to numeric milliseconds for data_loss.
            # A trial gaze may carry only precision measures and no time column at all, and a
            # non-Duration time column is left untouched so precision measures still compute
            # instead of the whole trial being dropped.
            samples = gaze.samples
            if 'time' in samples.columns and isinstance(samples.schema['time'], pl.Duration):
                samples = samples.with_columns(duration_to_ms('time').alias('time'))
            trial_df = samples.group_by(
                gaze.trial_columns,
            ).agg(agg_exprs_t).sort(gaze.trial_columns)
            if 'data_loss_ratio' in trial_df.columns and 'data_loss' not in trial_df.columns:
                trial_df = trial_df.rename({'data_loss_ratio': 'data_loss'})
            trial_df = trial_df.with_columns(
                # The dtype a DataFrame infers, as for the file rows (pl.lit gives Int32
                # for an int).
                pl.lit(subject_id, dtype=pl.Series([subject_id]).dtype).alias('subject_id'),
                *[
                    pl.lit(None, dtype=pl.Float64).alias(measure)
                    for measure in _ALL_MEASURES
                    if measure in requested and measure not in trial_df.columns
                ],
            )
            trial_rows.append(trial_df)
        except (pl.exceptions.PolarsError, ValueError, ArithmeticError) as exc:
            warnings.warn(
                f'trial-level measures failed{_for_subject(subject_id)} '
                f'({type(exc).__name__}: {exc}); skipping this file at trial level.',
                DataQualityWarning,
                stacklevel=5,
            )
            continue

    return trial_rows


def _warn_missing_ids(gaze_list: list[Any], level: str) -> bool:
    """Warn about gaze objects missing ``'<level>_id'``, return whether the level is omitted.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames, at least one.
    level : str
        ``'subject'`` or ``'session'``.

    Returns
    -------
    bool
        ``True`` if the level is omitted, which happens only for the session
        level when no gaze object has a session id. Otherwise the gaze objects
        without an id are grouped under a null id.
    """
    key = f'{level}_id'
    n_missing = sum(_metadata_entry(gaze, key) is None for gaze in gaze_list)
    if not n_missing:
        return False
    prefix = f"{level}-level aggregation requested but '{key}' is not a metadata key of "
    if level == 'session' and n_missing == len(gaze_list):
        warnings.warn(
            prefix + 'any gaze object; the session level will be omitted.',
            DataQualityWarning,
            stacklevel=5,
        )
        return True
    warnings.warn(
        prefix + f'{n_missing} of {len(gaze_list)} gaze objects; grouping them '
        f'under a null {level} id.',
        DataQualityWarning,
        stacklevel=5,
    )
    return False


def _aggregate_file_rows(
        gaze_list: list[Any],
        levels: set[str],
        requested: set[str],
) -> dict[str, pl.DataFrame]:
    """Compute one row per gaze object and aggregate the rows to the file-based levels.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames. Subject and session ids are read from each
        gaze object's ``metadata``.
    levels : set[str]
        The file-based levels to compute, a subset of ``{'dataset', 'subject',
        'session'}``.
    requested : set[str]
        Which measures to compute.

    Returns
    -------
    dict[str, pl.DataFrame]
        One DataFrame per level, if any measure is requested.
    """
    results: dict[str, pl.DataFrame] = {}
    rows_all = [
        _compute_file_row(
            gaze,
            _metadata_entry(gaze, 'subject_id'),
            _metadata_entry(gaze, 'session_id'),
            requested,
        )
        for gaze in gaze_list
    ]

    file_df = pl.DataFrame(rows_all)
    measure_cols = [measure for measure in _ALL_MEASURES if measure in requested]

    if 'dataset' in levels and measure_cols:
        dataset_row: dict[str, Any] = {
            col: file_df[col].cast(pl.Float64).mean() for col in measure_cols
        }
        results['dataset'] = pl.DataFrame([dataset_row])

    if 'subject' in levels and measure_cols:
        results['subject'] = file_df.group_by('subject_id').agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort('subject_id')

    if 'session' in levels and measure_cols:
        group_cols = ['subject_id', 'session_id']
        results['session'] = file_df.group_by(group_cols).agg(
            [pl.col(c).cast(pl.Float64).mean() for c in measure_cols],
        ).sort(group_cols)

    return results


def compute_measures(
        gaze_list: list[Any],
        levels: list[str],
        measures: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute data quality measures at the requested aggregation levels.

    Parameters
    ----------
    gaze_list : list[Any]
        All loaded gaze frames from the dataset. Subject and session aggregation
        read the ``'subject_id'`` and ``'session_id'`` entries of each frame's
        ``metadata``, which :py:meth:`~pymovements.Dataset.load` fills from the
        filename pattern. A :py:class:`DataQualityWarning` is emitted if a
        requested level's key is absent from any frame, and those frames are
        grouped under a null id. The session level is omitted if no frame has
        a session id.
    levels : list[str]
        Subset of ``['dataset', 'subject', 'session', 'trial']``.
    measures : list[str] | None
        Subset of ``['data_loss', 'std_rms', 'rms_s2s', 'bcea']``. None means all.
        The precision measures read the first ``(x, y)`` pair of the coordinate
        column, which is the left eye for binocular data.

    Returns
    -------
    dict[str, pl.DataFrame]
        Keyed by level. Each DataFrame has grouping columns plus one column per
        requested measure. A measure that could not be computed is null.
    """
    requested = set(measures) if measures is not None else set(_ALL_MEASURES)

    if not gaze_list:
        return {}

    # Warn when subject/session aggregation is requested but the metadata lacks the
    # expected identifier (datasets may name the groups differently). Those gaze objects
    # are grouped under a null id, the session level is omitted if none has a session id.
    file_levels = {'dataset', 'subject', 'session'} & set(levels)
    for level in ('subject', 'session'):
        if level in levels and _warn_missing_ids(gaze_list, level):
            file_levels.discard(level)

    results: dict[str, pl.DataFrame] = {}
    if file_levels:
        results.update(_aggregate_file_rows(gaze_list, file_levels, requested))

    if 'trial' in levels:
        trial_rows = _compute_trial_rows(gaze_list, requested)
        if trial_rows:
            # diagonal_relaxed: a null subject id of one file joins the typed ids of others.
            results['trial'] = pl.concat(trial_rows, how='diagonal_relaxed')

    return results


def _check_identifiers(kind: str, names: list[str], valid: list[str]) -> None:
    """Raise a ``ValueError`` listing the valid identifiers if any name is unknown."""
    unknown = set(names) - set(valid)
    if unknown:
        raise ValueError(
            f'Unknown {kind} identifier(s) {sorted(unknown)!r}. '
            f'Valid identifiers: {valid!r}',
        )


def run_report(
        gaze_source_pairs: list[tuple[Gaze, str]],
        *,
        checks: list[str] | None,
        measures: list[str] | None,
        levels: list[str] | None,
        raise_on_error: bool,
        output_path: Path | str | None,
        max_gap_factor: float = 5.0,
        max_deviation: float = 0.05,
        min_fraction: float = 0.95,
) -> DataQualityReport:
    """Run validation checks and measures for one or more gaze objects.

    Shared implementation behind :py:meth:`~pymovements.Gaze.report_data_quality`
    and :py:meth:`~pymovements.Dataset.report_data_quality`.

    Parameters
    ----------
    gaze_source_pairs : list[tuple[Gaze, str]]
        ``(gaze, source_path)`` pairs to report on. ``source_path`` identifies the
        gaze object in ``sources`` of each check result and in ``affected_files``
        of a raised ``ValidationError``. Subject and session ids are read from
        each gaze object's ``metadata``.
    checks : list[str] | None
        Check identifiers to run. ``None`` runs all eight.
    measures : list[str] | None
        Measure identifiers to compute. ``None`` computes all four.
    levels : list[str] | None
        Aggregation levels to compute. ``None`` uses all four, without ``'session'``
        if no gaze object has a ``'session_id'`` metadata entry.
    raise_on_error : bool
        Raise :py:class:`ValidationError` on the first ``'fail'`` or ``'error'`` result.
    output_path : Path | str | None
        If given, write BIDS derivative files here.
    max_gap_factor : float
        Passed to :py:func:`~pymovements.gaze.validation.check_max_gap`.
        (default: 5.0)
    max_deviation : float
        Passed to
        :py:func:`~pymovements.gaze.validation.check_sampling_rate_consistency`.
        (default: 0.05)
    min_fraction : float
        Passed to :py:func:`~pymovements.gaze.validation.check_gaze_range`.
        (default: 0.95)

    Returns
    -------
    DataQualityReport
        Populated report object.

    Raises
    ------
    ValidationError
        If *raise_on_error* is ``True`` and any check produces an error.
    ValueError
        If any name in *checks*, *measures* or *levels* is not a valid identifier.
    """
    if checks is not None:
        _check_identifiers('check', checks, list(_ALL_CHECKS))
    if measures is not None:
        _check_identifiers('measure', measures, list(_ALL_MEASURES))
    if levels is None:
        has_session_ids = any(
            _metadata_entry(gaze, 'session_id') is not None for gaze, _ in gaze_source_pairs
        )
        levels = [level for level in _ALL_LEVELS if level != 'session' or has_session_ids]
    else:
        _check_identifiers('level', levels, list(_ALL_LEVELS))
    checks_to_run = set(checks) if checks is not None else set(_ALL_CHECKS)

    check_results: list[CheckResult] = []

    with _record_warnings() as captured:
        for gaze, source_path in gaze_source_pairs:
            validate_results = gaze.validate(
                trial_columns_exist='trial_columns_exist' in checks_to_run,
                trial_columns_dtype='trial_columns_dtype' in checks_to_run,
                time_column_exists='time_column_exists' in checks_to_run,
                gaze_components_defined='gaze_components_defined' in checks_to_run,
                time_monotone='time_monotone' in checks_to_run,
                max_gap='max_gap' in checks_to_run,
                max_gap_factor=max_gap_factor,
                sampling_rate_consistency='sampling_rate_consistency' in checks_to_run,
                max_deviation=max_deviation,
                gaze_range='gaze_range' in checks_to_run,
                min_fraction=min_fraction,
                source_path=source_path,
            )
            for result in validate_results:
                check_results.append(result)
                if raise_on_error and result.severity in {'fail', 'error'}:
                    raise ValidationError(
                        check_id=result.code,
                        message=str(result.message),
                        affected_files=result.sources,
                    )

        gaze_list = [gaze for gaze, _ in gaze_source_pairs]
        measure_results = compute_measures(gaze_list, levels, measures)

    report = DataQualityReport(
        check_results=check_results,
        measures=measure_results,
        warning_log=captured,
    )

    if output_path is not None:
        report.save_bids_report(Path(output_path))

    return report
