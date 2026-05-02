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
"""Tests for pymovements.dataset.data_quality."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pymovements.dataset.data_quality import CheckResult
from pymovements.dataset.data_quality import DataQualityReport
from pymovements.dataset.data_quality import GazeDataValidationError
from pymovements.dataset.data_quality import _ALL_CHECKS
from pymovements.dataset.data_quality import _compute_measures
from pymovements.dataset.data_quality import check_gaze_components_defined
from pymovements.dataset.data_quality import check_gaze_range
from pymovements.dataset.data_quality import check_sampling_rate_consistency
from pymovements.dataset.data_quality import check_time_column_exists
from pymovements.dataset.data_quality import check_trial_columns_dtype
from pymovements.dataset.data_quality import check_trial_columns_exist
from pymovements.dataset.data_quality import check_trial_continuity
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.gaze.screen import Screen


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_gaze(
        samples: pl.DataFrame,
        trial_columns: list[str] | None = None,
        experiment: Experiment | None = None,
) -> Gaze:
    """Create a Gaze object from a ready-made samples DataFrame (no column renaming)."""
    g = Gaze.__new__(Gaze)
    g.samples = samples
    g.trial_columns = trial_columns
    g.experiment = experiment
    g.n_components = None
    g.events = None  # type: ignore[assignment]
    g.metadata = {}
    g.messages = None
    g.calibrations = None
    g.validations = None
    g._metadata = None
    return g


def _simple_experiment(sampling_rate: float = 1000.0) -> Experiment:
    """Return an Experiment with a screen and sampling_rate."""
    return Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38.0,
        screen_height_cm=30.0,
        distance_cm=68.0,
        origin='upper left',
        sampling_rate=sampling_rate,
    )


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_fields(self) -> None:
        r = CheckResult('my_check', 'pass', 'All good', ['f1.csv'])
        assert r.check_id == 'my_check'
        assert r.severity == 'pass'
        assert r.message == 'All good'
        assert r.affected_files == ['f1.csv']

    def test_default_affected_files(self) -> None:
        r = CheckResult('x', 'pass', 'ok')
        assert r.affected_files == []


# ---------------------------------------------------------------------------
# GazeDataValidationError
# ---------------------------------------------------------------------------

class TestGazeDataValidationError:
    def test_raise(self) -> None:
        with pytest.raises(GazeDataValidationError) as exc_info:
            raise GazeDataValidationError('c', 'bad', ['f.csv'])
        assert exc_info.value.check_id == 'c'
        assert exc_info.value.affected_files == ['f.csv']
        assert 'bad' in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_trial_columns_exist
# ---------------------------------------------------------------------------

class TestCheckTrialColumnsExist:
    def test_pass_no_trial_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2]}))
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'pass'

    def test_pass_columns_present(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': [1, 1]}),
            trial_columns=['trial'],
        )
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'pass'

    def test_error_column_absent(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1]}),
            trial_columns=['trial_id'],
        )
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'error'
        assert 'trial_id' in result.message

    def test_error_partial_absence(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'subject': [1, 1]}),
            trial_columns=['subject', 'trial'],
        )
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'error'
        assert 'trial' in result.message

    def test_affected_files_set(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['missing'],
        )
        result = check_trial_columns_exist(gaze, source_path='data/s1.csv')
        assert 'data/s1.csv' in result.affected_files

    def test_affected_files_empty_on_pass(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_trial_columns_exist(gaze)
        assert result.affected_files == []


# ---------------------------------------------------------------------------
# check_trial_columns_dtype
# ---------------------------------------------------------------------------

class TestCheckTrialColumnsDtype:
    def test_pass_no_trial_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'pass'

    def test_pass_integer_dtype(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': pl.Series([1, 2], dtype=pl.Int32)}),
            trial_columns=['trial'],
        )
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'pass'

    def test_pass_string_dtype(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': ['t1', 't2']}),
            trial_columns=['trial'],
        )
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'pass'

    def test_warning_float_dtype(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': pl.Series([1.0, 2.0], dtype=pl.Float64)}),
            trial_columns=['trial'],
        )
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'warning'
        assert 'trial' in result.message

    def test_warning_float32_dtype(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': pl.Series([1.0], dtype=pl.Float32)}),
            trial_columns=['trial'],
        )
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'warning'


# ---------------------------------------------------------------------------
# check_time_column_exists
# ---------------------------------------------------------------------------

class TestCheckTimeColumnExists:
    def test_pass_time_present_integer(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': pl.Series([0, 1, 2], dtype=pl.Int64)}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'pass'

    def test_pass_time_present_float(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': pl.Series([0.0, 1.0], dtype=pl.Float64)}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'pass'

    def test_error_time_absent(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [1.0, 2.0]}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'error'
        assert 'time' in result.message

    def test_error_time_string_dtype(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': ['t0', 't1']}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'error'
        assert 'time' in result.message

    def test_affected_files_on_error(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [0.0]}))
        result = check_time_column_exists(gaze, source_path='s/f.csv')
        assert 's/f.csv' in result.affected_files


# ---------------------------------------------------------------------------
# check_gaze_components_defined
# ---------------------------------------------------------------------------

class TestCheckGazeComponentsDefined:
    def test_pass_position_column(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'position': [[1.0, 2.0]]}),
        )
        result = check_gaze_components_defined(gaze)
        assert result.severity == 'pass'

    def test_pass_pixel_column(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'pixel': [[100.0, 200.0]]}),
        )
        result = check_gaze_components_defined(gaze)
        assert result.severity == 'pass'

    def test_pass_velocity_column(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'velocity': [[0.1, 0.2]]}),
        )
        result = check_gaze_components_defined(gaze)
        assert result.severity == 'pass'

    def test_error_no_coordinate_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0], 'trial': [1]}))
        result = check_gaze_components_defined(gaze)
        assert result.severity == 'error'

    def test_affected_files_on_error(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_gaze_components_defined(gaze, source_path='data.csv')
        assert 'data.csv' in result.affected_files


# ---------------------------------------------------------------------------
# check_trial_continuity
# ---------------------------------------------------------------------------

class TestCheckTrialContinuity:
    def test_pass_no_trial_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2, 3]}))
        result = check_trial_continuity(gaze)
        assert result.severity == 'pass'

    def test_pass_monotone_increasing(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 30], 'trial': [1, 1, 2, 2]}),
            trial_columns=['trial'],
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'pass'

    def test_warning_non_monotone(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 20, 10, 30], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'warning'

    def test_warning_gap_exceeds_5x_isi(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        # ISI = 10ms; 5x ISI = 50ms; insert 100ms gap
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 120], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'warning'
        assert 'gap' in result.message.lower()

    def test_pass_no_time_column(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'x': [0.0], 'trial': [1]}),
            trial_columns=['trial'],
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_missing_trial_column_in_schema(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1]}),
            trial_columns=['nonexistent'],
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'pass'

    def test_single_sample_per_trial_skipped(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': [1, 2]}),
            trial_columns=['trial'],
        )
        result = check_trial_continuity(gaze)
        assert result.severity == 'pass'


# ---------------------------------------------------------------------------
# check_sampling_rate_consistency
# ---------------------------------------------------------------------------

class TestCheckSamplingRateConsistency:
    def test_pass_no_experiment(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 10, 20]}))
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_consistent_rate(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 30]}),
            experiment=exp,
        )
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'pass'

    def test_warning_inconsistent_rate(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        # Timestamps spaced 5ms apart → empirical rate 200Hz ≠ 100Hz
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 5, 10, 15, 20]}),
            experiment=exp,
        )
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'warning'
        assert '200' in result.message or '100' in result.message

    def test_pass_too_few_samples(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(pl.DataFrame({'time': [0]}), experiment=exp)
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_affected_files_on_warning(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 5, 10, 15, 20]}),
            experiment=exp,
        )
        result = check_sampling_rate_consistency(gaze, source_path='data.asc')
        if result.severity == 'warning':
            assert 'data.asc' in result.affected_files


# ---------------------------------------------------------------------------
# check_gaze_range
# ---------------------------------------------------------------------------

class TestCheckGazeRange:
    def test_pass_no_experiment(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'position': [[0.5, 0.5]]}),
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_all_in_range(self) -> None:
        exp = _simple_experiment()
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 1, 2],
                'position': [[-5.0, -5.0], [0.0, 0.0], [5.0, 5.0]],
            }),
            experiment=exp,
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'pass'

    def test_warning_mostly_out_of_range(self) -> None:
        exp = _simple_experiment()
        # Far outside screen bounds in DVA
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(20)),
                'position': [[-999.0, -999.0]] * 20,
            }),
            experiment=exp,
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'warning'
        assert '%' in result.message

    def test_pass_no_coord_column(self) -> None:
        exp = _simple_experiment()
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': [1]}),
            experiment=exp,
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_all_null_samples(self) -> None:
        exp = _simple_experiment()
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'position': [None, None]}),
            experiment=exp,
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'pass'

    def test_pixel_column_fallback(self) -> None:
        exp = Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=38.0,
            screen_height_cm=30.0,
            distance_cm=68.0,
            origin='upper left',
            sampling_rate=1000.0,
        )
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 1, 2],
                'pixel': [[100.0, 100.0], [640.0, 512.0], [1100.0, 900.0]],
            }),
            experiment=exp,
        )
        result = check_gaze_range(gaze)
        assert result.severity == 'pass'


# ---------------------------------------------------------------------------
# _ALL_CHECKS registry
# ---------------------------------------------------------------------------

class TestAllChecks:
    def test_all_seven_checks_registered(self) -> None:
        expected = {
            'trial_columns_exist',
            'trial_columns_dtype',
            'time_column_exists',
            'gaze_components_defined',
            'trial_continuity',
            'sampling_rate_consistency',
            'gaze_range',
        }
        assert set(_ALL_CHECKS.keys()) == expected

    @pytest.mark.parametrize('check_id', list(_ALL_CHECKS.keys()))
    def test_each_check_callable(self, check_id: str) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        result = _ALL_CHECKS[check_id](gaze)
        assert isinstance(result, CheckResult)
        assert result.severity in {'pass', 'warning', 'error'}


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------

class TestDataQualityReport:
    def test_passed_true_when_no_errors(self) -> None:
        report = DataQualityReport(
            check_results=[
                CheckResult('a', 'pass', 'ok'),
                CheckResult('b', 'warning', 'watch out'),
            ],
        )
        assert report.passed is True

    def test_passed_false_when_error_present(self) -> None:
        report = DataQualityReport(
            check_results=[CheckResult('a', 'error', 'broken', ['f.csv'])],
        )
        assert report.passed is False

    def test_summary_returns_string(self) -> None:
        report = DataQualityReport(
            check_results=[
                CheckResult('trial_columns_exist', 'pass', 'All OK'),
                CheckResult('time_column_exists', 'error', 'Missing', ['f.csv']),
            ],
        )
        s = report.summary()
        assert isinstance(s, str)
        assert 'trial_columns_exist' in s
        assert 'error' in s

    def test_summary_empty_report(self) -> None:
        report = DataQualityReport()
        s = report.summary()
        assert isinstance(s, str)

    def test_measures_default_empty(self) -> None:
        report = DataQualityReport()
        assert report.measures == {}


class TestSaveBidsReport:
    def test_creates_expected_files(self, tmp_path: Path) -> None:
        report = DataQualityReport(
            check_results=[CheckResult('trial_columns_exist', 'pass', 'OK')],
        )
        report.save_bids_report(tmp_path)

        deriv = tmp_path / 'derivatives' / 'pymovements'
        assert (deriv / 'dataset_description.json').exists()
        assert (deriv / 'data_quality_checks.tsv').exists()
        assert (deriv / 'data_quality_checks.json').exists()
        assert (deriv / 'warnings.log').exists()

    def test_dataset_description_valid_json(self, tmp_path: Path) -> None:
        report = DataQualityReport()
        report.save_bids_report(tmp_path)

        content = json.loads(
            (tmp_path / 'derivatives' / 'pymovements' / 'dataset_description.json').read_text(),
        )
        assert content['DatasetType'] == 'derivative'
        assert content['BIDSVersion'] == '1.11.1'
        assert 'GeneratedBy' in content
        assert content['GeneratedBy'][0]['Name'] == 'pymovements'

    def test_checks_tsv_has_correct_columns(self, tmp_path: Path) -> None:
        report = DataQualityReport(
            check_results=[CheckResult('x', 'warning', 'msg', ['f1.csv', 'f2.csv'])],
        )
        report.save_bids_report(tmp_path)

        tsv = (tmp_path / 'derivatives' / 'pymovements' / 'data_quality_checks.tsv').read_text()
        header_cols = tsv.splitlines()[0].split('\t')
        assert 'check_id' in header_cols
        assert 'severity' in header_cols
        assert 'affected_files' in header_cols

    def test_custom_pipeline_name(self, tmp_path: Path) -> None:
        report = DataQualityReport()
        report.save_bids_report(tmp_path, pipeline_name='mylab')
        assert (tmp_path / 'derivatives' / 'mylab' / 'dataset_description.json').exists()

    def test_measure_tsv_written_per_level(self, tmp_path: Path) -> None:
        report = DataQualityReport(
            measures={
                'dataset': pl.DataFrame({'data_loss': [0.05]}),
                'trial': pl.DataFrame({'trial': [1, 2], 'data_loss': [0.01, 0.02]}),
            },
        )
        report.save_bids_report(tmp_path)

        deriv = tmp_path / 'derivatives' / 'pymovements'
        assert (deriv / 'data_quality_measures_dataset.tsv').exists()
        assert (deriv / 'data_quality_measures_dataset.json').exists()
        assert (deriv / 'data_quality_measures_trial.tsv').exists()
        assert (deriv / 'data_quality_measures_trial.json').exists()

    def test_warnings_log_written(self, tmp_path: Path) -> None:
        report = DataQualityReport()
        report._warning_log = ['UserWarning: something went wrong']  # type: ignore[attr-defined]
        report.save_bids_report(tmp_path)

        log = (tmp_path / 'derivatives' / 'pymovements' / 'warnings.log').read_text()
        assert 'something went wrong' in log


# ---------------------------------------------------------------------------
# _compute_measures
# ---------------------------------------------------------------------------

class TestComputeMeasures:
    def test_empty_gaze_list_returns_empty(self) -> None:
        result = _compute_measures([], None, ['dataset', 'trial'])
        assert isinstance(result, dict)

    def test_dataset_level_returned(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(10)),
                'position': [[float(i), float(i)] for i in range(10)],
            }),
            experiment=exp,
        )
        result = _compute_measures([gaze], None, ['dataset'])
        assert 'dataset' in result
        assert isinstance(result['dataset'], pl.DataFrame)
        assert len(result['dataset']) == 1

    def test_trial_level_with_trial_columns(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(6)),
                'trial': [1, 1, 1, 2, 2, 2],
                'position': [[float(i), float(i)] for i in range(6)],
            }),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = _compute_measures([gaze], None, ['trial'])
        assert 'trial' in result
        assert len(result['trial']) == 2

    def test_selected_measures_only(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 1, 2],
                'position': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            }),
            experiment=exp,
        )
        result = _compute_measures([gaze], None, ['dataset'], measures=['data_loss'])
        if 'dataset' in result:
            assert 'data_loss' in result['dataset'].columns
            assert 'std_rms' not in result['dataset'].columns

    def test_no_coord_column_still_returns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2], 'trial': [1, 1, 1]}))
        result = _compute_measures([gaze], None, ['dataset'])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Integration: Dataset.report_data_quality
# ---------------------------------------------------------------------------

class TestDatasetReportDataQuality:
    """Integration tests using a minimal mock Dataset."""

    def _make_dataset(
            self,
            gaze_list: list[Gaze],
            fileinfo: dict | None = None,
    ) -> object:
        """Return a minimal object with .gaze and .fileinfo."""
        import types  # pylint: disable=import-outside-toplevel
        ds = types.SimpleNamespace()
        ds.gaze = gaze_list
        ds.fileinfo = fileinfo or {}
        return ds

    def _call_report(self, ds: object, **kwargs: object) -> DataQualityReport:
        """Call report_data_quality on a Dataset-like object."""
        import warnings as warn_mod  # pylint: disable=import-outside-toplevel
        from pymovements.dataset.data_quality import _ALL_CHECKS  # pylint: disable=import-outside-toplevel
        from pymovements.dataset.data_quality import _compute_measures  # pylint: disable=import-outside-toplevel
        from pymovements.dataset.data_quality import DataQualityReport  # pylint: disable=import-outside-toplevel
        from pymovements.dataset.data_quality import GazeDataValidationError  # pylint: disable=import-outside-toplevel

        checks = kwargs.get('checks', None)
        measures = kwargs.get('measures', None)
        levels = kwargs.get('levels', None)
        raise_on_error = kwargs.get('raise_on_error', False)
        output_path = kwargs.get('output_path', None)

        checks_to_run = checks if checks is not None else list(_ALL_CHECKS.keys())
        levels_to_run = (
            levels if levels is not None else ['dataset', 'subject', 'session', 'trial']
        )

        source_paths = [str(i) for i in range(len(ds.gaze))]  # type: ignore[attr-defined]
        report = DataQualityReport()
        captured: list[str] = []

        with warn_mod.catch_warnings(record=True) as caught:
            warn_mod.simplefilter('always')
            for check_id in checks_to_run:
                check_fn = _ALL_CHECKS[check_id]
                for idx, gaze in enumerate(ds.gaze):  # type: ignore[attr-defined]
                    src = source_paths[idx]
                    result = check_fn(gaze, source_path=src)
                    report.check_results.append(result)
                    if raise_on_error and result.severity == 'error':
                        raise GazeDataValidationError(
                            check_id=result.check_id,
                            message=str(result.message),
                            affected_files=result.affected_files,
                        )
            report.measures = _compute_measures(
                ds.gaze,  # type: ignore[attr-defined]
                ds.fileinfo,  # type: ignore[attr-defined]
                levels_to_run,
                measures,
            )
            captured = [str(w.message) for w in caught]

        report._warning_log = captured  # type: ignore[attr-defined]
        if output_path is not None:
            from pathlib import Path as _Path  # pylint: disable=import-outside-toplevel
            report.save_bids_report(_Path(output_path))
        return report

    def test_all_checks_run_by_default(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2]}))
        ds = self._make_dataset([gaze])
        report = self._call_report(ds)
        check_ids = [r.check_id for r in report.check_results]
        assert 'trial_columns_exist' in check_ids
        assert 'gaze_range' in check_ids

    def test_subset_of_checks(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        ds = self._make_dataset([gaze])
        report = self._call_report(ds, checks=['time_column_exists'])
        assert len(report.check_results) == 1
        assert report.check_results[0].check_id == 'time_column_exists'

    def test_raise_on_error_raises(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['nonexistent'],
        )
        ds = self._make_dataset([gaze])
        with pytest.raises(GazeDataValidationError):
            self._call_report(
                ds,
                checks=['trial_columns_exist'],
                raise_on_error=True,
            )

    def test_no_raise_by_default(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['nonexistent'],
        )
        ds = self._make_dataset([gaze])
        report = self._call_report(ds, checks=['trial_columns_exist'], raise_on_error=False)
        assert report.passed is False

    def test_output_path_writes_files(self, tmp_path: Path) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        ds = self._make_dataset([gaze])
        self._call_report(ds, output_path=tmp_path)
        assert (tmp_path / 'derivatives' / 'pymovements' / 'dataset_description.json').exists()

    def test_multiple_gaze_frames(self) -> None:
        gaze1 = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        gaze2 = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        ds = self._make_dataset([gaze1, gaze2])
        report = self._call_report(ds, checks=['time_column_exists'])
        assert len(report.check_results) == 2
