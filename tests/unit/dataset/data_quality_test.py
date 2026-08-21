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
"""Tests for pymovements.gaze.quality and pymovements.gaze.validation."""
from __future__ import annotations

import json
import types
import warnings as warn_mod
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest

from pymovements.dataset import Dataset
from pymovements.dataset import DatasetDefinition
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.gaze.quality import _compute_data_loss_simple
from pymovements.gaze.quality import compute_measures
from pymovements.gaze.quality import DataQualityReport
from pymovements.gaze.quality import ValidationError
from pymovements.gaze.validation import _ALL_CHECKS
from pymovements.gaze.validation import check_gaze_components_defined
from pymovements.gaze.validation import check_gaze_range
from pymovements.gaze.validation import check_max_gap
from pymovements.gaze.validation import check_sampling_rate_consistency
from pymovements.gaze.validation import check_time_column_exists
from pymovements.gaze.validation import check_time_monotone
from pymovements.gaze.validation import check_trial_columns_dtype
from pymovements.gaze.validation import check_trial_columns_exist
from pymovements.gaze.validation import CheckResult

pytestmark = pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')


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
        assert r.code == 'my_check'
        assert r.severity == 'pass'
        assert r.message == 'All good'
        assert r.sources == ['f1.csv']

    def test_default_sources(self) -> None:
        r = CheckResult('x', 'pass', 'ok')
        assert not r.sources

    def test_frozen(self) -> None:
        r = CheckResult('x', 'pass', 'ok')
        with pytest.raises((AttributeError, TypeError)):
            r.code = 'y'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------

class TestValidationError:
    def test_raise(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError('c', 'bad', ['f.csv'])
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

    def test_fail_column_absent(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1]}),
            trial_columns=['trial_id'],
        )
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'fail'
        assert 'trial_id' in result.message

    def test_fail_partial_absence(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'subject': [1, 1]}),
            trial_columns=['subject', 'trial'],
        )
        result = check_trial_columns_exist(gaze)
        assert result.severity == 'fail'
        assert 'trial' in result.message

    def test_sources_set_on_fail(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['missing'],
        )
        result = check_trial_columns_exist(gaze, source_path='data/s1.csv')
        assert 'data/s1.csv' in result.sources

    def test_sources_set_on_pass(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_trial_columns_exist(gaze, source_path='data/s1.csv')
        assert 'data/s1.csv' in result.sources

    def test_sources_empty_when_no_source_path(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}), trial_columns=['missing'])
        result = check_trial_columns_exist(gaze, source_path='')
        assert not result.sources


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

    def test_trial_col_absent_from_schema_is_skipped(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['missing'],
        )
        result = check_trial_columns_dtype(gaze)
        assert result.severity == 'pass'


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

    def test_fail_time_absent(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [1.0, 2.0]}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'fail'
        assert 'time' in result.message

    def test_fail_time_string_dtype(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': ['t0', 't1']}))
        result = check_time_column_exists(gaze)
        assert result.severity == 'fail'
        assert 'time' in result.message

    def test_sources_on_fail(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [0.0]}))
        result = check_time_column_exists(gaze, source_path='s/f.csv')
        assert 's/f.csv' in result.sources

    def test_sources_empty_no_source_path(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [0.0]}))
        result = check_time_column_exists(gaze, source_path='')
        assert not result.sources


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

    def test_fail_no_coordinate_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0], 'trial': [1]}))
        result = check_gaze_components_defined(gaze)
        assert result.severity == 'fail'

    def test_sources_on_fail(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_gaze_components_defined(gaze, source_path='data.csv')
        assert 'data.csv' in result.sources

    def test_sources_empty_no_source_path(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}))
        result = check_gaze_components_defined(gaze, source_path='')
        assert not result.sources


# ---------------------------------------------------------------------------
# check_time_monotone
# ---------------------------------------------------------------------------

class TestCheckTimeMonotone:
    def test_pass_no_time_column(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [0.0]}))
        result = check_time_monotone(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_monotone_no_trial_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2, 3]}))
        result = check_time_monotone(gaze)
        assert result.severity == 'pass'

    def test_pass_monotone_with_trial_columns(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 30], 'trial': [1, 1, 2, 2]}),
            trial_columns=['trial'],
        )
        result = check_time_monotone(gaze)
        assert result.severity == 'pass'

    def test_fail_non_monotone(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 20, 10, 30], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
        )
        result = check_time_monotone(gaze)
        assert result.severity == 'fail'

    def test_fail_non_monotone_no_trial_columns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 20, 10, 30]}))
        result = check_time_monotone(gaze)
        assert result.severity == 'fail'

    def test_pass_single_sample_per_trial_skipped(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': [1, 2]}),
            trial_columns=['trial'],
        )
        result = check_time_monotone(gaze)
        assert result.severity == 'pass'

    def test_error_missing_trial_col_in_schema(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1]}),
            trial_columns=['nonexistent'],
        )
        result = check_time_monotone(gaze)
        assert result.severity == 'error'
        assert 'nonexistent' in result.message

    def test_sources_on_fail(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 20, 10], 'trial': [1, 1, 1]}),
            trial_columns=['trial'],
        )
        result = check_time_monotone(gaze, source_path='s.csv')
        assert result.severity == 'fail'
        assert 's.csv' in result.sources

    def test_error_missing_trial_col_in_max_gap(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20]}),
            trial_columns=['nonexistent'],
            experiment=exp,
        )
        result = check_max_gap(gaze)
        assert result.severity == 'error'
        assert 'nonexistent' in result.message


# ---------------------------------------------------------------------------
# check_max_gap
# ---------------------------------------------------------------------------

class TestCheckMaxGap:
    def test_error_no_time_column(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'x': [0.0]}))
        result = check_max_gap(gaze)
        assert result.severity == 'error'
        assert 'time' in result.message

    def test_error_no_sampling_rate(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 10, 20]}))
        result = check_max_gap(gaze)
        assert result.severity == 'error'
        assert 'sampling_rate' in result.message

    def test_pass_no_gap(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 30], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = check_max_gap(gaze)
        assert result.severity == 'pass'

    def test_warning_gap_exceeds_5x_isi(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        # ISI = 10ms; 5x ISI = 50ms; insert 100ms gap
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 120], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = check_max_gap(gaze)
        assert result.severity == 'warning'
        assert 'gap' in result.message.lower()

    def test_warning_gap_no_trial_columns(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 120]}),
            experiment=exp,
        )
        result = check_max_gap(gaze)
        assert result.severity == 'warning'

    def test_custom_max_gap_factor(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        # ISI = 10ms; default factor 5.0 → max 50ms; 60ms gap triggers at factor 5.0
        # but not at factor 10.0
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 80], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        assert check_max_gap(gaze, max_gap_factor=5.0).severity == 'warning'
        assert check_max_gap(gaze, max_gap_factor=10.0).severity == 'pass'

    def test_sources_on_warning(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20, 120], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = check_max_gap(gaze, source_path='file.csv')
        assert result.severity == 'warning'
        assert 'file.csv' in result.sources


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

    def test_sources_on_warning(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 5, 10, 15, 20]}),
            experiment=exp,
        )
        result = check_sampling_rate_consistency(gaze, source_path='data.asc')
        if result.severity == 'warning':
            assert 'data.asc' in result.sources

    def test_pass_no_positive_diffs(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(pl.DataFrame({'time': [5, 5, 5]}), experiment=exp)
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_pass_no_time_column(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(pl.DataFrame({'x': [0.0, 1.0]}), experiment=exp)
        result = check_sampling_rate_consistency(gaze)
        assert result.severity == 'pass'
        assert 'skipped' in result.message

    def test_custom_max_deviation(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        # Empirical 200 Hz vs declared 100 Hz → 100% deviation
        # Passes with a very large threshold only
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 5, 10, 15, 20]}),
            experiment=exp,
        )
        assert check_sampling_rate_consistency(gaze, max_deviation=0.05).severity == 'warning'
        assert check_sampling_rate_consistency(gaze, max_deviation=2.0).severity == 'pass'


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

    def test_sources_on_warning(self) -> None:
        exp = _simple_experiment()
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(20)),
                'position': [[-999.0, -999.0]] * 20,
            }),
            experiment=exp,
        )
        result = check_gaze_range(gaze, source_path='data.csv')
        assert result.severity == 'warning'
        assert 'data.csv' in result.sources

    def test_custom_min_fraction(self) -> None:
        exp = _simple_experiment()
        # All out of range → 0% in range
        gaze = _make_gaze(
            pl.DataFrame({'time': list(range(20)), 'position': [[-999.0, -999.0]] * 20}),
            experiment=exp,
        )
        assert check_gaze_range(gaze, min_fraction=0.95).severity == 'warning'
        # Lowering threshold to 0 → always pass
        assert check_gaze_range(gaze, min_fraction=0.0).severity == 'pass'


# ---------------------------------------------------------------------------
# _ALL_CHECKS registry
# ---------------------------------------------------------------------------

class TestAllChecks:
    def test_all_eight_checks_registered(self) -> None:
        expected = {
            'trial_columns_exist',
            'trial_columns_dtype',
            'time_column_exists',
            'gaze_components_defined',
            'time_monotone',
            'max_gap',
            'sampling_rate_consistency',
            'gaze_range',
        }
        assert set(_ALL_CHECKS.keys()) == expected

    @pytest.mark.parametrize('check_id', list(_ALL_CHECKS.keys()))
    def test_each_check_callable(self, check_id: str) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        result = _ALL_CHECKS[check_id](gaze, '')
        assert isinstance(result, CheckResult)
        assert result.severity in {'pass', 'warning', 'fail', 'error'}


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

    def test_passed_false_when_fail_present(self) -> None:
        report = DataQualityReport(
            check_results=[CheckResult('a', 'fail', 'broken', ['f.csv'])],
        )
        assert report.passed is False

    def test_passed_true_empty_report(self) -> None:
        report = DataQualityReport()
        assert report.passed is True

    def test_passed_is_derived_from_check_results(self) -> None:
        report = DataQualityReport(
            check_results=[CheckResult('a', 'error', 'broken')],
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
        assert not report.measures


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
        assert 'code' in header_cols
        assert 'severity' in header_cols
        assert 'sources' in header_cols

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
        report = DataQualityReport(warning_log=['UserWarning: something went wrong'])
        report.save_bids_report(tmp_path)

        log = (tmp_path / 'derivatives' / 'pymovements' / 'warnings.log').read_text()
        assert 'something went wrong' in log

    def test_empty_check_results_writes_header_only(self, tmp_path: Path) -> None:
        report = DataQualityReport()
        report.save_bids_report(tmp_path)
        tsv = (tmp_path / 'derivatives' / 'pymovements' / 'data_quality_checks.tsv').read_text()
        assert tsv.startswith('code')


# ---------------------------------------------------------------------------
# compute_measures
# ---------------------------------------------------------------------------

class TestComputeMeasures:
    def test_empty_gaze_list_returns_empty(self) -> None:
        result = compute_measures([], None, ['dataset', 'trial'])
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
        result = compute_measures([gaze], None, ['dataset'])
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
        result = compute_measures([gaze], None, ['trial'])
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
        result = compute_measures([gaze], None, ['dataset'], measures=['data_loss'])
        if 'dataset' in result:
            assert 'data_loss' in result['dataset'].columns
            assert 'std_rms' not in result['dataset'].columns

    def test_no_coord_column_still_returns(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2], 'trial': [1, 1, 1]}))
        result = compute_measures([gaze], None, ['dataset'])
        assert isinstance(result, dict)

    def test_session_level(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(4)),
                'position': [[float(i), float(i)] for i in range(4)],
            }),
            experiment=exp,
        )
        fileinfo = {
            'gaze': pl.DataFrame({
                'subject_id': ['s1'],
                'session_id': ['ses-1'],
                'filepath': ['/data/s1.csv'],
            }),
        }
        result = compute_measures([gaze], fileinfo, ['session'])
        assert isinstance(result, dict)

    def test_subject_level(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': list(range(4)),
                'position': [[float(i), float(i)] for i in range(4)],
            }),
            experiment=exp,
        )
        fileinfo = {
            'gaze': pl.DataFrame({
                'subject_id': ['s1'],
                'filepath': ['/data/s1.csv'],
            }),
        }
        result = compute_measures([gaze], fileinfo, ['subject'])
        assert 'subject' in result

    def test_trial_level_no_trial_columns_skips(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'position': [[0.0, 0.0], [1.0, 1.0]]}),
        )
        result = compute_measures([gaze], None, ['trial'])
        assert 'trial' not in result or len(result.get('trial', pl.DataFrame())) == 0

    def test_data_loss_simple_empty_samples(self) -> None:
        empty_df = pl.DataFrame({'position': []}).with_columns(
            pl.col('position').cast(pl.List(pl.Float64)),
        )
        gaze = _make_gaze(empty_df)
        result = _compute_data_loss_simple(gaze, 'position')
        assert result == 0.0

    def test_fileinfo_not_dataframe_handled(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'position': [[0.0, 0.0], [1.0, 1.0]]}),
            experiment=_simple_experiment(),
        )
        bad_fileinfo = {'gaze': 'not_a_dataframe'}
        result = compute_measures([gaze], bad_fileinfo, ['dataset'])
        assert isinstance(result, dict)

    def test_trial_no_coord_column_skips(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': [1, 1]}),
            trial_columns=['trial'],
        )
        result = compute_measures([gaze], None, ['trial'])
        assert 'trial' not in result

    def test_trial_empty_agg_exprs_skips(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'trial': [1, 1], 'position': [[0.0, 0.0], [1.0, 1.0]]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = compute_measures([gaze], None, ['trial'], measures=[])
        assert 'trial' not in result

    def test_trial_data_loss_ratio_rename(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 10, 20, 30, 40, 50],
                'trial': [1, 1, 1, 2, 2, 2],
                'position': [[float(i), float(i)] for i in range(6)],
            }),
            trial_columns=['trial'],
            experiment=exp,
        )
        result = compute_measures([gaze], None, ['trial'], measures=['data_loss'])
        if 'trial' in result:
            assert 'data_loss' in result['trial'].columns
            assert 'data_loss_ratio' not in result['trial'].columns

    def test_trial_missing_trial_col_in_schema_skips(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1], 'position': [[0.0, 0.0], [1.0, 1.0]]}),
            trial_columns=['nonexistent'],
        )
        result = compute_measures([gaze], None, ['trial'])
        assert 'trial' not in result

    def test_dataset_no_coord_col_skips_measures(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2]}))
        result = compute_measures([gaze], None, ['dataset'])
        assert isinstance(result, dict)

    def test_trial_no_sampling_rate_skips_data_loss_agg(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 1, 2, 3],
                'trial': [1, 1, 2, 2],
                'position': [[float(i), float(i)] for i in range(4)],
            }),
            trial_columns=['trial'],
            experiment=None,
        )
        result = compute_measures([gaze], None, ['trial'])
        assert isinstance(result, dict)

    def test_partial_precision_measures_only_std_rms(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1, 2], 'position': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}),
            experiment=exp,
        )
        result = compute_measures([gaze], None, ['dataset'], measures=['std_rms'])
        if 'dataset' in result:
            assert 'std_rms' in result['dataset'].columns
            assert 'rms_s2s' not in result['dataset'].columns
            assert 'bcea' not in result['dataset'].columns

    def test_partial_precision_measures_only_rms_s2s(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1, 2], 'position': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}),
            experiment=exp,
        )
        result = compute_measures([gaze], None, ['dataset'], measures=['rms_s2s'])
        if 'dataset' in result:
            assert 'rms_s2s' in result['dataset'].columns
            assert 'std_rms' not in result['dataset'].columns

    def test_partial_precision_measures_only_bcea(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1, 2], 'position': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}),
            experiment=exp,
        )
        result = compute_measures([gaze], None, ['dataset'], measures=['bcea'])
        if 'dataset' in result:
            assert 'bcea' in result['dataset'].columns
            assert 'std_rms' not in result['dataset'].columns

    def test_data_loss_polars_error_falls_back_to_simple(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 20], 'position': [[0.0, 0.0], [1.0, 1.0], None]}),
            experiment=exp,
        )
        with patch(
            'pymovements.gaze.quality.data_loss',
            side_effect=pl.exceptions.ComputeError('mock error'),
        ):
            result = compute_measures([gaze], None, ['dataset'], measures=['data_loss'])
        assert isinstance(result, dict)

    def test_precision_polars_error_sets_none(self) -> None:
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 1, 2], 'position': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}),
            experiment=exp,
        )
        with patch.object(
            pl.DataFrame,
            'select',
            side_effect=pl.exceptions.ComputeError('mock precision error'),
        ):
            result = compute_measures([gaze], None, ['dataset'], measures=['std_rms'])
        assert isinstance(result, dict)

    def test_trial_agg_polars_error_continues(self) -> None:
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 10, 20, 30],
                'trial': [1, 1, 2, 2],
                'position': [[float(i), float(i)] for i in range(4)],
            }),
            trial_columns=['trial'],
            experiment=exp,
        )
        with patch.object(
            pl.DataFrame,
            'group_by',
            side_effect=pl.exceptions.ComputeError('mock agg error'),
        ):
            result = compute_measures([gaze], None, ['trial'])
        assert isinstance(result, dict)
        assert 'trial' not in result


# ---------------------------------------------------------------------------
# Integration: Dataset.report_data_quality via Gaze.validate
# ---------------------------------------------------------------------------

class TestDatasetReportDataQuality:
    """Integration tests using a minimal mock Dataset."""

    def _make_dataset(
            self,
            gaze_list: list[Gaze],
            fileinfo: dict[str, Any] | None = None,
    ) -> object:
        ds = types.SimpleNamespace()
        ds.gaze = gaze_list
        ds.fileinfo = fileinfo or {}
        return ds

    def _call_report(
            self,
            ds: object,
            checks: list[str] | None = None,
            measures: list[str] | None = None,
            levels: list[str] | None = None,
            raise_on_error: bool = False,
            output_path: Path | None = None,
    ) -> DataQualityReport:
        gaze_list: list[Gaze] = getattr(ds, 'gaze', [])
        fileinfo: Any = getattr(ds, 'fileinfo', {})

        checks_to_run = set(checks) if checks is not None else set(_ALL_CHECKS.keys())
        levels_to_run = (
            levels if levels is not None else ['dataset', 'subject', 'session', 'trial']
        )

        source_paths = ['' for _ in gaze_list]
        check_results: list = []

        with warn_mod.catch_warnings(record=True) as caught:
            warn_mod.simplefilter('always')
            for idx, gaze in enumerate(gaze_list):
                src = source_paths[idx]
                results = gaze.validate(
                    trial_columns_exist='trial_columns_exist' in checks_to_run,
                    trial_columns_dtype='trial_columns_dtype' in checks_to_run,
                    time_column_exists='time_column_exists' in checks_to_run,
                    gaze_components_defined='gaze_components_defined' in checks_to_run,
                    time_monotone='time_monotone' in checks_to_run,
                    max_gap='max_gap' in checks_to_run,
                    sampling_rate_consistency='sampling_rate_consistency' in checks_to_run,
                    gaze_range='gaze_range' in checks_to_run,
                    source_path=src,
                )
                for result in results:
                    check_results.append(result)
                    if raise_on_error and result.severity in {'fail', 'error'}:
                        raise ValidationError(
                            check_id=result.code,
                            message=str(result.message),
                            affected_files=result.sources,
                        )
            measure_results = compute_measures(
                gaze_list,
                fileinfo,
                levels_to_run,
                measures,
            )
            captured = [str(w.message) for w in caught]

        report = DataQualityReport(
            check_results=check_results,
            measures=measure_results,
            warning_log=captured,
        )
        if output_path is not None:
            report.save_bids_report(output_path)
        return report

    def test_all_checks_run_by_default(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1, 2]}))
        ds = self._make_dataset([gaze])
        report = self._call_report(ds)
        check_codes = [r.code for r in report.check_results]
        assert 'trial_columns_exist' in check_codes
        assert 'gaze_range' in check_codes

    def test_subset_of_checks(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0, 1]}))
        ds = self._make_dataset([gaze])
        report = self._call_report(ds, checks=['time_column_exists'])
        assert len(report.check_results) == 1
        assert report.check_results[0].code == 'time_column_exists'

    def test_raise_on_error_raises(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['nonexistent'],
        )
        ds = self._make_dataset([gaze])
        with pytest.raises(ValidationError):
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

    def test_passed_updated_after_checks(self) -> None:
        gaze = _make_gaze(pl.DataFrame({'time': [0]}), trial_columns=['missing'])
        ds = self._make_dataset([gaze])
        report = self._call_report(ds, checks=['trial_columns_exist'])
        assert report.passed is False

    def test_empty_gaze_list(self) -> None:
        ds = self._make_dataset([])
        report = self._call_report(ds)
        assert not report.check_results
        assert report.passed is True


# ---------------------------------------------------------------------------
# Direct tests for Dataset.report_data_quality()
# ---------------------------------------------------------------------------

def _make_real_gaze() -> Gaze:
    return Gaze(
        samples=pl.DataFrame(
            {'time': [0, 10, 20], 'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0]},
        ),
        pixel_columns=['x', 'y'],
        experiment=_simple_experiment(sampling_rate=100.0),
    )


def _make_real_dataset(
        gaze_list: list[Gaze],
        fileinfo: dict[str, pl.DataFrame] | None = None,
) -> Dataset:
    ds = Dataset(DatasetDefinition, path='.')
    ds.gaze = gaze_list
    if fileinfo is not None:
        ds.fileinfo = fileinfo
    return ds


class TestDatasetReportDataQualityDirect:
    """Tests that call Dataset.report_data_quality() on real Dataset objects."""

    def test_basic_call_returns_report(self) -> None:
        """Basic call returns a DataQualityReport with all 8 check results."""
        ds = _make_real_dataset([_make_real_gaze()])
        report = ds.report_data_quality()
        assert isinstance(report, DataQualityReport)
        assert report.passed is True
        assert len(report.check_results) == 8

    def test_subset_checks(self) -> None:
        """Passing checks= runs only the listed checks."""
        ds = _make_real_dataset([_make_real_gaze()])
        report = ds.report_data_quality(checks=['time_column_exists'])
        assert len(report.check_results) == 1
        assert report.check_results[0].code == 'time_column_exists'

    def test_unknown_check_raises_value_error(self) -> None:
        """Unknown check identifier raises ValueError."""
        ds = _make_real_dataset([_make_real_gaze()])
        with pytest.raises(ValueError, match='Unknown check identifier'):
            ds.report_data_quality(checks=['not_a_valid_check'])

    def test_raise_on_error_raises_gaze_validation_error(self) -> None:
        """raise_on_error=True raises ValidationError on first error."""
        gaze = _make_gaze(
            pl.DataFrame({'time': [0]}),
            trial_columns=['nonexistent_col'],
        )
        ds = _make_real_dataset([gaze])
        with pytest.raises(ValidationError):
            ds.report_data_quality(
                checks=['trial_columns_exist'],
                raise_on_error=True,
            )

    def test_fileinfo_with_filepath_sets_source_paths(self) -> None:
        """Fileinfo with filepath column sets source paths on check results."""
        gaze = _make_real_gaze()
        fileinfo = {'gaze': pl.DataFrame({'filepath': ['subject1.csv']})}
        ds = _make_real_dataset([gaze], fileinfo=fileinfo)
        report = ds.report_data_quality(checks=['time_column_exists'])
        assert report.passed is True

    def test_output_path_writes_bids_report(self, tmp_path: Path) -> None:
        """output_path triggers writing of BIDS derivative files."""
        ds = _make_real_dataset([_make_real_gaze()])
        ds.report_data_quality(output_path=tmp_path)
        assert (tmp_path / 'derivatives' / 'pymovements' / 'dataset_description.json').exists()

    def test_levels_parameter_filters_measures(self) -> None:
        """levels= parameter restricts measure aggregation to requested levels."""
        ds = _make_real_dataset([_make_real_gaze()])
        report = ds.report_data_quality(levels=['dataset'], measures=['data_loss'])
        assert 'dataset' in report.measures

    def test_empty_gaze_list_returns_passed_report(self) -> None:
        """Empty gaze list produces a passed report with no check results."""
        ds = _make_real_dataset([])
        report = ds.report_data_quality()
        assert report.passed is True
        assert not report.check_results

    def test_custom_max_gap_factor_passed_through(self) -> None:
        """max_gap_factor is forwarded to the max_gap check."""
        exp = _simple_experiment(sampling_rate=100.0)
        # ISI=10ms; gap=15ms; default factor 5.0 → pass; tight factor 1.0 → warning
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 10, 25]}),
            experiment=exp,
        )
        ds = _make_real_dataset([gaze])
        report_default = ds.report_data_quality(checks=['max_gap'])
        report_tight = ds.report_data_quality(checks=['max_gap'], max_gap_factor=1.0)
        assert report_default.check_results[0].severity == 'pass'
        assert report_tight.check_results[0].severity == 'warning'

    def test_custom_max_deviation_passed_through(self) -> None:
        """max_deviation is forwarded to the sampling_rate_consistency check."""
        exp = _simple_experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            pl.DataFrame({'time': [0, 11, 22, 33, 44]}),
            experiment=exp,
        )
        ds = _make_real_dataset([gaze])
        report_tight = ds.report_data_quality(
            checks=['sampling_rate_consistency'], max_deviation=0.01,
        )
        assert report_tight.check_results[0].severity == 'warning'

    def test_custom_min_fraction_passed_through(self) -> None:
        """min_fraction is forwarded to the gaze_range check."""
        exp = _simple_experiment(sampling_rate=1000.0)
        gaze = _make_gaze(
            pl.DataFrame({
                'time': [0, 1, 2],
                'px': [9999.0, 9999.0, 9999.0],
                'py': [9999.0, 9999.0, 9999.0],
            }),
            experiment=exp,
        )
        gaze.samples = gaze.samples.with_columns(
            pl.concat_list(['px', 'py']).alias('pixel'),
        )
        ds = _make_real_dataset([gaze])
        report_loose = ds.report_data_quality(checks=['gaze_range'], min_fraction=0.0)
        assert report_loose.check_results[0].severity == 'pass'
