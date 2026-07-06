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
"""Tests for :py:meth:`~pymovements.gaze.Gaze.validate` and the validation module."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.gaze.quality import DataQualityReport
from pymovements.gaze.quality import GazeDataValidationError
from pymovements.gaze.validation import CheckResult

pytestmark = pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp(sampling_rate: float = 1000.0) -> Experiment:
    return Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38.0,
        screen_height_cm=30.0,
        distance_cm=68.0,
        origin='upper left',
        sampling_rate=sampling_rate,
    )


def _make_gaze(
        samples: pl.DataFrame,
        trial_columns: list[str] | None = None,
        experiment: Experiment | None = None,
) -> Gaze:
    """Bypass Gaze.__init__ validation to create test fixtures with invalid state."""
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


# ---------------------------------------------------------------------------
# Gaze.validate — basic contract
# ---------------------------------------------------------------------------

class TestGazeValidateContract:
    """Verify that validate() returns the expected structure regardless of content."""

    def test_returns_list(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        result = gaze.validate()
        assert isinstance(result, list)

    def test_all_items_are_check_results(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        for r in gaze.validate():
            assert isinstance(r, CheckResult)

    def test_default_returns_eight_results(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        assert len(gaze.validate()) == 8

    def test_all_severities_valid(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        for r in gaze.validate():
            assert r.severity in {'pass', 'warning', 'error'}

    def test_check_codes_unique_in_result(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        codes = [r.code for r in gaze.validate()]
        assert len(codes) == len(set(codes))


# ---------------------------------------------------------------------------
# Gaze.validate — disabling individual checks
# ---------------------------------------------------------------------------

class TestGazeValidateDisableChecks:
    """Toggling a check to False removes exactly that check from the output."""

    @pytest.mark.parametrize(
        'kwarg,check_code', [
            ('trial_columns_exist', 'trial_columns_exist'),
            ('trial_columns_dtype', 'trial_columns_dtype'),
            ('time_column_exists', 'time_column_exists'),
            ('gaze_components_defined', 'gaze_components_defined'),
            ('time_monotone', 'time_monotone'),
            ('max_gap', 'max_gap'),
            ('sampling_rate_consistency', 'sampling_rate_consistency'),
            ('gaze_range', 'gaze_range'),
        ],
    )
    def test_disable_single_check(self, kwarg: str, check_code: str) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        results = gaze.validate(**{kwarg: False})  # type: ignore[arg-type]
        codes = [r.code for r in results]
        assert check_code not in codes
        assert len(results) == 7

    def test_disable_all_returns_empty(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert not results

    def test_enable_only_one_returns_one(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=True,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert len(results) == 1
        assert results[0].code == 'time_column_exists'


# ---------------------------------------------------------------------------
# Gaze.validate — order of results
# ---------------------------------------------------------------------------

class TestGazeValidateOrder:
    def test_results_in_defined_order(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        expected_order = [
            'trial_columns_exist',
            'trial_columns_dtype',
            'time_column_exists',
            'gaze_components_defined',
            'time_monotone',
            'max_gap',
            'sampling_rate_consistency',
            'gaze_range',
        ]
        codes = [r.code for r in gaze.validate()]
        assert codes == expected_order

    def test_partial_order_preserved(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
        )
        codes = [r.code for r in results]
        assert codes == [
            'time_column_exists',
            'gaze_components_defined',
            'time_monotone',
            'max_gap',
            'sampling_rate_consistency',
            'gaze_range',
        ]


# ---------------------------------------------------------------------------
# Gaze.validate — source_path propagation
# ---------------------------------------------------------------------------

class TestGazeValidateSourcePath:
    def test_error_includes_source_path(self) -> None:
        gaze = _make_gaze(
            samples=pl.DataFrame({'time': [0]}),
            trial_columns=['missing_column'],
        )
        results = gaze.validate(source_path='data/subject01.csv')
        error_results = [r for r in results if r.severity == 'error']
        assert error_results, 'Expected at least one error result'
        for r in error_results:
            assert 'data/subject01.csv' in r.sources

    def test_pass_results_have_source_path(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1, 2]}))
        results = gaze.validate(source_path='data/file.csv')
        for r in results:
            if r.severity == 'pass':
                assert r.sources == ['data/file.csv']

    def test_empty_source_path_gives_empty_sources(self) -> None:
        gaze = _make_gaze(
            samples=pl.DataFrame({'time': [0]}),
            trial_columns=['missing'],
        )
        results = gaze.validate(source_path='')
        for r in results:
            assert r.sources == []


# ---------------------------------------------------------------------------
# Gaze.validate — per-check outcome scenarios
# ---------------------------------------------------------------------------

class TestGazeValidateCheckOutcomes:
    def test_trial_columns_exist_pass(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 1], 'trial': [1, 1]}),
            trial_columns=['trial'],
        )
        results = gaze.validate(
            trial_columns_exist=True,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_trial_columns_exist_error(self) -> None:
        gaze = _make_gaze(
            samples=pl.DataFrame({'time': [0, 1]}),
            trial_columns=['nonexistent'],
        )
        results = gaze.validate(
            trial_columns_exist=True,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'error'

    def test_trial_columns_dtype_warning_on_float(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0], 'trial': pl.Series([1.0], dtype=pl.Float64)}),
            trial_columns=['trial'],
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=True,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_time_column_exists_error(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'x': [1.0, 2.0]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=True,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'error'

    def test_gaze_components_defined_error(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0, 1]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=True,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'error'

    def test_gaze_components_defined_pass_with_position(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0], 'position': [[1.0, 2.0]]}),
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=True,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_time_monotone_warning_on_non_monotone(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 20, 10, 30], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=True,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_max_gap_warning_on_large_gap(self) -> None:
        exp = _exp(sampling_rate=100.0)
        # ISI=10ms; 5×ISI=50ms; 100ms gap → warning
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 20, 120], 'trial': [1, 1, 1, 1]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=True,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_max_gap_single_sample_trial_skipped(self) -> None:
        # Trial 2 has only 1 sample — the inner `continue` branch is exercised.
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 20, 30], 'trial': [1, 1, 1, 2]}),
            trial_columns=['trial'],
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=True,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_custom_max_gap_factor_tighter_triggers_warning(self) -> None:
        # Default factor 5.0 would pass; factor 1.5 should warn on a 2× ISI gap.
        exp = _exp(sampling_rate=100.0)
        # ISI=10ms; gap=15ms → 1.5× ISI; factor=1.0 triggers warning
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 25]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=True,
            max_gap_factor=1.0,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_custom_max_deviation_tighter_triggers_warning(self) -> None:
        # Declared 100 Hz; empirical ~95 Hz (5% deviation); tight tolerance 0.01 → warning
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 21, 32, 43]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=True,
            max_deviation=0.01,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_sampling_rate_consistency_pass(self) -> None:
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 20, 30]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=True,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_sampling_rate_consistency_warning(self) -> None:
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 5, 10, 15, 20]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=True,
            gaze_range=False,
        )
        assert results[0].severity == 'warning'

    def test_gaze_range_pass_no_experiment(self) -> None:
        gaze = Gaze(samples=pl.DataFrame({'time': [0], 'position': [[0.5, 0.5]]}))
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=True,
        )
        assert results[0].severity == 'pass'

    def test_gaze_range_warning_out_of_bounds(self) -> None:
        exp = _exp()
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': list(range(20)),
                'position': [[-999.0, -999.0]] * 20,
            }),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=True,
        )
        assert results[0].severity == 'warning'

    def test_gaze_range_pixel_col_no_px_dimensions(self) -> None:
        exp = Experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            samples=pl.DataFrame({'time': [0], 'pixel': [[100.0, 200.0]]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=True,
        )
        assert results[0].severity == 'pass'
        assert 'skipped' in results[0].message

    def test_gaze_range_screen_bounds_raises_type_error(self) -> None:
        exp = Experiment(sampling_rate=100.0)
        gaze = _make_gaze(
            samples=pl.DataFrame({'time': [0], 'position': [[0.0, 0.0]]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=True,
        )
        assert results[0].severity == 'pass'
        assert 'skipped' in results[0].message


# ---------------------------------------------------------------------------
# Gaze.validate — realistic Gaze objects (via normal constructor)
# ---------------------------------------------------------------------------

class TestGazeValidateWithRealGaze:
    """Use the real Gaze.__init__ to exercise the path via gaze.validate()."""

    def test_minimal_valid_gaze(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 1, 2]}),
        )
        results = gaze.validate(
            trial_columns_exist=True,
            trial_columns_dtype=True,
            time_column_exists=True,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        severities = {r.code: r.severity for r in results}
        assert severities['trial_columns_exist'] == 'pass'
        assert severities['trial_columns_dtype'] == 'pass'
        assert severities['time_column_exists'] == 'pass'

    def test_gaze_with_position_columns(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2],
                'x_pos': [0.1, 0.2, 0.3],
                'y_pos': [0.4, 0.5, 0.6],
            }),
            position_columns=['x_pos', 'y_pos'],
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=True,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_gaze_with_experiment_and_consistent_rate(self) -> None:
        exp = _exp(sampling_rate=1000.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 1, 2, 3, 4]}),
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=True,
            gaze_range=False,
        )
        assert results[0].severity == 'pass'

    def test_gaze_with_pixel_columns_and_experiment(self) -> None:
        exp = _exp()
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2],
                'px': [640.0, 641.0, 642.0],
                'py': [512.0, 513.0, 514.0],
            }),
            pixel_columns=['px', 'py'],
            experiment=exp,
        )
        results = gaze.validate(
            trial_columns_exist=False,
            trial_columns_dtype=False,
            time_column_exists=False,
            gaze_components_defined=True,
            time_monotone=False,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=True,
        )
        comp_result = next(r for r in results if r.code == 'gaze_components_defined')
        range_result = next(r for r in results if r.code == 'gaze_range')
        assert comp_result.severity == 'pass'
        assert range_result.severity == 'pass'

    def test_gaze_trial_columns_present(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 10, 20, 30, 40],
                'trial': [1, 1, 2, 2, 2],
            }),
            trial_columns=['trial'],
        )
        results = gaze.validate(
            trial_columns_exist=True,
            trial_columns_dtype=True,
            time_column_exists=False,
            gaze_components_defined=False,
            time_monotone=True,
            max_gap=False,
            sampling_rate_consistency=False,
            gaze_range=False,
        )
        severities = {r.code: r.severity for r in results}
        assert severities['trial_columns_exist'] == 'pass'
        assert severities['trial_columns_dtype'] == 'pass'
        assert severities['time_monotone'] == 'pass'

    def test_all_eight_checks_with_healthy_gaze(self) -> None:
        exp = _exp(sampling_rate=1000.0)
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2, 3],
                'x_pos': [0.0, 0.1, 0.2, 0.3],
                'y_pos': [0.0, 0.1, 0.2, 0.3],
                'trial': [1, 1, 2, 2],
            }),
            position_columns=['x_pos', 'y_pos'],
            trial_columns=['trial'],
            experiment=exp,
        )
        results = gaze.validate()
        assert len(results) == 8
        for r in results:
            assert r.severity in {'pass', 'warning'}, (
                f'{r.code} unexpectedly errored: {r.message}'
            )


# ---------------------------------------------------------------------------
# Gaze.report_data_quality
# ---------------------------------------------------------------------------

class TestGazeReportDataQuality:
    """Tests for :py:meth:`~pymovements.gaze.gaze.Gaze.report_data_quality`."""

    def _clean_gaze(self) -> Gaze:
        return Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2, 3, 4],
                'trial': [1, 1, 2, 2, 2],
                'x': [0.0, 0.1, 0.0, 0.0, 0.1],
                'y': [0.0, 0.0, 0.0, 0.0, 0.0],
            }),
            trial_columns='trial',
            time_column='time',
            position_columns=['x', 'y'],
            experiment=_exp(sampling_rate=1000.0),
        )

    def test_returns_data_quality_report(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality()
        assert isinstance(report, DataQualityReport)

    def test_passed_true_for_clean_gaze(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality()
        assert report.passed is True

    def test_check_results_populated(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(checks=['time_column_exists', 'gaze_components_defined'])
        assert len(report.check_results) == 2
        assert all(r.severity == 'pass' for r in report.check_results)

    def test_default_levels_are_dataset_and_trial(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(measures=['data_loss'])
        assert 'dataset' in report.measures
        assert 'trial' in report.measures
        assert 'subject' not in report.measures
        assert 'session' not in report.measures

    def test_custom_levels(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(levels=['dataset'], measures=['data_loss'])
        assert 'dataset' in report.measures
        assert 'trial' not in report.measures

    def test_custom_checks_subset(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(checks=['time_column_exists'])
        codes = [r.code for r in report.check_results]
        assert codes == ['time_column_exists']

    def test_all_checks_run_by_default(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality()
        assert len(report.check_results) == 8

    def test_invalid_check_raises_value_error(self) -> None:
        gaze = self._clean_gaze()
        with pytest.raises(ValueError, match='Unknown check identifier'):
            gaze.report_data_quality(checks=['not_a_real_check'])

    def test_raise_on_error_raises_exception(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': [1]}),
            trial_columns=['missing_col'],
        )
        with pytest.raises(GazeDataValidationError):
            gaze.report_data_quality(
                checks=['trial_columns_exist'],
                raise_on_error=True,
            )

    def test_raise_on_error_false_does_not_raise(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': [1]}),
            trial_columns=['missing_col'],
        )
        report = gaze.report_data_quality(
            checks=['trial_columns_exist'],
            raise_on_error=False,
        )
        assert report.passed is False

    def test_passed_false_when_error_check(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': [1]}),
            trial_columns=['missing_col'],
        )
        report = gaze.report_data_quality(checks=['trial_columns_exist'])
        assert report.passed is False

    def test_source_path_in_sources(self) -> None:
        gaze = _make_gaze(
            pl.DataFrame({'time': [0], 'trial': [1]}),
            trial_columns=['missing_col'],
        )
        report = gaze.report_data_quality(
            checks=['trial_columns_exist'],
            source_path='subject01/run01.csv',
        )
        errors = [r for r in report.check_results if r.severity == 'error']
        assert len(errors) == 1
        assert 'subject01/run01.csv' in errors[0].sources

    def test_warning_log_captured(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality()
        assert isinstance(report.warning_log, list)

    def test_output_path_writes_bids_files(self, tmp_path: Path) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(
            checks=['time_column_exists'],
            output_path=tmp_path,
        )
        assert isinstance(report, DataQualityReport)
        deriv = tmp_path / 'derivatives' / 'pymovements'
        assert (deriv / 'dataset_description.json').exists()
        assert (deriv / 'data_quality_checks.tsv').exists()
        assert (deriv / 'warnings.log').exists()

    def test_measures_none_means_all_four(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(levels=['dataset'], measures=None)
        if 'dataset' in report.measures:
            cols = report.measures['dataset'].columns
            assert any(c in cols for c in ('data_loss', 'std_rms', 'rms_s2s', 'bcea'))

    def test_empty_measures_list(self) -> None:
        gaze = self._clean_gaze()
        report = gaze.report_data_quality(levels=['dataset'], measures=[])
        assert 'dataset' not in report.measures

    def test_gaze_without_trial_columns_skips_trial_level(self) -> None:
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2],
                'x': [0.0, 0.1, 0.0],
                'y': [0.0, 0.0, 0.0],
            }),
            time_column='time',
            position_columns=['x', 'y'],
            experiment=_exp(sampling_rate=1000.0),
        )
        report = gaze.report_data_quality(levels=['dataset', 'trial'], measures=['data_loss'])
        assert 'dataset' in report.measures
        assert 'trial' not in report.measures

    def test_custom_max_gap_factor_passed_through_report(self) -> None:
        # ISI=10ms; gap=15ms; default factor 5.0 → pass; tight factor 1.0 → warning
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 10, 25]}),
            experiment=exp,
        )
        report_default = gaze.report_data_quality(checks=['max_gap'])
        report_tight = gaze.report_data_quality(checks=['max_gap'], max_gap_factor=1.0)
        assert report_default.check_results[0].severity == 'pass'
        assert report_tight.check_results[0].severity == 'warning'

    def test_custom_max_deviation_passed_through_report(self) -> None:
        # Empirical ~90.9 Hz vs declared 100 Hz (~9% deviation); tight tolerance 0.01 → warning
        exp = _exp(sampling_rate=100.0)
        gaze = Gaze(
            samples=pl.DataFrame({'time': [0, 11, 22, 33, 44]}),
            experiment=exp,
        )
        report_tight = gaze.report_data_quality(
            checks=['sampling_rate_consistency'], max_deviation=0.01,
        )
        assert report_tight.check_results[0].severity == 'warning'

    def test_custom_min_fraction_passed_through_report(self) -> None:
        # All samples out of pixel range; tight min_fraction=0.99 → warning; loose=0.0 → pass
        exp = _exp(sampling_rate=1000.0)
        gaze = Gaze(
            samples=pl.DataFrame({
                'time': [0, 1, 2],
                'px': [9999.0, 9999.0, 9999.0],
                'py': [9999.0, 9999.0, 9999.0],
            }),
            pixel_columns=['px', 'py'],
            experiment=exp,
        )
        report_tight = gaze.report_data_quality(
            checks=['gaze_range'], min_fraction=0.99,
        )
        report_loose = gaze.report_data_quality(
            checks=['gaze_range'], min_fraction=0.0,
        )
        assert report_tight.check_results[0].severity == 'warning'
        assert report_loose.check_results[0].severity == 'pass'
