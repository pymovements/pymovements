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
"""Test event_ratio functionality."""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.fixture(name='fixture_gaze_data')
def fixture_gaze_data_fixture() -> pm.Gaze:
    """Create gaze data with 8 samples."""
    samples = pl.DataFrame(
        {
            'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'pixel': [[i, i] for i in range(8)],
        },
    )
    return pm.Gaze(samples=samples)


class TestEventRatio:
    """Test event_ratio functionality."""

    @pytest.mark.parametrize(
        (
            'event_name_prefix',
            'event_names',
            'event_onsets',
            'event_offsets',
            'sampling_rate',
            'expected_ratio',
        ),
        [
            # Single blink event covering 2 out of 8 samples
            pytest.param('blink', ['blink'], [1.5], [2.5], 1.0, 0.25, id='single_blink'),
            # Two blink events covering 4 out of 8 samples
            pytest.param(
                'blink',
                ['blink', 'blink'],
                [1.5, 4.5],
                [2.5, 5.5],
                1.0,
                0.5,
                id='multiple_blinks',
            ),
            # No matching events
            pytest.param('blink', ['saccade'], [1.5], [2.5], 1.0, 0.0, id='no_matching_events'),
            # Different sampling rate - 3 out of 15 samples
            pytest.param(
                'blink',
                ['blink'],
                [0.75],
                [1.75],
                2.0,
                0.2,
                id='different_sampling_rate',
            ),
            # Event type prefix matching
            pytest.param(
                'blink',
                ['blink', 'blink_custom'],
                [0.5, 2.5],
                [1.5, 3.5],
                1.0,
                0.5,
                id='prefix_matching',
            ),
            # Event covering single sample
            pytest.param('blink', ['blink'], [1.0], [2.0], 1.0, 0.125, id='single_sample'),
        ],
    )
    def test_event_ratio(
        self,
        fixture_gaze_data,
        event_name_prefix,
        event_names,
        event_onsets,
        event_offsets,
        sampling_rate,
        expected_ratio,
    ):
        """Test event ratio calculation with various configurations."""
        events = pm.Events(name=event_names, onsets=event_onsets, offsets=event_offsets)
        gaze = pm.Gaze(samples=fixture_gaze_data.samples, events=events)
        result = gaze.measure_events_ratio(
            event_name_prefix,
            sampling_rate=sampling_rate,
        )

        expected = pl.DataFrame({f"event_ratio_{event_name_prefix}": [expected_ratio]})
        assert_frame_equal(
            result,
            expected,
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    @pytest.mark.parametrize(
        ('trial_columns', 'event_trials', 'expected_ratios'),
        [
            pytest.param(['trial'], [1], {1: 0.5, 2: 0.0}, id='single_trial_col'),
            pytest.param(['trial'], [1, 2], {1: 0.5, 2: 0.5}, id='multiple_trials_single_col'),
            pytest.param(
                ['trial', 'condition'], [(1, 'A')], {
                    (1, 'A'): 0.5, (2, 'B'): 0.0,
                }, id='multi_trial_cols',
            ),
            pytest.param(
                ['trial', 'extra_col'],
                [1],
                {(1, 'X'): 0.5, (2, 'Y'): 0.0},
                id='missing_trial_col_in_events',
            ),
        ],
    )
    def test_event_ratio_with_trials(self, request, trial_columns, event_trials, expected_ratios):
        """Test event ratio calculation with trial columns."""
        samples = pl.DataFrame(
            {
                'time': [0.0, 1.0, 0.0, 1.0],
                'trial': [1, 1, 2, 2],
                'condition': ['A', 'A', 'B', 'B'],
                'extra_col': ['X', 'X', 'Y', 'Y'],
                'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
            },
        )
        # 1.0, 2.0 covers exactly 1 sample in a 2-sample trial (time 0, 1) -> ratio 0.5
        events_data = []
        for trial_val in event_trials:
            row = {'name': 'blink', 'onset': 1.0, 'offset': 2.0}
            if (
                    'extra_col' in trial_columns and
                    'missing_trial_col_in_events' in request.node.callspec.id
            ):
                # For this specific case, we don't add extra_col to events
                row['trial'] = trial_val
            elif isinstance(trial_val, tuple):
                for col, val in zip(trial_columns, trial_val):
                    row[col] = val
            else:
                row[trial_columns[0]] = trial_val
            events_data.append(row)

        events = pm.Events(pl.DataFrame(events_data))
        events.trial_columns = [c for c in trial_columns if c in events.frame.columns]

        gaze = pm.Gaze(samples=samples, events=events, trial_columns=trial_columns)
        result = gaze.measure_events_ratio('blink', sampling_rate=1.0)

        # expected dataframe
        expected_data = []
        unique_trials = samples.select(trial_columns).unique().sort(trial_columns)
        for trial in unique_trials.iter_rows(named=True):
            if len(trial_columns) == 1:
                key = trial[trial_columns[0]]
            else:
                key = tuple(trial[col] for col in trial_columns)
            expected_data.append(
                {**trial, 'event_ratio_blink': expected_ratios.get(key, 0.0)},
            )

        expected = pl.DataFrame(expected_data)

        # column order by select
        result = result.select(sorted(result.columns)).sort(trial_columns)
        expected = expected.select(sorted(expected.columns)).sort(trial_columns)
        assert_frame_equal(result, expected, check_exact=False)

    @pytest.mark.parametrize(
        ('start_time', 'end_time', 'sampling_rate', 'expected_ratio'),
        [
            pytest.param(1.0, 4.0, 10.0, 11 / 31, id='window_1_4'),
            pytest.param(0.0, 1.0, 10.0, 1 / 11, id='window_0_1'),
            pytest.param(2.5, 3.5, 10.0, 0.0, id='between_events'),
            pytest.param(None, None, 1.0, 4 / 8, id='none_bounds'),
            pytest.param(None, 3.0, 1.0, 1 / 4, id='start_none'),
            pytest.param(1.0, 3.0, 10.0, 10 / 21, id='time_window_including_first_event_1'),
            pytest.param(0.0, 3.0, 10.0, 10 / 31, id='time_window_including_first_event_2'),
            pytest.param(2.5, 3.5, 100.0, 0.0, id='time_window_between_100'),
            pytest.param(
                1.5, 3.5, 10.0, 5 /
                21, id='time_window_partially_overlapping_first_event',
            ),
            pytest.param(0.0, 0.9, 10.0, 0.0, id='time_window_before_first_event'),
            pytest.param(7.0, 10.0, 10.0, 0.0, id='time_window_after_last_event'),
            pytest.param(1.0, 7.0, 10.0, 40 / 61, id='time_window_both_events_10'),
            pytest.param(1.0, 7.0, 1.0, 4 / 7, id='time_window_both_events_1'),
            pytest.param(0.5, 8.5, 10.0, 40 / 81, id='time_window_both_events_with_buffer_10'),
            pytest.param(0.5, 8.5, 1.0, 6 / 9, id='time_window_both_events_with_buffer_1'),
        ],
    )
    def test_event_ratio_time_bounds(
            self, fixture_gaze_data, start_time, end_time, sampling_rate, expected_ratio,
    ):
        """Test event ratio calculation with various time bounds."""
        events = pm.Events(name=['blink', 'blink'], onsets=[1, 4], offsets=[2, 7])
        gaze = pm.Gaze(samples=fixture_gaze_data.samples, events=events)
        result = gaze.measure_events_ratio(
            'blink',
            sampling_rate=sampling_rate,
            start_time=start_time,
            end_time=end_time,
        )
        expected = pl.DataFrame({'event_ratio_blink': [expected_ratio]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ('setup_fn', 'expected_data'),
        [
            # gaze.events is None, no trials
            pytest.param(
                lambda g: setattr(g, 'events', None),  # type: ignore[func-returns-value]
                {'event_ratio_blink': [0.0]},
                id='events_none_no_trials',
            ),
            # gaze.events is None, with trials
            pytest.param(
                lambda g: (
                    setattr(  # type: ignore[func-returns-value]
                        g, 'samples', pl.DataFrame({
                            'time': [0.0, 1.0, 0.0, 1.0],
                            'trial': [1, 1, 2, 2],
                            'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                        }),
                    ),
                    setattr(g, 'trial_columns', ['trial']),  # type: ignore[func-returns-value]
                    setattr(g, 'events', None),  # type: ignore[func-returns-value]
                ),
                [{'trial': 1, 'event_ratio_blink': 0.0}, {'trial': 2, 'event_ratio_blink': 0.0}],
                id='events_none_with_trials',
            ),
            # empty events frame
            pytest.param(
                lambda g: setattr(g, 'events', pm.Events()),  # type: ignore[func-returns-value]
                {'event_ratio_blink': [0.0]},
                id='empty_events_frame',
            ),
            # empty samples
            pytest.param(
                lambda g: setattr(
                    g,
                    'samples',
                    pl.DataFrame(
                        schema=g.samples.schema,
                    ),
                ),
                # type: ignore[func-returns-value]
                {'event_ratio_blink': [0.0]},
                id='empty_samples',
            ),
        ],
    )
    def test_event_ratio_edge_cases(self, fixture_gaze_data, setup_fn, expected_data):
        """Test edge cases for event ratio, including those for coverage."""
        setup_fn(fixture_gaze_data)
        result = fixture_gaze_data.measure_events_ratio('blink', sampling_rate=1.0)

        expected = pl.DataFrame(expected_data)
        # potential sorting for trial results
        if 'trial' in result.columns:
            result = result.sort('trial')
            expected = expected.sort('trial')

        assert_frame_equal(result, expected, check_exact=False)

    @pytest.mark.parametrize(
        ('kwargs', 'error_type', 'match'),
        [
            ({'event_name_prefix': ''}, ValueError, 'non-empty string'),
            ({'event_name_prefix': None}, ValueError, 'non-empty string'),
            ({'sampling_rate': 0}, ValueError, 'positive number'),
            ({'sampling_rate': -1}, ValueError, 'positive number'),
            ({'time_column': 'missing'}, ValueError, 'not found'),
            ({'time_column': 123}, TypeError, 'invalid type'),
        ],
    )
    def test_event_ratio_validation(self, fixture_gaze_data, kwargs, error_type, match):
        """Test input validation for measure_events_ratio."""
        base_kwargs = {'event_name_prefix': 'blink', 'sampling_rate': 1.0}
        base_kwargs.update(kwargs)
        with pytest.raises(error_type, match=match):
            fixture_gaze_data.measure_events_ratio(**base_kwargs)

    def test_event_ratio_overlapping_events(self, fixture_gaze_data):
        """Test that overlapping events raise a ValueError."""
        events = pm.Events(name=['blink', 'blink'], onsets=[1.5, 2.5], offsets=[3.5, 4.5])
        gaze = pm.Gaze(samples=fixture_gaze_data.samples, events=events)
        with pytest.raises(ValueError, match='Overlapping events detected'):
            gaze.measure_events_ratio('blink', sampling_rate=1.0)
