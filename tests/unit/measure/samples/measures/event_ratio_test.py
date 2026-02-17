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
            'event_name',
            'event_names',
            'event_onsets',
            'event_offsets',
            'expected_ratio',
        ),
        [
            # Single blink event covering time range [1, 3) = 2 time units
            # Total time range = 7 (from 0 to 7)
            pytest.param('blink', ['blink'], [1.0], [3.0], 2 / 7, id='single_blink'),
            # Two blink events: [1, 2) = 1 unit + [5, 6) = 1 unit = 2 total
            # Total time range = 7 (from 0 to 7)
            pytest.param(
                'blink',
                ['blink', 'blink'],
                [1.0, 5.0],
                [2.0, 6.0],
                2 / 7,
                id='multiple_blinks',
            ),
            # No matching events
            pytest.param(
                'blink',
                ['saccade'],
                [1.0],
                [3.0],
                0.0,
                id='no_matching_events',
            ),
            # Event covering single time unit [1, 1) = 0 units
            pytest.param('blink', ['blink'], [1.0], [1.0], 0.0, id='single_sample'),
        ],
    )
    def test_event_ratio(
        self,
        fixture_gaze_data,
        event_name,
        event_names,
        event_onsets,
        event_offsets,
        expected_ratio,
    ):
        """Test event ratio calculation with various configurations."""
        events = pm.Events(name=event_names, onsets=event_onsets, offsets=event_offsets)
        gaze = pm.Gaze(samples=fixture_gaze_data.samples, events=events)

        result = gaze.samples.select(gaze.measure_events_ratio(event_name))

        expected = pl.DataFrame({f"event_ratio_{event_name}": [expected_ratio]})
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ('samples', 'trial_columns', 'events_data', 'expected_ratios'),
        [
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [{'name': 'blink', 'onset': 1.0, 'offset': 2.0, 'trial': 1}],
                {1: 1.0, 2: 0.0},
                id='single_trial_with_event',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 1.0, 'offset': 1.0, 'trial': 1},
                    {'name': 'blink', 'onset': 1.0, 'offset': 1.0, 'trial': 2},
                ],
                {1: 0.0, 2: 0.0},
                id='both_trials_with_event',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                    'trial': [1, 1, 1, 2, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 0.0, 'offset': 1.0, 'trial': 1},
                ],
                {1: 0.5, 2: 0.0},
                id='event_partial_trial',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'session': [1, 1, 1, 1],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial', 'session'],
                [
                    {'name': 'blink', 'onset': 0.0, 'offset': 0.0, 'trial': 1, 'session': 1},
                ],
                {(1, 1): 0.0, (2, 1): 0.0},
                id='multiple_trial_columns',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 3.0],
                    'trial': [1, 1, 1, 1],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 0.0, 'offset': 1.0, 'trial': 1},
                    {'name': 'blink', 'onset': 2.0, 'offset': 3.0, 'trial': 1},
                ],
                {1: 2 / 3},
                id='adjacent_events',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 3.0],
                    'trial': [1, 1, 1, 1],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 0.0, 'offset': 2.0, 'trial': 1},
                    {'name': 'blink', 'onset': 2.0, 'offset': 3.0, 'trial': 1},
                ],
                {1: 1.0},
                id='overlapping_events',
                marks=pytest.mark.filterwarnings('ignore:Overlapping events detected'),
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 3.0],
                    'trial': [1, 1, 1, 1],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [],
                {1: 0.0},
                id='no_events_trial',
            ),
            pytest.param(
                pl.DataFrame(
                    {
                        'time': [0.0, 1.0, 0.0, 1.0],
                        'trial': [1, 1, 2, 2],
                        'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                    },
                ),
                ['trial'],
                [{'name': 'blink', 'onset': 0.0, 'offset': 1.0, 'trial': 1}],
                {1: 1.0, 2: 0.0},
                id='partial_trials_with_events',
            ),
        ],
    )
    def test_event_ratio_with_trials(self, samples, trial_columns, events_data, expected_ratios):
        """Test event ratio calculation with trial columns."""
        if events_data:
            events = pm.Events(pl.DataFrame(events_data))
            events.trial_columns = trial_columns
        else:
            events = None

        gaze = pm.Gaze(samples=samples, events=events, trial_columns=trial_columns)

        result = gaze.samples.group_by(trial_columns, maintain_order=True).agg(
            gaze.measure_events_ratio('blink').mean(),
        )

        expected_data = []
        unique_trials = samples.select(trial_columns).unique().sort(trial_columns)
        for trial in unique_trials.to_dicts():
            if len(trial_columns) == 1:
                key = trial[trial_columns[0]]
            else:
                key = tuple(trial[col] for col in trial_columns)
            expected_data.append(
                {**trial, 'event_ratio_blink': expected_ratios.get(key, 0.0)},
            )

        expected = pl.DataFrame(expected_data)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ('setup_fn', 'expected_data'),
        [
            # gaze.events is None
            pytest.param(
                lambda g: setattr(g, 'events', None),
                {'event_ratio_blink': [0.0]},
                id='events_none',
            ),
            # events exist but none match name
            pytest.param(
                lambda g: setattr(
                    g, 'events', pm.Events(name=['saccade'], onsets=[1.0], offsets=[2.0]),
                ),
                {'event_ratio_blink': [0.0]},
                id='no_matching_name',
            ),
            # events is an empty Events object
            pytest.param(
                lambda g: setattr(g, 'events', pm.Events()),
                {'event_ratio_blink': [0.0]},
                id='empty_events',
            ),
        ],
    )
    def test_event_ratio_edge_cases(self, fixture_gaze_data, setup_fn, expected_data):
        """Test edge cases for event ratio."""
        setup_fn(fixture_gaze_data)
        result = fixture_gaze_data.samples.select(fixture_gaze_data.measure_events_ratio('blink'))

        expected = pl.DataFrame(expected_data)
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ('kwargs', 'error_type', 'match'),
        [
            ({'name': ''}, ValueError, 'non-empty string'),
            ({'name': None}, ValueError, 'non-empty string'),
            ({'time_column': 'missing'}, ValueError, 'not found'),
            ({'time_column': 123}, TypeError, 'invalid type'),
        ],
    )
    def test_event_ratio_validation(self, fixture_gaze_data, kwargs, error_type, match):
        """Test input validation for measure_events_ratio."""
        base_kwargs = {'name': 'blink'}
        base_kwargs.update(kwargs)
        with pytest.raises(error_type, match=match):
            fixture_gaze_data.measure_events_ratio(**base_kwargs)

    @pytest.mark.parametrize(
        ('sampling_rate_arg', 'experiment_rate', 'expected'),
        [
            # Sampling rate from experiment (1000.0 Hz -> dt = 1.0)
            # Expected: (3-1+1) + (7-5+1) / (7-0+1) = 6/8 = 0.75
            pytest.param(None, 1000.0, 0.75, id='from_experiment'),
            # Explicit override (500.0 Hz -> dt = 2.0)
            # Expected: (3-1+2) + (7-5+2) / (7-0+2) = 8/9
            pytest.param(500.0, 1000.0, 8 / 9, id='explicit_override'),
            # No experiment, no rate (dt = 0.0)
            # Expected: (3-1) + (7-5) / (7-0) = 4/7
            pytest.param(None, None, 4 / 7, id='no_experiment_no_rate'),
        ],
    )
    def test_measure_events_ratio_sampling_rate_fallback(
        self,
        sampling_rate_arg,
        experiment_rate,
        expected,
    ):
        """Test sampling_rate fallback logic for measure_events_ratio."""
        samples = pl.DataFrame({
            'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'pixel': [[0, 0]] * 8,
        })
        events = pm.Events(
            name=['blink', 'blink'],
            onsets=[1.0, 5.0],
            offsets=[3.0, 7.0],
        )
        experiment = pm.Experiment(sampling_rate=experiment_rate) if experiment_rate else None
        gaze = pm.Gaze(samples=samples, events=events, experiment=experiment)

        result = gaze.samples.select(
            gaze.measure_events_ratio(
                'blink', sampling_rate=sampling_rate_arg,
            ),
        )

        assert result.to_series()[0] == pytest.approx(expected)
