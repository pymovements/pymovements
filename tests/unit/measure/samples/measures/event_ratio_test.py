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
            'event_durations',
            'expected_ratio',
        ),
        [
            # Single blink event covering [1, 4) = 3 time units
            # Total time range = 7 - 0 + 1 = 8
            pytest.param('blink', ['blink'], [1.0], [3.0], 3 / 8, id='single_blink'),
            # Two blink events: [1, 3) = 2 units + [5, 7) = 2 units = 4 total
            # Total time range = 8 (from 0 to 7 + dt)
            pytest.param(
                'blink',
                ['blink', 'blink'],
                [1.0, 5.0],
                [2.0, 2.0],
                4 / 8,
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
            # Event covering single sample [1, 2) = 1 unit
            # Total time range = 8
            pytest.param('blink', ['blink'], [1.0], [1.0], 1 / 8, id='single_sample'),
        ],
    )
    def test_event_ratio(
        self,
        fixture_gaze_data,
        event_name,
        event_names,
        event_onsets,
        event_durations,
        expected_ratio,
    ):
        """Test event ratio calculation with various configurations."""
        events = pm.Events(name=event_names, onsets=event_onsets, durations=event_durations)
        gaze = pm.Gaze(samples=fixture_gaze_data.samples, events=events)

        result = gaze.samples.select(gaze.measure_events_ratio(event_name))

        expected = pl.DataFrame({f'event_ratio_{event_name}': [expected_ratio]})
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
                [{'name': 'blink', 'onset': 1.0, 'duration': 2.0, 'trial': 1}],
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
                    {'name': 'blink', 'onset': 1.0, 'duration': 1.0, 'trial': 1},
                    {'name': 'blink', 'onset': 1.0, 'duration': 1.0, 'trial': 2},
                ],
                {1: 0.5, 2: 0.5},
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
                    {'name': 'blink', 'onset': 0.0, 'duration': 2.0, 'trial': 1},
                ],
                {1: 2 / 3, 2: 0.0},
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
                    {'name': 'blink', 'onset': 0.0, 'duration': 1.0, 'trial': 1, 'session': 1},
                ],
                {(1, 1): 0.5, (2, 1): 0.0},
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
                    {'name': 'blink', 'onset': 0.0, 'duration': 2.0, 'trial': 1},
                    {'name': 'blink', 'onset': 2.0, 'duration': 2.0, 'trial': 1},
                ],
                {1: 1.0},
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
                    {'name': 'blink', 'onset': 0.0, 'duration': 3.0, 'trial': 1},
                    {'name': 'blink', 'onset': 2.0, 'duration': 2.0, 'trial': 1},
                ],
                # overlapping intervals [0, 2] and [2, 3] are merged into [0, 3]
                # before summing, so the ratio is (3 - 0 + 1) / (3 - 0 + 1) = 1.0
                {1: 1.0},
                id='overlapping_events',
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
                [{'name': 'blink', 'onset': 0.0, 'duration': 2.0, 'trial': 1}],
                {1: 1.0, 2: 0.0},
                id='partial_trials_with_events',
            ),
        ],
    )
    def test_event_ratio_with_trials(self, samples, trial_columns, events_data, expected_ratios):
        """Test event ratio calculation with trial columns."""
        if events_data:
            events = pm.Events(pl.DataFrame(events_data), trial_columns=trial_columns)
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
                    g, 'events', pm.Events(name=['saccade'], onsets=[1.0], durations=[2.0]),
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
            # Expected: (3 + 3) / (7-0+1) = 6/8 = 0.75
            pytest.param(None, 1000.0, 0.75, id='from_experiment'),
            # Explicit override (500.0 Hz -> dt = 2.0), dt only extends the time range
            # Expected: (3 + 3) / (7-0+2) = 6/9
            pytest.param(500.0, 1000.0, 6 / 9, id='explicit_override'),
            # No experiment, no rate (dt = 1.0 inferred from mode)
            # Expected: (3 + 3) / (7-0+1) = 6/8 = 0.75
            pytest.param(None, None, 0.75, id='no_experiment_no_rate'),
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
            durations=[3.0, 3.0],
        )
        experiment = pm.Experiment(sampling_rate=experiment_rate) if experiment_rate else None
        gaze = pm.Gaze(samples=samples, events=events, experiment=experiment)

        result = gaze.samples.select(
            gaze.measure_events_ratio(
                'blink', sampling_rate=sampling_rate_arg,
            ),
        )

        assert result.to_series()[0] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ('samples', 'gaze_trial_columns', 'events_data', 'expected_ratio'),
        [
            # A blink in the first trial only. Session ratio ignores trial grouping.
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 3.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [{'name': 'blink', 'onset': 1.0, 'duration': 2.0, 'trial': 1}],
                0.5,
                id='single_blink_session_ratio',
            ),
            # Blinks in both trials combine into a single session-wide ratio.
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 3.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 0.0, 'duration': 2.0, 'trial': 1},
                    {'name': 'blink', 'onset': 2.0, 'duration': 2.0, 'trial': 2},
                ],
                1.0,
                id='blinks_in_both_trials_session_ratio',
            ),
            # No matching events yields a session-level zero.
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [{'name': 'saccade', 'onset': 0.0, 'duration': 2.0, 'trial': 1}],
                0.0,
                id='no_matching_events_session_ratio',
            ),
        ],
    )
    def test_event_ratio_trial_columns_empty(
        self,
        samples,
        gaze_trial_columns,
        events_data,
        expected_ratio,
    ):
        """trial_columns=[] yields a single session-level scalar and item() works."""
        events = pm.Events(pl.DataFrame(events_data), trial_columns=gaze_trial_columns)
        gaze = pm.Gaze(samples=samples, events=events, trial_columns=gaze_trial_columns)

        result = gaze.samples.select(gaze.measure_events_ratio('blink', trial_columns=[]))

        assert result.shape == (1, 1)
        assert result.item() == pytest.approx(expected_ratio)

    @pytest.mark.parametrize(
        ('samples', 'gaze_trial_columns', 'events_data', 'expected_ratios'),
        [
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial'],
                [{'name': 'blink', 'onset': 1.0, 'duration': 2.0, 'trial': 1}],
                {1: 1.0, 2: 0.0},
                id='single_trial_with_event',
            ),
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                    'trial': [1, 1, 1, 2, 2, 2],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                }),
                ['trial'],
                [
                    {'name': 'blink', 'onset': 0.0, 'duration': 2.0, 'trial': 1},
                ],
                {1: 2 / 3, 2: 0.0},
                id='event_partial_trial',
            ),
        ],
    )
    def test_event_ratio_default_trial_columns(
        self,
        samples,
        gaze_trial_columns,
        events_data,
        expected_ratios,
    ):
        """Not passing trial_columns preserves existing behavior (uses self.trial_columns)."""
        events = pm.Events(pl.DataFrame(events_data), trial_columns=gaze_trial_columns)
        gaze = pm.Gaze(samples=samples, events=events, trial_columns=gaze_trial_columns)

        default_result = gaze.samples.group_by(gaze_trial_columns, maintain_order=True).agg(
            gaze.measure_events_ratio('blink').mean(),
        )
        explicit_result = gaze.samples.group_by(gaze_trial_columns, maintain_order=True).agg(
            gaze.measure_events_ratio('blink', trial_columns=gaze.trial_columns).mean(),
        )

        assert_frame_equal(default_result, explicit_result)

        expected_data = []
        unique_trials = samples.select(gaze_trial_columns).unique().sort(gaze_trial_columns)
        for trial in unique_trials.to_dicts():
            expected_data.append(
                {
                    **trial,
                    'event_ratio_blink': expected_ratios.get(trial[gaze_trial_columns[0]], 0.0),
                },
            )

        expected = pl.DataFrame(expected_data)
        assert_frame_equal(default_result, expected)

    @pytest.mark.parametrize(
        (
            'samples', 'gaze_trial_columns', 'override_trial_columns', 'events_data',
            'expected_ratios',
        ),
        [
            # Gaze is trialized by (trial, session), but override groups by session only.
            pytest.param(
                pl.DataFrame({
                    'time': [0.0, 1.0, 0.0, 1.0],
                    'trial': [1, 1, 2, 2],
                    'session': [1, 1, 1, 1],
                    'pixel': [[0, 0], [1, 1], [2, 2], [3, 3]],
                }),
                ['trial', 'session'],
                ['session'],
                [
                    {'name': 'blink', 'onset': 0.0, 'duration': 1.0, 'trial': 1, 'session': 1},
                ],
                {1: 0.5},
                id='override_trial_columns_by_session',
            ),
        ],
    )
    def test_event_ratio_trial_columns_override(
        self,
        samples,
        gaze_trial_columns,
        override_trial_columns,
        events_data,
        expected_ratios,
    ):
        """Passing explicit trial_columns overrides self.trial_columns."""
        events = pm.Events(pl.DataFrame(events_data), trial_columns=gaze_trial_columns)
        gaze = pm.Gaze(samples=samples, events=events, trial_columns=gaze_trial_columns)

        result = gaze.samples.group_by(override_trial_columns, maintain_order=True).agg(
            gaze.measure_events_ratio('blink', trial_columns=override_trial_columns).mean(),
        )

        expected_data = []
        unique_groups = (
            samples.select(override_trial_columns).unique().sort(override_trial_columns)
        )
        for group in unique_groups.to_dicts():
            expected_data.append(
                {
                    **group,
                    'event_ratio_blink': expected_ratios.get(
                        group[override_trial_columns[0]], 0.0,
                    ),
                },
            )

        expected = pl.DataFrame(expected_data)
        assert_frame_equal(result, expected)


@pytest.fixture(name='detected_gaze')
def fixture_detected_gaze() -> pm.Gaze:
    """Create gaze at 250 Hz with two fixations detected by I-VT around a saccade."""
    velocities = [[0.0, 0.0]] * 5 + [[100.0, 100.0]] * 3 + [[0.0, 0.0]] * 6
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': [float(i * 4) for i in range(14)],
            'velocity': velocities,
        }),
        experiment=pm.Experiment(sampling_rate=250),
    )
    gaze.detect('ivt', velocity_threshold=20, minimum_duration=8)
    return gaze


@pytest.mark.parametrize(
    ('onset', 'duration'),
    [
        pytest.param('start', 'length', id='str'),
        pytest.param(pl.col('start'), pl.col('length'), id='expr'),
    ],
)
def test_measure_events_ratio_onset_and_duration_arguments(fixture_gaze_data, onset, duration):
    events = pm.Events(name=['blink'], onsets=[1.0], durations=[3.0])
    events.frame = events.frame.rename({'onset': 'start', 'duration': 'length'})
    fixture_gaze_data.events = events

    result = fixture_gaze_data.samples.select(
        fixture_gaze_data.measure_events_ratio('blink', onset=onset, duration=duration),
    )

    assert result.item() == pytest.approx(3 / 8)


def test_measure_events_ratio_null_duration_contributes_nothing(fixture_gaze_data):
    fixture_gaze_data.events = pm.Events(
        pl.DataFrame(
            {'name': ['blink', 'blink'], 'onset': [1.0, 5.0], 'duration': [None, 2.0]},
            schema={'name': pl.String, 'onset': pl.Float64, 'duration': pl.Float64},
        ),
    )

    result = fixture_gaze_data.samples.select(fixture_gaze_data.measure_events_ratio('blink'))

    assert result.item() == pytest.approx(2 / 8)


@pytest.mark.parametrize(
    ('kwargs', 'argument'),
    [
        pytest.param({'onset_column': 'onset'}, 'onset_column', id='onset_column'),
        pytest.param({'offset_column': 'offset'}, 'offset_column', id='offset_column'),
    ],
)
def test_measure_events_ratio_column_arguments_are_deprecated(
        detected_gaze, kwargs, argument, assert_deprecation_is_removed,
):
    with pytest.raises(DeprecationWarning) as info:
        detected_gaze.measure_events_ratio('fixation', **kwargs)

    assert_deprecation_is_removed(
        function_name=f"events2timeratio argument '{argument}'",
        warning_message=info.value.args[0],
        scheduled_version='0.34.0',
    )
