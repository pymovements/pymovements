# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test event processing classes."""

from math import sqrt

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from pymovements.exceptions import UnknownMeasure
from pymovements.measure.events import duration
from pymovements.measure.events import EventProcessor
from pymovements.measure.events import EventSamplesProcessor
from pymovements.measure.samples import peak_velocity


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_measures'),
    [
        pytest.param(['duration'], {}, [duration], id='arg_str_duration'),
        pytest.param([['duration']], {}, [duration], id='arg_list_duration'),
        pytest.param([], {'measures': 'duration'}, [duration], id='kwarg_measures_duration'),
    ],
)
def test_event_processor_init(args, kwargs, expected_measures):
    processor = EventProcessor(*args, **kwargs)

    assert len(processor.measures) == len(expected_measures)
    for measure, expected_measure in zip(processor.measures, expected_measures):
        assert str(measure) == str(expected_measure())


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'message'),
    [
        pytest.param(
            ['foo'],
            {},
            UnknownMeasure,
            "Measure 'foo' is unknown.",
            id='unknown_measure',
        ),
    ],
)
def test_event_processor_init_exceptions(args, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        EventProcessor(*args, **kwargs)


@pytest.mark.parametrize(
    ('events', 'measures', 'expected_dataframe'),
    [
        pytest.param(
            pl.DataFrame(schema={'onset': pl.Int64, 'offset': pl.Int64}),
            'duration',
            pl.DataFrame(schema={'duration': pl.Int64}),
            id='duration_no_event',
        ),
        pytest.param(
            pl.DataFrame(
                data={'onset': [0], 'offset': [1]},
                schema={'onset': pl.Int64, 'offset': pl.Int64},
            ),
            'duration',
            pl.DataFrame(data=[1], schema={'duration': pl.Int64}),
            id='duration_single_event',
        ),
        pytest.param(
            pl.DataFrame(
                data={'onset': [0, 100], 'offset': [1, 111]},
                schema={'onset': pl.Int64, 'offset': pl.Int64},
            ),
            'duration',
            pl.DataFrame(data=[1, 11], schema={'duration': pl.Int64}),
            id='duration_two_events',
        ),
    ],
)
def test_event_processor_process_correct_result(events, measures, expected_dataframe):
    processor = EventProcessor(measures)

    measure_result = processor.process(events)
    assert_frame_equal(measure_result, expected_dataframe)


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_measures'),
    [
        pytest.param(['peak_velocity'], {}, [peak_velocity], id='arg_str_peak_velocity'),
        pytest.param([['peak_velocity']], {}, [peak_velocity], id='arg_list_peak_velocity'),
        pytest.param(
            [],
            {'measures': 'peak_velocity'},
            [peak_velocity],
            id='kwarg_measures_peak_velocity',
        ),
    ],
)
def test_event_samples_processor_init(args, kwargs, expected_measures):
    processor = EventSamplesProcessor(*args, **kwargs)

    assert len(processor.measures) == len(expected_measures)
    for measure, expected_measure in zip(processor.measures, expected_measures):
        assert str(measure) == str(expected_measure())


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'message'),
    [
        pytest.param(
            ['foo'],
            {},
            UnknownMeasure,
            "Measure 'foo' is unknown.",
            id='unknown_event_measure',
        ),
        pytest.param(
            [('peak_velocity', {}, None)],
            {},
            ValueError,
            'Tuple must have a length of 2.',
            id='tuple_length_incorrect',
        ),
        pytest.param(
            [[('peak_velocity', {}), ('amplitude', {}, 1)]],
            {},
            ValueError,
            'Tuple must have a length of 2.',
            id='tuple_length_incorrect1',
        ),
        pytest.param(
            [(1, {})],
            {},
            TypeError,
            'First item of tuple must be a string',
            id='first_item_not_string',
        ),
        pytest.param(
            [('peak_velocity', 1)],
            {},
            TypeError,
            'Second item of tuple must be a dictionary',
            id='second_item_not_dict',
        ),
        pytest.param(
            [[(1, 1), (1, {})]],
            {},
            TypeError,
            'First item of tuple must be a string',
            id='first_item_not_string1',
        ),
        pytest.param(
            [[('peak_velocity', 1), ('amplitude', 1)]],
            {},
            TypeError,
            'Second item of tuple must be a dictionary',
            id='second_item_not_dict1',
        ),
        pytest.param(
            [['peak_velocity', 1]],
            {},
            TypeError,
            'Each item in the list must be either a string or a tuple',
            id='list_contains_invalid_item',
        ),
        pytest.param(
            [1],
            {},
            TypeError,
            'measures must be of type str, tuple, or list',
            id='measures_invalid_type',
        ),
    ],
)
def test_event_samples_processor_init_exceptions(args, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        EventSamplesProcessor(*args, **kwargs)


@pytest.mark.parametrize(
    ('events', 'samples', 'init_kwargs', 'process_kwargs', 'expected_dataframe'),
    [
        pytest.param(
            pl.DataFrame(
                schema={'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'time': [0, 1, 2, 3, 4],
                    'velocity': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                },
                schema={'time': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': None},
            pl.DataFrame(
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            marks=pytest.mark.filterwarnings(
                'ignore:No events available for processing.*:UserWarning',
            ),
            id='no_event_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'name': ['fixation'], 'onset': [0], 'offset': [4]},
                schema={'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'time': [0, 1, 2, 3, 4],
                    'velocity': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                },
                schema={'time': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': None},
            pl.from_dict(
                {'name': ['fixation'], 'onset': [0], 'offset': [4], 'peak_velocity': [0.0]},
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='no_identifier_one_fixation_default_columns_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'name': ['fixation', 'saccade'], 'onset': [0, 5], 'offset': [4, 7]},
            ),
            pl.from_dict(
                {
                    'time': [0, 1, 2, 3, 4, 5, 6, 7],
                    'velocity': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]],
                }
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': None},
            pl.from_dict(
                {
                    'name': ['fixation', 'saccade'],
                    'onset': [0, 5],
                    'offset': [4, 7],
                    'peak_velocity': [0.0, sqrt(2)],
                },
            ),
            id='no_identifier_two_events_default_columns_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'name': ['fixation', 'saccade', 'blink'], 'onset': [0, 3, 7], 'offset': [2, 6, 7]},
            ),
            pl.from_dict(
                {
                    'time': [0, 1, 2, 3, 4, 5, 6, 7],
                    'velocity': [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                }
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': None},
            pl.from_dict(
                {
                    'name': ['fixation', 'saccade', 'blink'],
                    'onset': [0, 3, 7],
                    'offset': [2, 6, 7],
                    'peak_velocity': [0.0, 1, sqrt(2)],
                },
            ),
            id='no_identifier_three_events_default_columns_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'name': ['saccade'], 'onset': [0], 'offset': [10]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'velocity': np.repeat([[0, 1]], 10, axis=0),
                },
                schema={'subject_id': pl.Int64, 'time': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': ['saccade'],
                    'onset': [0],
                    'offset': [10],
                    'peak_velocity': [1],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='one_identifier_single_event_complete_window_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'name': ['saccade'], 'onset': [0], 'offset': [5]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'velocity': np.column_stack(
                        [
                            np.concatenate([np.ones(5), np.zeros(5)]),
                            np.zeros(10),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': ['saccade'],
                    'onset': [0],
                    'offset': [5],
                    'peak_velocity': [1],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='one_identifier_single_event_half_window_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'name': ['fixation'], 'onset': [0], 'offset': [10]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'position': np.column_stack(
                        [
                            np.concatenate([np.ones(5), np.zeros(5)]),
                            np.concatenate([np.zeros(5), np.ones(5)]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'position': pl.List(pl.Float64),
                },
            ),
            {'measures': 'dispersion'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': ['fixation'],
                    'onset': [0],
                    'offset': [10],
                    'dispersion': [2],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'dispersion': pl.Float64,
                },
            ),
            id='one_identifier_single_event_complete_window_dispersion',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'name': ['saccade'], 'onset': [0], 'offset': [10]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'velocity': np.column_stack(
                        [
                            np.arange(0.1, 1.1, 0.1),
                            np.arange(0.1, 1.1, 0.1),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': ['saccade'],
                    'onset': [0],
                    'offset': [10],
                    'peak_velocity': [np.sqrt(2)],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='one_identifier_single_event_rising_velocity_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'velocity': np.column_stack(
                        [
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1, 1],
                    'name': ['A', 'B'],
                    'onset': [0, 80],
                    'offset': [10, 100],
                    'peak_velocity': [np.sqrt(2), 2 * np.sqrt(2)],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='one_identifier_two_events_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'velocity': np.column_stack(
                        [
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id', 'name': 'A'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': ['A'],
                    'onset': [0],
                    'offset': [10],
                    'peak_velocity': [np.sqrt(2)],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='one_identifier_two_events_peak_velocity_name_filter',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'position': np.column_stack(
                        [
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(20)]),
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(20)]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'position': pl.List(pl.Float64),
                },
            ),
            {'measures': 'location'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1, 1],
                    'name': ['A', 'B'],
                    'onset': [0, 80],
                    'offset': [10, 100],
                    'location': [[1, 1], [2, 2]],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'location': pl.List(pl.Float64),
                },
            ),
            id='one_identifier_two_events_location',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'position': np.column_stack(
                        [
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'position': pl.List(pl.Float64),
                },
            ),
            {'measures': ('location', {'method': 'mean'})},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1, 1],
                    'name': ['A', 'B'],
                    'onset': [0, 80],
                    'offset': [10, 100],
                    'location': [[1.0, 1.0], [11.9, 11.9]],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'location': pl.List(pl.Float64),
                },
            ),
            id='one_identifier_two_events_location_method_mean',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'position': np.column_stack(
                        [
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'position': pl.List(pl.Float64),
                },
            ),
            {'measures': ('location', {'method': 'median'})},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1, 1],
                    'name': ['A', 'B'],
                    'onset': [0, 80],
                    'offset': [10, 100],
                    'location': [[1, 1], [2, 2]],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'location': pl.List(pl.Float64),
                },
            ),
            id='one_identifier_two_events_location_method_median',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'pixel': np.column_stack(
                        [
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                            np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(19), [200]]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            {'measures': ('location', {'position_column': 'pixel'})},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1, 1],
                    'name': ['A', 'B'],
                    'onset': [0, 80],
                    'offset': [10, 100],
                    'location': [[1.0, 1.0], [11.9, 11.9]],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'location': pl.List(pl.Float64),
                },
            ),
            id='one_identifier_two_events_location_position_column_pixel',
        ),
        pytest.param(
            pl.from_dict(
                {
                    'task': ['A', 'B'],
                    'trial': [0, 0],
                    'name': ['fixation', 'saccade'],
                    'onset': [0, 7],
                    'offset': [3, 8],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'task': ['A'] * 10 + ['B'] * 10,
                    'trial': [0] * 20,
                    'time': list(range(10)) * 2,
                    'velocity': [
                        *[[1, 1]] * 3 + [[0, 0]] * 7,  # task A, trial 0
                        *[[0, 0]] * 7 + [[1, 0]] * 2 + [[0, 0]],  # task B, trial 0
                    ],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': ['task', 'trial']},
            pl.from_dict(
                {
                    'task': ['A', 'B'],
                    'trial': [0, 0],
                    'name': ['fixation', 'saccade'],
                    'onset': [0, 7],
                    'offset': [3, 8],
                    'peak_velocity': [sqrt(2), 1],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='two_identifiers_two_events_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {
                    'task': ['A', 'A', 'B', 'B'],
                    'trial': [0, 1, 0, 1],
                    'name': ['fixation', 'saccade', 'fixation', 'saccade'],
                    'onset': [0, 2, 5, 4],
                    'offset': [8, 6, 7, 9],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'task': ['A'] * 20 + ['B'] * 20,
                    'trial': [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10,
                    'time': list(range(10)) * 4,
                    'velocity': [
                        *[[1, 1]] * 8 + [[0, 0]] * 2,  # task A, trial 0
                        *[[1, 1]] * 2 + [[1, 0]] * 4 + [[0, 0]] * 4,  # task A, trial 1
                        *[[0, 0]] * 5 + [[1, 1]] * 2 + [[0, 0]] * 3,  # task B, trial 0
                        *[[1, 1]] * 4 + [[0, 0]] * 6,  # task B, trial 1
                    ],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': ['task', 'trial']},
            pl.from_dict(
                {
                    'task': ['A', 'A', 'B', 'B'],
                    'trial': [0, 1, 0, 1],
                    'name': ['fixation', 'saccade', 'fixation', 'saccade'],
                    'onset': [0, 2, 5, 4],
                    'offset': [8, 6, 7, 9],
                    'peak_velocity': [sqrt(2), 1, sqrt(2), 0],
                },
                schema={
                    'task': pl.Utf8,
                    'trial': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='two_identifiers_four_events_peak_velocity',
        ),
    ],
)
def test_event_samples_processor_process_correct_result(
    events,
    samples,
    init_kwargs,
    process_kwargs,
    expected_dataframe,
):
    processor = EventSamplesProcessor(**init_kwargs)
    measure_result = processor.process(events, samples, **process_kwargs)
    assert_frame_equal(measure_result, expected_dataframe, check_dtypes=False)


@pytest.mark.parametrize(
    ('events', 'samples', 'init_kwargs', 'process_kwargs', 'warning', 'message'),
    [
        pytest.param(
            pl.DataFrame(
                schema={'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'time': [0, 1, 2, 3, 4],
                    'position': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                },
                schema={'time': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            {'measures': 'amplitude'},
            {'identifiers': None},
            UserWarning,
            'No events available for processing. Creating empty columns for.*amplitude',
            id='no_events_amplitude',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': 'abcdef', 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(100),
                    'time': np.arange(100),
                    'velocity': np.column_stack(
                        [
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                            np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        ]
                    ),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'velocity': pl.List(pl.Float64),
                },
            ),
            {'measures': 'peak_velocity'},
            {'identifiers': 'subject_id', 'name': 'cde'},
            UserWarning,
            "No events found with name 'cde'.",
            marks=pytest.mark.filterwarnings(
                'ignore:No events available for processing.*:UserWarning',
            ),
            id='event_name_not_in_dataframe',
        ),
    ],
)
def test_event_samples_processor_process_warnings(
    events,
    samples,
    init_kwargs,
    process_kwargs,
    warning,
    message,
):
    processor = EventSamplesProcessor(**init_kwargs)

    with pytest.warns(warning, match=message):
        processor.process(events, samples, **process_kwargs)
