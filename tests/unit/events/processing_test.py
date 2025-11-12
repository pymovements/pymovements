# Copyright (c) 2023-2025 The pymovements Project Authors
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
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_property_definitions'),
    [
        pytest.param(['duration'], {}, ['duration'], id='arg_str_duration'),
        pytest.param([['duration']], {}, ['duration'], id='arg_list_duration'),
        pytest.param(
            [], {'event_properties': 'duration'}, ['duration'],
            id='kwarg_properties_duration',
        ),
    ],
)
def test_event_processor_init(args, kwargs, expected_property_definitions):
    processor = pm.EventProcessor(*args, **kwargs)

    assert processor.event_properties == expected_property_definitions


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            ['foo'], {},
            pm.exceptions.InvalidProperty, ('foo', 'invalid', 'duration'),
            id='unknown_event_property',
        ),
    ],
)
def test_event_processor_init_exceptions(args, kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.EventProcessor(*args, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('events_kwargs', 'event_properties', 'expected_dataframe'),
    [
        pytest.param(
            {'onsets': [], 'offsets': []},
            'duration',
            pl.DataFrame(schema={'duration': pl.Int64}),
            id='duration_no_event',
        ),
        pytest.param(
            {'onsets': [0], 'offsets': [1]},
            'duration',
            pl.DataFrame(data=[1], schema={'duration': pl.Int64}),
            id='duration_single_event',
        ),
        pytest.param(
            {'onsets': [0, 100], 'offsets': [1, 111]},
            'duration',
            pl.DataFrame(data=[1, 11], schema={'duration': pl.Int64}),
            id='duration_two_events',
        ),
    ],
)
def test_event_processor_process_correct_result(
        events_kwargs, event_properties, expected_dataframe,
):
    events = pm.Events(**events_kwargs)
    processor = pm.EventProcessor(event_properties)

    property_result = processor.process(events)
    assert_frame_equal(property_result, expected_dataframe)


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_property_definitions'),
    [
        pytest.param(['peak_velocity'], {}, [('peak_velocity', {})], id='arg_str_peak_velocity'),
        pytest.param([['peak_velocity']], {}, [('peak_velocity', {})], id='arg_list_peak_velocity'),
        pytest.param(
            [], {'event_properties': 'peak_velocity'}, [('peak_velocity', {})],
            id='kwarg_properties_peak_velocity',
        ),
    ],
)
def test_event_gaze_processor_init(args, kwargs, expected_property_definitions):
    processor = pm.EventGazeProcessor(*args, **kwargs)

    assert processor.event_properties == expected_property_definitions


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            ['foo'], {},
            pm.exceptions.InvalidProperty, ('foo', 'invalid', 'peak_velocity'),
            id='unknown_event_property',
        ),
        pytest.param(
            [('peak_velocity', {}, None)], {},
            ValueError, ('Tuple must have a length of 2.'),
            id='tuple_length_incorrect',
        ),
        pytest.param(
            [[('peak_velocity', {}), ('amplitude', {}, 1)]], {},
            ValueError, ('Tuple must have a length of 2.'),
            id='tuple_length_incorrect1',
        ),
        pytest.param(
            [(1, {})], {},
            TypeError, ('First item of tuple must be a string'),
            id='first_item_not_string',
        ),
        pytest.param(
            [('peak_velocity', 1)], {},
            TypeError, ('Second item of tuple must be a dictionary'),
            id='second_item_not_dict',
        ),
        pytest.param(
            [[(1, 1), (1, {})]], {},
            TypeError, ('First item of tuple must be a string'),
            id='first_item_not_string1',
        ),
        pytest.param(
            [[('peak_velocity', 1), ('amplitude', 1)]], {},
            TypeError, ('Second item of tuple must be a dictionary'),
            id='second_item_not_dict1',
        ),
        pytest.param(
            [['peak_velocity', 1]], {},
            TypeError, ('Each item in the list must be either a string or a tuple'),
            id='list_contains_invalid_item',
        ),
        pytest.param(
            [1], {},
            TypeError, ('event_properties must be of type str, tuple, or list'),
            id='event_properties_invalid_type',
        ),
    ],
)
def test_event_gaze_processor_init_exceptions(args, kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.EventGazeProcessor(*args, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('events', 'gaze', 'init_kwargs', 'process_kwargs', 'expected_dataframe'),
    [
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_vel': np.ones(10),
                        'y_vel': np.zeros(10),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
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
            id='peak_velocity_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_vel': np.ones(10),
                        'y_vel': np.zeros(10),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
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
            id='peak_velocity_custom_columns_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [5]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_vel': np.concatenate([np.ones(5), np.zeros(5)]),
                        'y_vel': np.zeros(10),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
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
            id='peak_velocity_single_event_half_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_pos': np.concatenate([np.ones(5), np.zeros(5)]),
                        'y_pos': np.concatenate([np.zeros(5), np.ones(5)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_pos': pl.Float64,
                        'y_pos': pl.Float64,
                    },
                ),
                position_columns=['x_pos', 'y_pos'],
            ),
            {'event_properties': 'dispersion'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
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
            id='dispersion_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_vel': np.concatenate([np.arange(0.1, 1.1, 0.1)]),
                        'y_vel': np.concatenate([np.arange(0.1, 1.1, 0.1)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
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
            id='peak_velocity_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        'y_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
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
            id='two_events_different_names_different_peak_velocity',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        'y_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
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
            id='two_events_peak_velocity_name_filter',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_pos': np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(20)]),
                        'y_pos': np.concatenate([np.ones(11), np.zeros(69), 2 * np.ones(20)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_pos': pl.Float64,
                        'y_pos': pl.Float64,
                    },
                ),
                position_columns=['x_pos', 'y_pos'],
            ),
            {'event_properties': 'location'},
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
            id='two_events_different_names_different_location',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_pos': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                        'y_pos': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_pos': pl.Float64,
                        'y_pos': pl.Float64,
                    },
                ),
                position_columns=['x_pos', 'y_pos'],
            ),
            {'event_properties': ('location', {'method': 'mean'})},
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
            id='two_events_location_method_mean',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_pos': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                        'y_pos': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_pos': pl.Float64,
                        'y_pos': pl.Float64,
                    },
                ),
                position_columns=['x_pos', 'y_pos'],
            ),
            {'event_properties': ('location', {'method': 'median'})},
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
            id='two_events_location_method_median',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': ['A', 'B'], 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_pix': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                        'y_pix': np.concatenate(
                            [np.ones(11), np.zeros(69), 2 * np.ones(19), [200]],
                        ),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_pix': pl.Float64,
                        'y_pix': pl.Float64,
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            {'event_properties': ('location', {'position_column': 'pixel'})},
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
            id='two_events_location_position_column_pixel',
        ),
    ],
)
def test_event_gaze_processor_process_correct_result(
        events, gaze, init_kwargs, process_kwargs, expected_dataframe,
):
    events = pm.Events(events)
    processor = pm.EventGazeProcessor(**init_kwargs)
    property_result = processor.process(events, gaze, **process_kwargs)
    assert_frame_equal(property_result, expected_dataframe)


@pytest.mark.parametrize(
    ('events', 'gaze', 'init_kwargs', 'process_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(10),
                        'time': np.arange(10),
                        'x_vel': np.ones(10),
                        'y_vel': np.zeros(10),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': []},
            ValueError,
            ('identifiers', 'list', 'must', 'not', 'empty'),
            id='empty_list',
        ),

        pytest.param(
            pl.from_dict(
                {'subject_id': [1, 1], 'name': 'abcdef', 'onset': [0, 80], 'offset': [10, 100]},
                schema={
                    'subject_id': pl.Int64, 'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64,
                },
            ),
            pm.Gaze(
                pl.from_dict(
                    {
                        'subject_id': np.ones(100),
                        'time': np.arange(100),
                        'x_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                        'y_vel': np.concatenate([np.ones(10), np.zeros(70), 2 * np.ones(20)]),
                    },
                    schema={
                        'subject_id': pl.Int64,
                        'time': pl.Int64,
                        'x_vel': pl.Float64,
                        'y_vel': pl.Float64,
                    },
                ),
                velocity_columns=['x_vel', 'y_vel'],
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id', 'name': 'cde'},
            RuntimeError,
            ('No events with name "cde" found in data frame',),
            id='event_name_not_in_dataframe',
        ),
    ],
)
def test_event_processor_process_exceptions(
        events, gaze, init_kwargs, process_kwargs, exception, msg_substrings,
):
    processor = pm.EventGazeProcessor(**init_kwargs)
    events = pm.Events(events)

    with pytest.raises(exception) as excinfo:
        processor.process(events, gaze, **process_kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()
