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
"""Tests pymovements.events.Events."""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events


EXPECTED_SCHEMA_AFTER_INIT = {
    'name': pl.Utf8,
    'onset': pl.Duration('us'),
    'offset': pl.Duration('us'),
    'duration': pl.Duration('us'),
}


@pytest.fixture(name='make_events', scope='function')
def fixture_make_events():
    """Make a fixture function to create simple Events objects."""

    def _make_events(names: list[str], properties: list[str] | None = None) -> Events:
        data = {
            'name': names,
            'onset': range(0, 2 * len(names), 2),
            'offset': range(1, 2 * len(names) + 1, 2),
        }
        events = Events(pl.from_dict(data))

        # adding columns afterward to not count them as non-property additional_columns
        if properties is not None:
            events.frame = events.frame.select(
                [pl.all()] + [
                    pl.int_ranges(0, 100 * len(names), 100).alias(property)
                    for property in properties
                ],
            )

        return events

    return _make_events


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'onsets': None, 'offsets': []},
            ValueError, ('onsets', 'offsets', 'both None', 'or', 'both not None'),
            id='onsets_none_offsets_list',
        ),
        pytest.param(
            {'onsets': [], 'offsets': None},
            ValueError, ('onsets', 'offsets', 'both None', 'or', 'both not None'),
            id='onsets_list_offsets_none',
        ),
        pytest.param(
            {'onsets': [], 'offsets': [0]},
            ValueError, ('onsets', 'offsets', 'length', 'equal'),
            id='onsets_empty_list_offsets_single_int',
        ),
        pytest.param(
            {'onsets': [1], 'offsets': []},
            ValueError, ('onsets', 'offsets', 'length', 'equal'),
            id='onsets_single_int_offsets_empty_list',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'name': None, 'onsets': 1, 'offsets': None},
            ValueError, ('data', 'onsets', 'mutually', 'exclusive'),
            id='data_with_onsets_raises_mutually_exclusive',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'offsets': 1},
            ValueError, ('data', 'offsets', 'mutually', 'exclusive'),
            id='data_with_offsets_raises_mutually_exclusive',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'name': 1},
            ValueError, ('data', 'name', 'mutually', 'exclusive'),
            id='data_with_name_raises_mutually_exclusive',
        ),
    ],
)
def test_init_exceptions(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        Events(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('args', 'kwargs'),
    [
        pytest.param([], {}, id='no_args_no_kwargs'),
        pytest.param([], {'onsets': [], 'offsets': []}, id='dict_with_empty_lists_kwarg'),
        pytest.param([], {'onsets': [0], 'offsets': [1]}, id='dict_with_single_event_kwarg'),
        pytest.param([], {'onsets': [0, 2], 'offsets': [1, 3]}, id='dict_with_two_events_kwarg'),
    ],
)
def test_init_expected_schema(args, kwargs):
    events = Events(*args, **kwargs)
    assert events.schema == EXPECTED_SCHEMA_AFTER_INIT


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_length'),
    [
        pytest.param([], {}, 0, id='no_args_no_kwargs'),
        pytest.param([], {'onsets': [], 'offsets': []}, 0, id='dict_with_empty_lists_kwarg'),
        pytest.param([], {'onsets': [0], 'offsets': [1]}, 1, id='dict_with_single_event_kwarg'),
        pytest.param([], {'onsets': [0, 2], 'offsets': [1, 3]}, 2, id='dict_with_two_events_kwarg'),
    ],
)
def test_init_has_expected_length(args, kwargs, expected_length):
    events = Events(*args, **kwargs)
    assert len(events) == expected_length


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_name'),
    [
        pytest.param(
            [pl.DataFrame()], {}, 'foo',
            id='dataframe_arg_dict_with_single_event_kwarg',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]}, 'bar',
            id='dict_with_single_event_with_name_kwarg',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 1], 'offsets': [1, 2]}, 'bar',
            id='dict_with_two_events_with_name_kwarg',
        ),
    ],
)
def test_init_has_correct_name(args, kwargs, expected_name):
    events = Events(*args, **kwargs)
    assert (events['name'].to_numpy() == expected_name).all()


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_names'),
    [
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 1], 'offsets': [1, 2]}, ['foo', 'bar'],
            id='dict_with_two_events_with_name_kwarg',
        ),
    ],
)
def test_init_has_correct_names(args, kwargs, expected_names):
    events = Events(*args, **kwargs)
    assert (events['name'] == expected_names).all()


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_df_data'),
    [
        pytest.param(
            [], {'onsets': [0], 'offsets': [1]},
            {'name': [''], 'onset': [0], 'offset': [1000], 'duration': [1000]},
            id='no_arg_dict_with_single_event_kwarg',
        ),
        pytest.param(
            [pl.DataFrame()], {},
            {},
            id='dataframe_arg_no_kwargs',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]},
            {'name': ['bar'], 'onset': [0], 'offset': [1000], 'duration': [1000]},
            id='dict_with_single_named_event',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 2], 'offsets': [1, 3]},
            {
                'name': ['bar', 'bar'], 'onset': [0, 2000],
                'offset': [1000, 3000], 'duration': [1000, 1000],
            },
            id='dict_with_two_events_same_name',
        ),
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 2], 'offsets': [1, 4]},
            {
                'name': ['foo', 'bar'], 'onset': [0, 2000],
                'offset': [1000, 4000], 'duration': [1000, 2000],
            },
            id='dict_with_two_differently_named_events',
        ),
    ],
)
def test_init_expected(args, kwargs, expected_df_data):
    events = Events(*args, **kwargs)

    expected_df = pl.DataFrame(data=expected_df_data, schema=EXPECTED_SCHEMA_AFTER_INIT)
    assert_frame_equal(events.frame, expected_df)


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_df'),
    [
        pytest.param(
            [], {'onsets': [0], 'offsets': [1]},
            pl.DataFrame(
                {'name': [''], 'onset': [0], 'offset': [1000], 'duration': [1000]},
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='no_arg_lists_with_single_event_kwarg',
        ),
        pytest.param(
            [pl.DataFrame()], {},
            pl.DataFrame({}, schema=EXPECTED_SCHEMA_AFTER_INIT),
            id='dataframe_arg_no_kwargs',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]},
            pl.DataFrame(
                {'name': ['bar'], 'onset': [0], 'offset': [1000], 'duration': [1000]},
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='lists_with_single_named_event',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 2], 'offsets': [1, 3]},
            pl.DataFrame(
                {
                    'name': ['bar', 'bar'], 'onset': [0, 2000],
                    'offset': [1000, 3000], 'duration': [1000, 1000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='lists_with_two_events_same_name',
        ),
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 2], 'offsets': [1, 4]},
            pl.DataFrame(
                {
                    'name': ['foo', 'bar'], 'onset': [0, 2000],
                    'offset': [1000, 4000], 'duration': [1000, 2000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='lists_with_two_differently_named_events',
        ),
        pytest.param(
            [], {'name': ['foo'], 'onsets': [0], 'offsets': [1], 'trials': [1]},
            pl.DataFrame(
                {
                    'trial': [1], 'name': ['foo'], 'onset': [0],
                    'offset': [1000], 'duration': [1000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='lists_one_event_trial_column_at_start',
        ),
        pytest.param(
            [], {
                'data': pl.DataFrame(
                    data={
                        'trial': [1], 'name': ['foo'], 'onset': [0], 'offset': [1],
                    },
                ),
                'trial_columns': 'trial',
            },
            pl.DataFrame(
                {
                    'trial': [1], 'name': ['foo'], 'onset': [0],
                    'offset': [1000], 'duration': [1000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='data_one_event_trial_column_at_start',
        ),
        pytest.param(
            [], {
                'data': pl.DataFrame(
                    data={
                        'name': ['foo'], 'onset': [0], 'offset': [1], 'trial': [1],
                    },
                ),
                'trial_columns': 'trial',
            },
            pl.DataFrame(
                {
                    'trial': [1], 'name': ['foo'], 'onset': [0],
                    'offset': [1000], 'duration': [1000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='data_one_event_trial_column_enforce_start',
        ),
        pytest.param(
            [], {
                'data': pl.from_dict({
                    'trial_id': [1, 1, 2],
                    'name': ['fixation', 'saccade', 'fixation'],
                    'onset': [100, 200, 300],
                    'offset': [150, 250, 350],
                    'custom_property': [1.5, 2.5, 1.5],
                }),
                'trial_columns': 'trial_id',
            },
            pl.DataFrame(
                {
                    'trial_id': [1, 1, 2],
                    'name': ['fixation', 'saccade', 'fixation'],
                    'onset': [100000, 200000, 300000],
                    'offset': [150000, 250000, 350000],
                    'custom_property': [1.5, 2.5, 1.5],
                    'duration': [50000, 50000, 50000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='data_with_trial_columns_preserves_custom_property',
        ),
        pytest.param(
            [], {
                'data': pl.from_dict({
                    'name': ['fixation', 'saccade', 'fixation'],
                    'onset': [100, 200, 300],
                    'offset': [150, 250, 350],
                    'trial_id': [1, 1, 2],
                    'custom_property': [1.5, 2.5, 1.5],
                }),
                'trial_columns': 'trial_id',
            },
            pl.DataFrame(
                {
                    'trial_id': [1, 1, 2],
                    'name': ['fixation', 'saccade', 'fixation'],
                    'onset': [100000, 200000, 300000],
                    'offset': [150000, 250000, 350000],
                    'custom_property': [1.5, 2.5, 1.5],
                    'duration': [50000, 50000, 50000],
                },
                schema_overrides=EXPECTED_SCHEMA_AFTER_INIT,
            ),
            id='data_with_trial_columns_enforce_start_and_preserve_custom',
        ),
    ],
)
def test_init_expected_df(args, kwargs, expected_df):
    events = Events(*args, **kwargs)

    assert_frame_equal(events.frame, expected_df)


@pytest.mark.parametrize(
    ('data', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({
                'name': ['fixation'],
                'onset': pl.Series([5], dtype=pl.Duration('ms')),
                'offset': pl.Series([12], dtype=pl.Duration('ms')),
            }),
            {
                'name': ['fixation'],
                'onset': [timedelta(milliseconds=5)],
                'offset': [timedelta(milliseconds=12)],
                'duration': [timedelta(milliseconds=7)],
            },
            id='duration_ms_input_normalized_to_us',
        ),
        pytest.param(
            pl.DataFrame({
                'name': ['fixation'],
                'onset': pl.Series([5_000_000], dtype=pl.Duration('ns')),
                'offset': pl.Series([12_500_000], dtype=pl.Duration('ns')),
            }),
            {
                'name': ['fixation'],
                'onset': [timedelta(milliseconds=5)],
                'offset': [timedelta(microseconds=12500)],
                'duration': [timedelta(microseconds=7500)],
            },
            id='duration_ns_input_normalized_to_us',
        ),
        pytest.param(
            pl.DataFrame({
                'name': ['fixation'],
                'onset': ['1.5'],
                'offset': ['3'],
            }),
            {
                'name': ['fixation'],
                'onset': [timedelta(microseconds=1500)],
                'offset': [timedelta(milliseconds=3)],
                'duration': [timedelta(microseconds=1500)],
            },
            id='string_columns_interpreted_as_milliseconds',
        ),
    ],
)
def test_init_normalizes_time_columns_to_duration_us(data, expected_data):
    events = Events(data)

    expected_df = pl.DataFrame(
        expected_data,
        schema={
            'name': pl.Utf8,
            'onset': pl.Duration('us'),
            'offset': pl.Duration('us'),
            'duration': pl.Duration('us'),
        },
    )
    assert_frame_equal(events.frame, expected_df)


@pytest.mark.parametrize(
    ('onsets', 'offsets', 'expected_onset_us', 'expected_offset_us'),
    [
        pytest.param([5], [10], 5000, 10000, id='numeric_list_interpreted_as_ms'),
        pytest.param([5.0], [10.5], 5000, 10500, id='numeric_float_interpreted_as_ms'),
        pytest.param(
            np.array([5], dtype='timedelta64[us]'),
            np.array([10], dtype='timedelta64[us]'),
            5, 10, id='timedelta64_us_by_physical_unit',
        ),
        pytest.param(
            np.array([5000], dtype='timedelta64[ns]'),
            np.array([10500], dtype='timedelta64[ns]'),
            5, 10, id='timedelta64_ns_by_physical_unit',
        ),
        pytest.param(
            [timedelta(milliseconds=5)],
            [timedelta(milliseconds=10)],
            5000, 10000, id='python_timedelta_by_physical_unit',
        ),
        # A missing onset/duration value arrives as NaN. Duration has no NaN, so a missing time
        # point is stored as null rather than raising.
        pytest.param([math.nan], [10.0], None, 10000, id='nan_onset_maps_to_null'),
        pytest.param([5.0], [math.nan], 5000, None, id='nan_offset_maps_to_null'),
    ],
)
def test_init_onsets_offsets_convert_by_physical_unit(
        onsets, offsets, expected_onset_us, expected_offset_us,
):
    events = Events(name='fixation', onsets=onsets, offsets=offsets)

    assert events.frame.schema['onset'] == pl.Duration('us')
    assert events.frame['onset'].dt.total_microseconds().to_list() == [expected_onset_us]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [expected_offset_us]


@pytest.mark.parametrize(
    ('time_unit', 'expected_onset_us', 'expected_offset_us'),
    [
        pytest.param('s', 5_000_000, 10_000_000, id='seconds'),
        pytest.param('ms', 5_000, 10_000, id='milliseconds'),
        pytest.param('us', 5, 10, id='microseconds'),
        pytest.param(None, 5_000, 10_000, id='none_defaults_to_milliseconds'),
    ],
)
def test_init_onsets_offsets_respect_time_unit(time_unit, expected_onset_us, expected_offset_us):
    events = Events(name='fixation', onsets=[5], offsets=[10], time_unit=time_unit)

    assert events.frame.schema['onset'] == pl.Duration('us')
    assert events.frame['onset'].dt.total_microseconds().to_list() == [expected_onset_us]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [expected_offset_us]
    assert events.frame['duration'].dt.total_microseconds().to_list() == [
        expected_offset_us - expected_onset_us,
    ]


def test_init_data_frame_respects_time_unit():
    events = Events(
        pl.DataFrame({'name': ['fixation'], 'onset': [5], 'offset': [10], 'duration': [5]}),
        time_unit='s',
    )

    assert events.frame['onset'].dt.total_microseconds().to_list() == [5_000_000]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [10_000_000]
    assert events.frame['duration'].dt.total_microseconds().to_list() == [5_000_000]


def test_init_time_unit_ignored_for_duration_input():
    # Duration input carries its own unit, so time_unit='s' must not rescale it.
    events = Events(
        pl.DataFrame({
            'name': ['fixation'],
            'onset': pl.Series([5], dtype=pl.Duration('us')),
            'offset': pl.Series([10], dtype=pl.Duration('us')),
        }),
        time_unit='s',
    )

    assert events.frame['onset'].dt.total_microseconds().to_list() == [5]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [10]


@pytest.mark.parametrize('time_unit', ['step', 'sec', 'minutes', ''])
def test_init_invalid_time_unit_raises_value_error(time_unit):
    with pytest.raises(ValueError, match='unsupported time unit'):
        Events(name='fixation', onsets=[5], offsets=[10], time_unit=time_unit)


def test_init_rounds_sub_microsecond_duration_input():
    # Sub-microsecond Duration input is rounded to the nearest microsecond, not truncated.
    events = Events(
        pl.DataFrame({
            'name': ['fixation'],
            'onset': pl.Series([1500], dtype=pl.Duration('ns')),   # 1.5 us -> 2 us
            'offset': pl.Series([2600], dtype=pl.Duration('ns')),  # 2.6 us -> 3 us
        }),
    )

    assert events.frame.schema['onset'] == pl.Duration('us')
    assert events.frame['onset'].dt.total_microseconds().to_list() == [2]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [3]
    assert events.frame['duration'].dt.total_microseconds().to_list() == [1]


def test_init_normalizes_large_duration_input_without_overflow():
    # A Duration('ms') beyond ~292 years would overflow an intermediate nanosecond count; the
    # normalization must stay exact for whole-microsecond values rather than silently wrapping.
    onset_ms = 400 * 365 * 24 * 3600 * 1000  # 400 years in milliseconds
    events = Events(
        pl.DataFrame({
            'name': ['fixation'],
            'onset': pl.Series([onset_ms], dtype=pl.Duration('ms')),
            'offset': pl.Series([onset_ms + 5], dtype=pl.Duration('ms')),
        }),
    )

    assert events.frame.schema['onset'] == pl.Duration('us')
    assert events.frame['onset'].dt.total_microseconds().to_list() == [onset_ms * 1000]
    assert events.frame['offset'].dt.total_microseconds().to_list() == [(onset_ms + 5) * 1000]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'name': 'fixation', 'onsets': [math.inf], 'offsets': [10.0]}, id='inf'),
        pytest.param(
            {'name': 'fixation', 'onsets': [5.0], 'offsets': [math.inf]}, id='inf_offset',
        ),
        pytest.param(
            {'name': 'fixation', 'onsets': [-math.inf], 'offsets': [10.0]}, id='neg_inf',
        ),
    ],
)
def test_init_infinite_time_values_raise_value_error(kwargs):
    with pytest.raises(ValueError, match='contain infinite values'):
        Events(**kwargs)


def test_init_invalid_time_unit_raises_value_error_for_duration_input():
    # The unit is validated up front, so a typo is rejected even though it is otherwise ignored
    # for Duration input.
    with pytest.raises(ValueError, match='unsupported time unit'):
        Events(
            pl.DataFrame({
                'name': ['fixation'],
                'onset': pl.Series([5], dtype=pl.Duration('us')),
                'offset': pl.Series([10], dtype=pl.Duration('us')),
            }),
            time_unit='banana',
        )


@pytest.mark.parametrize(
    ('kwargs', 'expected_trial_column_list'),
    [
        pytest.param(
            {'data': pl.DataFrame()},
            None,
            id='empty_df_no_trial_columns',
        ),
        pytest.param(
            {'onsets': [0], 'offsets': [1]},
            None,
            id='single_row_no_trial_columns',
        ),
        pytest.param(
            {'onsets': [0], 'offsets': [1], 'trials': None},
            None,
            id='single_row_trials_list',
        ),
        pytest.param(
            {'onsets': [0], 'offsets': [1], 'trials': ['A']},
            ['trial'],
            id='single_row_trials_list',
        ),
        pytest.param(
            {'data': pl.DataFrame({'onset': [0], 'offset': [1], 'trial': ['A']})},
            None,
            id='single_row_trial_column_not_specified',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'trial': ['A']}),
                'trial_columns': ['trial'],
            },
            ['trial'],
            id='single_row_trial_column_specified',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'group': [1], 'trial': ['C']}),
                'trial_columns': ['group', 'trial'],
            },
            ['group', 'trial'],
            id='single_row_two_trial_columns',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'trial': ['A']}),
                'trial_columns': 'trial',
            },
            ['trial'],
            id='single_row_trial_column_str',
        ),
    ],
)
def test_init_expected_trial_column_list(kwargs, expected_trial_column_list):
    events = Events(**kwargs)

    assert events.trial_columns == expected_trial_column_list


@pytest.mark.parametrize(
    ('kwargs', 'expected_trial_column_data'),
    [
        pytest.param(
            {'onsets': [0], 'offsets': [1], 'trials': ['A']},
            pl.Series('trial', ['A']),
            id='single_row_trials_list',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'trial': ['C']}),
                'trial_columns': 'trial',
            },
            pl.Series('trial', ['C']),
            id='single_row_trial_column_str',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'trial': ['B']}),
                'trial_columns': ['trial'],
            },
            pl.Series('trial', ['B']),
            id='single_row_trial_column_list_single',
        ),
        pytest.param(
            {
                'data': pl.DataFrame({'onset': [0], 'offset': [1], 'group': [1], 'trial': ['C']}),
                'trial_columns': ['group', 'trial'],
            },
            pl.DataFrame({'group': [1], 'trial': ['C']}),
            id='single_row_two_trial_columns',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(
                    {'onset': [0, 2], 'offset': [1, 3], 'trial': [1, 1]},
                ),
                'trial_columns': 'trial',
            },
            pl.DataFrame({'trial': [1, 1]}),
            id='two_rows_one_trial',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(
                    {'onset': [0, 2], 'offset': [1, 3], 'trial': [1, 2]},
                ),
                'trial_columns': 'trial',
            },
            pl.DataFrame({'trial': [1, 2]}),
            id='two_rows_one_trial',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(
                    {'onset': [0, 2], 'offset': [1, 3], 'trial': 1},
                ),
                'trial_columns': 'trial',
            },
            pl.DataFrame({'trial': [1, 1]}, schema_overrides={'trial': pl.Int32}),
            id='two_rows_plain_trial',
        ),
    ],
)
def test_init_expected_trial_column_data(kwargs, expected_trial_column_data):
    events = Events(**kwargs)

    if isinstance(expected_trial_column_data, pl.Series):
        expected_trial_column_data = pl.DataFrame(expected_trial_column_data)
    assert_frame_equal(events.frame[events.trial_columns], expected_trial_column_data)


@pytest.mark.parametrize(
    ('events_left', 'events_right', 'expected'),
    [
        pytest.param(
            Events(),
            Events(),
            True,
            id='empty_events',
        ),
        pytest.param(
            Events(),
            Events(onsets=[0], offsets=[1]),
            False,
            id='one_empty_one_not',
        ),
        pytest.param(
            Events(
                pl.from_dict({'name': ['saccade'], 'onset': [0], 'offset': [1]}),
            ),
            Events(name=['saccade'], onsets=[0], offsets=[1]),
            True,
            id='same_events',
        ),
        pytest.param(
            Events(
                pl.from_dict({'name': ['saccade', None], 'onset': [0, 1], 'offset': [1, 2]}),
            ),
            Events(
                pl.from_dict({'name': ['saccade', None], 'onset': [0, 1], 'offset': [1, 2]}),
            ),
            True,
            id='same_events_with_nulls',
        ),
        pytest.param(
            Events(name=['saccade'], onsets=[0], offsets=[1]),
            Events(name=['fixation'], onsets=[0], offsets=[1]),
            False,
            id='different_events',
        ),
        pytest.param(
            Events(name=['saccade'], onsets=[0], offsets=[1], trials=[0]),
            Events(name=['saccade'], onsets=[0], offsets=[1], trials=[1]),
            False,
            id='same_events_different_trials',
        ),
        pytest.param(
            Events(
                pl.from_dict({'trial': [0], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                trial_columns='trial',
            ),
            Events(name=['saccade'], onsets=[0], offsets=[1], trials=[0]),
            True,
            id='same_events_same_trials',
        ),
        pytest.param(
            Events(
                pl.from_dict({'trial': [0], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
                trial_columns='trial',
            ),
            Events(
                pl.from_dict({'trial': [0], 'name': ['saccade'], 'onset': [0], 'offset': [1]}),
            ),
            False,
            id='same_events_same_trials_different_trial_columns',
        ),
    ],
)
def test_equality_as_expected(events_left, events_right, expected):
    assert (events_left == events_right) == expected


def test_columns_same_as_frame():
    init_kwargs = {'onsets': [0], 'offsets': [1]}
    events = Events(**init_kwargs)

    assert events.columns == events.frame.columns


@pytest.mark.parametrize(
    'events',
    [
        pytest.param(
            Events(name='saccade', onsets=[0], offsets=[123]),
            id='simple_events_no_trials',
        ),
        pytest.param(
            Events(
                data=pl.from_dict(
                    {
                        'trial_id': [1],
                        'name': ['saccade'],
                        'onset': [0],
                        'offset': [123],
                        'custom_property': [42],
                    },
                ),
                trial_columns='trial_id',
            ),
            id='events_with_trial_columns_and_custom_property',  # regression test for #1349
        ),
    ],
)
def test_clone(events):
    events_copy = events.clone()

    # We want to have separate dataframes but with the exact same data.
    assert events is not events_copy
    assert events.frame is not events_copy.frame
    assert_frame_equal(events.frame, events_copy.frame)


def test_clones_trial_columns():
    events = Events(data=pl.DataFrame({'trial': 'trial'}), trial_columns='trial')
    events_copy = events.clone()

    assert events.trial_columns == events_copy.trial_columns


@pytest.mark.parametrize(
    ('events', 'kwargs', 'expected_df'),
    [
        pytest.param(
            Events(name='a', onsets=[0], offsets=[1]),
            {'column': 'trial', 'data': 1},
            Events(
                pl.DataFrame(
                    {'trial': [1], 'name': 'a', 'onset': [0], 'offset': [1]},
                ),
            ),
            id='single_row_trial_str',
        ),
        pytest.param(
            Events(name='a', onsets=[0], offsets=[1]),
            {'column': ['trial'], 'data': 1},
            Events(
                pl.DataFrame(
                    {'trial': [1], 'name': 'a', 'onset': [0], 'offset': [1]},
                ),
            ),
            id='single_row_trial_list_data_int',
        ),
        pytest.param(
            Events(name='a', onsets=[0], offsets=[1]),
            {'column': ['trial'], 'data': [1]},
            Events(
                pl.DataFrame(
                    {'trial': [1], 'name': 'a', 'onset': [0], 'offset': [1]},
                ),
            ),
            id='single_row_trial_list_single_identifier',
        ),
        pytest.param(
            Events(name='a', onsets=[0], offsets=[1]),
            {'column': ['group', 'trial'], 'data': ['A', 1]},
            Events(
                pl.DataFrame(
                    {'group': 'A', 'trial': [1], 'name': 'a', 'onset': [0], 'offset': [1]},
                ),
            ),
            id='single_row_trial_list_single_identifier',
        ),
        pytest.param(
            Events(name='a', onsets=[0, 8], offsets=[1, 9]),
            {'column': ['trial'], 'data': [1]},
            Events(
                pl.DataFrame(
                    {'trial': [1, 1], 'name': ['a', 'a'], 'onset': [0, 8], 'offset': [1, 9]},
                ),
            ),
            id='two_rows_trial_list_single_identifier',
        ),
    ],
)
def test_add_trial_column(events, kwargs, expected_df):
    events.add_trial_column(**kwargs)
    assert_frame_equal(events.frame, expected_df.frame)


@pytest.mark.parametrize(
    ('events', 'kwargs', 'exception', 'message'),
    [
        pytest.param(
            Events(name='a', onsets=[0], offsets=[1]),
            {'column': ['group', 'trial'], 'data': 1},
            TypeError,
            'data must be passed as a list of values in case of providing multiple columns',
            id='multiple_columns_data_not_list',
        ),
    ],
)
def test_add_trial_column_raises_exception(events, kwargs, exception, message):
    with pytest.raises(exception) as excinfo:
        events.add_trial_column(**kwargs)

    assert message == excinfo.value.args[0]


def test_split_by_str():
    events = Events(
        pl.DataFrame(
            {
                'trial_id': [0, 1, 1, 2],
                'name': ['fixation', 'fixation', 'fixation', 'fixation'],
                'onset': [0, 1, 2, 3],
                'offset': [1, 2, 44, 1340],
                'duration': [1, 1, 42, 1337],
            },
        ),
        trial_columns='trial_id',
    )

    split_event = events.split('trial_id')
    assert all(events.frame.n_unique('trial_id') == 1 for events in split_event)
    assert len(split_event) == 3
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 0), split_event[0].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 1), split_event[1].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 2), split_event[2].frame)


def test_split_by_list():
    events = Events(
        pl.DataFrame(
            {
                'trial_ida': [0, 1, 1, 2],
                'trial_idb': [0, 1, 2, 3],
                'name': ['fixation', 'fixation', 'fixation', 'fixation'],
                'onset': [0, 1, 2, 3],
                'offset': [1, 2, 44, 1340],
                'duration': [1, 1, 42, 1337],
            },
        ),
        trial_columns=['trial_ida', 'trial_idb'],
    )

    split_event = events.split(['trial_ida', 'trial_idb'])
    assert all(events.frame.n_unique(['trial_ida', 'trial_idb']) == 1 for events in split_event)
    assert len(split_event) == 4


def test_split_default():
    events = Events(
        pl.DataFrame(
            {
                'trial_id': [0, 1, 1, 2],
                'name': ['fixation', 'fixation', 'fixation', 'fixation'],
                'onset': [0, 1, 2, 3],
                'offset': [1, 2, 44, 1340],
                'duration': [1, 1, 42, 1337],
            },
        ),
        trial_columns='trial_id',
    )

    split_event = events.split()
    assert all(events.frame.n_unique('trial_id') == 1 for events in split_event)
    assert len(split_event) == 3
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 0), split_event[0].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 1), split_event[1].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 2), split_event[2].frame)


def test_split_default_no_trial_columns_raises_typeerror():
    events = Events(
        pl.DataFrame(
            {
                'trial_id': [0, 1, 1, 2],
                'name': ['fixation', 'fixation', 'fixation', 'fixation'],
                'onset': [0, 1, 2, 3],
                'offset': [1, 2, 44, 1340],
                'duration': [1, 1, 42, 1337],
            },
        ),
    )
    with pytest.raises(TypeError, match="Either 'by' or 'Events.trial_columns' must be specified"):
        events.split()


@pytest.mark.parametrize(
    ('events', 'by', 'expected_splits'),
    [
        pytest.param(
            Events(
                pl.from_dict({
                    'trial_id': [0, 1],
                    'name': ['fixation', 'saccade'],
                    'onset': [0, 10],
                    'offset': [1, 12],
                }),
            ),
            'trial_id',
            {
                (0,): Events(
                    pl.from_dict({
                        'trial_id': [0],
                        'name': ['fixation'],
                        'onset': [0],
                        'offset': [1],
                    }),
                ),
                (1,): Events(
                    pl.from_dict({
                        'trial_id': [1],
                        'name': ['saccade'],
                        'onset': [10],
                        'offset': [12],
                    }),
                ),
            },
            id='single_column',
        ),

        pytest.param(
            Events(onsets=[20, 30], offsets=[24, 40], name=['blink', 'fixation'], trials=[1, 2]),
            None,
            {
                (1,): Events(onsets=[20], offsets=[24], name=['blink'], trials=[1]),
                (2,): Events(onsets=[30], offsets=[40], name=['fixation'], trials=[2]),
            },
            id='single_column_default',
        ),

        pytest.param(
            Events(onsets=[20, 30], offsets=[24, 40], name=['blink', 'fixation'], trials=[1, None]),
            None,
            {
                (1,): Events(onsets=[20], offsets=[24], name=['blink'], trials=[1]),
                (None,): Events(onsets=[30], offsets=[40], name=['fixation'], trials=[None]),
            },
            id='single_column_two_trials_int_one_none',
        ),

        pytest.param(
            Events(
                onsets=[20, 30], offsets=[24, 40], name=['blink', 'fixation'], trials=['A', None],
            ),
            None,
            {
                ('A',): Events(onsets=[20], offsets=[24], name=['blink'], trials=['A']),
                (None,): Events(onsets=[30], offsets=[40], name=['fixation'], trials=[None]),
            },
            id='single_column_two_trials_str_one_none',
        ),

        pytest.param(
            Events(
                pl.from_dict({
                    'trial_id': [0, 1, 1, 2],
                    'task_id': ['A', 'B', 'C', 'D'],
                    'name': ['saccade', 'fixation', 'blink', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 44, 1340],
                }),
            ),
            'task_id',
            {
                ('A',): Events(
                    pl.from_dict({
                        'trial_id': [0],
                        'task_id': ['A'],
                        'name': ['saccade'],
                        'onset': [0],
                        'offset': [1],
                    }),
                ),
                ('B',): Events(
                    pl.from_dict({
                        'trial_id': [1],
                        'task_id': ['B'],
                        'name': ['fixation'],
                        'onset': [1],
                        'offset': [2],
                    }),
                ),
                ('C',): Events(
                    pl.from_dict({
                        'trial_id': [1],
                        'task_id': ['C'],
                        'name': ['blink'],
                        'onset': [2],
                        'offset': [44],
                    }),
                ),
                ('D',): Events(
                    pl.from_dict({
                        'trial_id': [2],
                        'task_id': ['D'],
                        'name': ['fixation'],
                        'onset': [3],
                        'offset': [1340],
                    }),
                ),
            },
            id='two_columns',
        ),

        pytest.param(
            Events(
                pl.from_dict({
                    'trial_id': [0, 1, 1, 2],
                    'task_id': ['A', 'B', 'C', 'D'],
                    'name': ['saccade', 'fixation', 'blink', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 44, 1340],
                }),
                trial_columns=['task_id', 'trial_id'],
            ),
            None,
            {
                ('A', 0): Events(
                    pl.from_dict({
                        'trial_id': [0],
                        'task_id': ['A'],
                        'name': ['saccade'],
                        'onset': [0],
                        'offset': [1],
                    }),
                    trial_columns=['task_id', 'trial_id'],
                ),
                ('B', 1): Events(
                    pl.from_dict({
                        'trial_id': [1],
                        'task_id': ['B'],
                        'name': ['fixation'],
                        'onset': [1],
                        'offset': [2],
                    }),
                    trial_columns=['task_id', 'trial_id'],
                ),
                ('C', 1): Events(
                    pl.from_dict({
                        'trial_id': [1],
                        'task_id': ['C'],
                        'name': ['blink'],
                        'onset': [2],
                        'offset': [44],
                    }),
                    trial_columns=['task_id', 'trial_id'],
                ),
                ('D', 2): Events(
                    pl.from_dict({
                        'trial_id': [2],
                        'task_id': ['D'],
                        'name': ['fixation'],
                        'onset': [3],
                        'offset': [1340],
                    }),
                    trial_columns=['task_id', 'trial_id'],
                ),
            },
            id='two_trial_columns_default',
        ),
    ],
)
def test_split_as_dict_returns_expected_dict(events, by, expected_splits):
    splits = events.split(by=by, as_dict=True)

    assert splits == expected_splits


def test_filter_by_name_literal_substring(make_events):
    events = make_events(['fixation.ivt', 'fixation', 'saccade.ivt', 'blink'])
    out = events.filter_by_name('fixation')
    assert set(out['name'].to_list()) == {'fixation.ivt', 'fixation'}


def test_fixations_filter(make_events):
    events = make_events(['fixation', 'fixation_ivt', 'saccade', 'blink'])
    out = events.fixations
    assert set(out['name'].to_list()) == {'fixation', 'fixation_ivt'}


def test_filter_by_name_prefix_regex(make_events):
    events = make_events(['fixation.ivt', 'fixation', 'saccade.ivt', 'blink'])
    out = events.filter_by_name(r'^fixation')
    assert set(out['name'].to_list()) == {'fixation.ivt', 'fixation'}


def test_filter_by_name_missing_column_raises_column_not_found_error(make_events):
    events = make_events(['microsaccade', 'microsaccade_x', 'saccade'])
    events.frame = events.frame.drop('name')
    expected_msg = "Events frame is missing the 'name' column."

    with pytest.raises(ValueError, match=expected_msg):
        events.filter_by_name('saccade')


def test_saccades_filter(make_events):
    events = make_events(['saccade', 'saccade_algo', 'fixation'])
    out = events.saccades
    assert set(out['name'].to_list()) == {'saccade', 'saccade_algo'}


def test_filter_by_name_exact_match_regex(make_events):
    events = make_events(['fixation.ivt', 'fixation', 'fixation_ivt', 'saccade'])
    out = events.filter_by_name(r'^fixation\.ivt$')
    assert out['name'].to_list() == ['fixation.ivt']


def test_filter_by_name_no_matches(make_events):
    events = make_events(['fixation', 'saccade'])
    out = events.filter_by_name(r'^blink$')
    assert out.height == 0


def test_blinks_filter(make_events):
    events = make_events(['blink', 'blink_fast', 'fixation'])
    out = events.blinks
    assert set(out['name'].to_list()) == {'blink', 'blink_fast'}


def test_microsaccades_filter(make_events):
    events = make_events(['microsaccade', 'microsaccade_x', 'saccade'])
    out = events.microsaccades
    assert set(out['name'].to_list()) == {'microsaccade', 'microsaccade_x'}


@pytest.mark.parametrize(
    ('init_names', 'init_properties', 'remove_properties', 'expected_columns'),
    [
        pytest.param(
            [], ['test1'],
            'test1',
            ['duration'],
            id='empty',
        ),
        pytest.param(
            ['fixation'], ['test1'],
            'test1',
            ['duration'],
            id='one',
        ),
        pytest.param(
            ['fixation'], ['test1', 'test2'],
            'test1',
            ['duration', 'test2'],
            id='one_out_of_two',
        ),
        pytest.param(
            ['fixation'], ['test1', 'test2'],
            ['test1', 'test2'],
            ['duration'],
            id='two_out_of_two',
        ),
    ],
)
def test_drop_event_properties_has_expected_columns(
        init_names, init_properties, remove_properties, expected_columns, make_events,
):
    events = make_events(names=init_names, properties=init_properties)
    events.drop(remove_properties)
    assert set(events.event_property_columns) == set(expected_columns)


@pytest.mark.parametrize(
    ('init_names', 'init_properties', 'remove_properties', 'exception', 'message'),
    [
        pytest.param(
            [], None,
            'foobar',
            ValueError, 'foobar.*does not exist',
            id='empty',
        ),
        pytest.param(
            [], None,
            'onset',
            ValueError, 'onset.*belongs to the minimal schema',
            id='onset',
        ),
    ],
)
def test_drop_event_properties_raises_exception(
        init_names, init_properties, remove_properties, exception, message, make_events,
):
    events = make_events(names=init_names, properties=init_properties)

    with pytest.raises(exception, match=message):
        events.drop(remove_properties)


@pytest.mark.parametrize(
    'locations, expected_x, expected_y',
    [
        pytest.param(
            [[1, 2], [3, 4]],
            [1, 3],
            [2, 4],
            id='two_rows_integers',
        ),
        pytest.param(
            [[None, None]],
            [None],
            [None],
            id='none_pairs_propagate',
        ),
    ],
)
def test_unnest_location_basic(
        locations: list[list[int | None]],
        expected_x: list[int | None],
        expected_y: list[int | None],
) -> None:
    """Events.unnest splits 'location' list into 'location_x'/'location_y' and drops input.

    This test covers typical integer values and None pairs - values are propagated as-is.
    """
    df = pl.DataFrame(
        {
            'name': ['fixation'] * len(locations),
            'onset': list(range(len(locations))),
            'offset': list(range(1, len(locations) + 1)),
            'location': locations,
        },
    )
    events = Events(data=df)

    events.unnest()

    assert 'location' not in events.frame.columns
    assert events.frame.get_column('location_x').to_list() == expected_x
    assert events.frame.get_column('location_y').to_list() == expected_y


def test_unnest_location_absent_is_noop() -> None:
    """If 'location' is absent, unnest should do nothing (no error, no new columns)."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
        },
    )
    events = Events(data=df)

    before_cols = set(events.frame.columns)
    events.unnest()
    after_cols = set(events.frame.columns)

    assert before_cols == after_cols


@pytest.mark.parametrize('max_gap', range(6))
@pytest.mark.parametrize(
    'events',
    [
        Events(
            pl.DataFrame(
                {
                    'name': [
                        'fixation',
                        'fixation',
                        'fixation',
                        'blink',
                        'saccade',
                        'blink',
                        'fixation',
                        'fixation',
                        'fixation',
                        'fixation',
                    ],
                    'onset': [0, 2, 5, 13, 21, 22, 30, 40, 53, 73],
                    'offset': [1, 3, 10, 20, 22, 29, 35, 49, 70, 90],
                    'other_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    'other_col_2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                },
            ),
        ),
    ],
)
def test_merge_subsequent_close_events_with_varying_max_gap(events, max_gap):
    events.merge_subsequent_close_events('fixation', max_gap=max_gap, verbose=True)
    assert (max_gap + len(events.frame)) == 10, \
        f'Expected {10 - max_gap} events after merging,' + \
        f' but got {len(events.frame)} for max_gap={max_gap}'


@pytest.mark.parametrize(
    ('max_gap', 'expected_n_events'),
    [
        # Two fixations are separated by a sub-millisecond gap of 10.6 ms. Merging happens only
        # when max_gap is at least the gap, so max_gap must be compared without truncating the gap
        # to whole milliseconds (which would collapse 10.6 to 10 and merge at max_gap=10).
        pytest.param(10, 2, id='integer_max_gap_below_subms_gap'),
        pytest.param(10.5, 2, id='float_max_gap_below_subms_gap'),
        pytest.param(11, 1, id='integer_max_gap_above_subms_gap'),
        pytest.param(10.6, 1, id='float_max_gap_equal_subms_gap'),
    ],
)
def test_merge_subsequent_close_events_subms_gap_and_float_max_gap(max_gap, expected_n_events):
    events = Events(
        pl.DataFrame(
            {
                'name': ['fixation', 'fixation'],
                'onset': [0.0, 20.6],
                'offset': [10.0, 30.0],
            },
        ),
    )
    events.merge_subsequent_close_events('fixation', max_gap=max_gap)
    assert len(events.frame) == expected_n_events


@pytest.mark.parametrize('verbose', [True, False])
@pytest.mark.parametrize(
    ('input_events', 'max_gap', 'expected_events'),
    [
        pytest.param(
            Events(),
            6,
            Events(),
            id='empty_events',
        ),
        pytest.param(
            Events(name='fixation', onsets=[0], offsets=[1]),
            6,
            Events(name='fixation', onsets=[0], offsets=[1]),
            id='single_event_left_unchanged',
        ),
        pytest.param(
            Events(name='fixation', onsets=[0, 3], offsets=[1, 5]),
            6,
            Events(name='fixation', onsets=[0], offsets=[5]),
            id='two_events_small_gap_merged',
        ),
        pytest.param(
            Events(name='fixation', onsets=[0, 103], offsets=[1, 105]),
            6,
            Events(name='fixation', onsets=[0, 103], offsets=[1, 105]),
            id='two_events_big_gap_not_merged',
        ),
        pytest.param(
            Events(name='fixation', onsets=[0, 4, 103], offsets=[1, 10, 105]),
            6,
            Events(name='fixation', onsets=[0, 103], offsets=[10, 105]),
            id='three_events_small_gap_two_merged',
        ),
        pytest.param(
            Events(name='fixation', onsets=[0, 4, 13], offsets=[1, 10, 15]),
            6,
            Events(name='fixation', onsets=[0], offsets=[15]),
            id='three_events_small_gap_three_merged',
        ),
        pytest.param(
            Events(name=['fixation', 'saccade', 'fixation'], onsets=[0, 2, 5], offsets=[1, 4, 12]),
            6,
            Events(name=['fixation', 'saccade'], onsets=[0, 2], offsets=[12, 4]),
            id='three_events_small_gap_two_merged_inbetween',
        ),
    ],
)
def test_merge_subsequent_close_events_result_dataframe(
        input_events, max_gap, verbose, expected_events,
):
    input_events.merge_subsequent_close_events('fixation', max_gap=max_gap, verbose=verbose)
    assert_frame_equal(input_events.frame, expected_events.frame)
