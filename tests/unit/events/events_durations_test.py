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
"""Tests the Events minimal schema, durations and offset conventions."""
from __future__ import annotations

import re
import warnings
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements import Events


def _ms(values):
    return pl.Series(
        [None if v is None else timedelta(milliseconds=v) for v in values],
        dtype=pl.Duration('us'),
    )


def test_init_durations_without_offsets_has_no_offset_column():
    events = Events(name='blink', onsets=[2], durations=[2])

    expected = pl.DataFrame({
        'onset': _ms([2]),
        'duration': _ms([2]),
        'name': ['blink'],
    })
    assert_frame_equal(events.frame, expected)


@pytest.mark.parametrize(
    'durations',
    [
        pytest.param([2, 3], id='numeric_ms'),
        pytest.param([timedelta(milliseconds=2), timedelta(milliseconds=3)], id='timedelta'),
    ],
)
def test_init_durations_accepted_like_onsets(durations):
    events = Events(onsets=[0, 10], durations=durations)

    assert events.frame['duration'].to_list() == [
        timedelta(milliseconds=2), timedelta(milliseconds=3),
    ]


def test_init_durations_respect_time_unit():
    events = Events(onsets=[0], durations=[2], time_unit='s')

    assert events.frame['duration'].to_list() == [timedelta(seconds=2)]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'onsets': [0, 10], 'offsets': [5, 20]}, id='keywords'),
        pytest.param({'onsets': [0, 10], 'offsets': [5, 20], 'durations': [6, 11]}, id='both'),
        pytest.param(
            {'data': pl.DataFrame({'onset': [0, 10], 'offset': [5, 20]})}, id='frame',
        ),
        pytest.param(
            {'data': pl.DataFrame({'onset': [0, 10], 'offset': [5, 20], 'duration': [6, 11]})},
            id='frame_both',
        ),
    ],
)
def test_init_supplied_offset_column_stays_stored_as_supplied(kwargs):
    events = Events(**kwargs)

    assert events.frame['offset'].to_list() == [
        timedelta(milliseconds=5), timedelta(milliseconds=20),
    ]


@pytest.mark.parametrize(
    ('kwargs', 'expected_name'),
    [
        pytest.param({'onsets': [0, 10]}, ['', ''], id='keywords'),
        pytest.param(
            {'data': pl.DataFrame({'onset': [0, 10]})},
            pl.Series([None, None], dtype=pl.Utf8),
            id='frame',
        ),
    ],
)
def test_init_onsets_only_has_null_durations(kwargs, expected_name):
    events = Events(**kwargs)

    expected = pl.DataFrame({
        'onset': _ms([0, 10]),
        'duration': _ms([None, None]),
        'name': expected_name,
    })
    assert_frame_equal(events.frame, expected)


@pytest.mark.parametrize('offsets_inclusive', [True, False])
@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'onsets': [0, 10]}, id='keywords'),
        pytest.param({'data': pl.DataFrame({'onset': [0, 10]})}, id='frame'),
    ],
)
def test_init_onsets_only_with_declared_convention_warns_and_has_null_durations(
        kwargs, offsets_inclusive,
):
    with pytest.warns(UserWarning, match="no 'offset' column") as record:
        events = Events(**kwargs, offsets_inclusive=offsets_inclusive, sampling_rate=1000)

    assert len(record) == 1
    assert events.frame['duration'].to_list() == [None, None]
    assert 'offset' not in events.columns


def test_init_empty_has_minimal_schema_only():
    assert Events().frame.schema == pl.Schema({
        'onset': pl.Duration('us'),
        'duration': pl.Duration('us'),
        'name': pl.Utf8,
    })


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'durations': [1]}, id='durations'),
        pytest.param({'offsets': [1]}, id='offsets'),
        pytest.param({'durations': [1], 'offsets': [1]}, id='both'),
    ],
)
def test_init_durations_or_offsets_without_onsets_raises(kwargs):
    with pytest.raises(ValueError, match='only be passed together with onsets'):
        Events(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        pytest.param(
            {'onsets': [0, 1], 'durations': [1]},
            'The sequences "onsets" and "durations" must be of equal length',
            id='durations',
        ),
        pytest.param(
            {'onsets': [0, 1], 'offsets': [1]},
            'The sequences "onsets" and "offsets" must be of equal length',
            id='offsets',
        ),
        pytest.param(
            {'onsets': [0, 1], 'durations': [1, 1], 'offsets': [1]},
            'The sequences "onsets" and "offsets" must be of equal length',
            id='offsets_both',
        ),
    ],
)
def test_init_length_not_matching_onsets_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Events(**kwargs)


@pytest.mark.parametrize('column', ['durations', 'offsets'])
def test_init_data_and_keyword_mutually_exclusive(column):
    with pytest.raises(ValueError, match=f'"data" and "{column}" are mutually exclusive'):
        Events(pl.DataFrame({'onset': [0]}), **{column: [1]})


def test_init_column_order_trial_columns_then_minimal_schema_then_extras():
    data = pl.DataFrame({
        'offset': [5], 'extra': ['a'], 'name': ['fixation'], 'trial': [1], 'onset': [0],
    })

    events = Events(data, trial_columns='trial')

    assert events.columns == ['trial', 'onset', 'duration', 'name', 'offset', 'extra']


def test_init_keyword_column_order():
    events = Events(name='blink', onsets=[2], offsets=[3], trials=[1])

    assert events.columns == ['trial', 'onset', 'duration', 'name', 'offset']


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'onsets': [0, 10], 'offsets': [5, 20]}, id='keywords'),
        pytest.param({'data': pl.DataFrame({'onset': [0, 10], 'offset': [5, 20]})}, id='frame'),
    ],
)
def test_init_undeclared_offsets_without_durations_keep_todays_values(kwargs):
    events = Events(**kwargs)

    assert events.frame['duration'].to_list() == [
        timedelta(milliseconds=5), timedelta(milliseconds=10),
    ]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'onsets': [0], 'offsets': [5], 'durations': [99]}, id='keywords'),
        pytest.param(
            {'data': pl.DataFrame({'onset': [0], 'offset': [5], 'duration': [99]})}, id='frame',
        ),
    ],
)
def test_init_undeclared_offsets_keep_supplied_durations_unchecked(kwargs):
    events = Events(**kwargs)

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=99)]


@pytest.mark.parametrize(
    ('offsets_inclusive', 'sampling_rate', 'expected_ms'),
    [
        pytest.param(True, 1000, [6, 11], id='inclusive_1000hz'),
        pytest.param(True, 500, [7, 12], id='inclusive_500hz'),
        pytest.param(False, None, [5, 10], id='exclusive_no_rate'),
        pytest.param(False, 1000, [5, 10], id='exclusive_rate_unused'),
    ],
)
@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'onsets': [0, 10], 'offsets': [5, 20]}, id='keywords'),
        pytest.param({'data': pl.DataFrame({'onset': [0, 10], 'offset': [5, 20]})}, id='frame'),
    ],
)
def test_init_declared_offsets_without_durations_derive(
        offsets_inclusive, sampling_rate, expected_ms, kwargs,
):
    events = Events(**kwargs, offsets_inclusive=offsets_inclusive, sampling_rate=sampling_rate)

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=v) for v in expected_ms]


def test_init_inclusive_derivation_rounds_sampling_interval_to_microseconds():
    events = Events(onsets=[0], offsets=[0], offsets_inclusive=True, sampling_rate=3)

    assert events.frame['duration'].to_list() == [timedelta(microseconds=333333)]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='no_durations'),
        pytest.param({'durations': [1], 'durations_from_offsets': True}, id='durations_replaced'),
    ],
)
def test_init_inclusive_derivation_without_sampling_rate_raises(kwargs):
    with pytest.raises(
        ValueError, match='deriving durations from inclusive offsets requires a sampling_rate',
    ):
        Events(onsets=[0], offsets=[5], offsets_inclusive=True, **kwargs)


@pytest.mark.parametrize(
    ('offsets_inclusive', 'sampling_rate', 'expected_ms'),
    [
        pytest.param(True, 1000, [6, 11], id='inclusive'),
        pytest.param(False, None, [5, 10], id='exclusive'),
    ],
)
@pytest.mark.parametrize('validate', [True, False])
def test_init_durations_from_offsets_replaces_durations(
        offsets_inclusive, sampling_rate, expected_ms, validate,
):
    events = Events(
        onsets=[0, 10], offsets=[5, 20], durations=[99, 99],
        offsets_inclusive=offsets_inclusive, sampling_rate=sampling_rate,
        durations_from_offsets=True, validate=validate,
    )

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=v) for v in expected_ms]


@pytest.mark.parametrize('offsets_inclusive', [True, False])
def test_init_declared_validate_false_keeps_durations_unchecked(offsets_inclusive):
    events = Events(
        onsets=[0, 10], offsets=[5, 20], durations=[99, 99],
        offsets_inclusive=offsets_inclusive, sampling_rate=1000, validate=False,
    )

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=99)] * 2


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'offsets_inclusive': True}, id='inclusive'),
        pytest.param({'offsets_inclusive': False}, id='exclusive'),
        pytest.param({'durations_from_offsets': True}, id='durations_from_offsets'),
        pytest.param(
            {'offsets_inclusive': True, 'durations_from_offsets': True}, id='both',
        ),
    ],
)
@pytest.mark.parametrize('validate', [True, False])
@pytest.mark.parametrize(
    'input_kwargs',
    [
        pytest.param({'onsets': [0], 'durations': [5]}, id='keywords'),
        pytest.param({'data': pl.DataFrame({'onset': [0], 'duration': [5]})}, id='frame'),
    ],
)
def test_init_offset_keywords_without_offset_column_warn_once_and_are_ignored(
        kwargs, validate, input_kwargs,
):
    with pytest.warns(UserWarning, match="no 'offset' column") as record:
        events = Events(**input_kwargs, **kwargs, sampling_rate=1000, validate=validate)

    assert len(record) == 1
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=5)]
    assert 'offset' not in events.columns


@pytest.mark.parametrize('validate', [True, False])
@pytest.mark.parametrize(
    ('durations', 'expected_ms'),
    [
        pytest.param(None, 5, id='no_durations'),
        pytest.param([99], 99, id='durations'),
    ],
)
def test_init_durations_from_offsets_on_undeclared_offsets_warns_once_and_is_ignored(
        validate, durations, expected_ms,
):
    with pytest.warns(UserWarning, match="'offset' column is undeclared") as record:
        events = Events(
            onsets=[0], offsets=[5], durations=durations,
            durations_from_offsets=True, validate=validate,
        )

    assert len(record) == 1
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=expected_ms)]


@pytest.mark.parametrize('sampling_rate', [0, -1])
def test_init_non_positive_sampling_rate_raises(sampling_rate):
    with pytest.raises(ValueError, match='sampling_rate must be positive'):
        Events(onsets=[0], offsets=[1], offsets_inclusive=True, sampling_rate=sampling_rate)


def _declared(durations, offsets, **kwargs):
    return Events(onsets=[0] * len(offsets), offsets=offsets, durations=durations, **kwargs)


@pytest.mark.parametrize(
    ('durations', 'offsets', 'kwargs'),
    [
        pytest.param([5, 0], [5, 0], {'offsets_inclusive': False}, id='exclusive'),
        pytest.param(
            [6, 1], [5, 0], {'offsets_inclusive': True, 'sampling_rate': 1000},
            id='inclusive_rate',
        ),
        pytest.param(
            [7, 2], [5, 0], {'offsets_inclusive': True}, id='inclusive_no_rate_common_residual',
        ),
    ],
)
def test_check_no_row_contradicting_passes(durations, offsets, kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*contradict')
        events = _declared(durations, offsets, **kwargs)

    assert events.frame['duration'].to_list() == [
        None if v is None else timedelta(milliseconds=v) for v in durations
    ]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'sampling_rate': 600}, id='rate'),
        pytest.param({}, id='no_rate'),
    ],
)
def test_check_nominal_timestamps_at_600_hz_pass(kwargs):
    # each value is rounded to microseconds on its own: correct rows have residual 1666 or 1667 µs
    interval_ms = 1000 / 600
    first_samples = [0, 1, 3, 7, 10]
    last_samples = [2, 5, 4, 11, 13]
    onsets = [i * interval_ms for i in first_samples]
    offsets = [j * interval_ms for j in last_samples]
    durations = [(j - i + 1) * interval_ms for i, j in zip(first_samples, last_samples)]

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*contradict')
        events = Events(
            onsets=onsets, offsets=offsets, durations=durations, offsets_inclusive=True, **kwargs,
        )

    residuals = events.frame.select(
        (pl.col('duration') - (pl.col('offset') - pl.col('onset'))).dt.total_microseconds(),
    ).to_series()
    assert set(residuals.to_list()) == {1666, 1667}


@pytest.mark.parametrize(
    ('durations', 'offsets', 'kwargs', 'expected_matches', 'unexpected'),
    [
        pytest.param(
            [5, 0], [5, 0], {'offsets_inclusive': True, 'sampling_rate': 1000},
            [
                r'every row contradicts offsets_inclusive=True',
                r'is 0 on every row, expected 1ms \(sampling_rate=1000\)',
                r'pre-v0\.29\.0',
                r'durations_from_offsets=True',
                r'offsets_inclusive=False',
                r'validate=False',
            ],
            [],
            id='inclusive_legacy',
        ),
        pytest.param(
            [9, 4], [5, 0], {'offsets_inclusive': True, 'sampling_rate': 1000},
            [r'is 4ms on every row, expected 1ms', r'durations_from_offsets=True'],
            ['pre-v0.29.0'],
            id='inclusive_other_residual',
        ),
        pytest.param(
            [5, 0], [5, 0], {'offsets_inclusive': True},
            [r'is 0 on every row', r'no sampling rate', r'pre-v0\.29\.0'],
            [],
            id='inclusive_no_rate',
        ),
        pytest.param(
            [6, 1], [5, 0], {'offsets_inclusive': False},
            [
                r'every row contradicts offsets_inclusive=False',
                r'is 1ms on every row, expected 0',
                r'offsets_inclusive=True',
                r'validate=False',
            ],
            ['durations_from_offsets', 'pre-v0.29.0'],
            id='exclusive',
        ),
        pytest.param(
            [6, 2], [5, 0], {'offsets_inclusive': False},
            [r'takes the values \[1ms, 2ms\]'],
            [],
            id='exclusive_distinct_residuals',
        ),
    ],
)
def test_check_every_row_contradicting_raises_with_recipes(
        durations, offsets, kwargs, expected_matches, unexpected,
):
    with pytest.raises(ValueError) as excinfo:
        _declared(durations, offsets, **kwargs)

    message = str(excinfo.value)
    for pattern in expected_matches:
        assert re.search(pattern, message), (pattern, message)
    for substring in unexpected:
        assert substring not in message


@pytest.mark.parametrize(
    ('durations', 'offsets', 'kwargs', 'expected_matches', 'unexpected'),
    [
        pytest.param(
            [6, 0, 2], [5, 0, 1], {'offsets_inclusive': True, 'sampling_rate': 1000},
            [
                r'1 of 3 rows contradict offsets_inclusive=True',
                r'to be 1ms \(sampling_rate=1000 from the sampling_rate keyword\)',
                r'residuals seen: \[0, 1ms\]',
                r'1 row looks like pre-v0\.29\.0 durations',
                r'durations_from_offsets=True',
            ],
            [],
            id='inclusive_legacy_row',
        ),
        pytest.param(
            [6, 3, 2], [5, 0, 1], {'offsets_inclusive': True, 'sampling_rate': 1000},
            [r'1 of 3 rows contradict', r'residuals seen: \[1ms, 3ms\]'],
            ['pre-v0.29.0'],
            id='inclusive_other_residual',
        ),
        pytest.param(
            [7, 3, 4], [5, 1, 1], {'offsets_inclusive': True},
            [
                r'1 of 3 rows contradict',
                r'no sampling rate',
                r'common positive residual 2ms',
                r'residuals seen: \[2ms, 3ms\]',
            ],
            ['pre-v0.29.0'],
            id='inclusive_no_rate_unequal_positive',
        ),
        pytest.param(
            [7, 1, 3], [5, 1, 1], {'offsets_inclusive': True},
            [r'1 of 3 rows contradict', r'common positive residual 2ms', r'pre-v0\.29\.0'],
            [],
            id='inclusive_no_rate_legacy_row',
        ),
        pytest.param(
            [5, 1], [5, 0], {'offsets_inclusive': False},
            [r'1 of 2 rows contradict offsets_inclusive=False', r'to be 0,'],
            ['pre-v0.29.0', 'durations_from_offsets'],
            id='exclusive',
        ),
    ],
)
def test_check_some_rows_contradicting_warn_once(
        durations, offsets, kwargs, expected_matches, unexpected,
):
    with pytest.warns(UserWarning, match='contradict') as record:
        events = _declared(durations, offsets, **kwargs)

    messages = [str(w.message) for w in record if 'contradict' in str(w.message)]
    assert len(messages) == 1
    for pattern in expected_matches:
        assert re.search(pattern, messages[0]), (pattern, messages[0])
    for substring in unexpected:
        assert substring not in messages[0]
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=v) for v in durations]


def test_check_null_rows_neither_contradict_nor_pass():
    # The only row with both values non-null contradicts, so all checked rows contradict.
    with pytest.raises(ValueError, match='every row contradicts'):
        with pytest.warns(UserWarning, match='null duration'):
            Events(
                onsets=[0, 0, 0], offsets=[5, None, 5], durations=[5, 5, None],
                offsets_inclusive=True, sampling_rate=1000,
            )


@pytest.mark.parametrize('validate', [True, False])
def test_check_null_duration_beside_offset_warns_once_and_is_not_filled(validate):
    with pytest.warns(UserWarning, match='null duration') as record:
        events = Events(
            onsets=[0, 0], offsets=[5, 5], durations=[6, None],
            offsets_inclusive=True, sampling_rate=1000, validate=validate,
        )

    assert len([w for w in record if 'null duration' in str(w.message)]) == 1
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=6), None]


def test_check_all_null_durations_beside_declared_offsets_pass_with_null_warning():
    data = pl.DataFrame({
        'onset': [0, 0], 'offset': [5, 5], 'duration': pl.Series([None, None], dtype=pl.Int64),
    })

    with pytest.warns(UserWarning, match='2 rows have a null duration') as record:
        events = Events(data, offsets_inclusive=False)

    assert not [w for w in record if 'contradict' in str(w.message)]
    assert events.frame['duration'].to_list() == [None, None]


def test_check_skipped_with_validate_false():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        events = Events(
            onsets=[0], offsets=[5], durations=[5],
            offsets_inclusive=True, sampling_rate=1000, validate=False,
        )

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=5)]


def test_drop_removes_offset_column():
    events = Events(onsets=[0], offsets=[5])

    events.drop('offset')

    assert events.columns == ['onset', 'duration', 'name']


@pytest.mark.parametrize('column', ['onset', 'duration', 'name'])
def test_drop_minimal_schema_column_raises(column):
    events = Events(onsets=[0], durations=[5])

    with pytest.raises(ValueError, match=r'minimal schema \(onset, duration, name\)'):
        events.drop(column)


@pytest.mark.parametrize(
    'events',
    [
        pytest.param(Events(onsets=[0], durations=[5]), id='keywords'),
        pytest.param(Events(pl.DataFrame({'onset': [0], 'extra': [1]})), id='frame'),
    ],
)
def test_event_property_columns_exclude_columns_present_at_init(events):
    events.frame = events.frame.with_columns(pl.lit(1.0).alias('amplitude'))

    assert events.event_property_columns == ['amplitude']


def test_merge_subsequent_close_events_keeps_durations():
    events = Events(
        name='fixation', onsets=[0, 20, 100], offsets=[10, 30, 110], durations=[12, 40, 5],
    )

    events.merge_subsequent_close_events(max_gap=15)

    assert events.frame['duration'].to_list() == [
        timedelta(milliseconds=60), timedelta(milliseconds=5),
    ]


def test_merge_subsequent_close_events_duration_ends_with_last_event():
    # the second event lies inside the first one: the merged event still ends with the last event
    events = Events(name='fixation', onsets=[0, 20], durations=[50, 10])

    events.merge_subsequent_close_events(max_gap=15)

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=30)]


@pytest.mark.parametrize(
    ('events', 'expected_ms'),
    [
        pytest.param(Events(), [], id='empty'),
        pytest.param(
            Events(name='fixation', onsets=[0, 20, 100], durations=[12, 40, 5]),
            [60, 5],
            id='rows',
        ),
    ],
)
def test_merge_subsequent_close_events_without_offset_gaps_from_durations(
        events, expected_ms,
):
    events.merge_subsequent_close_events(max_gap=15)

    assert 'offset' not in events.columns
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=v) for v in expected_ms]


def _rebuild_entry_events():
    return Events(
        pl.DataFrame({
            'trial': [1, 1, 2],
            'name': ['fixation', 'fixation', 'fixation'],
            'onset': [0, 200, 400],
            'offset': [100, 300, 500],
            'location': [[100.0, 105.0], [110.0, 195.0], [120.0, 302.0]],
        }),
        trial_columns='trial',
        metadata={'key': 'value'},
    )


def _rebuild_stimulus():
    return pm.stimulus.TextStimulus(
        aois=pl.DataFrame({
            'word': ['first', 'second', 'third'],
            'start_x': [90.0, 90.0, 90.0],
            'start_y': [80.0, 180.0, 280.0],
            'end_x': [200.0, 200.0, 200.0],
            'end_y': [120.0, 220.0, 320.0],
        }),
        aoi_column='word',
        start_x_column='start_x',
        start_y_column='start_y',
        end_x_column='end_x',
        end_y_column='end_y',
    )


def _rebuild_gaze(events):
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': list(range(0, 600)) * 2,
            'trial': [1] * 600 + [2] * 600,
            'x': [0.0] * 1200,
            'y': [0.0] * 1200,
        }),
        time_column='time',
        pixel_columns=['x', 'y'],
        trial_columns='trial',
    )
    if events is not None:
        gaze.events = events
    else:
        gaze.events.metadata = {'key': 'value'}
    return gaze


_REBUILD_ENTRY_COLUMNS = ['trial', 'onset', 'duration', 'name', 'offset', 'location']


@pytest.mark.parametrize(
    ('prepare', 'rebuild', 'expected_columns'),
    [
        pytest.param(
            _rebuild_entry_events, lambda events: events.clone(), _REBUILD_ENTRY_COLUMNS,
            id='clone',
        ),
        pytest.param(
            _rebuild_entry_events, lambda events: events.split(), _REBUILD_ENTRY_COLUMNS,
            id='split',
        ),
        pytest.param(
            _rebuild_entry_events, lambda events: events.split(as_dict=True),
            _REBUILD_ENTRY_COLUMNS, id='split_dict',
        ),
        pytest.param(
            _rebuild_entry_events,
            lambda events: events.correct_fixations(
                _rebuild_stimulus(), algorithm='attach', inplace=False,
            ),
            [*_REBUILD_ENTRY_COLUMNS, 'location_original', 'correction_algorithm'],
            id='correct_fixations',
        ),
        pytest.param(
            lambda: _rebuild_gaze(None),
            lambda gaze: gaze.detect('fill'),
            ['trial', 'onset', 'duration', 'name'],
            id='gaze_detect_shared_empty',
        ),
        pytest.param(
            lambda: _rebuild_gaze(_rebuild_entry_events()),
            lambda gaze: gaze.detect('fill'),
            _REBUILD_ENTRY_COLUMNS,
            id='gaze_detect_per_group',
        ),
    ],
)
def test_rebuild_sites_pass_validate_false_and_carry_metadata(
        monkeypatch, prepare, rebuild, expected_columns,
):
    entry = prepare()

    calls = []
    original_init = Events.__init__

    def spy_init(self, data=None, **kwargs):
        original_init(self, data, **kwargs)
        calls.append((self, data, kwargs))

    monkeypatch.setattr(Events, '__init__', spy_init)
    with warnings.catch_warnings():
        # The detector itself may report that it detected nothing.
        warnings.filterwarnings('ignore', message='.*No events were detected')
        rebuild(entry)

    rebuilds = [call for call in calls if call[1] is not None]
    assert rebuilds
    for rebuilt, data, kwargs in rebuilds:
        assert kwargs['validate'] is False
        assert set(kwargs) <= {'trial_columns', 'metadata', 'validate'}
        assert_frame_equal(rebuilt.frame, data)
        assert rebuilt.columns == expected_columns
        assert rebuilt.metadata.items() >= {'key': 'value'}.items()


def test_compute_event_properties_on_empty_events_without_offset_warns():
    gaze = _rebuild_gaze(None)
    assert 'offset' not in gaze.events.columns

    with pytest.warns(UserWarning, match='No events available'):
        gaze.compute_event_properties(('location', {'position_column': 'pixel'}))

    assert gaze.events.frame.is_empty()
