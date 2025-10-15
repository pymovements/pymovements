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
"""Tests pymovements.events.Events."""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events


@pytest.fixture(name='expected_schema_after_init')
def fixture_dataset():
    schema = {'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64, 'duration': pl.Int64}
    yield schema


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
def test_init_expected_schema(args, kwargs, expected_schema_after_init):
    events = Events(*args, **kwargs)
    assert events.schema == expected_schema_after_init


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
            {'name': [''], 'onset': [0], 'offset': [1], 'duration': [1]},
            id='no_arg_dict_with_single_event_kwarg',
        ),
        pytest.param(
            [pl.DataFrame()], {},
            {},
            id='dataframe_arg_no_kwargs',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]},
            {'name': ['bar'], 'onset': [0], 'offset': [1], 'duration': [1]},
            id='dict_with_single_named_event',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 2], 'offsets': [1, 3]},
            {'name': ['bar', 'bar'], 'onset': [0, 2], 'offset': [1, 3], 'duration': [1, 1]},
            id='dict_with_two_events_same_name',
        ),
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 2], 'offsets': [1, 4]},
            {'name': ['foo', 'bar'], 'onset': [0, 2], 'offset': [1, 4], 'duration': [1, 2]},
            id='dict_with_two_differently_named_events',
        ),
    ],
)
def test_init_expected(args, kwargs, expected_df_data, expected_schema_after_init):
    events = Events(*args, **kwargs)

    expected_df = pl.DataFrame(data=expected_df_data, schema=expected_schema_after_init)
    assert_frame_equal(events.frame, expected_df)


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_df'),
    [
        pytest.param(
            [], {'onsets': [0], 'offsets': [1]},
            pl.DataFrame({'name': [''], 'onset': [0], 'offset': [1], 'duration': [1]}),
            id='no_arg_lists_with_single_event_kwarg',
        ),
        pytest.param(
            [pl.DataFrame()], {},
            pl.DataFrame(
                {}, schema={
                    'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64, 'duration': pl.Int64,
                },
            ),
            id='dataframe_arg_no_kwargs',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]},
            pl.DataFrame({'name': ['bar'], 'onset': [0], 'offset': [1], 'duration': [1]}),
            id='lists_with_single_named_event',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 2], 'offsets': [1, 3]},
            pl.DataFrame(
                {'name': ['bar', 'bar'], 'onset': [0, 2], 'offset': [1, 3], 'duration': [1, 1]},
            ),
            id='lists_with_two_events_same_name',
        ),
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 2], 'offsets': [1, 4]},
            pl.DataFrame(
                {'name': ['foo', 'bar'], 'onset': [0, 2], 'offset': [1, 4], 'duration': [1, 2]},
            ),
            id='lists_with_two_differently_named_events',
        ),
        pytest.param(
            [], {'name': ['foo'], 'onsets': [0], 'offsets': [1], 'trials': [1]},
            pl.DataFrame(
                {'trial': [1], 'name': ['foo'], 'onset': [0], 'offset': [1], 'duration': [1]},
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
                {'trial': [1], 'name': ['foo'], 'onset': [0], 'offset': [1], 'duration': [1]},
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
                {'trial': [1], 'name': ['foo'], 'onset': [0], 'offset': [1], 'duration': [1]},
            ),
            id='data_one_event_trial_column_enforce_start',
        ),
    ],
)
def test_init_expected_df(args, kwargs, expected_df):
    events = Events(*args, **kwargs)

    assert_frame_equal(events.frame, expected_df)


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


def test_clone():
    events = Events(name='saccade', onsets=[0], offsets=[123])
    events_copy = events.clone()

    # We want to have separate dataframes but with the exact same data.
    assert events is not events_copy
    assert events.frame is not events_copy.frame
    assert_frame_equal(events.frame, events_copy.frame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_copy():
    events = Events(name='saccade', onsets=[0], offsets=[123])
    events_copy = events.copy()

    # We want to have separate dataframes but with the exact same data.
    assert events is not events_copy
    assert events.frame is not events_copy.frame
    assert_frame_equal(events.frame, events_copy.frame)


def test_copy_removed(assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        Events().copy()
    assert_deprecation_is_removed('Events.copy()', info.value.args[0], __version__)


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


def test_fixations_filter(make_events):
    events = make_events(['fixation', 'fixation_ivt', 'saccade', 'blink'])
    out = events.fixations
    assert set(out['name'].to_list()) == {'fixation', 'fixation_ivt'}


def test_saccades_filter(make_events):
    events = make_events(['saccade', 'saccade_algo', 'fixation'])
    out = events.saccades
    assert set(out['name'].to_list()) == {'saccade', 'saccade_algo'}


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
