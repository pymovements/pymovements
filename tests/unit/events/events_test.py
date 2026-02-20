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
import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events


@pytest.fixture(name='expected_schema_after_init')
def fixture_dataset():
    schema = {'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64, 'duration': pl.Int64}
    yield schema


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
def test_event_dataframe_init_exceptions(kwargs, exception, msg_substrings):
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
def test_event_dataframe_init_expected_schema(args, kwargs, expected_schema_after_init):
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
def test_event_dataframe_init_has_expected_length(args, kwargs, expected_length):
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
def test_event_dataframe_init_has_correct_name(args, kwargs, expected_name):
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
def test_event_dataframe_init_has_correct_names(args, kwargs, expected_names):
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
def test_event_dataframe_init_expected(args, kwargs, expected_df_data, expected_schema_after_init):
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
def test_event_dataframe_init_expected_df(args, kwargs, expected_df):
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
def test_event_dataframe_init_expected_trial_column_list(kwargs, expected_trial_column_list):
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
def test_event_dataframe_init_expected_trial_column_data(kwargs, expected_trial_column_data):
    events = Events(**kwargs)

    if isinstance(expected_trial_column_data, pl.Series):
        expected_trial_column_data = pl.DataFrame(expected_trial_column_data)
    assert_frame_equal(events.frame[events.trial_columns], expected_trial_column_data)


def test_event_dataframe_columns_same_as_frame():
    init_kwargs = {'onsets': [0], 'offsets': [1]}
    events = Events(**init_kwargs)

    assert events.columns == events.frame.columns


def test_event_dataframe_clone():
    events = Events(name='saccade', onsets=[0], offsets=[123])
    events_copy = events.clone()

    # We want to have separate dataframes but with the exact same data.
    assert events is not events_copy
    assert events.frame is not events_copy.frame
    assert_frame_equal(events.frame, events_copy.frame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_dataframe_copy():
    events = Events(name='saccade', onsets=[0], offsets=[123])
    events_copy = events.copy()

    # We want to have separate dataframes but with the exact same data.
    assert events is not events_copy
    assert events.frame is not events_copy.frame
    assert_frame_equal(events.frame, events_copy.frame)


def test_event_dataframe_copy_removed():
    with pytest.raises(DeprecationWarning) as info:
        Events().copy()

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'Events.copy() was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )


def test_event_dataframe_clones_trial_columns():
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
def test_event_dataframe_add_trial_column(events, kwargs, expected_df):
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
def test_event_dataframe_add_trial_column_raises_exception(events, kwargs, exception, message):
    with pytest.raises(exception) as excinfo:
        events.add_trial_column(**kwargs)

    assert message == excinfo.value.args[0]


def test_eventdataframe_split():
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

    split_event = events.split('trial_id')
    assert all(events.frame.n_unique('trial_id') == 1 for events in split_event)
    assert len(split_event) == 3
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 0), split_event[0].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 1), split_event[1].frame)
    assert_frame_equal(events.frame.filter(pl.col('trial_id') == 2), split_event[2].frame)


def test_eventdataframe_split_by_str():
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


def test_eventdataframe_split_by_list():
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


def test_event_dataframe_split_default():
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


def test_event_dataframe_split_default_no_trial_columns():
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
    with pytest.raises(TypeError):
        events.split()


def _edf(names):
    """Create an EventDataFrame for tests."""
    df = pl.DataFrame({
        'name': names,
        'onset': [0] * len(names),
        'offset': [1] * len(names),
    })
    return Events(data=df)

def test_filter_by_name_literal_substring():
    edf = _edf(["fixation.ivt", "fixation", "saccade.ivt", "blink"])
    out = edf.filter_by_name("fixation")
    assert set(out["name"].to_list()) == {"fixation.ivt", "fixation"}


def test_filter_by_name_prefix_regex():
    edf = _edf(["fixation.ivt", "fixation", "saccade.ivt", "blink"])
    out = edf.filter_by_name(r"^fixation")
    assert set(out["name"].to_list()) == {"fixation.ivt", "fixation"}


def test_filter_by_name_exact_match_regex():
    edf = _edf(["fixation.ivt", "fixation", "fixation_ivt", "saccade"])
    out = edf.filter_by_name(r"^fixation\.ivt$")
    assert out["name"].to_list() == ["fixation.ivt"]


def test_filter_by_name_no_matches():
    edf = _edf(["fixation", "saccade"])
    out = edf.filter_by_name(r"^blink$")
    assert out.height == 0