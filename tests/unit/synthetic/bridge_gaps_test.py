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
"""Test the bridging of gaps in event sequences."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.synthetic import bridge_gaps


@pytest.mark.parametrize(
    ('events', 'max_gap', 'expected'),
    [
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 250], durations=[250, 250]),
            None,
            Events(name=['fixation'] * 2, onsets=[0, 250], durations=[250, 250]),
            id='touching_events_are_unchanged',
        ),
        pytest.param(
            Events(name=['fixation'], onsets=[0], durations=[250]),
            None,
            Events(name=['fixation'], onsets=[0], durations=[250]),
            id='single_fixation_is_unchanged',
        ),
        pytest.param(
            Events(name=['fixation'], onsets=[0], durations=[250]),
            100,
            Events(name=['fixation'], onsets=[0], durations=[250]),
            id='single_fixation_with_max_gap_is_unchanged',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            None,
            Events(
                name=['fixation', 'saccade', 'fixation'],
                onsets=[0, 250, 300],
                durations=[250, 50, 260],
            ),
            id='gap_becomes_saccade',
        ),
        pytest.param(
            Events(name=['fixation'] * 3, onsets=[0, 300, 2400], durations=[250, 260, 300]),
            100,
            Events(
                name=['fixation', 'saccade', 'fixation', 'fixation'],
                onsets=[0, 250, 300, 2400],
                durations=[250, 50, 260, 300],
            ),
            id='gap_longer_than_max_gap_is_kept',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            50,
            Events(
                name=['fixation', 'saccade', 'fixation'],
                onsets=[0, 250, 300],
                durations=[250, 50, 260],
            ),
            id='gap_equal_to_max_gap_is_bridged',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            0,
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            id='zero_max_gap_bridges_nothing',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[300, 0], durations=[260, 250]),
            None,
            Events(
                name=['fixation', 'saccade', 'fixation'],
                onsets=[0, 250, 300],
                durations=[250, 50, 260],
            ),
            id='unsorted_events_are_sorted',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0.0, 300.5], durations=[250.25, 259.5]),
            None,
            Events(
                name=['fixation', 'saccade', 'fixation'],
                onsets=[0.0, 250.25, 300.5],
                durations=[250.25, 50.25, 259.5],
            ),
            id='sub_millisecond_times_are_kept',
        ),
        pytest.param(
            Events(
                name=['fixation'] * 4,
                onsets=[0, 300, 0, 500],
                durations=[250, 260, 200, 200],
                trials=[1, 1, 2, 2],
            ),
            None,
            Events(
                name=['fixation', 'saccade', 'fixation', 'fixation', 'saccade', 'fixation'],
                onsets=[0, 250, 300, 0, 200, 500],
                durations=[250, 50, 260, 200, 300, 200],
                trials=[1, 1, 1, 2, 2, 2],
            ),
            id='trials_are_bridged_separately',
        ),
        pytest.param(
            Events(
                name=['fixation'] * 3,
                onsets=[0, 0, 300],
                durations=[200, 250, 260],
                trials=[2, 1, 1],
            ),
            None,
            Events(
                name=['fixation', 'saccade', 'fixation', 'fixation'],
                onsets=[0, 250, 300, 0],
                durations=[250, 50, 260, 200],
                trials=[1, 1, 1, 2],
            ),
            id='trials_are_sorted',
        ),
        pytest.param(
            Events(),
            None,
            Events(),
            id='empty_events',
        ),
    ],
)
def test_bridge_gaps_returns(events, max_gap, expected):
    bridged = bridge_gaps(events, max_gap=max_gap)

    assert_frame_equal(bridged.frame, expected.frame)
    assert bridged.trial_columns == expected.trial_columns


def test_bridge_gaps_reads_the_end_of_an_event_from_its_duration():
    """PMEP 1: an event ends at onset + duration, so it needs no offset."""
    events = Events(
        pl.DataFrame({'name': ['fixation'] * 2, 'onset': [0, 300], 'duration': [250, 260]}),
    )

    bridged = bridge_gaps(events)

    assert bridged.frame['name'].to_list() == ['fixation', 'saccade', 'fixation']
    assert bridged.frame['onset'].dt.total_milliseconds().to_list() == [0, 250, 300]
    assert bridged.frame['duration'].dt.total_milliseconds().to_list() == [250, 50, 260]


def test_bridge_gaps_writes_no_offset_for_the_inserted_saccades():
    events = Events(
        name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560], offsets_inclusive=False,
    )

    bridged = bridge_gaps(events)

    assert bridged.frame['offset'].dt.total_milliseconds().to_list() == [250, None, 560]


def test_bridge_gaps_leaves_extra_columns_null_for_saccades():
    events = Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260])
    events.frame = events.frame.with_columns(pl.Series('location', [[0.0, 0.0], [10.0, 0.0]]))

    bridged = bridge_gaps(events)

    assert bridged.frame['location'].to_list() == [[0.0, 0.0], None, [10.0, 0.0]]


def test_bridge_gaps_does_not_modify_the_input():
    events = Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260])
    original = events.frame.clone()

    bridge_gaps(events)

    assert_frame_equal(events.frame, original)


def test_bridge_gaps_keeps_the_metadata():
    events = Events(
        name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260], metadata={'note': 'kept'},
    )

    bridged = bridge_gaps(events)

    assert bridged.metadata == {'note': 'kept'}
    assert bridged.metadata is not events.metadata


@pytest.mark.parametrize(
    'max_gap',
    [
        pytest.param(100, id='int'),
        pytest.param(100.0, id='float'),
        pytest.param(np.int64(100), id='numpy_int'),
    ],
)
def test_bridge_gaps_accepts_real_numbers(max_gap):
    events = Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260])

    assert bridge_gaps(events, max_gap=max_gap).frame.height == 3


@pytest.mark.parametrize(
    ('events', 'max_gap', 'exception', 'msg_substrings'),
    [
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 200], durations=[250, 360]),
            None,
            ValueError,
            ('overlap', 'fixation', '200.0 ms', '250.0 ms'),
            id='overlapping_events_raise_value_error',
        ),
        pytest.param(
            Events(
                pl.DataFrame({'name': ['fixation'] * 2, 'onset': [0, 200], 'duration': [250, 360]}),
            ),
            None,
            ValueError,
            ('overlap', 'fixation', '200.0 ms', '250.0 ms'),
            id='overlap_is_read_from_the_duration',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0.0, 200.25], durations=[250.5, 359.75]),
            None,
            ValueError,
            ('overlap', '200.25 ms', '250.5 ms'),
            id='overlap_message_keeps_sub_millisecond_times',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            -1,
            ValueError,
            ('max_gap', 'negative'),
            id='negative_max_gap_raises_value_error',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            '100',
            TypeError,
            ('max_gap', 'number', 'str'),
            id='string_max_gap_raises_type_error',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], durations=[250, 260]),
            True,
            TypeError,
            ('max_gap', 'number', 'bool'),
            id='bool_max_gap_raises_type_error',
        ),
    ],
)
def test_bridge_gaps_raises_error(events, max_gap, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        bridge_gaps(events, max_gap=max_gap)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_overlap_across_trials_is_allowed():
    events = Events(
        name=['fixation'] * 2, onsets=[0, 100], durations=[250, 460], trials=[1, 2],
    )

    assert bridge_gaps(events).frame.height == 2
