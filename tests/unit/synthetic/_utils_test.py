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
"""Test the sequence rules shared by the sample generators."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.synthetic._utils import check_contiguous
from pymovements.synthetic._utils import milliseconds
from pymovements.synthetic._utils import ordered_frame
from pymovements.synthetic._utils import per_trial
from pymovements.synthetic._utils import sample_times


def test_milliseconds_keeps_the_fraction():
    events = Events(name=['fixation'], onsets=[0.25], offsets=[1.5])

    assert milliseconds(events.frame['onset']).to_list() == [0.25]


def test_ordered_frame_sorts_by_trial_and_onset():
    events = Events(
        name=['fixation'] * 3, onsets=[300, 0, 0], offsets=[560, 250, 200], trials=[1, 2, 1],
    )

    frame = ordered_frame(events)

    assert frame['trial'].to_list() == [1, 1, 2]
    assert milliseconds(frame['onset']).to_list() == [0.0, 300.0, 0.0]


@pytest.mark.parametrize(
    'events',
    [
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 250], offsets=[250, 500]),
            id='touching_events',
        ),
        pytest.param(
            Events(name=['fixation', 'saccade'], onsets=[0.0, 250.5], offsets=[250.5, 300.0]),
            id='touching_sub_millisecond_events',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 0], offsets=[250, 200], trials=[1, 2]),
            id='trials_start_independently',
        ),
        pytest.param(Events(name=[], onsets=[], offsets=[]), id='empty_events'),
    ],
)
def test_check_contiguous_accepts(events):
    check_contiguous(events)


@pytest.mark.parametrize(
    ('events', 'msg_substrings'),
    [
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560]),
            ('gap', '250.0 ms', '300.0 ms', 'fixation', 'bridge_gaps'),
            id='gap',
        ),
        pytest.param(
            Events(name=['fixation', 'saccade'], onsets=[0.0, 250.75], offsets=[250.5, 300.0]),
            ('250.5 ms', '250.75 ms', 'saccade'),
            id='sub_millisecond_gap',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 200], offsets=[250, 560]),
            ('overlap',),
            id='overlap',
        ),
    ],
)
def test_check_contiguous_raises_value_error(events, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        check_contiguous(events)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def _event_count(frame):
    return pl.DataFrame({'n_events': [frame.height], 'first': [frame['name'][0]]})


def test_per_trial_without_trials_synthesizes_once():
    events = Events(name=['saccade', 'fixation'], onsets=[300, 0], offsets=[400, 250])

    samples = per_trial(events, _event_count)

    assert_frame_equal(samples, pl.DataFrame({'n_events': [2], 'first': ['fixation']}))


def test_per_trial_prefixes_the_trial_columns():
    events = Events(
        name=['fixation', 'saccade', 'fixation'],
        onsets=[0, 0, 250],
        offsets=[200, 250, 500],
        trials=[2, 1, 1],
    )

    samples = per_trial(events, _event_count)

    assert_frame_equal(
        samples,
        pl.DataFrame({'trial': [1, 2], 'n_events': [2, 1], 'first': ['saccade', 'fixation']}),
    )


def test_per_trial_passes_each_trial_ordered():
    events = Events(
        name=['fixation'] * 3, onsets=[300, 0, 0], offsets=[560, 250, 100], trials=[1, 1, 2],
    )
    seen = []

    def synthesize(frame):
        seen.append(milliseconds(frame['onset']).to_list())
        return pl.DataFrame({'time': [0.0]})

    per_trial(events, synthesize)

    assert seen == [[0.0, 300.0], [0.0]]


@pytest.mark.parametrize(
    ('events', 'msg_substring'),
    [
        pytest.param(Events(name=[], onsets=[], offsets=[]), 'empty', id='empty_events'),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 200], offsets=[250, 560]),
            'overlap',
            id='overlapping_events',
        ),
    ],
)
def test_per_trial_raises_value_error(events, msg_substring):
    with pytest.raises(ValueError) as excinfo:
        per_trial(events, _event_count)

    msg, = excinfo.value.args
    assert msg_substring in msg


@pytest.mark.parametrize(
    ('onset', 'offset', 'sampling_rate', 'expected'),
    [
        pytest.param(0.0, 5.0, 1000.0, [0.0, 1.0, 2.0, 3.0, 4.0], id='1000_hz'),
        pytest.param(100.0, 120.0, 250.0, [100.0, 104.0, 108.0, 112.0, 116.0], id='onset_kept'),
        pytest.param(0.0, 0.25, 1000.0, [0.0], id='shorter_than_one_sample'),
        pytest.param(0.0, 0.0, 1000.0, [0.0], id='zero_duration'),
    ],
)
def test_sample_times_returns(onset, offset, sampling_rate, expected):
    np.testing.assert_allclose(sample_times(onset, offset, sampling_rate), expected)
