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
"""Test the private helper that only the statistical generator uses.

``check_contiguous`` has no public consumer until the statistical generator lands, so it is
tested directly here. The other helpers in ``_utils`` are tested through ``bridge_gaps``,
``Range``, ``generate()`` and the geometrical generator.
"""
import polars as pl
import pytest

from pymovements import Events
from pymovements.synthetic import bridge_gaps
from pymovements.synthetic._utils import check_contiguous


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
        pytest.param(
            bridge_gaps(Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560])),
            id='bridged_events',
        ),
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
            Events(
                pl.DataFrame({'name': ['fixation'] * 2, 'onset': [0, 300], 'duration': [250, 260]}),
            ),
            ('gap', '250.0 ms', '300.0 ms'),
            id='gap_is_read_from_the_duration',
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


def test_check_contiguous_names_the_generator():
    events = Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560])

    with pytest.raises(ValueError) as excinfo:
        check_contiguous(events, generator='statistical')

    msg, = excinfo.value.args
    assert msg.startswith("generator 'statistical': ")
