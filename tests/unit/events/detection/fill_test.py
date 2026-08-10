# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Tests functionality of the IVT algorithm."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import fill


@pytest.mark.parametrize(
    ('kwargs', 'expected_error', 'expected_message'),
    [
        pytest.param(
            {
                'events': Events(),
                'timesteps': pl.repeat('b', 10, eager=True),
            },
            TypeError,
            r'timesteps dtype must be float or int but is String',
            id='timesteps_str_raises_type_error',
        ),
    ],
)
def test_fill_raise_error(kwargs, expected_error, expected_message):
    """Test if fill raises expected error."""
    with pytest.raises(expected_error, match=expected_message):
        fill(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0], offsets=[100]),
                'timesteps': pl.arange(0, 100, eager=True),
            },
            Events(),
            id='fixation_from_start_to_end_no_fill',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0], offsets=[100]),
                'timesteps': np.arange(0, 100),
            },
            Events(),
            id='fixation_from_start_to_end_no_fill_numpy',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[10], offsets=[100]),
                'timesteps': pl.arange(0, 100, eager=True),
            },
            Events(
                name='unclassified',
                onsets=[0],
                offsets=[9],
            ),
            id='fixation_10_ms_after_start_to_end_single_fill',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[10], offsets=[100]),
                'timesteps': np.arange(0, 100),
            },
            Events(
                name='unclassified',
                onsets=[0],
                offsets=[9],
            ),
            id='fixation_10_ms_after_start_to_end_single_fill_numpy',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0], offsets=[90]),
                'timesteps': pl.arange(0, 100, eager=True),
            },
            Events(
                name='unclassified',
                onsets=[90],
                offsets=[99],
            ),
            id='fixation_from_start_to_10_ms_before_end_single_fill',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0], offsets=[90]),
                'timesteps': np.arange(0, 100),
            },
            Events(
                name='unclassified',
                onsets=[90],
                offsets=[99],
            ),
            id='fixation_from_start_to_10_ms_before_end_single_fill_numpy',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0, 50], offsets=[40, 100]),
                'timesteps': pl.arange(0, 100, eager=True),
            },
            Events(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_at_40ms_single_fill',
        ),
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0, 50], offsets=[40, 100]),
                'timesteps': np.arange(0, 100),
            },
            Events(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_at_40ms_single_fill_numpy',
        ),
        pytest.param(
            {
                'events': Events(
                    name=['fixation', 'saccade'], onsets=[0, 50], offsets=[40, 100],
                ),
                'timesteps': pl.arange(0, 100, eager=True),
            },
            Events(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_then_saccade_until_end_single_fill',
        ),
        pytest.param(
            {
                'events': Events(
                    name=['fixation', 'saccade'], onsets=[0, 50], offsets=[40, 100],
                ),
                'timesteps': np.arange(0, 100),
            },
            Events(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_then_saccade_until_end_single_fill_numpy',
        ),
    ],
)
def test_fill_fills_events(kwargs, expected):
    events = fill(**kwargs)

    assert_frame_equal(events.frame, expected.frame)


def test_fill_with_numeric_event_frame():
    # An events frame mutated to numeric onset/offset columns (milliseconds) must be
    # matched against the timesteps without any Duration conversion.
    events = Events(name='fixation', onsets=[0], offsets=[49])
    events.frame = events.frame.with_columns(
        pl.col('onset', 'offset', 'duration').dt.total_milliseconds(),
    )

    filled = fill(events, timesteps=np.arange(0, 100))

    # The event covers timesteps 0..48 (fill treats the offset as exclusive here),
    # so the remaining timesteps 49..99 become one unclassified event.
    expected = Events(name='unclassified', onsets=[49], offsets=[99])
    assert_frame_equal(filled.frame, expected.frame)


def test_fill_matches_float_timesteps_not_representable_in_microseconds():
    # Event boundaries taken from float timesteps that do not land on whole microseconds
    # must still be matched by interval membership instead of exact equality.
    timesteps = np.arange(30) / 3.0
    events = Events(name='fixation', onsets=[timesteps[3]], offsets=[timesteps[10]])

    filled = fill(events, timesteps=timesteps, minimum_duration=0)

    # The event covers timesteps[3]..timesteps[9]; the samples before and from
    # timesteps[10] onward stay unclassified in two separate segments.
    onsets = filled.frame['onset'].dt.total_microseconds().to_list()
    offsets = filled.frame['offset'].dt.total_microseconds().to_list()
    assert onsets == [0, 3333]
    assert offsets == [667, 9667]


def test_fill_respects_sub_millisecond_sample_spacing():
    # At 2000 Hz (0.5 ms spacing) the event [5 ms, 10 ms) must cover exactly the samples
    # 5.0..9.5 ms; the classification must not assume a one-unit sample spacing.
    timesteps = np.arange(40) * 0.5
    events = Events(name='fixation', onsets=[5.0], offsets=[10.0])

    filled = fill(events, timesteps=timesteps)

    onsets = filled.frame['onset'].dt.total_microseconds().to_list()
    offsets = filled.frame['offset'].dt.total_microseconds().to_list()
    # Unclassified segments: 0..4.5 ms and 10..19.5 ms, both in microseconds.
    assert onsets == [0, 10_000]
    assert offsets == [4_500, 19_500]


def test_fill_accepts_duration_timesteps():
    # A Duration timesteps series is converted to milliseconds internally, matching the
    # millisecond on- and offsets of the Duration event frame.
    events = Events(name='fixation', onsets=[0], offsets=[49])
    duration_timesteps = pl.Series(np.arange(0, 100)).cast(pl.Duration('ms'))

    filled = fill(events, timesteps=duration_timesteps)

    expected = Events(name='unclassified', onsets=[49], offsets=[99])
    assert_frame_equal(filled.frame, expected.frame)
