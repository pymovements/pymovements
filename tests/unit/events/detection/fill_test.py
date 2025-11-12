# Copyright (c) 2022-2025 The pymovements Project Authors
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
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements.events import fill


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'events': Events(name='fixation', onsets=[0], offsets=[100]),
                'timesteps': np.arange(0, 100),
            },
            Events(),
            id='fixation_from_start_to_end_no_fill',
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
            id='fixation_10_ms_after_start_to_end_single_fill',
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
            id='fixation_from_start_to_10_ms_before_end_single_fill',
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
            id='fixation_10_ms_break_at_40ms_single_fill',
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
            id='fixation_10_ms_break_then_saccade_until_end_single_fill',
        ),
    ],
)
def test_fill_fills_events(kwargs, expected):
    events = fill(**kwargs)

    assert_frame_equal(events.frame, expected.frame)
