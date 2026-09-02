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
"""Tests for Gaze.summary."""
import polars as pl

from pymovements import Events
from pymovements import Gaze


def test_gaze_summary_without_events(capsys):
    """Ensure that summary prints the string representation and a zero event count."""
    gaze = Gaze(
        samples=pl.DataFrame({'time': [0, 1], 'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
        pixel_columns=['x', 'y'],
    )

    gaze.summary()

    assert capsys.readouterr().out == f'{gaze}\ntotal events: 0\n'


def test_gaze_summary_counts_events_by_name(capsys):
    """Ensure that summary prints event counts grouped by event name in alphabetical order."""
    gaze = Gaze(
        samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
        pixel_columns=['x', 'y'],
        events=Events(
            name=['saccade', 'fixation', 'fixation'],
            onsets=[0, 1, 2],
            offsets=[1, 2, 3],
        ),
    )

    gaze.summary()

    expected = f'{gaze}\ntotal events: 3\n  fixation: 2\n  saccade: 1\n'
    assert capsys.readouterr().out == expected
