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
"""Test the sample selection helper of pymovements.events."""
from __future__ import annotations

import pytest

from pymovements import Events
from pymovements import Experiment
from pymovements.events._utils.selection import resolve_sampling_rate


@pytest.mark.parametrize(
    ('events', 'experiment', 'expected'),
    [
        pytest.param(
            Events(name=['a'], onsets=[0], durations=[4], sampling_rate=500),
            Experiment(sampling_rate=1000),
            500,
            id='entry_before_experiment',
        ),
        pytest.param(
            Events(name=['a'], onsets=[0], durations=[4]),
            Experiment(sampling_rate=1000),
            1000,
            id='experiment_when_no_entry',
        ),
        pytest.param(
            Events(name=['a'], onsets=[0], durations=[4]),
            Experiment(),
            None,
            id='none_when_experiment_has_no_rate',
        ),
        pytest.param(
            Events(name=['a'], onsets=[0], durations=[4]),
            None,
            None,
            id='none_without_experiment',
        ),
    ],
)
def test_resolve_sampling_rate(events, experiment, expected):
    assert resolve_sampling_rate(events, experiment) == expected
