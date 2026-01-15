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
"""Test Gaze detect method warnings."""
import warnings

import numpy as np
import pytest

import pymovements as pm


def dummy_detect_no_events(**_kwargs):
    """Return no events."""
    return pm.Events()


def dummy_detect_with_events(**_kwargs):
    """Return one event."""
    return pm.Events(name='fixation', onsets=[0], offsets=[1])


def test_detect_no_events_warning():
    """Test that a warning is emitted when no events are detected."""
    gaze = pm.gaze.from_numpy(
        time=np.arange(100),
        position=np.zeros((2, 100)),
        experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
    )

    with pytest.warns(UserWarning, match='dummy_detect_no_events: No events were detected.'):
        gaze.detect(dummy_detect_no_events)


def test_detect_no_events_warning_trial_columns():
    """Test that a warning is emitted when no events are detected with trial columns."""
    gaze = pm.gaze.from_numpy(
        time=np.arange(100),
        position=np.zeros((2, 100)),
        trial=np.array(['A'] * 50 + ['B'] * 50),
        experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
    )

    with pytest.warns(UserWarning, match='dummy_detect_no_events: No events were detected.'):
        gaze.detect(dummy_detect_no_events)


def test_detect_with_events_no_warning():
    """Test that no warning is emitted when events are detected."""
    gaze = pm.gaze.from_numpy(
        time=np.arange(100),
        position=np.zeros((2, 100)),
        experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gaze.detect(dummy_detect_with_events)

    # Check that no UserWarning about "No events" was issued
    assert not any(
        isinstance(w.message, UserWarning) and 'No events were detected' in str(w.message)
        for w in record
    )


def test_detect_with_events_trial_columns_no_warning():
    """Test that no warning is emitted when events are detected in at least one trial."""
    gaze = pm.gaze.from_numpy(
        time=np.arange(100),
        position=np.zeros((2, 100)),
        trial=np.array(['A'] * 50 + ['B'] * 50),
        experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
    )

    # Custom detect that only returns events for the first trial it is called for.
    call_count = [0]

    def detect_only_once(**_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return pm.Events(name='fixation', onsets=[0], offsets=[1])
        return pm.Events()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gaze.detect(detect_only_once)

    # Check that no UserWarning about "No events" was issued
    assert not any(
        isinstance(w.message, UserWarning) and 'No events were detected' in str(w.message)
        for w in record
    )
