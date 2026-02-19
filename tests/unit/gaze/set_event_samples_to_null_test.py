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
"""Test Gaze.set_event_samples_to_null() method."""
from __future__ import annotations

import polars as pl
import pytest

import pymovements as pm


def test_basic_nullify_during_blink():
    """Blink event samples are set to null for pixel column."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(6), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 6,
        }),
        events=pm.Events(name='blink', onsets=[2], offsets=[3]),
    )
    gaze.set_event_samples_to_null('blink', padding=0)

    result = gaze.samples['pixel'].to_list()
    assert result[0] == [1.0, 2.0]
    assert result[1] == [1.0, 2.0]
    assert result[2] is None
    assert result[3] is None
    assert result[4] == [1.0, 2.0]
    assert result[5] == [1.0, 2.0]


def test_with_symmetric_padding():
    """Padding extends the null window symmetrically."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(8), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 8,
        }),
        events=pm.Events(name='blink', onsets=[3], offsets=[4]),
    )
    gaze.set_event_samples_to_null('blink', padding=1)

    result = gaze.samples['pixel'].to_list()
    # Padded range: 2-5
    assert result[0] == [1.0, 2.0]
    assert result[1] == [1.0, 2.0]
    assert result[2] is None
    assert result[3] is None
    assert result[4] is None
    assert result[5] is None
    assert result[6] == [1.0, 2.0]
    assert result[7] == [1.0, 2.0]


def test_with_asymmetric_padding():
    """Asymmetric padding extends differently before and after."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(10), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 10,
        }),
        events=pm.Events(name='blink', onsets=[4], offsets=[5]),
    )
    gaze.set_event_samples_to_null('blink', padding=(2, 1))

    result = gaze.samples['pixel'].to_list()
    # Padded range: 2-6
    assert result[0] == [1.0, 2.0]
    assert result[1] == [1.0, 2.0]
    assert result[2] is None
    assert result[3] is None
    assert result[4] is None
    assert result[5] is None
    assert result[6] is None
    assert result[7] == [1.0, 2.0]


def test_multi_trial():
    """Events only affect their own trial's samples."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series([0, 1, 2, 3, 0, 1, 2, 3], dtype=pl.Int64),
            'trial': [1, 1, 1, 1, 2, 2, 2, 2],
            'pixel': [[1.0, 2.0]] * 8,
        }),
        events=pm.Events(
            name='blink',
            onsets=[1],
            offsets=[2],
            trials=[1],
            trial_columns='trial',
        ),
        trial_columns='trial',
    )
    gaze.set_event_samples_to_null('blink', padding=0)

    result = gaze.samples['pixel'].to_list()
    # Trial 1: time 1-2 nullified
    assert result[0] == [1.0, 2.0]
    assert result[1] is None
    assert result[2] is None
    assert result[3] == [1.0, 2.0]
    # Trial 2: unchanged
    assert result[4] == [1.0, 2.0]
    assert result[5] == [1.0, 2.0]
    assert result[6] == [1.0, 2.0]
    assert result[7] == [1.0, 2.0]


def test_no_events_attribute_raises():
    """Raise AttributeError when events is None."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(5), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 5,
        }),
    )
    gaze.events = None

    with pytest.raises(AttributeError, match='no events'):
        gaze.set_event_samples_to_null('blink')


def test_no_matching_events_raises():
    """Raise ValueError when no events match the given name."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(5), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 5,
        }),
        events=pm.Events(name='saccade', onsets=[1], offsets=[2]),
    )

    with pytest.raises(ValueError, match="No events with name 'blink'"):
        gaze.set_event_samples_to_null('blink')


def test_only_specified_event_type_nullified():
    """Only the specified event type is nullified; other event types are unaffected."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(10), dtype=pl.Int64),
            'pixel': [[float(i), float(i)] for i in range(10)],
        }),
        events=pm.Events(
            data=pl.DataFrame({
                'name': ['blink', 'saccade'],
                'onset': [2, 6],
                'offset': [3, 8],
            }),
        ),
    )
    gaze.set_event_samples_to_null('blink', padding=0)

    result = gaze.samples['pixel'].to_list()
    # Only blink at 2-3 nullified
    assert result[2] is None
    assert result[3] is None
    # Saccade samples not nullified
    assert result[6] == [6.0, 6.0]
    assert result[7] == [7.0, 7.0]
    assert result[8] == [8.0, 8.0]


def test_time_and_trial_columns_preserved():
    """Time and trial columns must remain intact after nullification."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(6), dtype=pl.Int64),
            'trial': [1, 1, 1, 2, 2, 2],
            'pixel': [[1.0, 2.0]] * 6,
        }),
        events=pm.Events(
            name='blink',
            onsets=[1],
            offsets=[2],
            trials=[1],
            trial_columns='trial',
        ),
        trial_columns='trial',
    )
    original_time = gaze.samples['time'].to_list()
    original_trial = gaze.samples['trial'].to_list()

    gaze.set_event_samples_to_null('blink', padding=0)

    assert gaze.samples['time'].to_list() == original_time
    assert gaze.samples['trial'].to_list() == original_trial


def test_multiple_data_columns_nullified():
    """All data columns (not time/trial) are nullified during events."""
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': pl.Series(range(5), dtype=pl.Int64),
            'pixel': [[1.0, 2.0]] * 5,
            'position': [[0.5, 0.5]] * 5,
            'velocity': [[10.0, 10.0]] * 5,
        }),
        events=pm.Events(name='blink', onsets=[1], offsets=[2]),
    )
    gaze.set_event_samples_to_null('blink', padding=0)

    for col in ['pixel', 'position', 'velocity']:
        result = gaze.samples[col].to_list()
        assert result[1] is None
        assert result[2] is None
        assert result[0] is not None
        assert result[3] is not None


def test_scalar_columns_nullified():
    """Scalar (non-nested) data columns are also nullified."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        gaze = pm.Gaze(
            samples=pl.DataFrame({
                'time': pl.Series(range(5), dtype=pl.Int64),
                'pupil': [3.0, 3.5, 4.0, 3.5, 3.0],
            }),
            events=pm.Events(name='blink', onsets=[1], offsets=[2]),
        )
    gaze.set_event_samples_to_null('blink', padding=0)

    result = gaze.samples['pupil'].to_list()
    assert result[0] == 3.0
    assert result[1] is None
    assert result[2] is None
    assert result[3] == 3.5
    assert result[4] == 3.0
