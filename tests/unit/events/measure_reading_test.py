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
"""Tests for Events.measure_reading and Gaze.measure_reading."""
import polars as pl
import pytest

from pymovements.events import Events
from pymovements.gaze import Gaze
from pymovements.measure.reading import ReadingMeasures
from pymovements.stimulus.text import TextStimulus


def _make_stimulus(words, trial='trial_1'):
    n = len(words)
    return TextStimulus(
        aois=pl.DataFrame({
            'char': words,
            'char_idx': list(range(n)),
            'word_idx': list(range(n)),
            'word': words,
            'top_left_x': [10.0 + 10 * i for i in range(n)],
            'top_left_y': [10.0] * n,
            'width': [10.0] * n,
            'height': [20.0] * n,
            'trial': [trial] * n,
        }),
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        trial_column='trial',
    )


@pytest.fixture(name='stimulus')
def fixture_stimulus():
    return _make_stimulus(['The', 'quick', 'brown'])


@pytest.fixture(name='fixation_events')
def fixture_fixation_events():
    # fixations on word 0 (x=15) and word 1 (x=25); word 2 (x=35) is skipped.
    return Events(
        pl.DataFrame({
            'name': ['fixation', 'fixation'],
            'onset': [0, 200],
            'offset': [200, 400],
            'duration': [200, 200],
            'location': [[15.0, 15.0], [25.0, 15.0]],
            'trial': ['trial_1', 'trial_1'],
        }),
    )


def test_events_measure_reading_returns_reading_measures(fixation_events, stimulus):
    result = fixation_events.measure_reading(stimulus)

    assert isinstance(result, ReadingMeasures)
    assert result.frame['word'].to_list() == ['The', 'quick', 'brown']
    assert result.frame['TFT'].to_list() == [200, 200, 0]
    assert result.frame['TFC'].to_list() == [1, 1, 0]
    # the skipped word is flagged and landing positions were computed from char_idx
    assert result.frame['skipped'].to_list() == [0, 0, 1]
    assert result.frame['LP'].to_list() == [0, 0, None]


def test_events_measure_reading_does_not_mutate_events(fixation_events, stimulus):
    before = fixation_events.frame.clone()

    fixation_events.measure_reading(stimulus)

    assert fixation_events.frame.equals(before)
    assert 'word_idx' not in fixation_events.columns


def test_events_measure_reading_empty_returns_empty(stimulus):
    empty = Events()

    result = empty.measure_reading(stimulus)

    assert isinstance(result, ReadingMeasures)
    assert result.frame.is_empty()


def test_events_measure_reading_multiple_stimuli(stimulus):
    events = Events(
        pl.DataFrame({
            'name': ['fixation'] * 3,
            'onset': [0, 200, 400],
            'offset': [200, 400, 600],
            'duration': [200, 200, 200],
            'location': [[15.0, 15.0], [25.0, 15.0], [15.0, 15.0]],
            'trial': ['trial_1', 'trial_1', 'trial_2'],
        }),
    )
    stimuli = {
        'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1'),
        'trial_2': _make_stimulus(['fox', 'jumps'], trial='trial_2'),
    }

    result = events.measure_reading(stimuli).frame.sort(['trial', 'word_index'])

    assert result['trial'].to_list() == ['trial_1', 'trial_1', 'trial_1', 'trial_2', 'trial_2']
    assert result['word'].to_list() == ['The', 'quick', 'brown', 'fox', 'jumps']
    # trial_1: words 0 and 1 fixated, word 2 skipped; trial_2: only word 0 fixated
    assert result['TFC'].to_list() == [1, 1, 0, 1, 0]


def test_events_measure_reading_multiple_stimuli_requires_sequence_column(stimulus):
    events = Events(
        pl.DataFrame({
            'name': ['fixation'],
            'onset': [0],
            'offset': [200],
            'duration': [200],
            'location': [[15.0, 15.0]],
        }),
    )

    with pytest.raises(ValueError, match="no 'trial' column"):
        events.measure_reading({'trial_1': stimulus})


def test_gaze_measure_reading_delegates_to_events(fixation_events, stimulus):
    gaze = Gaze(events=fixation_events)

    gaze_result = gaze.measure_reading(stimulus)
    events_result = fixation_events.measure_reading(stimulus)

    assert isinstance(gaze_result, ReadingMeasures)
    assert gaze_result.frame.equals(events_result.frame)


def test_gaze_measure_reading_without_events_returns_empty(stimulus):
    gaze = Gaze(events=Events())
    gaze.events = None

    result = gaze.measure_reading(stimulus)

    assert isinstance(result, ReadingMeasures)
    assert result.frame.is_empty()
