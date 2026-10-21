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


def _make_stimulus(words, trial='trial_1', start_x=None):
    n = len(words)
    if start_x is None:
        start_x = [10.0 + 10 * i for i in range(n)]
    aois = pl.DataFrame({
        'char': words,
        'char_idx': list(range(n)),
        'word_idx': list(range(n)),
        'word': words,
        'top_left_x': start_x,
        'top_left_y': [10.0] * n,
        'width': [10.0] * n,
        'height': [20.0] * n,
    })
    if trial is not None:
        aois = aois.with_columns(pl.lit(trial).alias('trial'))
    return TextStimulus(
        aois=aois,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        trial_column='trial' if trial is not None else None,
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
    assert result.frame['LP'].to_list() == [1, 1, 0]


def test_events_measure_reading_does_not_mutate_events(fixation_events, stimulus):
    before = fixation_events.frame.clone()

    fixation_events.measure_reading(stimulus)

    assert fixation_events.frame.equals(before)
    assert 'word_idx' not in fixation_events.columns


def test_events_measure_reading_merges_sources(fixation_events, stimulus):
    fixation_events.metadata = {'Sources': ['events.tsv']}
    stimulus.metadata = {'Sources': ['aoi.tsv']}

    result = fixation_events.measure_reading(stimulus)

    assert result.metadata == {'Sources': ['events.tsv', 'aoi.tsv']}
    # the events keep their own metadata, the stimulus source is only recorded on the result
    assert fixation_events.metadata == {'Sources': ['events.tsv']}


def test_events_measure_reading_multiple_stimuli_merges_sources(two_trial_events):
    two_trial_events.metadata = {'Sources': ['events.tsv']}
    stimuli = {
        'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1'),
        'trial_2': _make_stimulus(['fox', 'jumps'], trial='trial_2'),
    }
    stimuli['trial_1'].metadata = {'Sources': ['aoi_1.tsv']}
    stimuli['trial_2'].metadata = {'Sources': ['aoi_2.tsv']}

    result = two_trial_events.measure_reading(stimuli, group_columns=['trial'])

    assert result.metadata == {'Sources': ['events.tsv', 'aoi_1.tsv', 'aoi_2.tsv']}
    assert two_trial_events.metadata == {'Sources': ['events.tsv']}


def test_events_measure_reading_empty_keeps_events_metadata(stimulus):
    empty = Events(metadata={'Sources': ['events.tsv']})
    stimulus.metadata = {'Sources': ['aoi.tsv']}

    result = empty.measure_reading(stimulus)

    # nothing was mapped, so the stimulus is not a source of the empty measures
    assert result.metadata == {'Sources': ['events.tsv']}


def test_events_measure_reading_remaps_already_mapped_events(fixation_events, stimulus):
    # words of the second stimulus sit at x = 20, 30 and 10, so the fixations at x = 15 and
    # x = 25 land on its words 2 and 0.
    other_stimulus = _make_stimulus(['x', 'y', 'z'], start_x=[20.0, 30.0, 10.0])
    fixation_events.map_to_aois(stimulus)

    result = fixation_events.measure_reading(other_stimulus)

    assert result.frame['word'].to_list() == ['x', 'y', 'z']
    assert result.frame['TFC'].to_list() == [1, 0, 1]


def test_events_measure_reading_empty_returns_empty(stimulus):
    empty = Events()

    result = empty.measure_reading(stimulus)

    assert isinstance(result, ReadingMeasures)
    assert result.frame.is_empty()


def test_events_measure_reading_multiple_stimuli(two_trial_events):
    stimuli = {
        'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1'),
        'trial_2': _make_stimulus(['fox', 'jumps'], trial='trial_2'),
    }

    result = two_trial_events.measure_reading(stimuli, group_columns=['trial'])
    result = result.frame.sort(['trial', 'word_index'])

    assert result['trial'].to_list() == ['trial_1', 'trial_1', 'trial_1', 'trial_2', 'trial_2']
    assert result['word'].to_list() == ['The', 'quick', 'brown', 'fox', 'jumps']
    # trial_1: words 0 and 1 fixated, word 2 skipped; trial_2: only word 0 fixated
    assert result['TFC'].to_list() == [1, 1, 0, 1, 0]


@pytest.fixture(name='two_trial_events')
def fixture_two_trial_events():
    # trial_1 fixates words 0 and 1, trial_2 fixates word 0.
    return Events(
        pl.DataFrame({
            'name': ['fixation'] * 3,
            'onset': [0, 200, 400],
            'offset': [200, 400, 600],
            'duration': [200, 200, 200],
            'location': [[15.0, 15.0], [25.0, 15.0], [15.0, 15.0]],
            'trial': ['trial_1', 'trial_1', 'trial_2'],
        }),
    )


def test_events_measure_reading_multiple_stimuli_null_key_raises():
    events = Events(
        pl.DataFrame({
            'name': ['fixation', 'fixation'],
            'onset': [0, 200],
            'offset': [200, 400],
            'duration': [200, 200],
            'location': [[15.0, 15.0], [25.0, 15.0]],
            'trial': ['trial_1', None],
        }),
    )
    stimuli = {'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1')}

    with pytest.raises(ValueError, match=r'without an entry in the aois dict: \[None\]'):
        events.measure_reading(stimuli, group_columns=['trial'])


def test_events_measure_reading_multiple_stimuli_missing_key_raises(two_trial_events):
    stimuli = {'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1')}

    with pytest.raises(ValueError, match=r"without an entry in the aois dict: \['trial_2'\]"):
        two_trial_events.measure_reading(stimuli, group_columns=['trial'])


def test_events_measure_reading_multiple_stimuli_unused_key_warns(fixation_events):
    stimuli = {
        'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1'),
        'trial_2': _make_stimulus(['fox', 'jumps'], trial='trial_2'),
    }

    with pytest.warns(UserWarning, match=r"keys without any matching fixations: \['trial_2'\]"):
        result = fixation_events.measure_reading(stimuli, group_columns=['trial'])

    result = result.frame.sort(['trial', 'word_index'])
    assert result['word'].to_list() == ['The', 'quick', 'brown', 'fox', 'jumps']
    assert result['TFC'].to_list() == [1, 1, 0, 0, 0]
    assert result['skipped'].to_list() == [0, 0, 1, 1, 1]


def test_events_measure_reading_multiple_stimuli_empty_dict_raises(fixation_events):
    with pytest.raises(ValueError, match='aois dict is empty'):
        fixation_events.measure_reading({}, group_columns=['trial'])


def test_events_measure_reading_multiple_stimuli_no_key_matches_raises(fixation_events):
    stimuli = {'trial_9': _make_stimulus(['fox', 'jumps'], trial='trial_9')}

    with pytest.raises(ValueError, match=r"without an entry in the aois dict: \['trial_1'\]"):
        fixation_events.measure_reading(stimuli, group_columns=['trial'])


@pytest.fixture(name='int_trial_events')
def fixture_int_trial_events():
    return Events(
        pl.DataFrame({
            'name': ['fixation', 'fixation'],
            'onset': [0, 200],
            'offset': [200, 400],
            'duration': [200, 200],
            'location': [[15.0, 15.0], [25.0, 15.0]],
            'trial': [1, 1],
        }),
    )


def test_events_measure_reading_multiple_stimuli_incompatible_key_dtype_raises(int_trial_events):
    stimuli = {'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_1')}

    with pytest.raises(ValueError, match='keys are not compatible with the fixation trial dtype'):
        int_trial_events.measure_reading(stimuli, group_columns=['trial'])


def test_events_measure_reading_multiple_stimuli_casts_keys_to_column_dtype(int_trial_events):
    stimuli = {'1': _make_stimulus(['The', 'quick', 'brown'], trial=1)}

    result = int_trial_events.measure_reading(stimuli, group_columns=['trial'])

    assert result.frame['trial'].to_list() == [1, 1, 1]
    assert result.frame['TFC'].to_list() == [1, 1, 0]


def test_events_measure_reading_multiple_stimuli_without_sequence_column(fixation_events):
    stimuli = {'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial=None)}

    result = fixation_events.measure_reading(stimuli, group_columns=['trial'])

    assert result.frame['trial'].to_list() == ['trial_1'] * 3
    assert result.frame['TFC'].to_list() == [1, 1, 0]


def test_events_measure_reading_multiple_stimuli_stimulus_column_dtype_differs(fixation_events):
    # the stimulus keeps its sequence column as a string while the key and the events are ints
    events = Events(fixation_events.frame.with_columns(pl.lit(1).alias('trial')))
    stimuli = {1: _make_stimulus(['The', 'quick', 'brown'], trial='1')}

    result = events.measure_reading(stimuli, group_columns=['trial'])

    assert result.frame['trial'].to_list() == [1, 1, 1]
    assert result.frame['TFC'].to_list() == [1, 1, 0]


def test_events_measure_reading_multiple_stimuli_key_mismatch_raises(fixation_events):
    stimuli = {'trial_1': _make_stimulus(['The', 'quick', 'brown'], trial='trial_2')}

    message = r"stimulus for key 'trial_1' has trial values \['trial_2'\]"
    with pytest.raises(ValueError, match=message):
        fixation_events.measure_reading(stimuli, group_columns=['trial'])


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
        events.measure_reading({'trial_1': stimulus}, group_columns=['trial'])


def test_events_measure_reading_multiple_stimuli_requires_group_columns(fixation_events, stimulus):
    with pytest.raises(ValueError, match='requires at least one group column'):
        fixation_events.measure_reading({'trial_1': stimulus})


def test_gaze_measure_reading_delegates_to_events(fixation_events, stimulus):
    gaze = Gaze(events=fixation_events)

    gaze_result = gaze.measure_reading(stimulus)
    events_result = fixation_events.measure_reading(stimulus)

    assert isinstance(gaze_result, ReadingMeasures)
    assert gaze_result.frame.equals(events_result.frame)


@pytest.mark.parametrize('events', [None, Events()], ids=['none', 'empty'])
def test_gaze_measure_reading_without_events_raises(events, stimulus):
    gaze = Gaze(events=Events())
    gaze.events = events

    with pytest.raises(AttributeError, match=r'has no events. Use detect\(\)'):
        gaze.measure_reading(stimulus)
