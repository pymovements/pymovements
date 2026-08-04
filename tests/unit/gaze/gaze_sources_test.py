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
"""Test sources metadata propagation from Gaze to derived objects."""
import polars as pl
import pytest

from pymovements import Events
from pymovements import Gaze
from pymovements.stimulus.text import TextStimulus


def _detect_fixation() -> Events:
    return Events(name='fixation', onsets=[0], offsets=[1])


@pytest.fixture(name='gaze')
def fixture_gaze():
    return Gaze(
        samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]}),
        time_column='time',
        pixel_columns=['x', 'y'],
        metadata={'sources': ['raw/sub_1.csv']},
    )


@pytest.fixture(name='text_stimulus')
def fixture_text_stimulus():
    return TextStimulus(
        aois=pl.DataFrame({
            'char': ['a'],
            'sx': [0.0],
            'sy': [0.0],
            'ex': [1.0],
            'ey': [1.0],
        }),
        aoi_column='char',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        metadata={'sources': ['stimuli/text_1_aoi.csv']},
    )


def test_gaze_init_propagates_sources_to_events(gaze):
    assert gaze.events.metadata['sources'] == ['raw/sub_1.csv']


def test_gaze_init_without_sources_leaves_events_metadata_empty():
    gaze = Gaze(
        samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]}),
        time_column='time',
        pixel_columns=['x', 'y'],
    )
    assert gaze.events.metadata == {}


def test_gaze_init_merges_sources_into_passed_events():
    events = Events(
        name='fixation', onsets=[0], offsets=[1],
        metadata={'sources': ['events/sub_1.feather']},
    )
    gaze = Gaze(
        samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]}),
        time_column='time',
        pixel_columns=['x', 'y'],
        events=events,
        metadata={'sources': ['raw/sub_1.csv']},
    )
    assert gaze.events.metadata['sources'] == ['events/sub_1.feather', 'raw/sub_1.csv']


def test_gaze_detect_keeps_sources_on_events(gaze):
    gaze.detect(_detect_fixation)
    assert gaze.events.metadata['sources'] == ['raw/sub_1.csv']


def test_gaze_detect_clear_repropagates_sources(gaze):
    gaze.detect(_detect_fixation, clear=True)
    assert gaze.events.metadata['sources'] == ['raw/sub_1.csv']


def test_gaze_detect_after_events_reset_repropagates_sources(gaze):
    # Simulates Dataset.clear_events(), which replaces the events container.
    gaze.events = Events()
    gaze.detect(_detect_fixation)
    assert gaze.events.metadata['sources'] == ['raw/sub_1.csv']


def test_gaze_map_to_aois_merges_stimulus_sources(gaze, text_stimulus):
    gaze.map_to_aois(text_stimulus, verbose=False)

    assert gaze.metadata['sources'] == ['raw/sub_1.csv', 'stimuli/text_1_aoi.csv']


def test_gaze_map_to_aois_with_none_metadata_adopts_stimulus_sources(gaze, text_stimulus):
    gaze.metadata = None

    gaze.map_to_aois(text_stimulus, verbose=False)

    assert gaze.metadata == {'sources': ['stimuli/text_1_aoi.csv']}


def test_gaze_clone_deepcopies_sources(gaze):
    gaze_clone = gaze.clone()

    assert gaze_clone.metadata['sources'] == ['raw/sub_1.csv']
    assert gaze_clone.events.metadata['sources'] == ['raw/sub_1.csv']

    gaze_clone.metadata['sources'].append('other.csv')
    assert gaze.metadata['sources'] == ['raw/sub_1.csv']


def test_gaze_split_propagates_sources():
    gaze = Gaze(
        samples=pl.DataFrame({
            'trial': [1, 1, 2],
            'time': [0, 1, 2],
            'x': [0.1, 0.2, 0.3],
            'y': [0.1, 0.2, 0.3],
        }),
        time_column='time',
        pixel_columns=['x', 'y'],
        trial_columns='trial',
        metadata={'sources': ['raw/sub_1.csv']},
    )

    splits = gaze.split(by='trial')

    assert len(splits) == 2
    for gaze_split in splits:
        assert gaze_split.metadata['sources'] == ['raw/sub_1.csv']
        assert gaze_split.events.metadata['sources'] == ['raw/sub_1.csv']
