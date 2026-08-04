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
"""Test metadata attribute of the Events class."""
import polars as pl

from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus


def test_events_metadata_defaults_to_empty_dict():
    events = Events()
    assert events.metadata == {}


def test_events_metadata_is_stored():
    events = Events(metadata={'sources': ['raw/sub_1.csv'], 'subject_id': 1})
    assert events.metadata == {'sources': ['raw/sub_1.csv'], 'subject_id': 1}


def test_events_clone_deepcopies_metadata():
    events = Events(
        name=['fixation'], onsets=[0], offsets=[1],
        metadata={'sources': ['raw/sub_1.csv']},
    )

    events_clone = events.clone()

    assert events_clone.metadata == events.metadata
    events_clone.metadata['sources'].append('other.csv')
    assert events.metadata['sources'] == ['raw/sub_1.csv']


def test_events_split_list_propagates_metadata():
    events = Events(
        name=['fixation', 'fixation'], onsets=[0, 2], offsets=[1, 3], trials=[1, 2],
        metadata={'sources': ['raw/sub_1.csv']},
    )

    splits = events.split(by='trial')

    assert len(splits) == 2
    for events_split in splits:
        assert events_split.metadata == {'sources': ['raw/sub_1.csv']}


def test_events_split_dict_propagates_metadata():
    events = Events(
        name=['fixation', 'fixation'], onsets=[0, 2], offsets=[1, 3], trials=[1, 2],
        metadata={'sources': ['raw/sub_1.csv']},
    )

    splits = events.split(by='trial', as_dict=True)

    assert len(splits) == 2
    for events_split in splits.values():
        assert events_split.metadata == {'sources': ['raw/sub_1.csv']}


def test_events_map_to_aois_merges_stimulus_sources(simple_stimulus: TextStimulus) -> None:
    simple_stimulus.metadata['sources'] = ['stimuli/text_1_aoi.csv']

    events = Events(
        data=pl.DataFrame({
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5, 5]],
        }),
        metadata={'sources': ['raw/sub_1.csv']},
    )

    events.map_to_aois(simple_stimulus, verbose=False)

    assert events.metadata['sources'] == ['raw/sub_1.csv', 'stimuli/text_1_aoi.csv']
