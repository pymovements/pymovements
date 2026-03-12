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
"""Shared fixtures for reading measure tests."""
import polars as pl
import pytest

from pymovements.events import Events
from pymovements.measure.reading.processing import annotate_fixations
from pymovements.measure.reading.words import all_tokens_from_aois
from pymovements.stimulus.text import TextStimulus


# Synthetic char-level AOI table
# "The quick" — 2 words, each 3 chars, on page_1
CHAR_AOI_DF = pl.DataFrame({
    'char': ['T', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k'],
    'char_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'word_idx': [0, 0, 0, 0, 1, 1, 1, 1, 1],  # blank space is part of the previous word
    'word': ['The', 'The', 'The', 'The', 'quick', 'quick', 'quick', 'quick', 'quick'],
    'top_left_x': [10., 20., 30., 40., 50., 60., 70., 80., 90.],
    'top_left_y': [10., 10., 10., 10., 10., 10., 10., 10., 10.],
    'width': [10., 10., 10., 10., 10., 10., 10., 10., 10.],
    'height': [20., 20., 20., 20., 20., 20., 20., 20., 20.],
    'page': ['page_1'] * 9,
    'trial': ['trial_1'] * 9,
})


def _make_stimulus() -> TextStimulus:
    return TextStimulus(
        aois=CHAR_AOI_DF,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        trial_column='trial',
    )


def _make_mapped_events() -> pl.DataFrame:
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'onset': [0, 200],
        'offset': [200, 400],
        'duration': [200, 200],
        'location': [[15., 15.], [55., 15.]],
        'trial': ['trial_1', 'trial_1'],
        'page': ['page_1', 'page_1'],
    })
    events = Events(data=events_df)
    events.map_to_aois(_make_stimulus())
    return events.frame


@pytest.fixture
def stimulus() -> TextStimulus:
    return _make_stimulus()


@pytest.fixture
def mapped_events() -> pl.DataFrame:
    return _make_mapped_events()


@pytest.fixture
def annotated() -> pl.DataFrame:
    return annotate_fixations(_make_mapped_events(), group_columns=['trial', 'page'])


@pytest.fixture
def all_tokens() -> pl.DataFrame:
    return all_tokens_from_aois(_make_stimulus().aois, trial='trial_1')
