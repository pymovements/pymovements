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
"""Reading measure tests."""
import polars as pl
import pytest

from pymovements.events import Events
from pymovements.measure.reading.measures import build_word_level_table
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import total_fixation_count
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


@pytest.fixture(name='stimulus', scope='function')
def fixture_stimulus() -> TextStimulus:
    def _make_stimulus() -> TextStimulus:
        return TextStimulus(
            aois=CHAR_AOI_DF.clone(),
            aoi_column='char',
            start_x_column='top_left_x',
            start_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
            trial_column='trial',
        )

    return _make_stimulus()


@pytest.fixture(name='mapped_events', scope='function')
def fixture_mapped_events(stimulus: TextStimulus) -> Events:
    def _make_mapped_events() -> Events:
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
        events.map_to_aois(stimulus)
        return events.frame

    return _make_mapped_events()


@pytest.fixture(name='annotated_events', scope='function')
def fixture_annotated_events(mapped_events):
    return annotate_fixations(mapped_events, group_columns=['trial', 'page'])


@pytest.fixture(name='all_tokens', scope='function')
def fixture_all_tokens(stimulus: TextStimulus) -> pl.DataFrame:
    return all_tokens_from_aois(stimulus.aois, trial='trial_1')


def test_fixture_mapped_events_has_word_and_char_columns(mapped_events):
    assert 'word_idx' in mapped_events.columns
    assert 'word' in mapped_events.columns
    assert 'char_idx' in mapped_events.columns


def test_annotate_fixations(annotated_events):
    assert 'is_first_pass' in annotated_events.columns
    assert 'run_id' in annotated_events.columns
    assert annotated_events.height == 2


def test_all_tokens_from_aois(all_tokens):
    assert 'word_idx' in all_tokens.columns
    assert 'word' in all_tokens.columns
    assert all_tokens.height == 2  # 2 unique words: The, quick


def test_build_word_level_table(annotated_events, all_tokens):
    result = build_word_level_table(words=all_tokens, fix=annotated_events)
    assert result.height == 2  # one row per word
    assert 'FFD' in result.columns
    assert 'TFT' in result.columns
    assert result.filter(pl.col('word') == 'The')['FFD'][0] == 200
    assert result.filter(pl.col('word') == 'quick')['FFD'][0] == 200


def test_compute_first_duration(annotated_events):
    result = first_duration(annotated_events)
    assert 'FD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FD'][0] == 200
    assert result.filter(pl.col('word_idx') == 1)['FD'][0] == 200


def test_compute_first_fixation_duration(annotated_events):
    result = first_fixation_duration(annotated_events)
    assert 'FFD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FFD'][0] == 200


def test_compute_first_pass_reading_time(annotated_events):
    result = first_pass_reading_time(annotated_events)
    assert 'FPRT' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FPRT'][0] == 200


def test_compute_total_fixation_count(annotated_events):
    result = total_fixation_count(annotated_events)
    assert 'TFC' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['TFC'][0] == 1
    assert result.filter(pl.col('word_idx') == 1)['TFC'][0] == 1
