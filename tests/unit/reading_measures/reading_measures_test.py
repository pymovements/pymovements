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

from pymovements.measure.reading.measures import build_word_level_table
from pymovements.measure.reading.measures import compute_first_duration
from pymovements.measure.reading.measures import compute_first_fixation_duration
from pymovements.measure.reading.measures import compute_first_pass_reading_time
from pymovements.measure.reading.measures import compute_total_fixation_count


def test_map_to_aois_adds_word_columns(mapped_events):
    assert 'word_idx' in mapped_events.columns
    assert 'word' in mapped_events.columns
    assert 'char_idx' in mapped_events.columns


def test_annotate_fixations(annotated):
    assert 'is_first_pass' in annotated.columns
    assert 'run_id' in annotated.columns
    assert annotated.height == 2


def test_all_tokens_from_aois(all_tokens):
    assert 'word_idx' in all_tokens.columns
    assert 'word' in all_tokens.columns
    assert all_tokens.height == 2  # 2 unique words: The, quick


def test_build_word_level_table(annotated, all_tokens):
    result = build_word_level_table(words=all_tokens, fix=annotated)
    assert result.height == 2  # one row per word
    assert 'FFD' in result.columns
    assert 'TFT' in result.columns
    assert result.filter(pl.col('word') == 'The')['FFD'][0] == 200
    assert result.filter(pl.col('word') == 'quick')['FFD'][0] == 200


def test_compute_first_duration(annotated):
    result = compute_first_duration(annotated)
    assert 'FD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FD'][0] == 200
    assert result.filter(pl.col('word_idx') == 1)['FD'][0] == 200


def test_compute_first_fixation_duration(annotated):
    result = compute_first_fixation_duration(annotated)
    assert 'FFD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FFD'][0] == 200


def test_compute_first_pass_reading_time(annotated):
    result = compute_first_pass_reading_time(annotated)
    assert 'FPRT' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FPRT'][0] == 200


def test_compute_total_fixation_count(annotated):
    result = compute_total_fixation_count(annotated)
    assert 'TFC' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['TFC'][0] == 1
    assert result.filter(pl.col('word_idx') == 1)['TFC'][0] == 1
