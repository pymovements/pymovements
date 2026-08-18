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
"""Tests for words module."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.reading.words import all_tokens_from_aois
from pymovements.measure.reading.words import mark_skipped_tokens
from pymovements.measure.reading.words import repair_word_labels


@pytest.mark.parametrize(
    ('df', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'word_idx': [0, 0, 0, 1, 1],
                'word': ['The', None, 'The', 'quick', ' '],
                'char_idx_in_line': [0, 1, 2, 0, 1],
            }),
            pl.DataFrame({
                'word_idx': [0, 0, 0, 1, 1],
                'word': ['The', 'The', 'The', 'quick', 'quick'],
                'char_idx_in_line': [0, 1, 2, 0, 1],
            }),
            id='basic_repair',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'page': [1, 1, 1],
                'line_idx': [0, 0, 0],
                'word_idx': [0, 0, 0],
                'word': [None, 'The', ' '],
                'char_idx_in_line': [0, 1, 2],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'page': [1, 1, 1],
                'line_idx': [0, 0, 0],
                'word_idx': [0, 0, 0],
                'word': ['The', 'The', 'The'],
                'char_idx_in_line': [0, 1, 2],
            }),
            id='with_grouping_cols',
        ),
        pytest.param(
            pl.DataFrame({
                'word_idx': [0, 1],
                'word': ['a', 'b'],
                'char_idx_in_line': [1, 0],
            }),
            pl.DataFrame({
                'word_idx': [0, 1],
                'word': ['a', 'b'],
                'char_idx_in_line': [1, 0],
            }).sort(['word_idx', 'char_idx_in_line']),
            id='sorting_check',
        ),
    ],
)
def test_repair_word_labels(df, expected):
    """Test repair_word_labels."""
    result = repair_word_labels(df)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('aois', 'trial', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'page': [1, 1, 1],
                'word_idx': [0, 0, 1],
                'word': ['The', 'The', 'quick'],
                'other': [1, 2, 3],
            }),
            'trial_1',
            pl.DataFrame({
                'trial': ['trial_1', 'trial_1'],
                'page': [1, 1],
                'word_idx': [0, 1],
                'word': ['The', 'quick'],
            }),
            id='add_trial',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['t1', 't1'],
                'page': [1, 1],
                'word_idx': [0, 1],
                'word': ['a', 'b'],
            }),
            None,
            pl.DataFrame({
                'trial': ['t1', 't1'],
                'page': [1, 1],
                'word_idx': [0, 1],
                'word': ['a', 'b'],
            }),
            id='existing_trial',
        ),
    ],
)
def test_all_tokens_from_aois(aois, trial, expected):
    """Test all_tokens_from_aois."""
    result = all_tokens_from_aois(aois, trial=trial)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('all_tokens', 'fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'page': [1, 1, 1],
                'word_idx': [0, 1, 2],
                'word': ['a', 'b', 'c'],
            }),
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': [1, 1],
                'word_idx': [0, 2],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'page': [1, 1, 1],
                'word_idx': [0, 1, 2],
                'word': ['a', 'b', 'c'],
                'skipped': [0, 1, 0],
            }).with_columns(pl.col('skipped').cast(pl.Int8)),
            id='basic_skipped',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1'],
                'page': [1],
                'word_idx': [0],
                'word': ['a'],
            }),
            pl.DataFrame(
                {
                    'trial': ['1'],
                    'page': [1],
                    'word_idx': [None],
                }, schema={'trial': pl.String, 'page': pl.Int64, 'word_idx': pl.Int64},
            ),
            pl.DataFrame({
                'trial': ['1'],
                'page': [1],
                'word_idx': [0],
                'word': ['a'],
                'skipped': [1],
            }).with_columns(pl.col('skipped').cast(pl.Int8)),
            id='null_word_idx_in_fixations',
        ),
    ],
)
def test_mark_skipped_tokens(all_tokens, fixations, expected):
    """Test mark_skipped_tokens."""
    result = mark_skipped_tokens(all_tokens, fixations)
    assert_frame_equal(result, expected)
