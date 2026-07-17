# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Fixation annotation tests."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.reading.annotation import annotate_delta_in
from pymovements.measure.reading.annotation import annotate_delta_out
from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.annotation import annotate_is_first_fixation
from pymovements.measure.reading.annotation import annotate_is_first_pass
from pymovements.measure.reading.annotation import annotate_is_reg_in
from pymovements.measure.reading.annotation import annotate_is_reg_out
from pymovements.measure.reading.annotation import annotate_next_word_idx
from pymovements.measure.reading.annotation import annotate_prev_word_idx
from pymovements.measure.reading.annotation import annotate_run_id


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': [],
                    'stimulus': [],
                    'page': [],
                    'name': [],
                    'word_idx': [],
                },
                schema={
                    'trial': pl.String,
                    'stimulus': pl.String,
                    'page': pl.String,
                    'name': pl.String,
                    'word_idx': pl.Int64,
                },
            ),
            pl.DataFrame(
                data={
                    'trial': [],
                    'stimulus': [],
                    'page': [],
                    'name': [],
                    'word_idx': [],
                    'run_id': [],
                },
                schema={
                    'trial': pl.String,
                    'stimulus': pl.String,
                    'page': pl.String,
                    'name': pl.String,
                    'word_idx': pl.Int64,
                    'run_id': pl.Int64,
                },
            ),
            id='empty',
        ),
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '1', '1'],
                    'word_idx': [1, 1, 2, 1],
                },
            ),
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '1', '1'],
                    'word_idx': [1, 1, 2, 1],
                    'run_id': [1, 1, 2, 3],
                },
                schema_overrides={'run_id': pl.Int64},
            ),
            id='runs',
        ),
    ],
)
def test_annotate_run_id(fixations, expected):
    result = annotate_run_id(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_prev_word_idx():
    fixations = pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2]})
    expected = pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2], 'prev_word_idx': [None, 1]})
    result = annotate_prev_word_idx(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_next_word_idx():
    fixations = pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2]})
    expected = pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2], 'next_word_idx': [2, None]})
    result = annotate_next_word_idx(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_delta_in():
    fixations = pl.DataFrame({'trial': ['1'], 'word_idx': [2], 'prev_word_idx': [1]})
    expected = pl.DataFrame({
        'trial': ['1'], 'word_idx': [2],
        'prev_word_idx': [1], 'delta_in': [1],
    })
    result = annotate_delta_in(fixations)
    assert_frame_equal(result, expected)


def test_annotate_delta_out():
    fixations = pl.DataFrame({'trial': ['1'], 'word_idx': [1], 'next_word_idx': [2]})
    expected = pl.DataFrame({
        'trial': ['1'], 'word_idx': [1],
        'next_word_idx': [2], 'delta_out': [1],
    })
    result = annotate_delta_out(fixations)
    assert_frame_equal(result, expected)


def test_annotate_is_reg_in():
    fixations = pl.DataFrame({'trial': ['1', '1'], 'delta_in': [1, -1]})
    expected = pl.DataFrame({'trial': ['1', '1'], 'delta_in': [1, -1], 'is_reg_in': [False, True]})
    result = annotate_is_reg_in(fixations)
    assert_frame_equal(result, expected)


def test_annotate_is_reg_out():
    fixations = pl.DataFrame({'trial': ['1', '1'], 'delta_out': [1, -1]})
    expected = pl.DataFrame({
        'trial': ['1', '1'], 'delta_out': [
            1, -1,
        ], 'is_reg_out': [False, True],
    })
    result = annotate_is_reg_out(fixations)
    assert_frame_equal(result, expected)


def test_annotate_is_first_fixation():
    fixations = pl.DataFrame({'trial': ['1', '1', '1'], 'word_idx': [1, 1, 2]})
    expected = pl.DataFrame({
        'trial': ['1', '1', '1'],
        'word_idx': [1, 1, 2],
        'is_first_fix': [True, False, True],
    })
    result = annotate_is_first_fixation(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_is_first_pass():
    fixations = pl.DataFrame({
        'trial': ['1', '1', '1', '1', '1'],
        'onset': [0, 1, 2, 3, 4],
        'word_idx': [1, 1, 2, 1, 3],
        'run_id': [1, 1, 2, 3, 4],
        'prev_word_idx': [None, 1, 1, 2, 1],
    })
    # word 1: first run (0,1) is first pass. Second run (3) is not (revisit).
    # word 2: first run (2) is first pass.
    # word 3: first run (4) is first pass.
    expected = fixations.with_columns(
        pl.Series('is_first_pass', [True, True, True, False, True]),
    )
    result = annotate_is_first_pass(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_is_first_pass_regression_skip():
    # word 1 -> word 3 -> word 2.
    # word 2 should NOT be first pass because word 3 was already seen.
    fixations = pl.DataFrame({
        'trial': ['1', '1', '1'],
        'onset': [0, 1, 2],
        'word_idx': [1, 3, 2],
        'run_id': [1, 2, 3],
        'prev_word_idx': [None, 1, 3],
    })
    expected = fixations.with_columns(
        pl.Series('is_first_pass', [True, True, False]),
    )
    result = annotate_is_first_pass(fixations, ['trial'])
    assert_frame_equal(result, expected)


def test_annotate_fixations():
    events = pl.DataFrame({
        'trial': ['1', '1', '1'],
        'stimulus': ['s', 's', 's'],
        'page': ['p', 'p', 'p'],
        'name': ['fixation', 'fixation', 'fixation'],
        'word_idx': [1, 2, 1],
        'onset': [0, 100, 200],
    })
    result = annotate_fixations(events, group_columns=['trial', 'stimulus', 'page'])

    expected_columns = [
        'trial', 'stimulus', 'page', 'name', 'word_idx', 'onset',
        'fixation_id', 'run_id', 'prev_word_idx', 'next_word_idx',
        'delta_in', 'delta_out', 'is_reg_in', 'is_reg_out',
        'is_first_fix', 'is_first_pass',
    ]
    assert all(col in result.columns for col in expected_columns)
    assert len(result) == 3
    assert result['run_id'].to_list() == [1, 2, 3]
    assert result['is_first_pass'].to_list() == [True, True, False]
