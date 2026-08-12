# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Tests for vertical drift correction algorithms in `drift_algorithms.py`."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pymovements.events.correction.drift_algorithms as da


@pytest.fixture
def sample_fixations_and_lines():
    """Return sample 2D fixation coordinates and line Y midlines."""
    line_Y = np.array([100.0, 200.0, 300.0])
    fixations_line1 = np.column_stack([np.linspace(100, 500, 4), np.full(4, 105.0)])
    fixations_line2 = np.column_stack([np.linspace(100, 500, 4), np.full(4, 195.0)])
    fixations_line3 = np.column_stack([np.linspace(100, 500, 4), np.full(4, 302.0)])
    fixations_XY = np.vstack([fixations_line1, fixations_line2, fixations_line3])
    return fixations_XY, line_Y


def test_attach(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.attach(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape
    np.testing.assert_array_equal(res[:4, 1], 100.0)
    np.testing.assert_array_equal(res[4:8, 1], 200.0)
    np.testing.assert_array_equal(res[8:, 1], 300.0)


def test_chain(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.chain(fixations_XY, line_Y, x_thresh=192, y_thresh=32)
    assert res.shape == fixations_XY.shape
    np.testing.assert_array_equal(res[:4, 1], 100.0)
    np.testing.assert_array_equal(res[4:8, 1], 200.0)


def test_cluster(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.cluster(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape
    np.testing.assert_array_equal(res[:4, 1], 100.0)
    np.testing.assert_array_equal(res[4:8, 1], 200.0)
    np.testing.assert_array_equal(res[8:, 1], 300.0)


def test_compare(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    word_XY = np.column_stack([np.tile(np.linspace(100, 500, 4), 3), np.repeat(line_Y, 4)])
    res = da.compare(fixations_XY, word_XY, x_thresh=300, n_nearest_lines=2)
    assert res.shape == fixations_XY.shape


def test_compare_clamps_n_nearest_lines_on_two_line_text():
    """Compare with default n_nearest_lines must handle texts with fewer lines."""
    fixations_XY = np.array([
        [100.0, 105.0], [800.0, 102.0], [100.0, 198.0], [800.0, 201.0],
    ])
    word_XY = np.array([
        [100.0, 100.0], [800.0, 100.0], [100.0, 200.0], [800.0, 200.0],
    ])
    res = da.compare(fixations_XY, word_XY)
    np.testing.assert_array_equal(res[:2, 1], 100.0)
    np.testing.assert_array_equal(res[2:, 1], 200.0)


def test_compare_single_line_text():
    fixations_XY = np.column_stack([np.linspace(100, 900, 5), np.full(5, 105.0)])
    word_XY = np.column_stack([np.linspace(100, 900, 5), np.full(5, 100.0)])
    res = da.compare(fixations_XY, word_XY)
    np.testing.assert_array_equal(res[:, 1], 100.0)


def test_compare_invalid_n_nearest_lines_raises():
    fixations_XY = np.array([[100.0, 105.0]])
    word_XY = np.array([[100.0, 100.0]])
    with pytest.raises(ValueError, match='n_nearest_lines must be at least 1'):
        da.compare(fixations_XY, word_XY, n_nearest_lines=0)


def test_merge_ltr_and_rtl(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    rank_warning = np.RankWarning if hasattr(np, 'RankWarning') else np.exceptions.RankWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rank_warning)
        res_ltr = da.merge(fixations_XY, line_Y, text_right_to_left=False)
        res_rtl = da.merge(fixations_XY, line_Y, text_right_to_left=True)
    assert res_ltr.shape == fixations_XY.shape
    assert res_rtl.shape == fixations_XY.shape


def test_regress(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.regress(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape
    np.testing.assert_array_equal(res[:4, 1], 100.0)


def test_segment_ltr_and_rtl(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res_ltr = da.segment(fixations_XY, line_Y, text_right_to_left=False)
    res_rtl = da.segment(fixations_XY, line_Y, text_right_to_left=True)
    assert res_ltr.shape == fixations_XY.shape
    assert res_rtl.shape == fixations_XY.shape


def test_segment_single_line_ltr_and_rtl():
    """Segment must assign all fixations to the single line, also for RTL reading."""
    line_Y = np.array([100.0])

    fixations_ltr = np.column_stack([np.linspace(100, 500, 5), np.full(5, 105.0)])
    res_ltr = da.segment(fixations_ltr, line_Y)
    np.testing.assert_array_equal(res_ltr[:, 1], 100.0)

    fixations_rtl = np.column_stack([np.linspace(500, 100, 5), np.full(5, 105.0)])
    res_rtl = da.segment(fixations_rtl, line_Y, text_right_to_left=True)
    np.testing.assert_array_equal(res_rtl[:, 1], 100.0)


def test_split_ltr_and_rtl(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res_ltr = da.split(fixations_XY, line_Y, text_right_to_left=False)
    res_rtl = da.split(fixations_XY, line_Y, text_right_to_left=True)
    assert res_ltr.shape == fixations_XY.shape
    assert res_rtl.shape == fixations_XY.shape


def test_stretch(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.stretch(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape


def test_warp(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    word_XY = np.column_stack([np.tile(np.linspace(100, 500, 4), 3), np.repeat(line_Y, 4)])
    res = da.warp(fixations_XY, word_XY)
    assert res.shape == fixations_XY.shape


def test_slice(sample_fixations_and_lines):
    fixations_XY, line_Y = sample_fixations_and_lines
    res = da.slice(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape


def test_slice_prune_proto_lines():
    """Test slice algorithm when pruning proto lines is required."""
    line_Y = np.array([200.0])
    fixations_run1 = np.column_stack([np.linspace(100, 500, 10), np.full(10, 200.0)])
    fixations_run2 = np.column_stack([np.linspace(100, 200, 2), np.full(2, 150.0)])
    fixations_run3 = np.column_stack([np.linspace(100, 300, 4), np.full(4, 250.0)])
    fixations_XY = np.vstack([fixations_run1, fixations_run2, fixations_run3])

    res = da.slice(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape
    np.testing.assert_array_equal(res[:, 1], 200.0)


def test_slice_phantom_and_current_merge():
    """Test slice algorithm merging current proto line and using phantom proto lines."""
    line_Y = np.array([200.0, 300.0])
    fixations_run1 = np.column_stack([np.linspace(100, 500, 10), np.full(10, 200.0)])
    fixations_run2 = np.column_stack([np.linspace(100, 200, 5), np.full(5, 205.0)])
    fixations_run3 = np.column_stack([np.linspace(100, 300, 5), np.full(5, 250.0)])
    fixations_run4 = np.column_stack([np.linspace(100, 400, 5), np.full(5, 380.0)])
    fixations_XY = np.vstack([fixations_run1, fixations_run2, fixations_run3, fixations_run4])

    res = da.slice(fixations_XY, line_Y)
    assert res.shape == fixations_XY.shape


def test_wisdom_of_the_crowd():
    assign1 = [100.0, 200.0, 300.0]
    assign2 = [100.0, 200.0, 300.0]
    assign3 = [100.0, 150.0, 300.0]
    res = da.wisdom_of_the_crowd([assign1, assign2, assign3])
    assert res == [100.0, 200.0, 300.0]


def test_wisdom_of_the_crowd_tie():
    """Test wisdom_of_the_crowd when there is a tie between candidates."""
    res = da.wisdom_of_the_crowd([[100.0], [200.0]])
    assert res == [100.0]

    assign1 = [100.0]
    assign2 = [200.0]
    assign3 = [200.0]
    assign4 = [300.0]
    assign5 = [300.0]
    res = da.wisdom_of_the_crowd([assign1, assign2, assign3, assign4, assign5])
    assert res == [200.0]


def test_dynamic_time_warping():
    seq1 = np.array([[0, 0], [1, 1], [2, 2]])
    seq2 = np.array([[0, 0], [1, 1], [2, 2]])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert cost == 0.0
    assert len(path) == len(seq1)

    # Test unequal sequence length and non-diagonal backtrack paths
    seq1 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    seq2 = np.array([[0.0, 0.0], [2.0, 2.0]])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert len(path) == len(seq1)

    seq1 = np.array([[0.0, 0.0], [2.0, 2.0]])
    seq2 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert len(path) == len(seq1)
