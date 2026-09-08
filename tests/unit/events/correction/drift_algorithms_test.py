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

import numpy as np
import polars as pl
import pytest

import pymovements.events.correction.drift_algorithms as da


def make_location_frame(x_values, y_values):
    """Build a fixation frame with a 'location' column of [x, y] lists."""
    return pl.DataFrame({'x': x_values, 'y': y_values}).select(
        pl.concat_list(['x', 'y']).alias('location'),
    )


def make_word_locations(x_values, y_values):
    """Build a series of [x, y] word locations."""
    return make_location_frame(x_values, y_values).to_series()


@pytest.fixture
def sample_fixations_and_lines():
    """Return a sample fixation frame and line Y midlines."""
    line_ys = pl.Series([100.0, 200.0, 300.0])
    x_values = np.tile(np.linspace(100, 500, 4), 3)
    y_values = np.repeat([105.0, 195.0, 302.0], 4)
    return make_location_frame(x_values, y_values), line_ys


def corrected_y(fixations, expression):
    """Evaluate a drift correction expression on a fixation frame."""
    return fixations.select(expression).to_series().to_list()


def test_attach(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.attach(line_ys))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_chain(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.chain(line_ys, x_thresh=192, y_thresh=32))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_cluster(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.cluster(line_ys))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_compare(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    word_locations = make_word_locations(
        np.tile(np.linspace(100, 500, 4), 3), np.repeat(line_ys.to_numpy(), 4),
    )
    res = corrected_y(
        fixations, da.compare(word_locations, x_thresh=300, n_nearest_lines=2),
    )
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_compare_clamps_n_nearest_lines_on_two_line_text():
    """Compare with default n_nearest_lines must handle texts with fewer lines."""
    fixations = make_location_frame(
        [100.0, 800.0, 100.0, 800.0], [105.0, 102.0, 198.0, 201.0],
    )
    word_locations = make_word_locations(
        [100.0, 800.0, 100.0, 800.0], [100.0, 100.0, 200.0, 200.0],
    )
    res = corrected_y(fixations, da.compare(word_locations))
    assert res == [100.0, 100.0, 200.0, 200.0]


def test_compare_single_line_text():
    fixations = make_location_frame(np.linspace(100, 900, 5), np.full(5, 105.0))
    word_locations = make_word_locations(np.linspace(100, 900, 5), np.full(5, 100.0))
    res = corrected_y(fixations, da.compare(word_locations))
    assert res == [100.0] * 5


def test_compare_invalid_n_nearest_lines_raises():
    word_locations = make_word_locations([100.0], [100.0])
    with pytest.raises(ValueError, match='n_nearest_lines must be at least 1'):
        da.compare(word_locations, n_nearest_lines=0)


@pytest.mark.parametrize(
    ('directionality', 'expected'),
    [
        pytest.param(
            'left-to-right',
            [100.0] * 4 + [200.0] * 4 + [300.0] * 4,
            id='left-to-right',
        ),
        # For RTL every within-line step is a sequence boundary, so the twelve
        # one-fixation sequences must be merged back; only same-line pairs pass the
        # gradient constraint (any cross-line pair has |gradient| >= 90 / 400 >
        # g_thresh), recovering the lines.
        pytest.param(
            'right-to-left',
            [100.0] * 4 + [200.0] * 4 + [300.0] * 4,
            id='right-to-left',
        ),
    ],
)
def test_merge(sample_fixations_and_lines, directionality, expected):
    # With filterwarnings=error this test also asserts that merge does not leak
    # RankWarnings from poorly conditioned two-fixation line fits.
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.merge(line_ys, directionality=directionality))
    assert res == expected


def test_regress(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.regress(line_ys))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_segment_ltr(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.segment(line_ys, directionality='left-to-right'))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_segment_rtl(sample_fixations_and_lines):
    # On this left-to-right fixture the RTL return sweep candidates are the within-line
    # x-steps, which differ from each other only at floating point rounding level (they
    # come from np.linspace), so which two rank largest is not hand-derivable; the exact
    # output is therefore not pinned here (see test_segment_single_line for a pinned
    # RTL case).
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.segment(line_ys, directionality='right-to-left'))
    assert len(res) == fixations.height


@pytest.mark.parametrize(
    ('directionality', 'x_values', 'expected'),
    [
        pytest.param(
            'left-to-right',
            np.linspace(100, 500, 5),
            [100.0] * 5,
            id='left-to-right',
        ),
        pytest.param(
            'right-to-left',
            np.linspace(500, 100, 5),
            [100.0] * 5,
            id='right-to-left',
        ),
    ],
)
def test_segment_single_line(directionality, x_values, expected):
    """Segment must assign all fixations to the single line, also for RTL reading."""
    line_ys = pl.Series([100.0])
    fixations = make_location_frame(x_values, np.full(5, 105.0))
    res = corrected_y(fixations, da.segment(line_ys, directionality=directionality))
    assert res == expected


@pytest.mark.parametrize(
    ('directionality', 'expected'),
    [
        pytest.param(
            'left-to-right',
            [100.0] * 4 + [200.0] * 4 + [300.0] * 4,
            id='left-to-right',
        ),
        # For RTL the nine positive within-line x-steps are classified as return sweeps
        # and the two -400 line changes are not, yielding segments 0-4, 5-8 and 9-11:
        # segment means 105, 105, 105, 150, 195 then 195, 248.5, 302 then 302, 302, 302
        # snap to 100 (a 150 tie snaps to the first, upper line), 200 and 300.
        pytest.param(
            'right-to-left',
            [100.0] * 5 + [200.0] * 4 + [300.0] * 3,
            id='right-to-left',
        ),
    ],
)
def test_split(sample_fixations_and_lines, directionality, expected):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.split(line_ys, directionality=directionality))
    assert res == expected


def test_stretch(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    res = corrected_y(fixations, da.stretch(line_ys))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_warp(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    word_locations = make_word_locations(
        np.tile(np.linspace(100, 500, 4), 3), np.repeat(line_ys.to_numpy(), 4),
    )
    res = corrected_y(fixations, da.warp(word_locations))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_slice(sample_fixations_and_lines):
    fixations, line_ys = sample_fixations_and_lines
    # The fixture's 100 px line spacing meets the default n_thresh=90 boundary, so no run
    # merges into an adjacent proto line and an empty proto line remains above the first
    # line's fixations: every run is assigned one line too far down.
    res_default = corrected_y(fixations, da.slice(line_ys))
    assert res_default == [200.0] * 4 + [300.0] * 8

    # With n_thresh above the line spacing the runs merge onto their own lines.
    res = corrected_y(fixations, da.slice(line_ys, n_thresh=150))
    assert res == [100.0] * 4 + [200.0] * 4 + [300.0] * 4


def test_slice_prune_proto_lines():
    """Test slice algorithm when pruning proto lines is required."""
    line_ys = pl.Series([200.0])
    x_values = np.concatenate([
        np.linspace(100, 500, 10), np.linspace(100, 200, 2), np.linspace(100, 300, 4),
    ])
    y_values = np.concatenate([np.full(10, 200.0), np.full(2, 150.0), np.full(4, 250.0)])
    fixations = make_location_frame(x_values, y_values)

    res = corrected_y(fixations, da.slice(line_ys))
    assert res == [200.0] * fixations.height


def test_slice_phantom_and_current_merge():
    """Test slice algorithm merging current proto line and using phantom proto lines."""
    line_ys = pl.Series([200.0, 300.0])
    x_values = np.concatenate([
        np.linspace(100, 500, 10), np.linspace(100, 200, 5),
        np.linspace(100, 300, 5), np.linspace(100, 400, 5),
    ])
    y_values = np.concatenate([
        np.full(10, 200.0), np.full(5, 205.0), np.full(5, 250.0), np.full(5, 380.0),
    ])
    fixations = make_location_frame(x_values, y_values)

    res = corrected_y(fixations, da.slice(line_ys))
    assert len(res) == fixations.height


def test_location_expression_argument(sample_fixations_and_lines):
    """Algorithms accept a location expression instead of a column name."""
    _, line_ys = sample_fixations_and_lines
    fixations = pl.DataFrame({
        'location_x': [100.0, 200.0],
        'location_y': [105.0, 198.0],
    })
    location = pl.concat_list([pl.col('location_x'), pl.col('location_y')])
    res = corrected_y(fixations, da.attach(line_ys, location=location))
    assert res == [100.0, 200.0]


def test_wisdom_of_the_crowd():
    votes = pl.DataFrame({
        'a': [100.0, 200.0, 300.0],
        'b': [100.0, 200.0, 300.0],
        'c': [100.0, 150.0, 300.0],
    })
    res = votes.select(da.wisdom_of_the_crowd(['a', 'b', 'c'])).to_series().to_list()
    assert res == [100.0, 200.0, 300.0]


def test_wisdom_of_the_crowd_tie():
    """Test wisdom_of_the_crowd when there is a tie between candidates."""
    votes = pl.DataFrame({'a': [100.0], 'b': [200.0]})
    res = votes.select(da.wisdom_of_the_crowd(['a', 'b'])).to_series().to_list()
    assert res == [100.0]

    votes = pl.DataFrame({
        'a': [100.0], 'b': [200.0], 'c': [200.0], 'd': [300.0], 'e': [300.0],
    })
    res = votes.select(
        da.wisdom_of_the_crowd(['a', 'b', 'c', 'd', 'e']),
    ).to_series().to_list()
    assert res == [200.0]


def test_dynamic_time_warping():
    seq1 = make_word_locations([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    seq2 = make_word_locations([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert cost == 0.0
    assert len(path) == len(seq1)

    # Test unequal sequence length and non-diagonal backtrack paths
    seq1 = make_word_locations([0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 2.0])
    seq2 = make_word_locations([0.0, 2.0], [0.0, 2.0])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert len(path) == len(seq1)

    # Numeric one-dimensional sequences are supported as well.
    seq1 = pl.Series([0.0, 2.0])
    seq2 = pl.Series([0.0, 1.0, 1.5, 2.0])
    cost, path = da.dynamic_time_warping(seq1, seq2)
    assert len(path) == len(seq1)
