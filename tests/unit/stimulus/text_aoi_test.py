# Copyright (c) 2025 The pymovements Project Authors
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
"""Test AOI functionality."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements.stimulus.text import TextStimulus


@pytest.mark.parametrize(
    ('x', 'y', 'expected'),
    [
        pytest.param(5, 5, 'L1', id='inside_overlap_picks_first'),
        pytest.param(15, 5, None, id='outside_none'),
    ],
)
def test_get_aoi_overlap_warns_and_picks_first(
    stimulus_overlap: TextStimulus, x: int, y: int, expected: str | None,
) -> None:
    row = {'__x': x, '__y': y}
    if expected is not None:
        with pytest.warns(UserWarning, match='Multiple AOIs matched'):
            out = stimulus_overlap.get_aoi(row=row, x_eye='__x', y_eye='__y')
    else:
        out = stimulus_overlap.get_aoi(row=row, x_eye='__x', y_eye='__y')

    labels = out.get_column('label').to_list()
    assert len(labels) == 1
    assert labels[0] == expected


@pytest.mark.parametrize(
    ('x', 'y', 'expected', 'expect_warning'),
    [
        pytest.param(5, 5, 'W1', True, id='inside_overlap_warns_and_picks_first'),
        pytest.param(15, 5, None, False, id='outside_no_warning'),
    ],
)
def test_get_aoi_overlap_warns_and_picks_first_width_height(
    x: int, y: int, expected: str | None, expect_warning: bool,
) -> None:
    """Overlap handling for width/height-configured stimulus."""
    df = pl.DataFrame(
        {
            'label': ['W1', 'W2'],
            'sx': [0, 0],
            'sy': [0, 0],
            # Two AOIs of identical size/position -> complete overlap
            'width': [10, 10],
            'height': [10, 10],
        },
    )
    stim = TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        width_column='width',
        height_column='height',
    )

    row = {'x': x, 'y': y}
    if expect_warning:
        with pytest.warns(
                UserWarning, match='Multiple AOIs matched this point; selecting the first',
        ):
            out = stim.get_aoi(row=row, x_eye='x', y_eye='y')
    else:
        out = stim.get_aoi(row=row, x_eye='x', y_eye='y')

    # Always exactly one output row
    assert out.height == 1
    label = out.select('label').item()
    assert label == expected


@pytest.mark.parametrize(
    ('row'),
    [
        pytest.param({'x': 0, 'y': 0}, id='origin'),
        pytest.param({'x': 5, 'y': 5}, id='middle'),
    ],
)
def test_get_aoi_raises_value_error_when_no_size_columns(row: dict[str, int]) -> None:
    """If neither width/height nor end_x/end_y are set, get_aoi must raise ValueError."""
    df = pl.DataFrame(
        {
            'label': ['Z'],
            'sx': [0],
            'sy': [0],
            # no width/height and no end columns
        },
    )
    stim = TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        width_column=None,
        height_column=None,
        end_x_column=None,
        end_y_column=None,
    )

    with pytest.raises(
            ValueError,
            match='either TextStimulus.width or TextStimulus.end_x_column must be defined',
    ):
        _ = stim.get_aoi(row=row, x_eye='x', y_eye='y')
