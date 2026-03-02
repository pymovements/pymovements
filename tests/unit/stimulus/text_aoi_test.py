# Copyright (c) 2025-2026 The pymovements Project Authors
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

from math import nan
from typing import Any

import polars as pl
import pytest

from pymovements.stimulus.text import TextStimulus


@pytest.fixture(name='text_stimulus_overlap')
def fixture_text_stimulus_overlap() -> TextStimulus:
    """Text stimulus with overlapping AOIs."""
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
    return TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        width_column='width',
        height_column='height',
    )


def test_get_aoi_overlap_warns(text_stimulus_overlap: TextStimulus) -> None:
    """Overlap handling for width/height-configured stimulus warns."""
    row = {'x': 5, 'y': 5}
    with pytest.warns(
        UserWarning,
        match=r'Multiple AOIs matched this point \(x=',
    ):
        out = text_stimulus_overlap.get_aoi(row=row, x_eye='x', y_eye='y')

    labels = out.get_column('label').to_list()
    assert len(labels) == 2
    assert labels[0] == 'W1'


def test_get_aoi_overlap_no_warning(text_stimulus_overlap: TextStimulus) -> None:
    """Overlap handling for width/height-configured stimulus without overlap."""
    row = {'x': 15, 'y': 5}
    out = text_stimulus_overlap.get_aoi(row=row, x_eye='x', y_eye='y')

    labels = out.get_column('label').to_list()
    assert len(labels) == 1
    assert labels[0] is None


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


@pytest.mark.parametrize(
    ('mode', 'x', 'y'),
    [
        pytest.param('width_height', None, 0, id='width_height-x-none'),
        pytest.param('width_height', 0, None, id='width_height-y-none'),
        pytest.param('width_height', 'bad', 0, id='width_height-x-str'),
        pytest.param('width_height', 0, 'bad', id='width_height-y-str'),
        pytest.param('width_height', nan, 0, id='width_height-x-nan'),
        pytest.param('width_height', 0, nan, id='width_height-y-nan'),
        pytest.param('end', None, 0, id='end-x-none'),
        pytest.param('end', 0, None, id='end-y-none'),
        pytest.param('end', 'bad', 0, id='end-x-str'),
        pytest.param('end', 0, 'bad', id='end-y-str'),
        pytest.param('end', nan, 0, id='end-x-nan'),
        pytest.param('end', 0, nan, id='end-y-nan'),
    ],
)
def test_get_aoi_invalid_coordinates_warns_and_returns_none(mode: str, x: Any, y: Any) -> None:
    """Invalid/non-numeric coords should emit a warning and return a single None row.

    This covers both AOI specification modes: width/height and end_x/end_y.
    """
    if mode == 'width_height':
        df = pl.DataFrame(
            {
                'label': ['A'],
                'sx': [0],
                'sy': [0],
                'width': [10],
                'height': [10],
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
    else:
        df = pl.DataFrame(
            {
                'label': ['A'],
                'sx': [0],
                'sy': [0],
                'ex': [10],
                'ey': [10],
            },
        )
        stim = TextStimulus(
            aois=df,
            aoi_column='label',
            start_x_column='sx',
            start_y_column='sy',
            end_x_column='ex',
            end_y_column='ey',
        )

    with pytest.warns(UserWarning, match='Invalid eye coordinates'):
        out = stim.get_aoi(row={'x': x, 'y': y}, x_eye='x', y_eye='y')

    assert out.height == 1
    assert out.select(stim.aoi_column).item() is None


@pytest.mark.parametrize('row_has_trial_key,trial_value', [(False, None), (True, None)])
def test_get_aoi_skips_trial_filter_when_row_trial_missing_or_none(
    stimulus_with_trials: TextStimulus,
    row_has_trial_key: bool,
    trial_value: float,
) -> None:
    # Row targets coordinates inside AOI 'A'. We either omit the 'trial' key or set it to None.
    row = {'__x': 5.0, '__y': 5.0}
    if row_has_trial_key:
        row['trial'] = trial_value

    aoi = stimulus_with_trials.get_aoi(row=row, x_eye='__x', y_eye='__y')

    # Without an effective trial value, AOIs are not pre-filtered - the spatial match
    # should still return the single matching AOI 'A'.
    assert aoi.select('label').item() == 'A'
