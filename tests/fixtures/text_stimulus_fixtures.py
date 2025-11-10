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
"""Shared TextStimulus test fixtures."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements.stimulus.text import TextStimulus


def _make_text_stimulus(
    df: pl.DataFrame, *, trial_col: str | None = None,
    page_col: str | None = None,
) -> TextStimulus:
    """Help to construct a TextStimulus from a minimal AOI dataframe.

    All fixtures use common column names: 'label', 'sx', 'sy', 'ex', 'ey' and optional
    'trial' / 'page'.
    """
    return TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        trial_column=trial_col,
        page_column=page_col,
    )


@pytest.fixture(name='stimulus_both_columns')
def _stimulus_both_columns() -> TextStimulus:  # noqa: D403
    """AOIs with both trial and page columns for filtering tests.

    Two AOIs with identical spatial boxes but different (trial, page) keys,
    and a third AOI at a different position.
    """
    df = pl.DataFrame(
        {
            'label': ['A1', 'A2', 'B1'],
            'sx': [0, 0, 10],
            'sy': [0, 0, 0],
            'ex': [10, 10, 20],
            'ey': [10, 10, 10],
            'trial': [1, 2, 1],
            'page': ['X', 'X', 'Y'],
        },
    )
    return _make_text_stimulus(df, trial_col='trial', page_col='page')


@pytest.fixture(name='stimulus_only_trial')
def _stimulus_only_trial() -> TextStimulus:  # noqa: D403
    """AOIs that are split by trial only (no page column)."""
    df = pl.DataFrame(
        {
            'label': ['T1', 'T2'],
            'sx': [0, 0],
            'sy': [0, 0],
            'ex': [10, 10],
            'ey': [10, 10],
            'trial': [1, 2],
        },
    )
    return _make_text_stimulus(df, trial_col='trial')


@pytest.fixture(name='stimulus_only_page')
def _stimulus_only_page() -> TextStimulus:  # noqa: D403
    """AOIs that are split by page only (no trial column)."""
    df = pl.DataFrame(
        {
            'label': ['PX', 'PY'],
            'sx': [0, 0],
            'sy': [0, 0],
            'ex': [10, 10],
            'ey': [10, 10],
            'page': ['X', 'Y'],
        },
    )
    return _make_text_stimulus(df, page_col='page')


@pytest.fixture(name='simple_stimulus')
def _simple_stimulus() -> TextStimulus:
    """Single AOI box [0,10) x [0,10) labeled 'A'."""
    df = pl.DataFrame(
        {
            'label': ['A'],
            'sx': [0],
            'sy': [0],
            'ex': [10],
            'ey': [10],
        },
    )
    return _make_text_stimulus(df)


@pytest.fixture(name='stimulus_with_trial_page')
def _stimulus_with_trial_page() -> TextStimulus:
    """Two AOIs on different (trial, page), same spatial box with labels 'TX' and 'TY'."""
    df = pl.DataFrame(
        {
            'label': ['TX', 'TY'],
            'sx': [0, 0],
            'sy': [0, 0],
            'ex': [10, 10],
            'ey': [10, 10],
            'trial': [1, 1],
            'page': ['X', 'Y'],
        },
    )
    return _make_text_stimulus(df, trial_col='trial', page_col='page')


@pytest.fixture(name='stimulus_overlap')
def _stimulus_overlap() -> TextStimulus:
    # Two overlapping AOIs covering the same box; order is deterministic.
    df = pl.DataFrame(
        {
            'label': ['L1', 'L2'],
            'sx': [0, 0],
            'sy': [0, 0],
            'ex': [10, 10],
            'ey': [10, 10],
        },
    )
    return _make_text_stimulus(df)
