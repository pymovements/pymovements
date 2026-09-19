# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test TextStimulus.with_line_idx()."""
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_series_equal

from pymovements.stimulus import TextStimulus


def make_stimulus(aois: pl.DataFrame, **kwargs: Any) -> TextStimulus:
    """Build a TextStimulus over character boxes given as x0/y0/x1/y1.

    Parameters
    ----------
    aois: pl.DataFrame
        AOI frame with the columns ``char``, ``x0``, ``y0``, ``x1`` and ``y1``.
    **kwargs: Any
        Passed through to :py:class:`~pymovements.stimulus.TextStimulus`.

    Returns
    -------
    TextStimulus
        The stimulus under test.

    """
    return TextStimulus(
        aois,
        aoi_column='char',
        start_x_column='x0',
        start_y_column='y0',
        end_x_column='x1',
        end_y_column='y1',
        **kwargs,
    )


def grid(rows: list[list[float]], height: float = 10.0) -> pl.DataFrame:
    """Build an AOI frame from one list of top-y values per intended line.

    Parameters
    ----------
    rows: list[list[float]]
        One list of ``start_y`` values per line, one entry per character of that line.
    height: float
        Box height used for every character. (default: 10.0)

    Returns
    -------
    pl.DataFrame
        AOI frame with the columns ``char``, ``x0``, ``y0``, ``x1`` and ``y1``.

    """
    chars: list[str] = []
    x0: list[float] = []
    y0: list[float] = []
    x1: list[float] = []
    y1: list[float] = []
    for line in rows:
        for index, top in enumerate(line):
            chars.append(f'c{len(chars)}')
            x0.append(float(index * 10))
            x1.append(float(index * 10 + 10))
            y0.append(float(top))
            y1.append(float(top + height))
    return pl.DataFrame({'char': chars, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def test_with_line_idx_numbers_lines_from_the_top():
    stimulus = make_stimulus(grid([[40, 40], [0, 0], [20, 20]]))

    assert stimulus.with_line_idx().aois['line_idx'].to_list() == [2, 2, 0, 0, 1, 1]


def test_with_line_idx_groups_boxes_of_unequal_height_on_one_line():
    # PoTeC's first and last line have 78/79px boxes where the rest has 82px. Grouping on the
    # exact top edge would split such a line in two; grouping on the centre must not.
    aois = pl.DataFrame({
        'char': ['a', 'b', 'c'],
        'x0': [0.0, 10.0, 20.0],
        'y0': [0.0, 1.0, 2.0],
        'x1': [10.0, 20.0, 30.0],
        'y1': [10.0, 9.0, 12.0],
    })

    assert make_stimulus(aois).with_line_idx().aois['line_idx'].to_list() == [0, 0, 0]


def test_with_line_idx_separates_lines_across_a_blank_line():
    stimulus = make_stimulus(grid([[0, 0], [40, 40]]))

    assert stimulus.with_line_idx().aois['line_idx'].to_list() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ('tolerance', 'expected'),
    [
        pytest.param(0.5, [0, 0, 1, 1], id='default_splits'),
        pytest.param(2.0, [0, 0, 0, 0], id='large_tolerance_merges'),
    ],
)
def test_with_line_idx_tolerance_is_measured_in_line_heights(tolerance, expected):
    stimulus = make_stimulus(grid([[0, 0], [12, 12]]))

    line_idx = stimulus.with_line_idx(tolerance=tolerance).aois['line_idx']

    assert line_idx.to_list() == expected


def test_with_line_idx_keeps_and_casts_an_existing_column():
    aois = grid([[0, 0], [20, 20]]).with_columns(pl.Series('line_idx', [1, 1, 0, 0]))

    line_idx = make_stimulus(aois).with_line_idx().aois['line_idx']

    assert_series_equal(line_idx, pl.Series('line_idx', [1, 1, 0, 0], dtype=pl.Int64))


def test_with_line_idx_overwrite_recomputes_an_existing_column():
    aois = grid([[0, 0], [20, 20]]).with_columns(pl.Series('line_idx', [1, 1, 0, 0]))

    line_idx = make_stimulus(aois).with_line_idx(overwrite=True).aois['line_idx']

    assert line_idx.to_list() == [0, 0, 1, 1]


def test_with_line_idx_uses_a_custom_column_name():
    stimulus = make_stimulus(grid([[0, 0], [20, 20]]))

    aois = stimulus.with_line_idx(line_column='line').aois

    assert aois['line'].to_list() == [0, 0, 1, 1]
    assert 'line_idx' not in aois.columns


def test_with_line_idx_does_not_mutate_the_original():
    stimulus = make_stimulus(grid([[0, 0], [20, 20]]))

    stimulus.with_line_idx()

    assert 'line_idx' not in stimulus.aois.columns


def test_with_line_idx_keeps_metadata():
    stimulus = make_stimulus(grid([[0, 0]]), metadata={'source': 'test'})

    assert stimulus.with_line_idx().metadata == {'source': 'test'}


def test_with_line_idx_raises_on_gapped_existing_column():
    aois = grid([[0, 0], [20, 20]]).with_columns(pl.Series('line_idx', [0, 0, 2, 2]))

    with pytest.raises(ValueError, match='must be numbered 0..k-1 without gaps'):
        make_stimulus(aois).with_line_idx()


def test_with_line_idx_raises_on_several_pages():
    aois = grid([[0, 0], [20, 20]]).with_columns(pl.Series('page', [0, 0, 1, 1]))

    with pytest.raises(ValueError, match='several values'):
        make_stimulus(aois, page_column='page').with_line_idx()


def test_with_line_idx_accepts_a_single_page():
    aois = grid([[0, 0], [20, 20]]).with_columns(pl.Series('page', [3, 3, 3, 3]))

    line_idx = make_stimulus(aois, page_column='page').with_line_idx().aois['line_idx']

    assert line_idx.to_list() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ('tolerance', 'message'),
    [
        pytest.param(0.0, 'tolerance must be positive', id='zero'),
        pytest.param(-1.0, 'tolerance must be positive', id='negative'),
    ],
)
def test_with_line_idx_raises_on_invalid_tolerance(tolerance, message):
    stimulus = make_stimulus(grid([[0, 0]]))

    with pytest.raises(ValueError, match=message):
        stimulus.with_line_idx(tolerance=tolerance)


def test_with_line_idx_raises_on_empty_aois():
    empty = grid([[0]]).clear()

    with pytest.raises(ValueError, match='empty AOI dataframe'):
        make_stimulus(empty).with_line_idx()


def test_with_line_idx_raises_when_no_box_is_usable():
    """Every rectangle dropped means nothing to derive: say which, then say why."""
    aois = pl.DataFrame({
        'char': ['a', 'b'],
        'x0': [0.0, 10.0], 'y0': [0.0, 0.0],
        'x1': [10.0, 20.0], 'y1': [0.0, 0.0],
    })

    with pytest.warns(UserWarning, match='2 of 2 areas of interest'):
        with pytest.raises(ValueError, match='empty AOI dataframe'):
            make_stimulus(aois).with_line_idx()


def test_with_line_idx_works_from_width_and_height_columns():
    aois = pl.DataFrame({
        'char': ['a', 'b'],
        'x0': [0.0, 0.0], 'y0': [0.0, 20.0],
        'w': [10.0, 10.0], 'h': [10.0, 10.0],
    })
    stimulus = TextStimulus(
        aois, aoi_column='char',
        start_x_column='x0', start_y_column='y0',
        width_column='w', height_column='h',
    )

    assert stimulus.with_line_idx().aois['line_idx'].to_list() == [0, 1]


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        pytest.param(
            {'start_x_column': 'x0', 'start_y_column': 'y0', 'end_y_column': 'y1'},
            'TextStimulus.width or TextStimulus.end_x_column',
            id='no_x_bounds',
        ),
        pytest.param(
            {'start_x_column': 'x0', 'start_y_column': 'y0', 'end_x_column': 'x1'},
            'TextStimulus.height or TextStimulus.end_y_column',
            id='no_y_bounds',
        ),
    ],
)
def test_with_line_idx_raises_without_bounds(kwargs, message):
    stimulus = TextStimulus(grid([[0, 0]]), aoi_column='char', **kwargs)

    with pytest.raises(ValueError, match=message):
        stimulus.with_line_idx()


@pytest.mark.parametrize(
    ('broken_y0', 'broken_y1', 'reason'),
    [
        pytest.param(None, None, 'null coordinates', id='null'),
        pytest.param(float('nan'), float('nan'), 'NaN coordinates', id='nan'),
        pytest.param(0.0, -5.0, 'a height of zero or less', id='non_positive_height'),
    ],
)
def test_with_line_idx_drops_unusable_boxes_instead_of_inventing_a_line(
        broken_y0, broken_y1, reason,
):
    """An unusable rectangle must not take part in the grouping, and must not shift the rest.

    Before this was handled, each of the three cases silently produced a line that is not on
    the page: null joined line 0, NaN and a negative height each opened one of their own, and
    the count went from two lines to three. The count feeds ``_min_fixation_count()`` and the
    number of clusters ``cluster`` fits, so it reaches the correction algorithms.
    """
    aois = pl.DataFrame({
        'char': ['a', 'b', 'c', 'd'],
        'x0': [0.0, 10.0, 0.0, 10.0],
        'y0': [0.0, broken_y0, 20.0, 20.0],
        'x1': [10.0, 20.0, 10.0, 20.0],
        'y1': [10.0, broken_y1, 30.0, 30.0],
    })
    stimulus = TextStimulus(
        aois, aoi_column='char',
        start_x_column='x0', start_y_column='y0', end_x_column='x1', end_y_column='y1',
    )

    with pytest.warns(UserWarning, match='1 of 4 areas of interest'):
        line_idx = stimulus.with_line_idx().aois['line_idx']

    assert line_idx.to_list() == [0, None, 1, 1], reason
    assert line_idx.drop_nulls().n_unique() == 2
