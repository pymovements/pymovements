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
"""Test Events.map_to_lines()."""
import numpy as np
import polars as pl
import pytest

import pymovements as pm
from pymovements.stimulus import TextStimulus


def page(
    n_lines: int = 3, n_chars: int = 5, width: float = 10.0, height: float = 10.0,
    pitch: float = 20.0,
) -> TextStimulus:
    """Build a rectangular page of character AOIs.

    Parameters
    ----------
    n_lines: int
        Number of text lines. (default: 3)
    n_chars: int
        Characters per line. (default: 5)
    width: float
        Character box width. (default: 10.0)
    height: float
        Character box height. (default: 10.0)
    pitch: float
        Vertical distance between the tops of consecutive lines. (default: 20.0)

    Returns
    -------
    TextStimulus
        A stimulus whose AOIs already carry ``line_idx``.

    """
    chars: list[str] = []
    x0: list[float] = []
    y0: list[float] = []
    x1: list[float] = []
    y1: list[float] = []
    line: list[int] = []
    for line_index in range(n_lines):
        for char_index in range(n_chars):
            chars.append(f'{line_index}_{char_index}')
            x0.append(char_index * width)
            x1.append(char_index * width + width)
            y0.append(line_index * pitch)
            y1.append(line_index * pitch + height)
            line.append(line_index)
    aois = pl.DataFrame({
        'char': chars, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'line_idx': line,
    })
    return TextStimulus(
        aois, aoi_column='char',
        start_x_column='x0', start_y_column='y0',
        end_x_column='x1', end_y_column='y1',
    )


def fixations(xy: list[tuple[float, float]], names: list[str] | None = None) -> pm.Events:
    """Build an Events frame of fixations at the given coordinates.

    Parameters
    ----------
    xy: list[tuple[float, float]]
        One ``(x, y)`` pair per event.
    names: list[str] | None
        Event names; all ``'fixation'`` when None. (default: None)

    Returns
    -------
    pm.Events
        The events under test.

    """
    names = names if names is not None else ['fixation'] * len(xy)
    return pm.Events(
        pl.DataFrame({
            'name': names,
            'onset': list(range(len(xy))),
            'offset': list(range(1, len(xy) + 1)),
            'location_x': [float(x) for x, _ in xy],
            'location_y': [float(y) for _, y in xy],
        }),
    )


def test_map_to_lines_contain_assigns_the_line_of_the_hit_aoi():
    events = fixations([(5, 5), (5, 25), (45, 45)])

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, 1, 2]


def test_map_to_lines_contain_leaves_a_miss_null():
    # y=15 is in the gap between line 0 (0..10) and line 1 (20..30).
    events = fixations([(5, 15), (-100, 5), (5, 5)])

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [None, None, 0]


def test_map_to_lines_uses_half_open_bounds():
    # The box of line 0 spans y in [0, 10); 10.0 already belongs to the gap.
    events = fixations([(0, 0), (0, 10), (50, 5)])

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, None, None]


def test_map_to_lines_ignores_non_fixations():
    events = fixations([(5, 5), (5, 25)], names=['fixation', 'saccade'])

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, None]


def test_map_to_lines_accepts_a_fixation_name_prefix():
    events = fixations([(5, 5), (5, 25)], names=['fixation.ivt', 'blink'])

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, None]


def test_map_to_lines_derives_the_line_index_when_missing():
    stimulus = page()
    without = TextStimulus(
        stimulus.aois.drop('line_idx'), aoi_column='char',
        start_x_column='x0', start_y_column='y0',
        end_x_column='x1', end_y_column='y1',
    )
    events = fixations([(5, 5), (5, 25), (5, 45)])

    events.map_to_lines(without)

    assert events.frame['line_idx'].to_list() == [0, 1, 2]


def test_map_to_lines_reads_a_location_list_column():
    events = pm.Events(
        pl.DataFrame({
            'name': ['fixation', 'fixation'],
            'onset': [0, 1], 'offset': [1, 2],
            'location': [[5.0, 5.0], [5.0, 25.0]],
        }),
    )

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, 1]


def test_map_to_lines_handles_null_coordinates():
    events = pm.Events(
        pl.DataFrame({
            'name': ['fixation', 'fixation'],
            'onset': [0, 1], 'offset': [1, 2],
            'location_x': [5.0, None],
            'location_y': [5.0, None],
        }),
    )

    events.map_to_lines(page())

    assert events.frame['line_idx'].to_list() == [0, None]


def test_map_to_lines_writes_a_custom_column_name():
    events = fixations([(5, 5)])

    events.map_to_lines(page(), line_column='line_idx')

    assert 'line_idx' in events.frame.columns


def test_map_to_lines_raises_without_coordinates():
    events = pm.Events(pl.DataFrame({'name': ['fixation'], 'onset': [0], 'offset': [1]}))

    with pytest.raises(ValueError, match="need 'location_x' and 'location_y'"):
        events.map_to_lines(page())


def test_map_to_lines_raises_on_several_pages():
    stimulus = page()
    paged = TextStimulus(
        stimulus.aois.with_columns(pl.col('line_idx').alias('page')), aoi_column='char',
        start_x_column='x0', start_y_column='y0',
        end_x_column='x1', end_y_column='y1',
        page_column='page',
    )
    events = fixations([(5, 5)])

    with pytest.raises(ValueError, match='several values'):
        events.map_to_lines(paged)


def test_map_to_lines_contain_matches_map_to_aois():
    """The vectorised containment test must agree with the per-row lookup, point for point.

    ``map_to_lines()`` reformulates ``map_to_aois`` as a join so that it does
    not cost a quarter of a millisecond per fixation. That is only allowed if the two agree,
    including on the awkward points: exactly on a box edge, in the gap between lines, outside
    the page and at negative coordinates.
    """
    stimulus = page(n_lines=4, n_chars=6)
    rng = np.random.default_rng(42)
    random_points = [
        (float(x), float(y))
        for x, y in zip(rng.uniform(-30, 90, 200), rng.uniform(-30, 110, 200))
    ]
    edge_points = [
        (0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (59.9, 69.9), (60.0, 70.0),
        (5.0, 15.0), (-1.0, 5.0), (5.0, -1.0), (1000.0, 1000.0),
    ]
    points = random_points + edge_points

    by_line = fixations(points)
    by_line.map_to_lines(stimulus)

    by_aoi = fixations(points)
    by_aoi.map_to_aois(stimulus, verbose=False)

    assert by_line.frame['line_idx'].to_list() == by_aoi.frame['line_idx'].to_list()
