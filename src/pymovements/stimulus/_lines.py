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
"""Line geometry derived from area-of-interest rectangles.

Lines are numbered ``0..k-1`` from the top and live in a ``line_idx`` column, the same name
that :py:mod:`pymovements.events.correction` uses to group AOIs into text lines. Tolerances and
distances are expressed in line heights rather than in pixels, so that they carry over between
stimuli of different font sizes.

This is deliberately separate from ``events/correction/_aoi.py``, which derives line centres
too: that module belongs to the correction package and works on frames whose geometry has
already been normalised, whereas the line index here hangs on the stimulus and has to be
available before any correction runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from pymovements.stimulus.text import TextStimulus

LINE_IDX_COLUMN = 'line_idx'

#: Position of an AOI in the stimulus' own dataframe, carried through the geometry so that a
#: line index can be written back to the row it belongs to rather than to the row that happens
#: to sit at the same position.
AOI_ROW_COLUMN = '__pm_aoi_row'

_ROW = '__pm_row'
_AOI = '__pm_aoi'
_CENTER_Y = '__pm_center_y'


def normalized_bounds(stimulus: TextStimulus) -> pl.DataFrame:
    """Return the usable AOI rectangles of a stimulus as half-open ``[start, end)`` intervals.

    Rectangles with a non-finite coordinate or a height of zero or less are **left out**: a
    single one of them would otherwise take part in the line grouping and invent a text line
    that is not there. Each returned row carries its position in ``stimulus.aois`` in
    :py:data:`AOI_ROW_COLUMN`, so that callers can tell which AOIs were dropped and write a
    result back to the right row. Callers that want to report the loss compare the returned
    height with ``stimulus.aois.height``.

    Parameters
    ----------
    stimulus: TextStimulus
        Text stimulus defining AOI rectangles.

    Returns
    -------
    pl.DataFrame
        Frame with ``__pm_aoi_row`` and the float columns ``start_x``, ``start_y``, ``end_x``
        and ``end_y``, in the row order of ``stimulus.aois``, minus the unusable rows.

    Raises
    ------
    ValueError
        If the stimulus defines neither an end column nor a size column for one of the axes.
    """
    start_x = pl.col(stimulus.start_x_column).cast(pl.Float64)
    start_y = pl.col(stimulus.start_y_column).cast(pl.Float64)

    if stimulus.end_x_column is not None:
        end_x = pl.col(stimulus.end_x_column).cast(pl.Float64)
    elif stimulus.width_column is not None:
        end_x = start_x + pl.col(stimulus.width_column).cast(pl.Float64)
    else:
        raise ValueError('either TextStimulus.width or TextStimulus.end_x_column must be defined')

    if stimulus.end_y_column is not None:
        end_y = pl.col(stimulus.end_y_column).cast(pl.Float64)
    elif stimulus.height_column is not None:
        end_y = start_y + pl.col(stimulus.height_column).cast(pl.Float64)
    else:
        raise ValueError('either TextStimulus.height or TextStimulus.end_y_column must be defined')

    bounds = stimulus.aois.select(
        start_x.alias('start_x'),
        start_y.alias('start_y'),
        end_x.alias('end_x'),
        end_y.alias('end_y'),
    ).with_row_index(AOI_ROW_COLUMN)

    usable = (
        pl.all_horizontal(
            pl.col('start_x').is_finite(),
            pl.col('start_y').is_finite(),
            pl.col('end_x').is_finite(),
            pl.col('end_y').is_finite(),
        )
        & (pl.col('end_y') > pl.col('start_y'))
    )
    return bounds.filter(usable.fill_null(False))  # noqa: FBT003


def median_line_height(bounds: pl.DataFrame) -> float:
    """Return the median AOI height, used as the unit for tolerances and distances.

    Parameters
    ----------
    bounds: pl.DataFrame
        AOI rectangles as returned by :py:func:`normalized_bounds`.

    Returns
    -------
    float
        Median of ``end_y - start_y``.

    Raises
    ------
    ValueError
        If ``bounds`` is empty, which is what is left when no rectangle of the stimulus is
        usable.
    """
    if bounds.height == 0:
        raise ValueError('cannot derive line geometry from an empty AOI dataframe')

    # Positive by construction: normalized_bounds() keeps only rectangles with end_y > start_y,
    # so the median over what is left cannot be zero or negative.
    return float((bounds['end_y'] - bounds['start_y']).median())


def derive_line_idx(bounds: pl.DataFrame, tolerance: float) -> pl.Series:
    """Group AOIs into text lines by their vertical centre.

    Sorted by vertical centre, a new line starts wherever the gap to the previous centre
    exceeds ``tolerance`` line heights. Grouping on the centre rather than on the top edge
    tolerates boxes of different height inside one line. Neither PoTeC nor EMTeC has such
    boxes — in both, every character of a line shares one centre — so for them the two ways
    of grouping agree.

    Parameters
    ----------
    bounds: pl.DataFrame
        AOI rectangles as returned by :py:func:`normalized_bounds`.
    tolerance: float
        Maximum gap within one line, in line heights.

    Returns
    -------
    pl.Series
        Integer line index per AOI, ``0..k-1`` from the top, in the row order of ``bounds``.

    Raises
    ------
    ValueError
        If ``tolerance`` is not positive.
    """
    if tolerance <= 0:
        raise ValueError(f'tolerance must be positive, got {tolerance}')

    height = median_line_height(bounds)
    frame = (
        bounds
        .with_row_index(_ROW)
        .with_columns(((pl.col('start_y') + pl.col('end_y')) / 2).alias(_CENTER_Y))
        .sort(_CENTER_Y)
    )
    line_idx = (
        (pl.col(_CENTER_Y) - pl.col(_CENTER_Y).shift(1) > tolerance * height)
        .fill_null(False)  # noqa: FBT003
        .cum_sum()
        .cast(pl.Int64)
        .alias(LINE_IDX_COLUMN)
    )
    return frame.with_columns(line_idx).sort(_ROW)[LINE_IDX_COLUMN]


def lines_containing(
        x: pl.Series,
        y: pl.Series,
        bounds: pl.DataFrame,
        line_idx: pl.Series,
) -> pl.Series:
    """Return the line of the AOI that contains each point, or null.

    Containment uses the same half-open interval as
    :py:meth:`~pymovements.stimulus.TextStimulus.get_aoi`, ``start <= coord < end``, on both
    axes. Where several AOIs overlap a point, the first one in AOI row order wins, matching
    ``get_aoi(..., max_matches=1)``.

    This is an expression-based reformulation of the per-row lookup in
    :py:meth:`~pymovements.Events.map_to_aois`; it exists because the per-row version costs
    about a quarter of a millisecond per fixation, which is minutes on a corpus.

    Parameters
    ----------
    x: pl.Series
        Horizontal coordinates.
    y: pl.Series
        Vertical coordinates.
    bounds: pl.DataFrame
        AOI rectangles as returned by :py:func:`normalized_bounds`.
    line_idx: pl.Series
        Line index per AOI.

    Returns
    -------
    pl.Series
        Integer line index per point, null where no AOI contains it.
    """
    points = pl.DataFrame({
        '__pm_x': x.cast(pl.Float64),
        '__pm_y': y.cast(pl.Float64),
    }).with_row_index(_ROW)
    aois = (
        bounds
        .with_columns(line_idx.cast(pl.Int64).alias(LINE_IDX_COLUMN))
        .with_row_index(_AOI)
    )

    hits = points.join_where(
        aois,
        pl.col('__pm_x') >= pl.col('start_x'),
        pl.col('__pm_x') < pl.col('end_x'),
        pl.col('__pm_y') >= pl.col('start_y'),
        pl.col('__pm_y') < pl.col('end_y'),
    )
    first_hit = (
        hits
        .group_by(_ROW)
        .agg(pl.col(LINE_IDX_COLUMN).sort_by(_AOI).first())
    )
    return (
        points
        .join(first_hit, on=_ROW, how='left')
        .sort(_ROW)[LINE_IDX_COLUMN]
        .cast(pl.Int64)
    )
