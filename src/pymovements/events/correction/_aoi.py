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
"""Provides AOI geometry helpers for deriving text line and word positions."""
from __future__ import annotations

import polars as pl


def with_line_centers(aois: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Annotate each AOI row with the y-center of the text line it belongs to.

    Lines are identified by 'line_idx' if present, otherwise by the top y-coordinate.
    Assumes that the line of text is vertically centered within each AOI bounding box.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to annotate.

    Returns
    -------
    tuple[pl.DataFrame, str]
        AOIs dataframe with an added 'line_center' column in original row order, and the
        name of the column identifying lines.

    Raises
    ------
    ValueError
        If the AOIs dataframe has neither a 'start_y' nor a 'top_left_y' column, or no
        'height' column.
    """
    if 'start_y' in aois.columns:
        y_col = 'start_y'
    elif 'top_left_y' in aois.columns:
        y_col = 'top_left_y'
    else:
        raise ValueError(
            "AOIs dataframe requires a 'start_y' or 'top_left_y' column to derive text "
            'line positions.',
        )
    if 'height' not in aois.columns:
        raise ValueError(
            "AOIs dataframe requires a 'height' column, or 'start_y' and 'end_y' columns "
            'to derive it, to compute text line centers.',
        )
    line_key = 'line_idx' if 'line_idx' in aois.columns else y_col

    aois_with_line_centers = (
        aois.filter(pl.col(line_key).is_not_null())
        .with_columns(
            (pl.col(y_col) + pl.col('height') / 2.0)
            .mean()
            .over(line_key)
            .alias('line_center'),
        )
    )
    return aois_with_line_centers, line_key


def get_lines_of_text_from_aois(aois: pl.DataFrame) -> list[float]:
    """Calculate line positions of text based on AOIs.

    Assumes that the line of text is vertically centered within each AOI.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to calculate line positions from.

    Returns
    -------
    list[float]
        Line center y-coordinates of the text.
    """
    aois_with_line_centers, line_key = with_line_centers(aois)
    return (
        aois_with_line_centers
        .unique(subset=line_key)
        .sort(line_key)['line_center']
        .to_list()
    )


def get_word_locations_from_aois(
    aois: pl.DataFrame,
    character_level: bool = False,
) -> pl.Series:
    """Calculate word center locations from AOIs for DTW-based drift algorithms.

    Following the word position convention of Carr et al. :cite:p:`Carr2022`, the
    y-coordinate of each word is the center of the text line the word belongs to, not the
    center of the word's own bounding box. This keeps the y-coordinates identical to the
    line positions returned by get_lines_of_text_from_aois.

    With ``character_level=True``, the AOIs are aggregated to one location per word via
    the 'word' column, spanning from the first to the last character of the word.
    Directly adjacent repetitions of the same word within a line cannot be distinguished
    and are aggregated into a single word location.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to calculate word locations from.
    character_level: bool
        Set to True when the AOIs are finer than words, e.g. one row per character. The
        AOIs are then aggregated to one location per word via the 'word' column, which
        must be present. If False, each AOI row yields its own location.
        (default: False)

    Returns
    -------
    pl.Series
        Series of [x, y] word center locations.

    Raises
    ------
    ValueError
        If ``character_level`` is True but the AOIs dataframe has no 'word' column.
    """
    if character_level and 'word' not in aois.columns:
        raise ValueError(
            "character_level is True, but the AOIs dataframe has no 'word' column to "
            'aggregate the character-level AOIs by.',
        )

    aois_with_line_centers, line_key = with_line_centers(aois)

    if character_level:
        word_run = pl.struct([pl.col(line_key), pl.col('word')]).rle_id()
        return (
            aois_with_line_centers
            .group_by(word_run.alias('word_run'), maintain_order=True)
            .agg(
                ((pl.col('start_x').min() + pl.col('end_x').max()) / 2.0).alias('word_x'),
                pl.col('line_center').first(),
            )
            .select(pl.concat_list(['word_x', 'line_center']).alias('word_location'))
            .to_series()
        )

    return aois_with_line_centers.select(
        pl.concat_list([
            (pl.col('start_x') + pl.col('end_x')) / 2.0,
            pl.col('line_center'),
        ]).alias('word_location'),
    ).to_series()


def has_word_x_coords(aois: pl.DataFrame) -> bool:
    """Check if word X coordinates are available in the aois DataFrame."""
    return 'start_x' in aois.columns and 'end_x' in aois.columns


def normalize_aois(aois: pl.DataFrame) -> pl.DataFrame:
    """Derive missing AOI geometry columns from the available ones.

    Derives 'height' from 'start_y' and 'end_y', and 'end_x' from 'start_x' and 'width',
    whenever the derived column is missing but its sources are present.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to normalize.

    Returns
    -------
    pl.DataFrame
        AOIs dataframe with derived geometry columns.
    """
    derived_columns = []
    if 'height' not in aois.columns and {'start_y', 'end_y'} <= set(aois.columns):
        derived_columns.append((pl.col('end_y') - pl.col('start_y')).alias('height'))
    if 'end_x' not in aois.columns and {'start_x', 'width'} <= set(aois.columns):
        derived_columns.append((pl.col('start_x') + pl.col('width')).alias('end_x'))
    if derived_columns:
        aois = aois.with_columns(derived_columns)
    return aois


def count_text_lines(aois: pl.DataFrame, word_locations: pl.Series | None) -> int | None:
    """Count the text lines of a trial, or None if no line information is available."""
    if (
        ('start_y' in aois.columns or 'top_left_y' in aois.columns)
        and 'height' in aois.columns
    ):
        return len(get_lines_of_text_from_aois(aois))
    if word_locations is not None:
        return word_locations.cast(pl.List(pl.Float64)).list.get(1).n_unique()
    return None
