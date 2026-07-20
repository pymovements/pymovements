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
"""Word-level AOI utilities for reading measure computation."""
from __future__ import annotations

import polars as pl


def repair_word_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure consistent word string labels within each word index group.

    Normalizes the ``word`` column so that all characters belonging to the same
    ``(trial, page, line_idx, word_idx)`` group share an identical word label.

    Whitespace-only or empty ``word`` entries are treated as missing and are
    forward- and backward-filled within each word group. The ``word_idx``
    column is not modified.

    This is primarily used to assign a proper word label to characters such as
    inter-word spaces that already have a valid ``word_idx``, ensuring
    downstream processing operates on consistent word-level labels.

    Parameters
    ----------
    df : pl.DataFrame
        Character-level AOI table containing at least the columns
        ``word_idx``, ``word``, and ``char_idx_in_line``.
        Optionally grouped by ``trial``, ``page``, and ``line_idx``.

    Returns
    -------
    pl.DataFrame
        A copy of the input DataFrame with normalized ``word`` labels.
        No rows are added or removed, and ``word_idx`` assignments remain
        unchanged.
    """
    # Use only grouping columns that actually exist.
    base_cols = ['trial', 'page', 'line_idx', 'word_idx']
    group_cols = [c for c in base_cols if c in df.columns]

    return (
        df.sort(group_cols + ['char_idx_in_line'])
        .with_columns(
            pl.when(pl.col('word').is_null() | (pl.col('word').str.strip_chars() == ''))
            .then(None)
            .otherwise(pl.col('word'))
            .alias('_word_tmp'),
        )
        .with_columns(
            pl.col('_word_tmp')
            .forward_fill()
            .backward_fill()
            .over(group_cols)
            .alias('word'),
        )
        .drop('_word_tmp')
    )


def all_tokens_from_aois(
    aois: pl.DataFrame,
    trial: str | None = None,
) -> pl.DataFrame:
    """Return every AOI token on the page in word-index order.

    Includes words, spaces, and punctuation — every row that has a
    ``word_idx``. If ``trial`` is provided and the ``trial`` column is absent
    from ``aois``, it is added as a constant column.

    Parameters
    ----------
    aois : pl.DataFrame
        AOI table containing at least the columns ``page``, ``word_idx``,
        and ``word``.
    trial : str | None
        Trial identifier to attach when the ``trial`` column is absent from
        ``aois``. If ``None`` and the column is already present, it is used
        as-is. (default: None)

    Returns
    -------
    pl.DataFrame
        Deduplicated table with columns ``trial``, ``page``, ``word_idx``,
        and ``word``, sorted by ``word_idx``.
    """
    aois = (
        aois.with_columns([pl.lit(trial).cast(pl.Utf8).alias('trial')])
        if 'trial' not in aois.columns
        else aois
    )

    return aois.select(['trial', 'page', 'word_idx', 'word']).unique().sort('word_idx')


def mark_skipped_tokens(
    all_tokens: pl.DataFrame,
    fixations: pl.DataFrame,
) -> pl.DataFrame:
    """Mark tokens that were never fixated as skipped.

    Performs a left join of ``all_tokens`` against the set of fixated
    ``(trial, page, word_idx)`` triples and adds a binary ``skipped`` column.

    Parameters
    ----------
    all_tokens : pl.DataFrame
        Full token table as returned by :func:`all_tokens_from_aois`,
        containing at least ``trial``, ``page``, and ``word_idx``.
    fixations : pl.DataFrame
        Fixation events containing at least ``trial``, ``page``, and
        ``word_idx``. Rows with null ``word_idx`` are ignored.

    Returns
    -------
    pl.DataFrame
        ``all_tokens`` with an additional ``skipped`` column (``Int8``):
        ``1`` if the token was never fixated, ``0`` otherwise.
    """
    fixated_tokens = (
        fixations.select(['trial', 'page', 'word_idx'])
        .drop_nulls()
        .unique()
        .with_columns(pl.lit(1).alias('fixated'))
    )

    out = all_tokens.join(
        fixated_tokens,
        on=['trial', 'page', 'word_idx'],
        how='left',
    )

    return out.with_columns(
        pl.when(pl.col('fixated').is_null())
        .then(1)   # not fixated → skipped
        .otherwise(0)  # fixated → not skipped
        .cast(pl.Int8)
        .alias('skipped'),
    ).drop('fixated')
