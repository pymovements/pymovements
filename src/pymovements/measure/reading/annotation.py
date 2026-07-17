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
"""Fixation annotation utilities for reading measure computation."""
from __future__ import annotations

import polars as pl


def annotate_run_id(fixations: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Annotate fixations with run IDs.

    A run is a contiguous sequence of fixations on the same word.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``word_idx`` column.
    group_columns : list[str]
        Column names used to partition the data into independent reading sequences.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``run_id`` column.
    """
    return fixations.with_columns(
        (pl.col('word_idx') != pl.col('word_idx').shift().over(group_columns))
        .fill_null(True).cast(pl.Int8)
        .cum_sum().over(group_columns).alias('run_id'),
    )


def annotate_prev_word_idx(fixations: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Annotate fixations with the word index of the previous fixation.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``word_idx`` column.
    group_columns : list[str]
        Column names used to partition the data into independent reading sequences.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``prev_word_idx`` column.
    """
    return fixations.with_columns(
        pl.col('word_idx').shift().over(group_columns).alias('prev_word_idx'),
    )


def annotate_next_word_idx(fixations: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Annotate fixations with the word index of the next fixation.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``word_idx`` column.
    group_columns : list[str]
        Column names used to partition the data into independent reading sequences.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``next_word_idx`` column.
    """
    return fixations.with_columns(
        pl.col('word_idx').shift(-1).over(group_columns).alias('next_word_idx'),
    )


def annotate_delta_in(fixations: pl.DataFrame) -> pl.DataFrame:
    """Annotate fixations with the difference in word index from the previous fixation.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain ``word_idx`` and ``prev_word_idx`` columns.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``delta_in`` column.
    """
    return fixations.with_columns(
        (pl.col('word_idx') - pl.col('prev_word_idx')).alias('delta_in'),
    )


def annotate_delta_out(fixations: pl.DataFrame) -> pl.DataFrame:
    """Annotate fixations with the difference in word index to the next fixation.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain ``word_idx`` and ``next_word_idx`` columns.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``delta_out`` column.
    """
    return fixations.with_columns(
        (pl.col('next_word_idx') - pl.col('word_idx')).alias('delta_out'),
    )


def annotate_is_reg_in(fixations: pl.DataFrame) -> pl.DataFrame:
    """Annotate fixations with a flag indicating a regression in.

    A regression in occurs when the fixation arrives from a higher-index word.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``delta_in`` column.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``is_reg_in`` column.
    """
    return fixations.with_columns(
        (pl.col('delta_in') < 0).alias('is_reg_in'),
    )


def annotate_is_reg_out(fixations: pl.DataFrame) -> pl.DataFrame:
    """Annotate fixations with a flag indicating a regression out.

    A regression out occurs when the fixation departs to a lower-index word.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``delta_out`` column.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``is_reg_out`` column.
    """
    return fixations.with_columns(
        (pl.col('delta_out') < 0).alias('is_reg_out'),
    )


def annotate_is_first_fixation(fixations: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Annotate fixations with a flag indicating the first fixation on a word.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain a ``word_idx`` column.
    group_columns : list[str]
        Column names used to partition the data into independent reading sequences.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``is_first_fix`` column.
    """
    return fixations.with_columns(
        pl.col('word_idx')
        .cum_count()
        .over(group_columns + ['word_idx'])
        .eq(1)
        .alias('is_first_fix'),
    )


def annotate_is_first_pass(fixations: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Annotate fixations with a flag indicating if it belongs to a first-pass reading.

    First-pass is defined at the *run* level. A run qualifies as first-pass if:
    1. It is the first time the reader enters the word.
    2. The word is entered from the left (forward reading direction).
    3. No words with a higher index have been fixated before.

    Parameters
    ----------
    fixations : pl.DataFrame
        Fixation-level DataFrame. Must contain ``word_idx``, ``run_id``, ``prev_word_idx`` and
        ``onset`` columns.
    group_columns : list[str]
        Column names used to partition the data into independent reading sequences.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an additional ``is_first_pass`` column.
    """
    def _mark_first_pass(fixation_group: pl.DataFrame) -> pl.DataFrame:
        """Mark fixations that belong to the first-pass reading of a word.

        First-pass is defined at the *run* level. A run qualifies as
        first-pass if all three conditions hold:

        1. It is the first time the reader enters the word.
        2. The word is entered from the left (forward reading direction).
        3. No words with a higher index have been fixated before (i.e.
           the word has not been exited or skipped over).

        All fixations within such a run are labelled
        ``is_first_pass = True``. Any later revisit to the word, or an
        entry from the right, is *not* part of first-pass.

        Parameters
        ----------
        fixation_group : pl.DataFrame
            Single-group fixation DataFrame sorted by ``onset``,
            annotated with ``run_id`` and ``prev_word_idx``.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with an additional boolean column
            ``is_first_pass``.
        """
        fixation_group = fixation_group.sort('onset')

        first_pass_flags: list[bool] = []

        prev_run = None
        rightmost_word_seen = None
        current_run_is_first_pass = False

        # set of words that have been entered at the start of any prior run
        words_ever_entered: set[int] = set()

        for row in fixation_group.iter_rows(named=True):
            w = row['word_idx']
            run = row['run_id']
            prev_w = row['prev_word_idx']

            new_run = run != prev_run

            if new_run:
                entered_from_left = (prev_w is None) or (w > prev_w)

                no_higher_word_seen = (rightmost_word_seen is None) or (
                    w >= rightmost_word_seen
                )

                first_time_entering_word = w not in words_ever_entered

                current_run_is_first_pass = (
                    entered_from_left
                    and no_higher_word_seen
                    and first_time_entering_word
                )

                words_ever_entered.add(w)

            first_pass_flags.append(current_run_is_first_pass)

            if rightmost_word_seen is None or w > rightmost_word_seen:
                rightmost_word_seen = w

            prev_run = run

        return fixation_group.with_columns(pl.Series('is_first_pass', first_pass_flags))

    return fixations.group_by(group_columns, maintain_order=True).map_groups(_mark_first_pass)


def annotate_fixations(
    events: pl.DataFrame,
    group_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Annotate fixations with run- and pass-level information.

    Computes the following per-fixation annotations:

    * **run_id**: integer ID for each contiguous sequence of fixations on the same word.
    * **prev_word_idx / next_word_idx**: word indices of the immediately preceding and following
      fixations.
    * **is_reg_in / is_reg_out**: whether the fixation arrives from a higher-index word
      (regression in) or departs to a lower-index word (regression out).
    * **is_first_fix**: whether this is the first fixation ever on the word within the trial.
    * **is_first_pass**: whether the fixation belongs to the first-pass reading episode of the word
      (see :func:`_mark_first_pass`).

    Parameters
    ----------
    events : pl.DataFrame
        DataFrame containing pymovements fixation events mapped to AOIs.
        Must contain at least ``name``, ``word_idx``, and ``onset``
        columns, plus whatever columns are listed in ``group_columns``.
    group_columns : list[str] | None
        Column names used to partition the data into independent reading
        sequences (e.g. one trial per page). If ``None``, defaults to
        ``['trial', 'stimulus', 'page']``.

    Returns
    -------
    pl.DataFrame
        Fixation-level DataFrame with the original columns plus
        ``fixation_id``, ``run_id``, ``prev_word_idx``,
        ``next_word_idx``, ``delta_in``, ``delta_out``,
        ``is_reg_in``, ``is_reg_out``, ``is_first_fix``, and
        ``is_first_pass``.
    """
    if group_columns is None:
        group_columns = ['trial', 'stimulus', 'page']

    fixations = (
        events.filter((pl.col('name') == 'fixation') & (pl.col('word_idx').is_not_null()))
        .with_row_index('fixation_id')
        .sort(group_columns + ['onset'])
    )

    # -------------------------------------------------
    # Reading runs (contiguous fixations on the same word)
    # -------------------------------------------------
    fixations = annotate_run_id(fixations, group_columns)

    # -----------------------------------------------------
    # Neighbouring fixated words (for regression detection)
    # -----------------------------------------------------
    fixations = annotate_prev_word_idx(fixations, group_columns)
    fixations = annotate_next_word_idx(fixations, group_columns)
    fixations = annotate_delta_in(fixations)  # requires prev_word_idx annotation
    fixations = annotate_delta_out(fixations)  # requires next_word_idx annotation
    fixations = annotate_is_reg_in(fixations)  # requires delta_in annotation
    fixations = annotate_is_reg_out(fixations)  # requires delta_out annotation

    # -------------------------------------------------
    # First fixation on word
    # -------------------------------------------------
    fixations = annotate_is_first_fixation(fixations, group_columns)

    # -------------------------------------------------
    # First-pass flag (word-level first reading episode)
    # -------------------------------------------------
    fixations = annotate_is_first_pass(fixations, group_columns)

    return fixations
