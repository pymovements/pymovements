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
"""Fixation annotation expressions for reading measure computation.

To compute all reading measures from fixations and an AOI table at once, use
:func:`~pymovements.measure.reading.compute_reading_measures`, which annotates the fixations
implicitly. The functions in this module are its building blocks, useful for custom analyses on
the fixation level.

Every function except :func:`annotate_fixations` returns a polars expression producing one
annotation column. The expressions do not alter any DataFrame themselves: the consumer applies
them via ``with_columns`` and supplies the partitioning into independent reading sequences with
``.over(...)`` where the docstring calls for it, e.g.::

    fixations.with_columns(run_id().over(['trial']))

Input columns can be given as column names or as arbitrary polars expressions. The expressions
expect the fixation table to be sorted by ``onset`` within each sequence.
:func:`annotate_fixations` is the consuming function that applies all annotations in dependency
order.
"""
from __future__ import annotations

import warnings

import polars as pl

from pymovements._utils._expressions import as_expr


def _over(expr: pl.Expr, group_columns: list[str] | None) -> pl.Expr:
    """Apply a window over the group columns, or leave the expression global without groups."""
    return expr.over(group_columns) if group_columns else expr


def run_id(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Assign run IDs to fixations.

    A run is a contiguous sequence of fixations on the same word. Apply ``.over(group_columns)``
    to partition into independent reading sequences.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``run_id`` column.
    """
    return (
        (as_expr(word_idx) != as_expr(word_idx).shift())
        .fill_null(True)
        .cast(pl.Int8)
        .cum_sum()
        .alias('run_id')
    )


def prev_word_idx(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Get the word index of the previous fixation.

    Apply ``.over(group_columns)`` to partition into independent reading sequences.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``prev_word_idx`` column.
    """
    return as_expr(word_idx).shift().alias('prev_word_idx')


def next_word_idx(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Get the word index of the next fixation.

    Apply ``.over(group_columns)`` to partition into independent reading sequences.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``next_word_idx`` column.
    """
    return as_expr(word_idx).shift(-1).alias('next_word_idx')


def delta_in(
    word_idx: str | pl.Expr = 'word_idx',
    prev_word_idx: str | pl.Expr = 'prev_word_idx',
) -> pl.Expr:
    """Compute the difference in word index from the previous fixation.

    Row-wise, no window needed.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    prev_word_idx : str | pl.Expr
        Column name or expression of the previous fixation's word index.
        (default: ``'prev_word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``delta_in`` column.
    """
    return (as_expr(word_idx) - as_expr(prev_word_idx)).alias('delta_in')


def delta_out(
    word_idx: str | pl.Expr = 'word_idx',
    next_word_idx: str | pl.Expr = 'next_word_idx',
) -> pl.Expr:
    """Compute the difference in word index to the next fixation.

    Row-wise, no window needed.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    next_word_idx : str | pl.Expr
        Column name or expression of the next fixation's word index.
        (default: ``'next_word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``delta_out`` column.
    """
    return (as_expr(next_word_idx) - as_expr(word_idx)).alias('delta_out')


def is_reg_in(delta_in: str | pl.Expr = 'delta_in') -> pl.Expr:
    """Flag fixations that arrive from a higher-index word (regression in).

    Row-wise, no window needed.

    Parameters
    ----------
    delta_in : str | pl.Expr
        Column name or expression of the word index difference from the previous fixation.
        (default: ``'delta_in'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_reg_in`` column.
    """
    return (as_expr(delta_in) < 0).alias('is_reg_in')


def is_reg_out(delta_out: str | pl.Expr = 'delta_out') -> pl.Expr:
    """Flag fixations that depart to a lower-index word (regression out).

    Row-wise, no window needed.

    Parameters
    ----------
    delta_out : str | pl.Expr
        Column name or expression of the word index difference to the next fixation.
        (default: ``'delta_out'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_reg_out`` column.
    """
    return (as_expr(delta_out) < 0).alias('is_reg_out')


def is_first_fixation(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Flag the first fixation on each word.

    Apply ``.over(group_columns + ['word_idx'])`` so the flag is evaluated per word within each
    reading sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_first_fix`` column.
    """
    return as_expr(word_idx).cum_count().eq(1).alias('is_first_fix')


def is_first_pass(
    group_columns: list[str] | None = None,
    word_idx: str | pl.Expr = 'word_idx',
    run_id: str | pl.Expr = 'run_id',
) -> pl.Expr:
    """Flag fixations that belong to the first-pass reading of their word.

    A run of fixations qualifies as first-pass if it is the word's *first* run and no word with
    a higher index has been fixated before the run starts. Entering from the left is implied:
    at a run start the previous word differs from the current one and cannot exceed the running
    maximum. The no-higher-word condition is constant across a run (within a run the running
    maximum either already exceeded the word or is the word itself), so it needs no run-level
    broadcast.

    Unlike the other annotation expressions, this one combines two different windows internally
    and therefore takes the group columns as a parameter instead of a trailing ``.over(...)``.

    Parameters
    ----------
    group_columns : list[str] | None
        Column names used to partition the data into independent reading sequences. If ``None``
        or empty, the whole table is treated as a single sequence. (default: None)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    run_id : str | pl.Expr
        Column name or expression of the run ID (see :func:`run_id`).
        (default: ``'run_id'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_first_pass`` column.
    """
    group_columns = list(group_columns or [])

    no_higher_word_seen = (
        as_expr(word_idx) >= _over(as_expr(word_idx).cum_max().shift(), group_columns)
    ).fill_null(True)

    first_run_of_word = (
        as_expr(run_id) == as_expr(run_id).min().over(group_columns + [as_expr(word_idx)])
    )

    return (no_higher_word_seen & first_run_of_word).alias('is_first_pass')


def regression_path_word(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Get the word whose regression path each fixation belongs to.

    The regression-path window of a word starts when the word is first entered in first pass
    (which is exactly the moment it becomes the running maximum of fixated word indices, as
    first-pass entry requires that no higher word has been fixated before) and ends when a
    fixation lands right of it (which is exactly when the running maximum increases past it).
    The windows of different words are therefore disjoint and partition the sequence, and every
    fixation belongs to the regression path of exactly one word: the current running maximum.

    Apply ``.over(group_columns)`` to partition into independent reading sequences.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``regression_path_word`` column.
    """
    return as_expr(word_idx).cum_max().alias('regression_path_word')


def annotate_fixations(
    events: pl.DataFrame,
    group_columns: list[str] | None = None,
    event_name: str = 'fixation',
    word_idx: str | pl.Expr = 'word_idx',
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
      (see :func:`~pymovements.measure.reading.is_first_pass`).
    * **regression_path_word**: the word whose regression-path window the fixation belongs to
      (see :func:`~pymovements.measure.reading.regression_path_word`).

    Parameters
    ----------
    events : pl.DataFrame
        DataFrame containing pymovements fixation events mapped to AOIs.
        Must contain at least ``name``, ``word_idx``, and ``onset``
        columns, plus whatever columns are listed in ``group_columns``.
    group_columns : list[str] | None
        Column names used to partition the data into independent reading
        sequences (e.g. one trial per page). If ``None`` or empty, the
        whole table is treated as a single sequence. (default: None)
    event_name : str
        Name of the events to annotate. Rows with a different ``name`` are
        dropped. (default: ``'fixation'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.DataFrame
        Fixation-level DataFrame with the original columns plus
        ``fixation_id``, ``run_id``, ``prev_word_idx``,
        ``next_word_idx``, ``delta_in``, ``delta_out``,
        ``is_reg_in``, ``is_reg_out``, ``is_first_fix``,
        ``is_first_pass``, and ``regression_path_word``.
    """
    group_columns = list(group_columns or [])

    fixations = (
        events.filter((pl.col('name') == event_name) & (as_expr(word_idx).is_not_null()))
        .with_row_index('fixation_id')
        # fixation_id breaks onset ties deterministically (it preserves the input order), so the
        # run/pass annotations are reproducible even when two fixations share an onset.
        .sort(group_columns + ['onset', 'fixation_id'])
    )

    if fixations.is_empty() and not events.is_empty():
        warnings.warn(
            f'no fixations left to annotate: no row has name == {event_name!r} together with a '
            'non-null word index. All reading measures will be zero.',
        )

    return (
        fixations
        .with_columns(
            _over(run_id(word_idx), group_columns),
            _over(prev_word_idx(word_idx), group_columns),
            _over(next_word_idx(word_idx), group_columns),
            _over(regression_path_word(word_idx), group_columns),
            is_first_fixation(word_idx).over(group_columns + [word_idx]),
        )
        .with_columns(
            delta_in(word_idx),  # requires prev_word_idx annotation
            delta_out(word_idx),  # requires next_word_idx annotation
        )
        .with_columns(
            is_reg_in(),  # requires delta_in annotation
            is_reg_out(),  # requires delta_out annotation
            is_first_pass(group_columns, word_idx),  # requires run_id annotation
        )
    )
