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

# Expression parameters are deliberately named after the annotation columns they default to,
# which shadows the sibling factory functions producing those columns.
# pylint: disable=redefined-outer-name


def _over(expr: pl.Expr, group_columns: list[str] | None) -> pl.Expr:
    """Apply a window over the group columns, or leave the expression global without groups."""
    return expr.over(group_columns) if group_columns else expr


def run_id(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Assign run IDs to fixations.

    A run is a contiguous sequence of fixations on the same word. Apply ``.over(group_columns)``
    to partition into independent reading sequences.

    .. warning:: Requires onset-sorted input within each sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``run_id`` column.

    Examples
    --------
    The id increments on every word change, so the two consecutive fixations on word 1 share a
    run and the later return to word 0 opens a new one:

    >>> import polars as pl
    >>> from pymovements.measure.reading import run_id
    >>> fixations = pl.DataFrame({'word_idx': [0, 1, 1, 0, 2]})
    >>> fixations.with_columns(run_id())
    shape: (5, 2)
    ┌──────────┬────────┐
    │ word_idx ┆ run_id │
    │ ---      ┆ ---    │
    │ i64      ┆ i64    │
    ╞══════════╪════════╡
    │ 0        ┆ 1      │
    │ 1        ┆ 2      │
    │ 1        ┆ 2      │
    │ 0        ┆ 3      │
    │ 2        ┆ 4      │
    └──────────┴────────┘

    Partitioned with ``.over(...)``, the numbering restarts for each reading sequence:

    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 1, 0, 0],
    ... })
    >>> fixations.with_columns(run_id().over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬────────┐
    │ trial ┆ word_idx ┆ run_id │
    │ ---   ┆ ---      ┆ ---    │
    │ i64   ┆ i64      ┆ i64    │
    ╞═══════╪══════════╪════════╡
    │ 1     ┆ 0        ┆ 1      │
    │ 1     ┆ 1        ┆ 2      │
    │ 1     ┆ 1        ┆ 2      │
    │ 2     ┆ 0        ┆ 1      │
    │ 2     ┆ 0        ┆ 1      │
    └───────┴──────────┴────────┘
    """
    word_idx_expr = as_expr(word_idx)
    return (
        (word_idx_expr != word_idx_expr.shift())
        .fill_null(True)
        .cast(pl.Int8)
        .cum_sum()
        .alias('run_id')
    )


def prev_word_idx(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Get the word index of the previous fixation.

    Apply ``.over(group_columns)`` to partition into independent reading sequences.

    .. warning:: Requires onset-sorted input within each sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``prev_word_idx`` column.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.measure.reading import prev_word_idx
    >>> fixations = pl.DataFrame({'word_idx': [0, 1, 1, 0, 2]})
    >>> fixations.with_columns(prev_word_idx())
    shape: (5, 2)
    ┌──────────┬───────────────┐
    │ word_idx ┆ prev_word_idx │
    │ ---      ┆ ---           │
    │ i64      ┆ i64           │
    ╞══════════╪═══════════════╡
    │ 0        ┆ null          │
    │ 1        ┆ 0             │
    │ 1        ┆ 1             │
    │ 0        ┆ 1             │
    │ 2        ┆ 0             │
    └──────────┴───────────────┘

    Partitioned with ``.over(...)``, each reading sequence starts fresh with a null:

    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 1, 0, 1],
    ... })
    >>> fixations.with_columns(prev_word_idx().over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬───────────────┐
    │ trial ┆ word_idx ┆ prev_word_idx │
    │ ---   ┆ ---      ┆ ---           │
    │ i64   ┆ i64      ┆ i64           │
    ╞═══════╪══════════╪═══════════════╡
    │ 1     ┆ 0        ┆ null          │
    │ 1     ┆ 1        ┆ 0             │
    │ 1     ┆ 1        ┆ 1             │
    │ 2     ┆ 0        ┆ null          │
    │ 2     ┆ 1        ┆ 0             │
    └───────┴──────────┴───────────────┘
    """
    word_idx_expr = as_expr(word_idx)
    return word_idx_expr.shift().alias('prev_word_idx')


def next_word_idx(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Get the word index of the next fixation.

    Apply ``.over(group_columns)`` to partition into independent reading sequences.

    .. warning:: Requires onset-sorted input within each sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``next_word_idx`` column.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.measure.reading import next_word_idx
    >>> fixations = pl.DataFrame({'word_idx': [0, 1, 1, 0, 2]})
    >>> fixations.with_columns(next_word_idx())
    shape: (5, 2)
    ┌──────────┬───────────────┐
    │ word_idx ┆ next_word_idx │
    │ ---      ┆ ---           │
    │ i64      ┆ i64           │
    ╞══════════╪═══════════════╡
    │ 0        ┆ 1             │
    │ 1        ┆ 1             │
    │ 1        ┆ 0             │
    │ 0        ┆ 2             │
    │ 2        ┆ null          │
    └──────────┴───────────────┘

    Partitioned with ``.over(...)``, each reading sequence ends with a null:

    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 1, 0, 1],
    ... })
    >>> fixations.with_columns(next_word_idx().over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬───────────────┐
    │ trial ┆ word_idx ┆ next_word_idx │
    │ ---   ┆ ---      ┆ ---           │
    │ i64   ┆ i64      ┆ i64           │
    ╞═══════╪══════════╪═══════════════╡
    │ 1     ┆ 0        ┆ 1             │
    │ 1     ┆ 1        ┆ 1             │
    │ 1     ┆ 1        ┆ null          │
    │ 2     ┆ 0        ┆ 1             │
    │ 2     ┆ 1        ┆ null          │
    └───────┴──────────┴───────────────┘
    """
    word_idx_expr = as_expr(word_idx)
    return word_idx_expr.shift(-1).alias('next_word_idx')


def delta_in(
    word_idx: str | pl.Expr = 'word_idx',
    prev_word_idx: str | pl.Expr = 'prev_word_idx',
) -> pl.Expr:
    """Compute the difference in word index from the previous fixation.

    Row-wise on materialized input columns, so no window is needed then. A window-dependent
    input expression such as :func:`prev_word_idx` reintroduces the need for ``.over(...)``.

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

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.measure.reading import delta_in
    >>> fixations = pl.DataFrame({
    ...     'word_idx': [0, 1, 1, 0, 2],
    ...     'prev_word_idx': [None, 0, 1, 1, 0],
    ... })
    >>> fixations.with_columns(delta_in())
    shape: (5, 3)
    ┌──────────┬───────────────┬──────────┐
    │ word_idx ┆ prev_word_idx ┆ delta_in │
    │ ---      ┆ ---           ┆ ---      │
    │ i64      ┆ i64           ┆ i64      │
    ╞══════════╪═══════════════╪══════════╡
    │ 0        ┆ null          ┆ null     │
    │ 1        ┆ 0             ┆ 1        │
    │ 1        ┆ 1             ┆ 0        │
    │ 0        ┆ 1             ┆ -1       │
    │ 2        ┆ 0             ┆ 2        │
    └──────────┴───────────────┴──────────┘

    The previous word index need not be materialized as a column: any expression works, e.g.
    composing the :func:`prev_word_idx` factory directly. The composed expression then contains
    a window and needs ``.over(...)`` to partition into reading sequences:

    >>> from pymovements.measure.reading import prev_word_idx
    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 0, 1],
    ... })
    >>> fixations.with_columns(delta_in(prev_word_idx=prev_word_idx()).over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬──────────┐
    │ trial ┆ word_idx ┆ delta_in │
    │ ---   ┆ ---      ┆ ---      │
    │ i64   ┆ i64      ┆ i64      │
    ╞═══════╪══════════╪══════════╡
    │ 1     ┆ 0        ┆ null     │
    │ 1     ┆ 1        ┆ 1        │
    │ 1     ┆ 0        ┆ -1       │
    │ 2     ┆ 0        ┆ null     │
    │ 2     ┆ 1        ┆ 1        │
    └───────┴──────────┴──────────┘
    """
    word_idx_expr = as_expr(word_idx)
    prev_word_idx_expr = as_expr(prev_word_idx)
    return (word_idx_expr - prev_word_idx_expr).alias('delta_in')


def delta_out(
    word_idx: str | pl.Expr = 'word_idx',
    next_word_idx: str | pl.Expr = 'next_word_idx',
) -> pl.Expr:
    """Compute the difference in word index to the next fixation.

    Row-wise on materialized input columns, so no window is needed then. A window-dependent
    input expression such as :func:`next_word_idx` reintroduces the need for ``.over(...)``.

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

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.measure.reading import delta_out
    >>> fixations = pl.DataFrame({
    ...     'word_idx': [0, 1, 1, 0, 2],
    ...     'next_word_idx': [1, 1, 0, 2, None],
    ... })
    >>> fixations.with_columns(delta_out())
    shape: (5, 3)
    ┌──────────┬───────────────┬───────────┐
    │ word_idx ┆ next_word_idx ┆ delta_out │
    │ ---      ┆ ---           ┆ ---       │
    │ i64      ┆ i64           ┆ i64       │
    ╞══════════╪═══════════════╪═══════════╡
    │ 0        ┆ 1             ┆ 1         │
    │ 1        ┆ 1             ┆ 0         │
    │ 1        ┆ 0             ┆ -1        │
    │ 0        ┆ 2             ┆ 2         │
    │ 2        ┆ null          ┆ null      │
    └──────────┴───────────────┴───────────┘

    The next word index need not be materialized as a column: any expression works, e.g.
    composing the :func:`next_word_idx` factory directly. The composed expression then contains
    a window and needs ``.over(...)`` to partition into reading sequences:

    >>> from pymovements.measure.reading import next_word_idx
    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 0, 1],
    ... })
    >>> fixations.with_columns(delta_out(next_word_idx=next_word_idx()).over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬───────────┐
    │ trial ┆ word_idx ┆ delta_out │
    │ ---   ┆ ---      ┆ ---       │
    │ i64   ┆ i64      ┆ i64       │
    ╞═══════╪══════════╪═══════════╡
    │ 1     ┆ 0        ┆ 1         │
    │ 1     ┆ 1        ┆ -1        │
    │ 1     ┆ 0        ┆ null      │
    │ 2     ┆ 0        ┆ 1         │
    │ 2     ┆ 1        ┆ null      │
    └───────┴──────────┴───────────┘
    """
    word_idx_expr = as_expr(word_idx)
    next_word_idx_expr = as_expr(next_word_idx)
    return (next_word_idx_expr - word_idx_expr).alias('delta_out')


def is_reg_in(delta_in: str | pl.Expr = 'delta_in') -> pl.Expr:
    """Flag fixations that arrive from a higher-index word (regression in).

    Row-wise on materialized input columns, so no window is needed then. A window-dependent
    input expression such as a composed :func:`delta_in` reintroduces the need for
    ``.over(...)``.

    Parameters
    ----------
    delta_in : str | pl.Expr
        Column name or expression of the word index difference from the previous fixation.
        (default: ``'delta_in'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_reg_in`` column.

    Examples
    --------
    The first fixation has no predecessor, so its ``delta_in`` (and thus ``is_reg_in``) is null:

    >>> import polars as pl
    >>> from pymovements.measure.reading import is_reg_in
    >>> fixations = pl.DataFrame({'delta_in': [None, 1, 0, -1, 2]})
    >>> fixations.with_columns(is_reg_in())
    shape: (5, 2)
    ┌──────────┬───────────┐
    │ delta_in ┆ is_reg_in │
    │ ---      ┆ ---       │
    │ i64      ┆ bool      │
    ╞══════════╪═══════════╡
    │ null     ┆ null      │
    │ 1        ┆ false     │
    │ 0        ┆ false     │
    │ -1       ┆ true      │
    │ 2        ┆ false     │
    └──────────┴───────────┘

    Composing the :func:`delta_in` and :func:`prev_word_idx` factories computes the flag straight
    from the word indices; the composed expression then contains a window and needs ``.over(...)``
    to partition into reading sequences:

    >>> from pymovements.measure.reading import delta_in, prev_word_idx
    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 0, 1],
    ... })
    >>> fixations.with_columns(
    ...     is_reg_in(delta_in(prev_word_idx=prev_word_idx())).over('trial'),
    ... )
    shape: (5, 3)
    ┌───────┬──────────┬───────────┐
    │ trial ┆ word_idx ┆ is_reg_in │
    │ ---   ┆ ---      ┆ ---       │
    │ i64   ┆ i64      ┆ bool      │
    ╞═══════╪══════════╪═══════════╡
    │ 1     ┆ 0        ┆ null      │
    │ 1     ┆ 1        ┆ false     │
    │ 1     ┆ 0        ┆ true      │
    │ 2     ┆ 0        ┆ null      │
    │ 2     ┆ 1        ┆ false     │
    └───────┴──────────┴───────────┘
    """
    delta_in_expr = as_expr(delta_in)
    return (delta_in_expr < 0).alias('is_reg_in')


def is_reg_out(delta_out: str | pl.Expr = 'delta_out') -> pl.Expr:
    """Flag fixations that depart to a lower-index word (regression out).

    Row-wise on materialized input columns, so no window is needed then. A window-dependent
    input expression such as a composed :func:`delta_out` reintroduces the need for
    ``.over(...)``.

    Parameters
    ----------
    delta_out : str | pl.Expr
        Column name or expression of the word index difference to the next fixation.
        (default: ``'delta_out'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_reg_out`` column.

    Examples
    --------
    The last fixation has no successor, so its ``delta_out`` (and thus ``is_reg_out``) is null:

    >>> import polars as pl
    >>> from pymovements.measure.reading import is_reg_out
    >>> fixations = pl.DataFrame({'delta_out': [1, 0, -1, 2, None]})
    >>> fixations.with_columns(is_reg_out())
    shape: (5, 2)
    ┌───────────┬────────────┐
    │ delta_out ┆ is_reg_out │
    │ ---       ┆ ---        │
    │ i64       ┆ bool       │
    ╞═══════════╪════════════╡
    │ 1         ┆ false      │
    │ 0         ┆ false      │
    │ -1        ┆ true       │
    │ 2         ┆ false      │
    │ null      ┆ null       │
    └───────────┴────────────┘

    Composing the :func:`delta_out` and :func:`next_word_idx` factories computes the flag straight
    from the word indices; the composed expression then contains a window and needs ``.over(...)``
    to partition into reading sequences:

    >>> from pymovements.measure.reading import delta_out, next_word_idx
    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 0, 1],
    ... })
    >>> fixations.with_columns(
    ...     is_reg_out(delta_out(next_word_idx=next_word_idx())).over('trial'),
    ... )
    shape: (5, 3)
    ┌───────┬──────────┬────────────┐
    │ trial ┆ word_idx ┆ is_reg_out │
    │ ---   ┆ ---      ┆ ---        │
    │ i64   ┆ i64      ┆ bool       │
    ╞═══════╪══════════╪════════════╡
    │ 1     ┆ 0        ┆ false      │
    │ 1     ┆ 1        ┆ true       │
    │ 1     ┆ 0        ┆ null       │
    │ 2     ┆ 0        ┆ false      │
    │ 2     ┆ 1        ┆ null       │
    └───────┴──────────┴────────────┘
    """
    delta_out_expr = as_expr(delta_out)
    return (delta_out_expr < 0).alias('is_reg_out')


def is_first_fixation(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Flag the first fixation on each word.

    Apply ``.over(group_columns + ['word_idx'])`` so the flag is evaluated per word within each
    reading sequence.

    .. warning:: Requires onset-sorted input within each sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``is_first_fix`` column.

    Examples
    --------
    Apply ``.over('word_idx')`` (or ``.over(group_columns + ['word_idx'])``) so the flag marks the
    first fixation of each word rather than only the first row overall:

    >>> import polars as pl
    >>> from pymovements.measure.reading import is_first_fixation
    >>> fixations = pl.DataFrame({'word_idx': [0, 1, 1, 0, 2]})
    >>> fixations.with_columns(is_first_fixation().over('word_idx'))
    shape: (5, 2)
    ┌──────────┬──────────────┐
    │ word_idx ┆ is_first_fix │
    │ ---      ┆ ---          │
    │ i64      ┆ bool         │
    ╞══════════╪══════════════╡
    │ 0        ┆ true         │
    │ 1        ┆ true         │
    │ 1        ┆ false        │
    │ 0        ┆ false        │
    │ 2        ┆ true         │
    └──────────┴──────────────┘

    With group columns in the window, refixations only count within their own sequence, so the
    same word is flagged again in a new trial:

    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 0, 1, 0, 0],
    ... })
    >>> fixations.with_columns(is_first_fixation().over(['trial', 'word_idx']))
    shape: (5, 3)
    ┌───────┬──────────┬──────────────┐
    │ trial ┆ word_idx ┆ is_first_fix │
    │ ---   ┆ ---      ┆ ---          │
    │ i64   ┆ i64      ┆ bool         │
    ╞═══════╪══════════╪══════════════╡
    │ 1     ┆ 0        ┆ true         │
    │ 1     ┆ 0        ┆ false        │
    │ 1     ┆ 1        ┆ true         │
    │ 2     ┆ 0        ┆ true         │
    │ 2     ┆ 0        ┆ false        │
    └───────┴──────────┴──────────────┘
    """
    word_idx_expr = as_expr(word_idx)
    return word_idx_expr.cum_count().eq(1).alias('is_first_fix')


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

    .. warning:: Requires onset-sorted input within each sequence.

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

    Examples
    --------
    Refixating the rightmost word read so far stays first-pass (the two ``word_idx == 2`` rows
    sharing run 3), while returning to it after a word further right has been read does not (the
    last row, run 5). The input must already carry ``run_id`` (see :func:`run_id`) and be
    onset-sorted:

    >>> import polars as pl
    >>> from pymovements.measure.reading import is_first_pass
    >>> fixations = pl.DataFrame({
    ...     'word_idx': [0, 1, 2, 2, 3, 2],
    ...     'run_id':   [1, 2, 3, 3, 4, 5],
    ... })
    >>> fixations.with_columns(is_first_pass())
    shape: (6, 3)
    ┌──────────┬────────┬───────────────┐
    │ word_idx ┆ run_id ┆ is_first_pass │
    │ ---      ┆ ---    ┆ ---           │
    │ i64      ┆ i64    ┆ bool          │
    ╞══════════╪════════╪═══════════════╡
    │ 0        ┆ 1      ┆ true          │
    │ 1        ┆ 2      ┆ true          │
    │ 2        ┆ 3      ┆ true          │
    │ 2        ┆ 3      ┆ true          │
    │ 3        ┆ 4      ┆ true          │
    │ 2        ┆ 5      ┆ false         │
    └──────────┴────────┴───────────────┘

    With ``group_columns``, each sequence is evaluated independently, so the regressed-to word 0
    of trial 1 is first-pass again in trial 2:

    >>> fixations = pl.DataFrame({
    ...     'trial':    [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 0, 1],
    ...     'run_id':   [1, 2, 3, 1, 2],
    ... })
    >>> fixations.with_columns(is_first_pass(group_columns=['trial']))
    shape: (5, 4)
    ┌───────┬──────────┬────────┬───────────────┐
    │ trial ┆ word_idx ┆ run_id ┆ is_first_pass │
    │ ---   ┆ ---      ┆ ---    ┆ ---           │
    │ i64   ┆ i64      ┆ i64    ┆ bool          │
    ╞═══════╪══════════╪════════╪═══════════════╡
    │ 1     ┆ 0        ┆ 1      ┆ true          │
    │ 1     ┆ 1        ┆ 2      ┆ true          │
    │ 1     ┆ 0        ┆ 3      ┆ false         │
    │ 2     ┆ 0        ┆ 1      ┆ true          │
    │ 2     ┆ 1        ┆ 2      ┆ true          │
    └───────┴──────────┴────────┴───────────────┘
    """
    group_columns = list(group_columns or [])
    word_idx_expr = as_expr(word_idx)
    run_id_expr = as_expr(run_id)

    no_higher_word_seen = (
        word_idx_expr >= _over(word_idx_expr.cum_max().shift(), group_columns)
    ).fill_null(True)

    first_run_of_word = (
        run_id_expr == run_id_expr.min().over(group_columns + [word_idx_expr])
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

    .. warning:: Requires onset-sorted input within each sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Expression producing the ``regression_path_word`` column.

    Examples
    --------
    Every fixation is attributed to the current running maximum of fixated word indices, so the
    regression to word 0 still belongs to word 1's regression path:

    >>> import polars as pl
    >>> from pymovements.measure.reading import regression_path_word
    >>> fixations = pl.DataFrame({'word_idx': [0, 1, 1, 0, 2]})
    >>> fixations.with_columns(regression_path_word())
    shape: (5, 2)
    ┌──────────┬──────────────────────┐
    │ word_idx ┆ regression_path_word │
    │ ---      ┆ ---                  │
    │ i64      ┆ i64                  │
    ╞══════════╪══════════════════════╡
    │ 0        ┆ 0                    │
    │ 1        ┆ 1                    │
    │ 1        ┆ 1                    │
    │ 0        ┆ 1                    │
    │ 2        ┆ 2                    │
    └──────────┴──────────────────────┘

    Partitioned with ``.over(...)``, the running maximum restarts for each reading sequence:

    >>> fixations = pl.DataFrame({
    ...     'trial': [1, 1, 1, 2, 2],
    ...     'word_idx': [0, 1, 0, 2, 0],
    ... })
    >>> fixations.with_columns(regression_path_word().over('trial'))
    shape: (5, 3)
    ┌───────┬──────────┬──────────────────────┐
    │ trial ┆ word_idx ┆ regression_path_word │
    │ ---   ┆ ---      ┆ ---                  │
    │ i64   ┆ i64      ┆ i64                  │
    ╞═══════╪══════════╪══════════════════════╡
    │ 1     ┆ 0        ┆ 0                    │
    │ 1     ┆ 1        ┆ 1                    │
    │ 1     ┆ 0        ┆ 1                    │
    │ 2     ┆ 2        ┆ 2                    │
    │ 2     ┆ 0        ┆ 2                    │
    └───────┴──────────┴──────────────────────┘
    """
    word_idx_expr = as_expr(word_idx)
    return word_idx_expr.cum_max().alias('regression_path_word')


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
    * **is_first_fix**: whether this is the first fixation ever on the word within the reading
      sequence.
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

    Examples
    --------
    Four fixations over a three-word text, with a regression from the second word back to the
    first:

    >>> import polars as pl
    >>> from pymovements.measure.reading import annotate_fixations
    >>> fixations = pl.DataFrame({
    ...     'name': ['fixation'] * 4,
    ...     'onset': [0, 250, 500, 750],
    ...     'word_idx': [0, 1, 0, 2],
    ... })
    >>> annotated = annotate_fixations(fixations)
    >>> annotated.select('word_idx', 'run_id', 'is_first_pass', 'regression_path_word')
    shape: (4, 4)
    ┌──────────┬────────┬───────────────┬──────────────────────┐
    │ word_idx ┆ run_id ┆ is_first_pass ┆ regression_path_word │
    │ ---      ┆ ---    ┆ ---           ┆ ---                  │
    │ i64      ┆ i64    ┆ bool          ┆ i64                  │
    ╞══════════╪════════╪═══════════════╪══════════════════════╡
    │ 0        ┆ 1      ┆ true          ┆ 0                    │
    │ 1        ┆ 2      ┆ true          ┆ 1                    │
    │ 0        ┆ 3      ┆ false         ┆ 1                    │
    │ 2        ┆ 4      ┆ true          ┆ 2                    │
    └──────────┴────────┴───────────────┴──────────────────────┘

    With ``group_columns``, each trial is annotated independently: run IDs restart, and word 1
    of trial 2 counts as first-pass even though trial 1 already fixated it. Within trial 2 the
    later fixation on word 0 arrives from the right and is thus not first-pass:

    >>> fixations = pl.DataFrame({
    ...     'name': ['fixation'] * 4,
    ...     'onset': [0, 250, 0, 250],
    ...     'word_idx': [0, 1, 1, 0],
    ...     'trial': [1, 1, 2, 2],
    ... })
    >>> annotated = annotate_fixations(fixations, group_columns=['trial'])
    >>> annotated.select('trial', 'word_idx', 'run_id', 'is_first_pass')
    shape: (4, 4)
    ┌───────┬──────────┬────────┬───────────────┐
    │ trial ┆ word_idx ┆ run_id ┆ is_first_pass │
    │ ---   ┆ ---      ┆ ---    ┆ ---           │
    │ i64   ┆ i64      ┆ i64    ┆ bool          │
    ╞═══════╪══════════╪════════╪═══════════════╡
    │ 1     ┆ 0        ┆ 1      ┆ true          │
    │ 1     ┆ 1        ┆ 2      ┆ true          │
    │ 2     ┆ 1        ┆ 1      ┆ true          │
    │ 2     ┆ 0        ┆ 2      ┆ false         │
    └───────┴──────────┴────────┴───────────────┘
    """
    group_columns = list(group_columns or [])
    word_idx_expr = as_expr(word_idx)

    fixations = (
        events.filter((pl.col('name') == event_name) & (word_idx_expr.is_not_null()))
        .with_row_index('fixation_id')
    )

    if not fixations.is_empty():
        onsets_sorted = fixations.select(
            (_over(pl.col('onset').diff(), group_columns) >= 0).all(),
        ).item()
        if not onsets_sorted:
            warnings.warn(
                'fixation onsets are not sorted within a reading sequence; sorting by onset. '
                'If these fixations span several trials or pages, pass group_columns.',
            )

    fixations = (
        fixations
        # Every downstream expression assumes onset-sorted rows: the annotations use running
        # windows (cum_max / cum_count / shift) and the word-level measures read the first row of
        # each group (.first()), so both encode "temporally first" as "first by row position".
        # fixation_id breaks onset ties deterministically (it preserves the input order), so these
        # stay reproducible even when two fixations share an onset.
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
