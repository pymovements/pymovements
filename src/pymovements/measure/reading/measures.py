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
"""Reading measure computation functions.

Every function returns a polars aggregation expression to be used inside a
``group_by(...).agg(...)`` over word groups, e.g.::

    fixations.group_by(['trial', 'word_idx']).agg([
        total_fixation_count(),
        first_fixation_duration(),
    ])

The grouping itself is chosen by the caller, so any partitioning (or none at all) works. Input
columns can be given as column names or as arbitrary polars expressions. The expressions expect
the fixation table to be annotated (see
:func:`~pymovements.measure.reading.annotate_fixations`) and sorted by ``onset`` within each
group, which the annotation step guarantees.

The regression-path measures are the exception to the word grouping: their windows span
fixations on other words, so they aggregate over ``rpd_target`` groups instead of ``word_idx``
groups (see :func:`~pymovements.measure.reading.rpd_target`).
"""
from __future__ import annotations

import polars as pl

from pymovements._utils._expressions import as_expr


# ---------------------------
# Basic fixation-based counts
# ---------------------------


def total_fixation_count() -> pl.Expr:
    """Total number of fixations on each word (TFC).

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``TFC`` column.
    """
    return pl.len().cast(pl.UInt64).alias('TFC')


def first_pass_fixation_count(is_first_pass: str | pl.Expr = 'is_first_pass') -> pl.Expr:
    """Count the fixations during the first pass (FPFC).

    Parameters
    ----------
    is_first_pass : str | pl.Expr
        Column name or expression of the first-pass flag.
        (default: ``'is_first_pass'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``FPFC`` column.
    """
    return as_expr(is_first_pass).sum().cast(pl.UInt64).alias('FPFC')


def first_duration(duration: str | pl.Expr = 'duration') -> pl.Expr:
    """Duration of the first fixation on each word (FD), regardless of reading pass.

    Requires onset-sorted input.

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``FD`` column.
    """
    return as_expr(duration).first().alias('FD')


def first_reading_time(
    duration: str | pl.Expr = 'duration',
    run_id: str | pl.Expr = 'run_id',
) -> pl.Expr:
    """Sum of fixation durations during the first run (FRT).

    FRT is the total dwell time from first entering a word until first leaving it (i.e., the
    first contiguous run of fixations).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    run_id : str | pl.Expr
        Column name or expression of the run ID.
        (default: ``'run_id'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``FRT`` column.
    """
    return (
        as_expr(duration)
        .filter(as_expr(run_id) == as_expr(run_id).min())
        .sum()
        .alias('FRT')
    )


def first_fixation_duration(
    duration: str | pl.Expr = 'duration',
    is_first_pass: str | pl.Expr = 'is_first_pass',
) -> pl.Expr:
    """Duration of the first fixation during first pass only (FFD).

    Requires onset-sorted input.

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    is_first_pass : str | pl.Expr
        Column name or expression of the first-pass flag.
        (default: ``'is_first_pass'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``FFD`` column.
    """
    return as_expr(duration).filter(as_expr(is_first_pass)).first().alias('FFD')


def first_pass_reading_time(
    duration: str | pl.Expr = 'duration',
    is_first_pass: str | pl.Expr = 'is_first_pass',
) -> pl.Expr:
    """Sum of fixation durations during the first pass (FPRT).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    is_first_pass : str | pl.Expr
        Column name or expression of the first-pass flag.
        (default: ``'is_first_pass'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``FPRT`` column.
    """
    return as_expr(duration).filter(as_expr(is_first_pass)).sum().alias('FPRT')


def rereading_time(
    duration: str | pl.Expr = 'duration',
    is_first_pass: str | pl.Expr = 'is_first_pass',
) -> pl.Expr:
    """Sum of fixation durations outside the first pass (RRT).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    is_first_pass : str | pl.Expr
        Column name or expression of the first-pass flag.
        (default: ``'is_first_pass'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RRT`` column.
    """
    return as_expr(duration).filter(~as_expr(is_first_pass)).sum().alias('RRT')


# ---------------------------
# Transition-based measures
# ---------------------------


def regression_count_in(is_reg_in: str | pl.Expr = 'is_reg_in') -> pl.Expr:
    """Regression count into each word (TRC_in).

    Parameters
    ----------
    is_reg_in : str | pl.Expr
        Column name or expression of the regression-in flag.
        (default: ``'is_reg_in'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``TRC_in`` column.
    """
    return as_expr(is_reg_in).sum().cast(pl.UInt64).alias('TRC_in')


def regression_count_out(is_reg_out: str | pl.Expr = 'is_reg_out') -> pl.Expr:
    """Regression count out of each word (TRC_out).

    Parameters
    ----------
    is_reg_out : str | pl.Expr
        Column name or expression of the regression-out flag.
        (default: ``'is_reg_out'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``TRC_out`` column.
    """
    return as_expr(is_reg_out).sum().cast(pl.UInt64).alias('TRC_out')


def landing_position(char_idx: str | pl.Expr = 'char_idx') -> pl.Expr:
    """Character index of the first fixation on each word (LP).

    Requires onset-sorted input.

    Parameters
    ----------
    char_idx : str | pl.Expr
        Column name or expression of the fixated character index.
        (default: ``'char_idx'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``LP`` column.
    """
    return as_expr(char_idx).first().alias('LP')


def saccade_length_in(
    word_idx: str | pl.Expr = 'word_idx',
    prev_word_idx: str | pl.Expr = 'prev_word_idx',
) -> pl.Expr:
    """Saccade length at word entry (SL_in).

    SL_in is the signed word distance between the current word and the previously fixated word
    at the moment of the very first fixation on the current word. Requires onset-sorted input.
    Null when the word starts the sequence.

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
        Aggregation expression producing the ``SL_in`` column.
    """
    return (as_expr(word_idx) - as_expr(prev_word_idx)).first().alias('SL_in')


def saccade_length_out(
    word_idx: str | pl.Expr = 'word_idx',
    next_word_idx: str | pl.Expr = 'next_word_idx',
    run_id: str | pl.Expr = 'run_id',
) -> pl.Expr:
    """Saccade length at first-pass word exit (SL_out).

    SL_out is the signed word distance from the current word to the next fixated word, measured
    at the last fixation of the first run. Requires onset-sorted input. Zero when the word ends
    the sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    next_word_idx : str | pl.Expr
        Column name or expression of the next fixation's word index.
        (default: ``'next_word_idx'``)
    run_id : str | pl.Expr
        Column name or expression of the run ID.
        (default: ``'run_id'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``SL_out`` column.
    """
    return (
        (as_expr(next_word_idx) - as_expr(word_idx))
        .filter(as_expr(run_id) == as_expr(run_id).min())
        .last()
        .fill_null(0)
        .alias('SL_out')
    )


# ---------------------------
# Regression-path measures
# ---------------------------


def regression_path_duration_inclusive(duration: str | pl.Expr = 'duration') -> pl.Expr:
    """Sum all fixation durations within the regression-path window of a word (RPD_inc).

    The window spans from first entering the word until the first fixation to its right,
    *including* fixations on the word itself. Aggregate over ``rpd_target`` groups instead of
    ``word_idx`` groups (see :func:`~pymovements.measure.reading.rpd_target`).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RPD_inc`` column.
    """
    return as_expr(duration).sum().alias('RPD_inc')


def regression_path_duration_exclusive(
    duration: str | pl.Expr = 'duration',
    word_idx: str | pl.Expr = 'word_idx',
    rpd_target: str | pl.Expr = 'rpd_target',
) -> pl.Expr:
    """Sum the regressed-time fixation durations within the regression-path window (RPD_exc).

    Same window as :func:`regression_path_duration_inclusive`, but *excluding* fixations on the
    word itself (i.e., time spent on regressed words only). Aggregate over ``rpd_target``
    groups.

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    rpd_target : str | pl.Expr
        Column name or expression of the regression-path target word index.
        (default: ``'rpd_target'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RPD_exc`` column.
    """
    return (
        as_expr(duration)
        .filter(as_expr(word_idx) != as_expr(rpd_target))
        .sum()
        .alias('RPD_exc')
    )


def right_bounded_reading_time(
    duration: str | pl.Expr = 'duration',
    word_idx: str | pl.Expr = 'word_idx',
    rpd_target: str | pl.Expr = 'rpd_target',
) -> pl.Expr:
    """Sum the fixation durations on a word before any word to its right is visited (RBRT).

    Aggregate over ``rpd_target`` groups (see
    :func:`~pymovements.measure.reading.rpd_target`).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    rpd_target : str | pl.Expr
        Column name or expression of the regression-path target word index.
        (default: ``'rpd_target'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RBRT`` column.
    """
    return (
        as_expr(duration)
        .filter(as_expr(word_idx) == as_expr(rpd_target))
        .sum()
        .alias('RBRT')
    )
