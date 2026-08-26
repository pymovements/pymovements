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

To compute all reading measures from fixations and an AOI table at once, use
:func:`~pymovements.measure.reading.compute_reading_measures`, which drives the full pipeline.
The functions in this module are its building blocks, useful for computing individual measures
or custom aggregations.

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
fixations on other words, so they aggregate over ``regression_path_word`` groups instead of
``word_idx`` groups (see :func:`~pymovements.measure.reading.regression_path_word`). The
sequence-level summary measures are the other exception: they describe a whole reading
sequence, so they aggregate over reading-sequence groups (e.g. ``['trial', 'page']``).
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
    """Total number of fixations during the first pass (FPFC).

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
    """One-based character position of the first fixation on each word (LP).

    Requires onset-sorted input. The aggregation emits the ``char_idx`` of the word's first
    fixation plus one. Within :func:`~pymovements.measure.reading.compute_reading_measures` the
    word's start character is subtracted, making the position relative to the word (its first
    character is 1), and words that were never fixated are filled with 0. The resulting values
    match the landing position of the PoTeC reference implementation.

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
    return (as_expr(char_idx).first() + 1).alias('LP')


def saccade_length_in(
    word_idx: str | pl.Expr = 'word_idx',
    prev_word_idx: str | pl.Expr = 'prev_word_idx',
    is_first_fix: str | pl.Expr = 'is_first_fix',
) -> pl.Expr:
    """Saccade length at word entry (SL_in).

    SL_in is the signed word distance between the current word and the previously fixated word
    at the moment of the very first fixation on the current word, selected via the
    ``is_first_fix`` annotation (see :func:`~pymovements.measure.reading.is_first_fixation`).
    Null when the word starts the sequence.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    prev_word_idx : str | pl.Expr
        Column name or expression of the previous fixation's word index.
        (default: ``'prev_word_idx'``)
    is_first_fix : str | pl.Expr
        Column name or expression of the first-fixation flag.
        (default: ``'is_first_fix'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``SL_in`` column.
    """
    return (
        (as_expr(word_idx) - as_expr(prev_word_idx))
        .filter(as_expr(is_first_fix))
        .first()
        .alias('SL_in')
    )


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
    """Sum of all fixation durations within the regression-path window of a word (RPD_inc).

    The window spans from first entering the word until the first fixation to its right,
    *including* fixations on the word itself. Aggregate over ``regression_path_word`` groups
    instead of ``word_idx`` groups (see
    :func:`~pymovements.measure.reading.regression_path_word`).

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
    regression_path_word: str | pl.Expr = 'regression_path_word',
) -> pl.Expr:
    """Sum of the regressed-time fixation durations within the regression-path window (RPD_exc).

    Same window as :func:`regression_path_duration_inclusive`, but *excluding* fixations on the
    word itself (i.e., time spent on regressed words only). Aggregate over ``regression_path_word``
    groups.

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    regression_path_word : str | pl.Expr
        Column name or expression of the regression-path target word index.
        (default: ``'regression_path_word'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RPD_exc`` column.
    """
    return (
        as_expr(duration)
        .filter(as_expr(word_idx) != as_expr(regression_path_word))
        .sum()
        .alias('RPD_exc')
    )


def right_bounded_reading_time(
    duration: str | pl.Expr = 'duration',
    word_idx: str | pl.Expr = 'word_idx',
    regression_path_word: str | pl.Expr = 'regression_path_word',
) -> pl.Expr:
    """Sum of fixation durations on a word before any word to its right is visited (RBRT).

    Aggregate over ``regression_path_word`` groups (see
    :func:`~pymovements.measure.reading.regression_path_word`).

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)
    regression_path_word : str | pl.Expr
        Column name or expression of the regression-path target word index.
        (default: ``'regression_path_word'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``RBRT`` column.
    """
    return (
        as_expr(duration)
        .filter(as_expr(word_idx) == as_expr(regression_path_word))
        .sum()
        .alias('RBRT')
    )


# ---------------------------
# Sequence-level summary measures
# ---------------------------


def non_aoi_fixation_count_ratio(word_idx: str | pl.Expr = 'word_idx') -> pl.Expr:
    """Ratio of fixations outside any AOI, by count (NAFCR).

    NAFCR is a summary of a whole reading sequence, not a word-level measure: aggregate it over
    reading-sequence groups (e.g. ``['trial', 'page']``) instead of word groups.

    A fixation counts as outside all AOIs if and only if its word index is null, which is the
    convention produced by :py:meth:`~pymovements.Events.map_to_aois`. Datasets that encode
    outside-AOI fixations with a sentinel value such as ``-1`` must convert those values to null
    first, otherwise the ratio is silently underestimated.

    The input must contain only fixation events, so filter the event frame to fixations first:
    non-fixation rows carry null AOI columns after :py:meth:`~pymovements.Events.map_to_aois`
    and would inflate the ratio.

    Parameters
    ----------
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``NAFCR`` column (proportion of fixations without
        a mapped word, 0.0 to 1.0).
    """
    return as_expr(word_idx).is_null().mean().alias('NAFCR')


def non_aoi_fixation_duration_ratio(
    duration: str | pl.Expr = 'duration',
    word_idx: str | pl.Expr = 'word_idx',
) -> pl.Expr:
    """Ratio of fixation duration outside any AOI (NAFDR).

    NAFDR is a summary of a whole reading sequence, not a word-level measure: aggregate it over
    reading-sequence groups (e.g. ``['trial', 'page']``) instead of word groups. Null when the
    total fixation duration of the group is zero.

    A fixation counts as outside all AOIs if and only if its word index is null, which is the
    convention produced by :py:meth:`~pymovements.Events.map_to_aois`. Datasets that encode
    outside-AOI fixations with a sentinel value such as ``-1`` must convert those values to null
    first, otherwise the ratio is silently underestimated.

    The input must contain only fixation events, so filter the event frame to fixations first:
    non-fixation rows carry null AOI columns after :py:meth:`~pymovements.Events.map_to_aois`
    and would inflate the ratio.

    Parameters
    ----------
    duration : str | pl.Expr
        Column name or expression of the fixation duration.
        (default: ``'duration'``)
    word_idx : str | pl.Expr
        Column name or expression of the fixated word index.
        (default: ``'word_idx'``)

    Returns
    -------
    pl.Expr
        Aggregation expression producing the ``NAFDR`` column (proportion of fixation duration
        without a mapped word, 0.0 to 1.0).
    """
    total_duration = as_expr(duration).sum()
    non_aoi_duration = as_expr(duration).filter(as_expr(word_idx).is_null()).sum()
    # A group with zero total duration has no defined ratio; dividing would yield NaN, so it
    # becomes null instead. Empty groups cannot occur inside group_by, so unlike this zero
    # total duration case they need no guard.
    return (
        pl.when(total_duration > 0)
        .then(non_aoi_duration / total_duration)
        .otherwise(None)
        .alias('NAFDR')
    )
