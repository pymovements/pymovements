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
"""Reading measure computation functions."""
from __future__ import annotations

import polars as pl


# ---------------------------
# Basic fixation-based counts
# ---------------------------


def compute_total_fixation_count(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the total number of fixations on each word (TFC).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``, and
        ``word_idx`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``TFC``.
    """
    return fix.group_by(['trial', 'page', 'word_idx']).len().rename({'len': 'TFC'})


def compute_first_pass_fixation_count(
    fix: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the number of fixations during the first pass (FPFC).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, and ``is_first_pass`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``FPFC``.
    """
    return (
        fix.filter(pl.col('is_first_pass'))
        .group_by(['trial', 'page', 'word_idx'])
        .len()
        .rename({'len': 'FPFC'})
    )


def compute_first_duration(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the duration of the first fixation on each word (FD).

    The first fixation is determined by the earliest ``onset`` value,
    regardless of reading pass.

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``onset``, and ``duration`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``FD``.
    """
    return fix.group_by(['trial', 'page', 'word_idx']).agg(
        pl.col('duration').sort_by('onset').first().alias('FD'),
    )


def compute_first_reading_time(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the sum of fixation durations during the first run (FRT).

    FRT is the total dwell time from first entering a word until first
    leaving it (i.e., the first contiguous run of fixations).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``run_id``, and ``duration`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``FRT``.
    """
    return (
        fix.group_by(['trial', 'page', 'word_idx', 'run_id'])
        .agg(pl.col('duration').sum().alias('run_duration'))
        .sort(['trial', 'page', 'word_idx', 'run_id'])
        .group_by(['trial', 'page', 'word_idx'])
        .first()
        .select(['trial', 'page', 'word_idx', 'run_duration'])
        .rename({'run_duration': 'FRT'})
    )


def compute_first_fixation_duration(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the duration of the first fixation during first pass only (FFD).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``is_first_pass``, ``onset``, and ``duration``
        columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``FFD``.
    """
    return (
        fix.filter(pl.col('is_first_pass'))
        .group_by(['trial', 'page', 'word_idx'])
        .agg(pl.col('duration').sort_by('onset').first().alias('FFD'))
    )


def compute_first_pass_reading_time(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the sum of fixation durations during the first pass (FPRT).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``is_first_pass``, and ``duration`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``FPRT``.
    """
    return (
        fix.filter(pl.col('is_first_pass'))
        .group_by(['trial', 'page', 'word_idx'])
        .agg(pl.col('duration').sum().alias('FPRT'))
    )


def compute_rereading_time(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the sum of fixation durations outside the first pass (RRT).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``is_first_pass``, and ``duration`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``RRT``.
    """
    return (
        fix.filter(~pl.col('is_first_pass'))
        .group_by(['trial', 'page', 'word_idx'])
        .agg(pl.col('duration').sum().alias('RRT'))
    )


# ---------------------------
# Transition-based measures
# ---------------------------


def compute_trc_in_out(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute regression counts into and out of each word (TRC_in, TRC_out).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``is_reg_in``, and ``is_reg_out`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``,
        ``TRC_in``, and ``TRC_out``.
    """
    return fix.group_by(['trial', 'page', 'word_idx']).agg(
        [
            pl.col('is_reg_in').sum().alias('TRC_in'),
            pl.col('is_reg_out').sum().alias('TRC_out'),
        ],
    )


def compute_landing_position(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the character index of the first fixation on each word (LP).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``onset``, and ``char_idx`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``LP``.
    """
    return fix.group_by(['trial', 'page', 'word_idx']).agg(
        pl.col('char_idx').sort_by('onset').first().alias('LP'),
    )


def compute_sl_in(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the saccade length at word entry (SL_in).

    SL_in is the signed word distance between the current word and the
    previously fixated word at the moment of the very first fixation on
    the current word.

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``is_first_fix``, and ``prev_word_idx`` columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``SL_in``.
    """
    return (
        fix.filter(pl.col('is_first_fix'))
        .with_columns((pl.col('word_idx') - pl.col('prev_word_idx')).alias('SL_in'))
        .select(['trial', 'page', 'word_idx', 'SL_in'])
    )


def compute_sl_out(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute the saccade length at first-pass word exit (SL_out).

    SL_out is the signed word distance from the current word to the next
    fixated word, measured at the last fixation of the first run.

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``run_id``, ``onset``, and ``next_word_idx``
        columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``, and
        ``SL_out``.
    """
    first_run = fix.group_by(['trial', 'page', 'word_idx']).agg(
        pl.col('run_id').min().alias('first_run'),
    )

    last_fix = (
        fix.join(first_run, on=['trial', 'page', 'word_idx'])
        .filter(pl.col('run_id') == pl.col('first_run'))
        .group_by(['trial', 'page', 'word_idx'])
        .agg(pl.all().sort_by('onset').last())
    )

    return last_fix.with_columns(
        (pl.col('next_word_idx') - pl.col('word_idx')).fill_null(0).alias('SL_out'),
    ).select(['trial', 'page', 'word_idx', 'SL_out'])


# ---------------------------
# Regression-path measures
# ---------------------------


def compute_rpd_measures(fix: pl.DataFrame) -> pl.DataFrame:
    """Compute regression-path duration and related measures (RPD, RBRT).

    Computes three measures for each word:

    * **RPD_inc** – sum of all fixation durations from first entering
      the word until the first fixation to its right, *including*
      fixations on the word itself.
    * **RPD_exc** – same window, *excluding* fixations on the word
      itself (i.e., time spent on regressed words only).
    * **RBRT** – sum of fixation durations on the word before any word
      to its right is visited (right-bounded reading time).

    Parameters
    ----------
    fix : pl.DataFrame
        Fixation table containing at least ``trial``, ``page``,
        ``word_idx``, ``onset``, ``duration``, and ``is_first_pass``
        columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``trial``, ``page``, ``word_idx``,
        ``RPD_inc``, ``RPD_exc``, and ``RBRT``.
    """
    fix = fix.collect() if isinstance(fix, pl.LazyFrame) else fix

    def per_group(df: pl.DataFrame) -> pl.DataFrame:
        rows = []

        for w in df['word_idx'].unique().to_list():
            first = (
                df.filter((pl.col('word_idx') == w) & (pl.col('is_first_pass')))
                .sort('onset')
                .head(1)
            )

            if first.height == 0:
                rows.append((w, 0, 0, 0))
                continue

            start_onset = first['onset'][0]

            after = df.filter(pl.col('onset') >= start_onset)

            exit_right = after.filter(pl.col('word_idx') > w).sort('onset').head(1)

            if exit_right.height > 0:
                stop_onset = exit_right['onset'][0]
                window = after.filter(pl.col('onset') < stop_onset)
            else:
                window = after

            rbrt = window.filter(pl.col('word_idx') == w)['duration'].sum()
            rpd_exc = window.filter(pl.col('word_idx') != w)['duration'].sum()
            rpd_inc = rbrt + rpd_exc

            rows.append((w, rpd_inc, rpd_exc, rbrt))

        return pl.DataFrame(
            rows,
            schema=['word_idx', 'RPD_inc', 'RPD_exc', 'RBRT'],
            orient='row',
        ).with_columns(
            [
                pl.lit(df['trial'][0]).alias('trial'),
                pl.lit(df['page'][0]).alias('page'),
            ],
        )

    return fix.group_by('trial', 'page', maintain_order=True).map_groups(per_group)


# ---------------------------
# Word-level table
# ---------------------------


def build_word_level_table(
    words: pl.DataFrame,
    fix: pl.DataFrame,
) -> pl.DataFrame:
    """Join all reading measures onto a word-level table.

    Computes every individual reading measure and left-joins them onto
    ``words``, filling missing values with ``0``. Derived measures
    (TFT, FPF, RR, SFD) are appended as final columns.

    Parameters
    ----------
    words : pl.DataFrame
        Base word table containing at least ``trial``, ``page``, and
        ``word_idx`` columns (one row per word).
    fix : pl.DataFrame
        Annotated fixation table as produced by
        :func:`~pymovements.measure.reading.processing.annotate_fixations`.

    Returns
    -------
    pl.DataFrame
        Word-level table with all reading measures as additional
        columns: ``TFC``, ``FD``, ``FFD``, ``FPRT``, ``FRT``,
        ``RRT``, ``FPFC``, ``TRC_in``, ``TRC_out``, ``LP``,
        ``SL_in``, ``SL_out``, ``RPD_inc``, ``RPD_exc``, ``RBRT``,
        ``TFT``, ``FPF``, ``RR``, and ``SFD``.
    """
    tfc = compute_total_fixation_count(fix)
    fd = compute_first_duration(fix)
    ffd = compute_first_fixation_duration(fix)
    fprt = compute_first_pass_reading_time(fix)
    frt = compute_first_reading_time(fix)
    rrt = compute_rereading_time(fix)
    fpfc = compute_first_pass_fixation_count(fix)
    trc = compute_trc_in_out(fix)
    lp = compute_landing_position(fix)
    sl_in = compute_sl_in(fix)
    sl_out = compute_sl_out(fix)
    rpd = compute_rpd_measures(fix)

    return (
        words.join(tfc, on=['trial', 'page', 'word_idx'], how='left')
        .join(fd, on=['trial', 'page', 'word_idx'], how='left')
        .join(ffd, on=['trial', 'page', 'word_idx'], how='left')
        .join(fprt, on=['trial', 'page', 'word_idx'], how='left')
        .join(frt, on=['trial', 'page', 'word_idx'], how='left')
        .join(rrt, on=['trial', 'page', 'word_idx'], how='left')
        .join(fpfc, on=['trial', 'page', 'word_idx'], how='left')
        .join(trc, on=['trial', 'page', 'word_idx'], how='left')
        .join(lp, on=['trial', 'page', 'word_idx'], how='left')
        .join(sl_in, on=['trial', 'page', 'word_idx'], how='left')
        .join(sl_out, on=['trial', 'page', 'word_idx'], how='left')
        .join(rpd, on=['trial', 'page', 'word_idx'], how='left')
        .with_columns(
            [
                pl.col('TFC').fill_null(0),
                pl.col('FD').fill_null(0),
                pl.col('FFD').fill_null(0),
                pl.col('FPRT').fill_null(0),
                pl.col('FRT').fill_null(0),
                pl.col('RRT').fill_null(0),
                pl.col('FPFC').fill_null(0),
                pl.col('TRC_in').fill_null(0),
                pl.col('TRC_out').fill_null(0),
                pl.col('LP').fill_null(0),
                pl.col('SL_in').fill_null(0),
                pl.col('SL_out').fill_null(0),
                pl.col('RPD_inc').fill_null(0),
                pl.col('RPD_exc').fill_null(0),
                pl.col('RBRT').fill_null(0),
            ],
        )
        # ---- derived measures ----
        .with_columns(
            [
                # total fixation time
                (pl.col('FPRT') + pl.col('RRT')).alias('TFT'),
                # binary indicators
                (pl.col('FPRT') > 0).cast(pl.Int8).alias('FPF'),
                (pl.col('RRT') > 0).cast(pl.Int8).alias('RR'),
                # single-fixation duration
                pl.when(pl.col('FPFC') == 1)
                .then(pl.col('FFD'))
                .otherwise(0)
                .alias('SFD'),
            ],
        )
    )
