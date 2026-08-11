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
"""Reading measure processing functions."""
from __future__ import annotations

import polars as pl

from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_fixation_count
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import first_reading_time
from pymovements.measure.reading.measures import regression_count_in
from pymovements.measure.reading.measures import regression_count_out
from pymovements.measure.reading.measures import regression_path_duration
from pymovements.measure.reading.measures import rereading_time
from pymovements.measure.reading.measures import saccade_length_in
from pymovements.measure.reading.measures import saccade_length_out
from pymovements.measure.reading.measures import total_fixation_count
from pymovements.measure.reading.words import all_tokens_from_aois

# Grouping columns used internally to keep independent reading sequences apart. When the input
# does not carry these columns, a single constant group is synthesized (see _normalize_*).
_GROUP_COLUMNS = ['trial', 'page']

# Measures that are joined onto the word table and filled with 0 for unfixated words.
_JOINED_MEASURES = [
    'TFC', 'FD', 'FFD', 'FPRT', 'FRT', 'RRT', 'FPFC', 'TRC_in', 'TRC_out',
    'SL_in', 'SL_out', 'RPD_inc', 'RPD_exc', 'RBRT',
]

# Final output columns, in order.
_OUTPUT_COLUMNS = [
    'word_index', 'word', 'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT',
    'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
    'TRC_in', 'SL_in', 'SL_out', 'TFC',
]


def compute_reading_measures(
        fixations: pl.DataFrame,
        aois: pl.DataFrame,
        *,
        word_index_column: str = 'word_idx',
        word_column: str = 'word',
) -> pl.DataFrame:
    """Compute reading measures from fixation sequences.

    This function expects fixations annotated with AOI data. See
    :py:meth:`~pymovements.Events.map_to_aois` for further details.

    The fixations are annotated with run- and pass-level information (see
    :func:`~pymovements.measure.reading.annotate_fixations`) and each reading measure is then
    aggregated per word and joined onto the word table derived from ``aois``. Fixation order is
    taken from the ``onset`` column; if it is absent, the row order of ``fixations`` is used. If
    the fixations carry ``trial`` and ``page`` columns, they are used to keep independent reading
    sequences apart; otherwise the fixations are treated as a single sequence.

    The returned ``word_index`` preserves the word indexing of ``aois`` (zero-based, one-based, or
    any other start), so word indices are not shifted.

    Parameters
    ----------
    fixations : pl.DataFrame
        DataFrame with fixation data, containing the column specified by ``word_index_column`` and
        a ``duration`` column.
    aois : pl.DataFrame
        DataFrame with AOI data, containing the columns specified by ``word_index_column`` and
        ``word_column``.
    word_index_column : str
        Shared column name in ``fixations`` and ``aois`` that corresponds to the word index of the
        text.
        (default: ``'word_idx'``)
    word_column : str
        Column in ``aois`` with the content within each AOI.
        (default: ``'word'``)

    Returns
    -------
    pl.DataFrame
        DataFrame with computed reading measures.
    """
    words = _normalize_words(aois, word_index_column=word_index_column, word_column=word_column)
    fixations = _normalize_fixations(fixations, word_index_column=word_index_column)

    annotated = annotate_fixations(fixations, group_columns=_GROUP_COLUMNS)

    table = _assemble_word_level_measures(words, annotated)

    return table.select(_OUTPUT_COLUMNS).sort('word_index')


def _normalize_fixations(fixations: pl.DataFrame, *, word_index_column: str) -> pl.DataFrame:
    """Bring a fixation table into the internal schema expected by the annotation pipeline."""
    if word_index_column != 'word_idx':
        fixations = fixations.rename({word_index_column: 'word_idx'})

    # Word indices are integer by nature; non-integer entries become null and are dropped by the
    # annotation step. This also keeps the join dtype consistent with the word table.
    fixations = fixations.with_columns(pl.col('word_idx').cast(pl.Int64, strict=False))

    constant_columns = []
    if 'name' not in fixations.columns:
        constant_columns.append(pl.lit('fixation').alias('name'))
    if 'trial' not in fixations.columns:
        constant_columns.append(pl.lit('_trial').alias('trial'))
    if 'page' not in fixations.columns:
        constant_columns.append(pl.lit('_page').alias('page'))
    if constant_columns:
        fixations = fixations.with_columns(constant_columns)

    # Reading order defaults to row order when no explicit onset is available.
    if 'onset' not in fixations.columns:
        fixations = fixations.with_row_index('onset')

    return fixations


def _normalize_words(
        aois: pl.DataFrame,
        *,
        word_index_column: str,
        word_column: str,
) -> pl.DataFrame:
    """Build the deduplicated word table from an AOI table in the internal schema."""
    rename = {}
    if word_index_column != 'word_idx':
        rename[word_index_column] = 'word_idx'
    if word_column != 'word':
        rename[word_column] = 'word'
    if rename:
        aois = aois.rename(rename)

    aois = aois.with_columns(pl.col('word_idx').cast(pl.Int64, strict=False))

    if 'page' not in aois.columns:
        aois = aois.with_columns(pl.lit('_page').alias('page'))

    return all_tokens_from_aois(aois, trial='_trial')


def _assemble_word_level_measures(words: pl.DataFrame, fixations: pl.DataFrame) -> pl.DataFrame:
    """Aggregate every reading measure and join it onto the word table."""
    on = ['trial', 'page', 'word_idx']

    measures = [
        total_fixation_count(fixations),
        first_duration(fixations),
        first_fixation_duration(fixations),
        first_pass_reading_time(fixations),
        first_reading_time(fixations),
        rereading_time(fixations),
        first_pass_fixation_count(fixations),
        regression_count_in(fixations),
        regression_count_out(fixations),
        saccade_length_in(fixations),
        saccade_length_out(fixations),
        regression_path_duration(fixations),
    ]

    table = words
    for measure in measures:
        table = table.join(measure, on=on, how='left', nulls_equal=True)

    table = table.with_columns([pl.col(column).fill_null(0) for column in _JOINED_MEASURES])

    return table.with_columns([
        # total fixation time
        (pl.col('FPRT') + pl.col('RRT')).alias('TFT'),
        # single-fixation duration: only defined when the word had a single first-pass fixation
        pl.when(pl.col('FPFC') == 1).then(pl.col('FFD')).otherwise(0).alias('SFD'),
        # binary indicators
        (pl.col('FPRT') > 0).cast(pl.Int64).alias('FPF'),
        (pl.col('RRT') > 0).cast(pl.Int64).alias('RR'),
        (pl.col('RPD_exc') > 0).cast(pl.Int64).alias('FPReg'),
    ]).with_columns([
        # fixated-at-all indicator (depends on the derived TFT column above)
        (pl.col('TFT') > 0).cast(pl.Int64).alias('Fix'),
    ]).rename({'word_idx': 'word_index'})
