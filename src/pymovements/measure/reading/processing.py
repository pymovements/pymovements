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

import warnings
from typing import Any

import polars as pl

from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_fixation_count
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import first_reading_time
from pymovements.measure.reading.measures import landing_position
from pymovements.measure.reading.measures import regression_count_in
from pymovements.measure.reading.measures import regression_count_out
from pymovements.measure.reading.measures import regression_path_duration_exclusive
from pymovements.measure.reading.measures import regression_path_duration_inclusive
from pymovements.measure.reading.measures import rereading_time
from pymovements.measure.reading.measures import right_bounded_reading_time
from pymovements.measure.reading.measures import saccade_length_in
from pymovements.measure.reading.measures import saccade_length_out
from pymovements.measure.reading.measures import total_fixation_count

# Measure output columns, in order. Group columns and word identity are prepended on output.
_MEASURE_COLUMNS = [
    'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT',
    'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'skipped', 'FPF', 'RR', 'FPReg', 'TRC_out',
    'TRC_in', 'SL_in', 'SL_out', 'LP', 'TFC',
]


def compute_reading_measures(
        fixations: pl.DataFrame,
        aois: pl.DataFrame | dict[Any, pl.DataFrame],
        *,
        word_index_column: str = 'word_idx',
        word_column: str = 'word',
        event_name: str = 'fixation',
        group_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Compute reading measures from fixation sequences.

    This function expects fixations annotated with AOI data. See
    :py:meth:`~pymovements.Events.map_to_aois` for further details.

    The fixations are annotated with run- and pass-level information (see
    :func:`~pymovements.measure.reading.annotate_fixations`) and each reading measure is then
    aggregated per word and joined onto the word table derived from ``aois``. Fixation order is
    taken from the ``onset`` column; if it is absent, the row order of ``fixations`` is used.

    Independent reading sequences are kept apart via the ``group_columns``. The first group
    column identifies the recording (trial role), the remaining group columns describe the
    stimulus layout (page role):

    * If both inputs carry a group column, it is used for grouping and preserved in the output.
    * If ``aois`` is a dict, its keys are the values of the first group column and each entry is
      the AOI table of that group. The fixations must then carry the first group column and
      every fixation group must have an entry in the dict.
    * If the fixations carry the first group column but a plain ``aois`` frame does not, the AOI
      table is broadcast to every fixation group and a warning is issued.
    * Any other one-sided group column raises a ``ValueError``, as the join would otherwise
      silently produce all-zero measures.
    * If neither input carries a group column, the fixations are treated as a single sequence.

    The returned ``word_index`` preserves the word indexing of ``aois`` (zero-based, one-based, or
    any other start), so word indices are not shifted. Fixations whose word index has no entry in
    the AOI table (for example sentinel values like ``-1`` for out-of-text fixations) are treated
    like fixations with a null word index: they are excluded from the fixation sequence and do not
    affect any measure.

    The landing position ``LP`` is the zero-based character position of the first fixation within
    the word, computed from the ``char_idx`` columns of fixations and AOI table. If either input
    has no ``char_idx`` column, ``LP`` is null.

    Parameters
    ----------
    fixations : pl.DataFrame
        DataFrame with fixation data, containing the column specified by ``word_index_column`` and
        a ``duration`` column.
    aois : pl.DataFrame | dict[Any, pl.DataFrame]
        DataFrame with AOI data, containing the columns specified by ``word_index_column`` and
        ``word_column``. Alternatively a dict mapping values of the first group column to such
        DataFrames.
    word_index_column : str
        Shared column name in ``fixations`` and ``aois`` that corresponds to the word index of the
        text.
        (default: ``'word_idx'``)
    word_column : str
        Column in ``aois`` with the content within each AOI.
        (default: ``'word'``)
    event_name : str
        Name of the fixation events to compute measures from. Rows of ``fixations`` with a
        different ``name`` are dropped. If ``fixations`` has no ``name`` column, all rows are
        used.
        (default: ``'fixation'``)
    group_columns : list[str] | None
        Column names used to partition the data into independent reading sequences. The first
        column takes the trial role, the remaining columns take the page role (see above). An
        empty list disables grouping entirely and treats the input as a single reading
        sequence. If ``None``, defaults to ``['trial', 'page']``.

    Returns
    -------
    pl.DataFrame
        DataFrame with computed reading measures, one row per word, prefixed by the group columns
        that were present in the input.

    Raises
    ------
    ValueError
        If ``group_columns`` contains reserved column names, if a group column is present in
        only one of the inputs (except for the broadcast case described above), if its dtypes
        do not match, or if the ``aois`` dict is inconsistent with the fixation groups.
    """
    if group_columns is None:
        group_columns = ['trial', 'page']
    # Columns the pipeline produces or consumes cannot double as group columns: the annotation
    # step would silently overwrite them and the measures would be computed on wrong groups.
    reserved_columns = {
        # word identity columns: renamed onto the internal schema, 'word_index' is the output
        word_index_column, word_column, 'word_idx', 'word', 'word_index',
        # fixation schema columns consumed by the pipeline
        'name', 'onset', 'duration',
        # annotation columns produced by annotate_fixations
        'fixation_id', 'run_id', 'prev_word_idx', 'next_word_idx', 'delta_in', 'delta_out',
        'is_reg_in', 'is_reg_out', 'is_first_fix', 'is_first_pass', 'regression_path_word',
        # internal working columns
        'word_start_char', '_is_aoi', '_group',
        # measure output columns
        'FPFC', *_MEASURE_COLUMNS,
    }
    if reserved := reserved_columns.intersection(group_columns):
        raise ValueError(f'group_columns must not contain the reserved columns {sorted(reserved)}.')

    fixations = _normalize_fixations(
        fixations, word_index_column=word_index_column, event_name=event_name,
    )
    words, output_group_columns = _build_word_table(
        aois,
        fixations,
        word_index_column=word_index_column,
        word_column=word_column,
        group_columns=group_columns,
    )

    # Without group columns there is no grouping: the whole input is one reading sequence. The
    # pipeline still needs one partitioning column, so a single constant group is synthesized.
    internal_group_columns = group_columns if group_columns else ['_group']

    fixations = _with_constant_group_columns(fixations, internal_group_columns)
    words = _with_constant_group_columns(words, internal_group_columns)

    fixations = _drop_non_aoi_word_indices(
        fixations, words, internal_group_columns, word_index_column,
    )

    annotated = annotate_fixations(
        fixations,
        group_columns=internal_group_columns,
        event_name=event_name,
        word_idx=word_index_column,
    )

    table = _assemble_word_level_measures(
        words, annotated, internal_group_columns, word_index_column,
    )

    output_columns = output_group_columns + ['word_index', 'word'] + _MEASURE_COLUMNS
    return table.select(output_columns).sort(output_group_columns + ['word_index'])


def _with_constant_group_columns(frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Synthesize missing group columns as constant columns."""
    constant_columns = [
        pl.lit(f'_{column}').alias(column)
        for column in group_columns if column not in frame.columns
    ]
    if constant_columns:
        frame = frame.with_columns(constant_columns)
    return frame


def _normalize_fixations(
        fixations: pl.DataFrame,
        *,
        word_index_column: str,
        event_name: str,
) -> pl.DataFrame:
    """Bring a fixation table into the internal schema expected by the annotation pipeline."""
    # Word indices are integer by nature; non-integer entries (unparsable strings, NaN,
    # fractional numbers) become null and are dropped by the annotation step. This also keeps
    # the join dtype consistent with the word table.
    word_idx = pl.col(word_index_column).cast(pl.Int64, strict=False)
    if fixations.schema[word_index_column].is_float():
        # The cast truncates fractional values, which would silently count the fixation on the
        # next lower word; null them out instead. NaN compares unequal and becomes null too.
        word_idx = pl.when(pl.col(word_index_column) == word_idx).then(word_idx)
    fixations = fixations.with_columns(word_idx.alias(word_index_column))

    if 'name' not in fixations.columns:
        fixations = fixations.with_columns(pl.lit(event_name).alias('name'))

    # Reading order defaults to row order when no explicit onset is available.
    if 'onset' not in fixations.columns:
        fixations = fixations.with_row_index('onset')

    return fixations


def _normalize_aoi_frame(
        aois: pl.DataFrame,
        *,
        word_index_column: str,
        word_column: str,
) -> pl.DataFrame:
    """Bring an AOI table into the internal schema."""
    rename = {}
    if word_index_column != 'word_idx':
        rename[word_index_column] = 'word_idx'
    if word_column != 'word':
        rename[word_column] = 'word'
    if rename:
        aois = aois.rename(rename)

    return aois.with_columns(pl.col('word_idx').cast(pl.Int64, strict=False))


def _build_word_table(
        aois: pl.DataFrame | dict[Any, pl.DataFrame],
        fixations: pl.DataFrame,
        *,
        word_index_column: str,
        word_column: str,
        group_columns: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Build the word table from ``aois`` and validate group column symmetry with the fixations.

    Returns the word table (with group columns synthesized where neither input carries them) and
    the list of group columns that are real and belong into the output.
    """
    if isinstance(aois, dict):
        words, output_group_columns = _word_table_from_dict(
            aois,
            fixations,
            word_index_column=word_index_column,
            word_column=word_column,
            group_columns=group_columns,
        )
    else:
        words, output_group_columns = _word_table_from_frame(
            aois,
            fixations,
            word_index_column=word_index_column,
            word_column=word_column,
            group_columns=group_columns,
        )

    return words, output_group_columns


def _word_table_from_frame(
        aois: pl.DataFrame,
        fixations: pl.DataFrame,
        *,
        word_index_column: str,
        word_column: str,
        group_columns: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Build the word table from a single AOI frame."""
    aois = _normalize_aoi_frame(aois, word_index_column=word_index_column, word_column=word_column)

    if not group_columns:
        return _word_table(aois, group_columns), []

    sequence_column = group_columns[0]
    if sequence_column in aois.columns and sequence_column not in fixations.columns:
        raise ValueError(
            f'aois has a {sequence_column!r} column but fixations do not. Add a matching '
            f'{sequence_column!r} column to the fixations or remove it from the AOI table.',
        )
    for column in group_columns[1:]:
        if (column in aois.columns) != (column in fixations.columns):
            with_column, without_column = (
                ('aois', 'fixations') if column in aois.columns else ('fixations', 'aois')
            )
            raise ValueError(
                f'{with_column} has a {column!r} column but {without_column} does not. The '
                f'measures would be computed on mismatched groups; add a matching {column!r} '
                'column to both inputs or to neither.',
            )

    shared_columns = [column for column in group_columns if column in aois.columns]
    _check_group_dtypes(fixations, aois, shared_columns)

    words = _word_table(aois, group_columns)

    if sequence_column in fixations.columns and sequence_column not in words.columns:
        sequence_values = fixations.select(sequence_column).unique(maintain_order=True)
        warnings.warn(
            f'aois has no {sequence_column!r} column while the fixations do; the AOI table is '
            f'broadcast to all {sequence_values.height} fixation {sequence_column} value(s). '
            f'Pass aois as a dict keyed by {sequence_column} to assign a separate AOI table '
            'per value.',
        )
        words = sequence_values.join(words, how='cross')

    output_group_columns = [column for column in group_columns if column in fixations.columns]
    return words, output_group_columns


def _word_table_from_dict(
        aois_dict: dict[Any, pl.DataFrame],
        fixations: pl.DataFrame,
        *,
        word_index_column: str,
        word_column: str,
        group_columns: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Build the word table from a dict mapping sequence values to AOI frames."""
    if not group_columns:
        raise ValueError(
            'aois given as dict requires at least one group column for its keys.',
        )

    sequence_column = group_columns[0]
    layout_columns = group_columns[1:]

    if not aois_dict:
        raise ValueError('aois dict is empty.')
    if sequence_column not in fixations.columns:
        raise ValueError(
            f'aois is a dict keyed by {sequence_column}, but the fixations have no '
            f'{sequence_column!r} column.',
        )

    sequence_dtype = fixations.schema[sequence_column]
    try:
        sequence_keys = pl.Series(sequence_column, list(aois_dict.keys())).cast(sequence_dtype)
    except pl.exceptions.PolarsError as exception:
        raise ValueError(
            f'aois dict keys are not compatible with the fixation {sequence_column} dtype '
            f'{sequence_dtype}: {exception}',
        ) from exception

    tables = []
    entries_with_column = dict.fromkeys(layout_columns, 0)
    entries_with_char_idx = 0
    for sequence_key, frame in zip(sequence_keys, aois_dict.values()):
        frame = _normalize_aoi_frame(
            frame, word_index_column=word_index_column, word_column=word_column,
        )
        if sequence_column in frame.columns:
            raise ValueError(
                f'aois dict entries must not contain a {sequence_column!r} column; the dict '
                'key defines it.',
            )
        for column in layout_columns:
            entries_with_column[column] += int(column in frame.columns)
        entries_with_char_idx += int('char_idx' in frame.columns)
        _check_group_dtypes(
            fixations, frame, [
                column for column in layout_columns
                if column in frame.columns and column in fixations.columns
            ],
        )
        tables.append(
            _word_table(frame, group_columns).with_columns(
                pl.lit(sequence_key, dtype=sequence_dtype).alias(sequence_column),
            ),
        )

    # Mixing character-level and word-level entries would make the word tables incompatible.
    if entries_with_char_idx not in (0, len(tables)):
        raise ValueError(
            "either all or no aois dict entries must have a 'char_idx' column.",
        )

    present_layout_columns = []
    for column in layout_columns:
        if entries_with_column[column] not in (0, len(tables)):
            raise ValueError(
                f'either all or no aois dict entries must have a {column!r} column.',
            )
        if entries_with_column[column] > 0:
            present_layout_columns.append(column)
        if (entries_with_column[column] > 0) != (column in fixations.columns):
            with_column, without_column = (
                ('aois', 'fixations') if entries_with_column[column] > 0
                else ('fixations', 'aois')
            )
            raise ValueError(
                f'{with_column} has a {column!r} column but {without_column} does not. The '
                f'measures would be computed on mismatched groups; add a matching {column!r} '
                'column to both inputs or to neither.',
            )

    words = pl.concat(tables)

    missing_keys = (
        fixations.select(sequence_column).unique().join(
            words.select(sequence_column).unique(), on=sequence_column, how='anti',
        )
    )
    if missing_keys.height > 0:
        raise ValueError(
            f'fixations contain {sequence_column} values without an entry in the aois dict: '
            f'{sorted(missing_keys[sequence_column].to_list())}',
        )

    output_group_columns = [sequence_column] + present_layout_columns
    return words, output_group_columns


def _check_group_dtypes(
        fixations: pl.DataFrame,
        aois: pl.DataFrame,
        columns: list[str],
) -> None:
    """Raise if a shared group column has different dtypes in fixations and AOI table."""
    for column in columns:
        if fixations.schema[column] != aois.schema[column]:
            raise ValueError(
                f'dtype mismatch for the {column!r} column: {fixations.schema[column]} in '
                f'fixations but {aois.schema[column]} in aois. Cast one side so both match.',
            )


def _word_table(aois: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Build the deduplicated word table from a normalized AOI frame.

    Keeps one row per word (first AOI row wins on inconsistent word labels), drops AOI rows
    without a word index, and records the first character index of each word when the AOI table
    is character-level.
    """
    key_columns = [column for column in group_columns if column in aois.columns] + ['word_idx']

    aois = aois.drop_nulls('word_idx')

    words = (
        aois.unique(subset=key_columns, keep='first', maintain_order=True)
        .select(key_columns + ['word'])
    )

    if 'char_idx' in aois.columns:
        word_starts = aois.group_by(key_columns, maintain_order=True).agg(
            pl.col('char_idx').min().alias('word_start_char'),
        )
        words = words.join(word_starts, on=key_columns, how='left', nulls_equal=True)
    else:
        words = words.with_columns(pl.lit(None, dtype=pl.Int64).alias('word_start_char'))

    return words


def _drop_non_aoi_word_indices(
        fixations: pl.DataFrame,
        words: pl.DataFrame,
        group_columns: list[str],
        word_index_column: str,
) -> pl.DataFrame:
    """Null out fixation word indices without an AOI entry to keep them out of sequence logic."""
    aoi_keys = (
        words.select(group_columns + ['word_idx'])
        .unique()
        .with_columns(pl.lit(True).alias('_is_aoi'))
    )
    return (
        fixations.join(
            aoi_keys,
            left_on=group_columns + [word_index_column],
            right_on=group_columns + ['word_idx'],
            how='left',
            nulls_equal=True,
        )
        .with_columns(
            pl.when(pl.col('_is_aoi'))
            .then(pl.col(word_index_column))
            .otherwise(None)
            .alias(word_index_column),
        )
        .drop('_is_aoi')
    )


def _assemble_word_level_measures(
        words: pl.DataFrame,
        fixations: pl.DataFrame,
        group_columns: list[str],
        word_index_column: str,
) -> pl.DataFrame:
    """Aggregate every reading measure and join it onto the word table."""
    on = group_columns + ['word_idx']

    aggregations = [
        total_fixation_count(),
        first_duration(),
        first_fixation_duration(),
        first_pass_reading_time(),
        first_reading_time(),
        rereading_time(),
        first_pass_fixation_count(),
        regression_count_in(),
        regression_count_out(),
        saccade_length_in(word_idx=word_index_column),
        saccade_length_out(word_idx=word_index_column),
    ]
    if 'char_idx' in fixations.columns:
        aggregations.append(landing_position())

    word_level = fixations.group_by(
        [*group_columns, pl.col(word_index_column).alias('word_idx')],
    ).agg(aggregations)

    # The regression-path measures aggregate over regression_path_word groups: a word's
    # regression-path window spans fixations on other words, each attributed to the running
    # rightmost word.
    regression_paths = (
        fixations.group_by(group_columns + ['regression_path_word'])
        .agg([
            regression_path_duration_inclusive(),
            regression_path_duration_exclusive(word_idx=word_index_column),
            right_bounded_reading_time(word_idx=word_index_column),
        ])
        .rename({'regression_path_word': 'word_idx'})
    )

    table = (
        words
        .join(word_level, on=on, how='left', nulls_equal=True)
        .join(regression_paths, on=on, how='left', nulls_equal=True)
    )

    if 'LP' in table.columns:
        # Landing position within the word, zero-based. Null when the AOI table is word-level.
        table = table.with_columns((pl.col('LP') - pl.col('word_start_char')).alias('LP'))
    else:
        table = table.with_columns(pl.lit(None, dtype=pl.Int64).alias('LP'))
    table = table.drop('word_start_char')

    # Unfixated words get 0 for every joined measure column. LP is the exception: 0 is a valid
    # landing position, so unfixated words keep null.
    joined_measure_columns = [
        column
        for frame in (word_level, regression_paths)
        for column in frame.columns
        if column not in on and column != 'LP'
    ]
    table = table.with_columns([pl.col(column).fill_null(0) for column in joined_measure_columns])

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
        # total skipping: the word received no fixation at all
        (pl.col('TFC') == 0).cast(pl.Int64).alias('skipped'),
    ]).rename({'word_idx': 'word_index'})
