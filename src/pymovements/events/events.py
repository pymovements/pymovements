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
"""Provides the Events class."""
# pylint: disable=too-many-lines
from __future__ import annotations

import warnings
from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import Any
from typing import Literal
from typing import overload
from warnings import warn

import numpy as np
import polars
from tqdm import tqdm

from pymovements._utils import checks
from pymovements._utils.column_nesting import unnest_list_columns
from pymovements._utils.html import repr_html
from pymovements._utils.nulls import row_is_null
from pymovements._utils.sources import merge_sources
from pymovements._utils.time import normalize_duration_to_us
from pymovements._utils.time import numeric_to_duration_us
from pymovements.events.correction import fixation_correction
from pymovements.measure.reading import compute_reading_measures
from pymovements.measure.reading import ReadingMeasures
from pymovements.stimulus.text import TextStimulus

# The time columns that are converted to ``polars.Duration('us')`` on construction.
_TIME_COLUMNS = ('onset', 'duration', 'offset')

# onset, offset and duration are rounded to microseconds one by one
_RESIDUAL_TOLERANCE_US = 1


def _format_microseconds(value: int) -> str:
    """Format a residual in microseconds, in milliseconds where it is a whole number of them."""
    if value == 0:
        return '0'
    if value % 1000 == 0:
        return f'{value // 1000}ms'
    return f'{value}us'


def _format_residuals(values: list[int]) -> str:
    """Format a sorted list of residuals, shortened after ten values."""
    shown = ', '.join(_format_microseconds(value) for value in values[:10])
    if len(values) > 10:
        shown += f', ... ({len(values) - 10} more)'
    return f'[{shown}]'


def _aois_frame_from_text_stimulus(stimulus: TextStimulus) -> polars.DataFrame:
    """Map the configured column names of a TextStimulus to drift correction column names.

    Parameters
    ----------
    stimulus: TextStimulus
        Text stimulus whose AOIs dataframe is extracted.

    Returns
    -------
    polars.DataFrame
        AOIs dataframe with columns renamed to the names expected by
        :py:mod:`~pymovements.events.correction`.
    """
    column_mapping = {
        stimulus.start_x_column: 'start_x',
        stimulus.start_y_column: 'start_y',
        stimulus.end_x_column: 'end_x',
        stimulus.end_y_column: 'end_y',
        stimulus.width_column: 'width',
        stimulus.height_column: 'height',
    }
    rename_mapping = {
        source: target
        for source, target in column_mapping.items()
        if source is not None and source != target and source in stimulus.aois.columns
    }
    return stimulus.aois.rename(rename_mapping)


def _build_time_series(values: list[int | float] | np.ndarray) -> polars.Series:
    """Build an onset/offset series, preserving temporal input for later normalization.

    Numeric values are interpreted as milliseconds by the constructor. ``timedelta`` and
    ``timedelta64`` input keeps its inferred ``polars.Duration`` dtype so it is converted
    by physical unit instead of being reinterpreted as milliseconds. Empty input is
    inferred as ``Null`` by polars and cast to ``Float64`` so the millisecond scaffold applies.
    """
    series = values if isinstance(values, polars.Series) else polars.Series(values)
    if series.dtype == polars.Null:
        series = series.cast(polars.Float64)
    return series


_AOI_ROW_ID = '__pymovements_aoi_row_id'
_AOI_MATCH_ORDER = '__pymovements_aoi_match_order'
_AOI_X = '__pymovements_aoi_x'
_AOI_Y = '__pymovements_aoi_y'
_AOI_NAME = '__pymovements_aoi_name'
_AOI_JOIN_KEY_PREFIX = '__pymovements_aoi_join_key_'


def _aoi_join_key_name(column: str) -> str:
    """Return the reserved name of the cast join-key column for ``column``."""
    return f'{_AOI_JOIN_KEY_PREFIX}{column}'


def _is_numeric_like(dtype: polars.PolarsDataType) -> bool:
    """Return whether ``dtype`` holds numeric values (booleans compare like integers)."""
    return dtype.is_numeric() or dtype == polars.Boolean


def _is_string_like(dtype: polars.PolarsDataType) -> bool:
    """Return whether ``dtype`` holds string-like values."""
    return (
        dtype == polars.String
        or dtype == polars.Categorical
        or isinstance(dtype, polars.Enum)
    )


def _key_cast_is_valid(
        original: polars.Expr,
        source_dtype: polars.PolarsDataType,
        target_dtype: polars.PolarsDataType,
) -> polars.Expr:
    """Return whether a key value survives casting to the AOI key dtype unchanged.

    Nulls are wildcards and stay valid. A value that cannot be cast, like the string ``'x'``
    against an integer column, matches no AOI, as in the per-row lookup. A numeric value must
    also survive the cast back, so precision loss (for example ``1.5`` against an integer
    column) matches no AOI either.
    """
    if source_dtype == target_dtype:
        return polars.lit(True)
    if _is_string_like(source_dtype) and _is_string_like(target_dtype):
        return polars.lit(True)
    cast = original.cast(target_dtype, strict=False)
    is_valid = cast.is_not_null()
    if _is_numeric_like(source_dtype):
        round_trip = cast.cast(source_dtype, strict=False)
        is_valid = is_valid & (round_trip == original)
    return original.is_null() | is_valid


def _event_coordinate_expr(
        frame: polars.DataFrame,
        component: Literal['x', 'y'],
) -> tuple[polars.Expr, bool]:
    """Resolve one fixation coordinate and report whether its source is numeric.

    Only numeric component columns or numeric ``location`` list entries count as coordinates.
    Anything else resolves to null so it receives no match, as in the per-row lookup.
    """
    index = 0 if component == 'x' else 1
    component_column = f'location_{component}'
    if component_column in frame.columns:
        source_dtype = frame.schema[component_column]
        if _is_numeric_like(source_dtype):
            return polars.col(component_column).cast(polars.Float64), True
        return polars.lit(None, dtype=polars.Float64), False
    if 'location' in frame.columns:
        source_dtype = frame.schema['location']
        if isinstance(source_dtype, polars.List) and _is_numeric_like(source_dtype.inner):
            return polars.col('location').list.get(index).cast(polars.Float64), True
        return polars.lit(None, dtype=polars.Float64), False
    return polars.lit(None, dtype=polars.Float64), False


def _map_fixations_to_aois(
        frame: polars.DataFrame,
        aoi_dataframe: TextStimulus,
        aoi_columns: list[str],
        aoi_schema: dict[str, polars.PolarsDataType],
        verbose: bool,
) -> polars.DataFrame:
    """Return the AOI columns for all fixations in ``frame``, aligned to its row order.

    Non-fixation rows, rows with invalid coordinates and rows without a matching AOI receive
    null values. Fixation-to-AOI matching is done with a single vectorized join and spatial
    filter instead of a per-row Python loop.
    """
    key_columns = [
        column
        for column in (aoi_dataframe.trial_column, aoi_dataframe.page_column)
        if column is not None and column in frame.columns
    ]

    x_expr, x_is_numeric = _event_coordinate_expr(frame, 'x')
    y_expr, y_is_numeric = _event_coordinate_expr(frame, 'y')

    candidates = frame.select(
        polars.int_range(polars.len()).alias(_AOI_ROW_ID),
        polars.col('name').alias(_AOI_NAME),
        x_expr.alias(_AOI_X),
        y_expr.alias(_AOI_Y),
        # Keep the original key values for the null-wildcard lookup below.
        *[polars.col(column) for column in key_columns],
        # Cast to the AOI key dtypes so the join keys always share a common dtype.
        *[
            polars.col(column)
            .cast(aoi_dataframe.aois.schema[column], strict=False)
            .alias(_aoi_join_key_name(column))
            for column in key_columns
        ],
    )
    is_fixation = (
        polars.col(_AOI_NAME).is_not_null()
        & polars.col(_AOI_NAME).str.starts_with('fixation')
    )
    has_valid_xy = (
        polars.col(_AOI_X).is_not_null()
        & polars.col(_AOI_Y).is_not_null()
        & polars.col(_AOI_X).is_not_nan()
        & polars.col(_AOI_Y).is_not_nan()
    )

    n_fixations = candidates.filter(is_fixation).height
    if x_is_numeric and y_is_numeric:
        n_invalid = candidates.filter(is_fixation & ~has_valid_xy).height
    else:
        n_invalid = n_fixations
    if n_invalid:
        warnings.warn(
            f'Invalid eye coordinates in {n_invalid} of {n_fixations} fixations for AOI '
            'lookup. Returning no match for these fixations.',
            UserWarning,
        )

    candidates = candidates.filter(is_fixation & has_valid_xy)

    # Rows whose configured key columns contain null are matched against all AOIs of the
    # respective other key (legacy behavior); handle them via the per-row lookup below.
    # Non-null keys that do not survive the cast match no AOI, as in the per-row lookup.
    if key_columns:
        keys_present = polars.all_horizontal(
            [polars.col(column).is_not_null() for column in key_columns],
        )
        keys_valid = polars.all_horizontal(
            [
                _key_cast_is_valid(
                    polars.col(column),
                    frame.schema[column],
                    aoi_dataframe.aois.schema[column],
                )
                for column in key_columns
            ],
        )
        constrained = candidates.filter(keys_present & keys_valid)
        wildcard = candidates.filter(~keys_present)
    else:
        constrained = candidates
        wildcard = candidates.head(0)

    aois = aoi_dataframe.aois.with_columns(
        polars.int_range(polars.len()).alias(_AOI_MATCH_ORDER),
    )
    if key_columns:
        matches = constrained.join(
            aois,
            left_on=[_aoi_join_key_name(column) for column in key_columns],
            right_on=key_columns,
            how='inner',
            suffix='_aoi',
        )
    else:
        matches = constrained.join(aois, how='cross', suffix='_aoi')

    if aoi_dataframe.width_column is not None:
        spatial = (
            (polars.col(aoi_dataframe.start_x_column) <= polars.col(_AOI_X))
            & (
                polars.col(_AOI_X)
                < polars.col(aoi_dataframe.start_x_column) + polars.col(aoi_dataframe.width_column)
            )
            & (polars.col(aoi_dataframe.start_y_column) <= polars.col(_AOI_Y))
            & (
                polars.col(_AOI_Y)
                < polars.col(aoi_dataframe.start_y_column) + polars.col(aoi_dataframe.height_column)
            )
        )
    else:
        spatial = (
            (polars.col(aoi_dataframe.start_x_column) <= polars.col(_AOI_X))
            & (polars.col(_AOI_X) < polars.col(aoi_dataframe.end_x_column))
            & (polars.col(aoi_dataframe.start_y_column) <= polars.col(_AOI_Y))
            & (polars.col(_AOI_Y) < polars.col(aoi_dataframe.end_y_column))
        )
    matches = matches.filter(spatial)

    multiple = (
        matches.group_by(_AOI_ROW_ID).len().filter(polars.col('len') > 1)
        .join(candidates.select(_AOI_ROW_ID, _AOI_X, _AOI_Y), on=_AOI_ROW_ID, how='left')
        .sort(_AOI_ROW_ID)
    )
    for candidate in multiple.iter_rows(named=True):
        warnings.warn(
            f'Multiple AOIs matched this point (x={candidate[_AOI_X]}, y={candidate[_AOI_Y]}).',
            UserWarning,
        )

    matched = (
        matches.sort([_AOI_ROW_ID, _AOI_MATCH_ORDER])
        .group_by(_AOI_ROW_ID)
        .agg([polars.col(column).first() for column in aoi_columns])
    )

    if wildcard.height:
        fallback_records = [
            _aoi_lookup_record(aoi_dataframe, row, key_columns, aoi_columns)
            for row in tqdm(
                wildcard.iter_rows(named=True),
                desc='Mapping events to AOIs',
                unit='event',
                ncols=80,
                disable=not verbose,
            )
        ]
        fallback_records = [record for record in fallback_records if record is not None]
        if fallback_records:
            fallback = polars.DataFrame(
                fallback_records,
                schema={_AOI_ROW_ID: candidates.schema[_AOI_ROW_ID], **aoi_schema},
            )
            matched = polars.concat([matched, fallback])

    result = frame.with_row_index(_AOI_ROW_ID)
    if matched.height:
        result = result.join(matched, on=_AOI_ROW_ID, how='left')
    else:
        result = result.with_columns(
            [polars.lit(None).cast(aoi_schema[column]).alias(column) for column in aoi_columns],
        )
    return result.sort(_AOI_ROW_ID).select(aoi_columns).cast(aoi_schema)


def _aoi_lookup_record(
        aoi_dataframe: TextStimulus,
        row: dict[str, Any],
        key_columns: list[str],
        aoi_columns: list[str],
) -> dict[str, Any] | None:
    """Look up a single wildcard (null key) event row via the per-row AOI lookup."""
    lookup = {column: row[column] for column in key_columns}
    lookup[_AOI_X] = row[_AOI_X]
    lookup[_AOI_Y] = row[_AOI_Y]
    try:
        aoi_row = aoi_dataframe.get_aoi(
            row=lookup, x_eye=_AOI_X, y_eye=_AOI_Y, max_matches=1,
        )
    except (KeyError, TypeError):
        return None
    record: dict[str, Any] = {_AOI_ROW_ID: row[_AOI_ROW_ID]}
    record.update({column: aoi_row[column][0] for column in aoi_columns})
    return record


@repr_html(['frame', 'trial_columns'])
class Events:
    """A data structure for event data.

    Each row has at least an onset, a duration and an event name.

    Parameters
    ----------
    data: polars.DataFrame | None
        A dataframe to be transformed to a polars dataframe. Numeric ``onset``, ``offset``
        and ``duration`` columns are interpreted according to ``time_unit`` and converted to
        ``polars.Duration`` columns with microsecond precision. ``polars.Duration`` columns
        are taken over unchanged. This argument is mutually exclusive with ``name``, ``onsets``,
        ``durations``, ``offsets`` and ``trials``. (default: None)
    name: str | list[str] | None
        Name of events. (default: None)
    onsets: list[int | float] | np.ndarray | None
        List of onsets. Numeric values are interpreted according to ``time_unit``. ``timedelta``
        and ``timedelta64`` input is converted by its physical unit. (default: None)
    durations: list[int | float] | np.ndarray | None
        List of durations, accepted like ``onsets``. Only together with ``onsets`` and of the
        same length. (default: None)
    offsets: list[int | float] | np.ndarray | None
        List of offsets, accepted like ``onsets``. Only together with ``onsets`` and of the same
        length. An ``offset`` column is stored only if offsets are supplied. (default: None)
    offsets_inclusive: bool | None
        The convention of the offsets. True if an offset is the last sample of the event, False
        if it is the first sample after the event. None leaves the convention to the convention
        entry in ``metadata``, see the notes below. (default: None)
    durations_from_offsets: bool
        If True, durations are derived from declared offsets and replace supplied durations.
        (default: False)
    sampling_rate: float | None
        The sampling rate in Hz. Its interval is the duration of the last sample of an inclusive
        offset. Required to derive durations from inclusive offsets. If None, the sampling-rate
        entry in ``metadata`` is used. (default: None)
    validate: bool
        If False, supplied durations are kept unchecked beside declared offsets, a null
        ``duration`` beside a non-null ``offset`` is not reported, and a convention entry
        without its column is kept without a warning. (default: True)
    trials: list[int | float | str | None] | np.ndarray | None
        List of trial identifiers. (default: None)
    trial_columns: list[str] | str | None
        List of trial columns in the passed dataframe. (default: None)
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata. A ``Sources`` entry lists the files the
        event data was generated from. The offset convention and sampling-rate entries are
        described in the notes below. The dictionary is deep-copied, so the events share
        nothing with the passed one. (default: None)
    time_unit: str | None
        The unit of the numeric ``onset``, ``offset`` and ``duration`` input: ``'s'`` for
        seconds, ``'ms'`` for milliseconds or ``'us'`` for microseconds. Ignored for
        ``polars.Duration`` input, which already carries its unit. If None, milliseconds are
        assumed. (default: None)

    Attributes
    ----------
    frame: polars.DataFrame
        A dataframe of events. The columns are the trial columns, ``onset``, ``duration`` and
        ``name``, followed by all other columns.
    trial_columns: list[str] | None
        The name of the trial columns in the data frame. If not None, processing methods
        will be applied to each trial separately.
    metadata: dict[str, Any]
        Dictionary containing additional metadata. A ``Sources`` entry lists the files the
        event data was generated from.
    schema: polars.type_aliases.SchemaDict
        Schema of the event dataframe.
    columns: list[str]
        List of column names in the event dataframe.
    event_property_columns: list[str]
        Event property columns for this dataframe.
    fixations: polars.DataFrame
        Fixation events.
    saccades: polars.DataFrame
        Saccade events.
    blinks: polars.DataFrame
        Blink events.
    microsaccades: polars.DataFrame
        Microsaccade events.

    Raises
    ------
    ValueError
        If ``durations`` or ``offsets`` are passed without ``onsets``, if their lengths do not
        match the length of ``onsets``, if ``time_unit`` is not one of ``'s'``, ``'ms'`` or
        ``'us'``, if ``sampling_rate`` is not positive, if the sampling-rate entry in
        ``metadata`` is not a positive number, if durations are derived from inclusive offsets
        without a sampling rate, if every row contradicts the declared offset convention, or if
        ``offsets_inclusive`` or ``sampling_rate`` differs from its entry in ``metadata``.

    Notes
    -----
    How the ``duration`` column is built from the supplied columns:

    .. list-table::
        :header-rows: 1
        :widths: 15 15 15 55

        * - ``offset``
          - convention
          - ``duration``
          - outcome
        * - no
          - any
          - no
          - a ``duration`` column of nulls, no ``offset`` column
        * - no
          - any
          - yes
          - the durations are kept
        * - yes
          - declared
          - no
          - the durations are derived under the declared convention
        * - yes
          - declared
          - yes
          - with ``durations_from_offsets=True`` the derivation replaces the durations,
            otherwise the durations are kept and checked unless ``validate=False``
        * - yes
          - undeclared
          - no
          - the durations are derived as ``offset - onset``
        * - yes
          - undeclared
          - yes
          - the durations are kept unchecked

    An inclusive offset adds one sampling interval to ``offset - onset``, an exclusive offset
    adds nothing. The check compares each row with both values non-null against this rule and
    allows 1 µs, because onset, offset and duration are rounded to microseconds one by one.
    Without a sampling rate the durations of inclusive offsets must exceed ``offset - onset`` by
    one common positive interval. If every row contradicts the declaration, a ``ValueError``
    names the ways to fix it. If some rows do, one warning names them.

    ``metadata`` carries the convention and the sampling rate, so a frame rebuilt from
    ``frame`` and ``metadata`` needs neither keyword:

    .. list-table::
        :header-rows: 1
        :widths: 30 35 35

        * - entry
          - written by
          - shape
        * - offset convention
          - ``offsets_inclusive``, only if the frame has an ``offset`` column
          - ``metadata['offset'] = {'inclusive': True}``, keyed by the column name
        * - sampling rate
          - ``sampling_rate``
          - ``metadata['sampling_rate'] = 1000.0``

    The convention of ``offset`` is declared by ``offsets_inclusive``, by its entry, or by both
    when they agree. The sampling rate resolves from ``sampling_rate`` first, then from its
    entry. A keyword that differs from its entry raises a ``ValueError``, also on a frame without
    an ``offset`` column. Several offset columns may carry their own convention entries. A
    convention entry for a column the frame does not have is kept, and with ``validate=True``
    one warning names the column. :py:meth:`~pymovements.Events.drop` removes the convention
    entry together with its column.

    Modifying ``frame`` directly is not supported: ``metadata`` is not kept consistent with it.

    Examples
    --------
    We define an event dataframe with given names of events and lists of onsets and offsets.
    Without a declared convention, durations are computed as ``offset - onset``.

    >>> event = Events(
    ...    name=['fixation', 'fixation', 'fixation', 'fixation', ],
    ...    onsets=[1988147, 1988351, 1988592, 1988788],
    ...    offsets=[1988322, 1988546, 1988736, 1989013]
    ... )
    >>> event
    shape: (4, 4)
    ┌──────────────┬──────────────┬──────────┬──────────────┐
    │ onset        ┆ duration     ┆ name     ┆ offset       │
    │ ---          ┆ ---          ┆ ---      ┆ ---          │
    │ duration[μs] ┆ duration[μs] ┆ str      ┆ duration[μs] │
    ╞══════════════╪══════════════╪══════════╪══════════════╡
    │ 33m 8s 147ms ┆ 175ms        ┆ fixation ┆ 33m 8s 322ms │
    │ 33m 8s 351ms ┆ 195ms        ┆ fixation ┆ 33m 8s 546ms │
    │ 33m 8s 592ms ┆ 144ms        ┆ fixation ┆ 33m 8s 736ms │
    │ 33m 8s 788ms ┆ 225ms        ┆ fixation ┆ 33m 9s 13ms  │
    └──────────────┴──────────────┴──────────┴──────────────┘

    If the offsets are the last sample of each event, we declare them inclusive and pass the
    sampling rate. Each duration then includes the last sample:

    >>> Events(
    ...     name='blink', onsets=[2], offsets=[3], offsets_inclusive=True, sampling_rate=1000,
    ... )
    shape: (1, 4)
    ┌──────────────┬──────────────┬───────┬──────────────┐
    │ onset        ┆ duration     ┆ name  ┆ offset       │
    │ ---          ┆ ---          ┆ ---   ┆ ---          │
    │ duration[μs] ┆ duration[μs] ┆ str   ┆ duration[μs] │
    ╞══════════════╪══════════════╪═══════╪══════════════╡
    │ 2ms          ┆ 2ms          ┆ blink ┆ 3ms          │
    └──────────────┴──────────────┴───────┴──────────────┘

    Events can also be defined by onsets and durations alone, without an offset column:

    >>> Events(name='blink', onsets=[2], durations=[2])
    shape: (1, 3)
    ┌──────────────┬──────────────┬───────┐
    │ onset        ┆ duration     ┆ name  │
    │ ---          ┆ ---          ┆ ---   │
    │ duration[μs] ┆ duration[μs] ┆ str   │
    ╞══════════════╪══════════════╪═══════╡
    │ 2ms          ┆ 2ms          ┆ blink │
    └──────────────┴──────────────┴───────┘

    If both durations and declared offsets are given, the durations are checked against the
    offsets. Durations computed as ``offset - onset`` for inclusive offsets are one sample short:

    >>> Events(
    ...     name='blink', onsets=[2], offsets=[3], durations=[1],
    ...     offsets_inclusive=True, sampling_rate=1000,
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: every row contradicts offsets_inclusive=True: ...

    The convention and the sampling rate are kept in ``metadata``. A frame rebuilt with this
    metadata is declared and rated without any keyword:

    >>> events = Events(onsets=[0], offsets=[120], offsets_inclusive=True, sampling_rate=1000)
    >>> events.metadata
    {'offset': {'inclusive': True}, 'sampling_rate': 1000.0}
    >>> rebuilt = Events(events.frame.drop('duration'), metadata=events.metadata)
    >>> rebuilt.frame['duration'].to_list()
    [datetime.timedelta(microseconds=121000)]
    """

    frame: polars.DataFrame

    trial_columns: list[str] | None

    metadata: dict[str, Any]

    _minimal_schema: dict[str, Any] = {
        'onset': polars.Duration('us'),
        'duration': polars.Duration('us'),
        'name': polars.Utf8,
    }

    def __init__(
            self,
            data: polars.DataFrame | None = None,
            *,
            name: str | list[str] | None = None,
            onsets: list[int | float] | np.ndarray | None = None,
            durations: list[int | float] | np.ndarray | None = None,
            offsets: list[int | float] | np.ndarray | None = None,
            offsets_inclusive: bool | None = None,
            durations_from_offsets: bool = False,
            sampling_rate: float | None = None,
            validate: bool = True,
            trials: list[int | float | str | None] | np.ndarray | None = None,
            trial_columns: list[str] | str | None = None,
            metadata: dict[str, Any] | None = None,
            time_unit: str | None = None,
    ):
        self.trial_columns: list[str] | None  # otherwise mypy gets confused.
        # A deep copy: neither the entries written below nor later changes to nested values
        # reach the caller's dict, and changes to the caller's dict never reach these events.
        self.metadata = deepcopy(metadata) if metadata is not None else {}

        # Numeric onset/offset/duration input is interpreted in this unit; None means
        # milliseconds. Duration input carries its own unit, so time_unit is ignored for it.
        if time_unit is None:
            time_unit = 'ms'

        # Validate the unit up front so a typo is rejected regardless of whether the input is
        # numeric or already Duration (where the unit is otherwise ignored), matching Gaze.
        if time_unit not in ('s', 'ms', 'us'):
            raise ValueError(
                f"unsupported time unit '{time_unit}'. "
                "Supported units are 's' for seconds, 'ms' for milliseconds "
                "and 'us' for microseconds.",
            )

        if sampling_rate is not None and not sampling_rate > 0:
            raise ValueError(f'sampling_rate must be positive, but is {sampling_rate}.')

        if data is not None:
            checks.check_is_mutual_exclusive(data=data, onsets=onsets)
            checks.check_is_mutual_exclusive(data=data, durations=durations)
            checks.check_is_mutual_exclusive(data=data, offsets=offsets)
            checks.check_is_mutual_exclusive(data=data, name=name)
            checks.check_is_mutual_exclusive(data=data, name=trials)

            durations_supplied = 'duration' in data.columns
            data = data.clone()
            data = self._add_minimal_schema_columns(data)
            data_dict = data.to_dict()

            if isinstance(trial_columns, str):
                self.trial_columns = [trial_columns]
            else:
                self.trial_columns = trial_columns

        else:
            durations_supplied = durations is not None

            if onsets is not None:
                # Make sure that durations, offsets and names match the onsets in length.
                if durations is not None:
                    checks.check_is_length_matching(onsets=onsets, durations=durations)
                if offsets is not None:
                    checks.check_is_length_matching(onsets=onsets, offsets=offsets)
                if isinstance(name, Sequence) and not isinstance(name, str):
                    checks.check_is_length_matching(onsets=onsets, name=name)

                # These reassignments are necessary for a correct conversion into a dataframe.
                if len(onsets) == 0:
                    name = []
                if name is None:
                    name = ''
                if isinstance(name, str):
                    name = [name] * len(onsets)

                data_dict = {
                    'onset': _build_time_series(onsets),
                    'duration': (
                        _build_time_series(durations) if durations is not None
                        else polars.Series([None] * len(onsets), dtype=polars.Float64)
                    ),
                    'name': polars.Series(name, dtype=polars.Utf8),
                }
                if offsets is not None:
                    data_dict['offset'] = _build_time_series(offsets)

                if trials is not None:
                    data_dict['trial'] = polars.Series('trial', trials)
                    self.trial_columns = ['trial']
                else:
                    self.trial_columns = None

            else:
                # Onsets are the anchor: without them there is nothing to attach durations or
                # offsets to.
                if durations is not None or offsets is not None:
                    raise ValueError(
                        'durations and offsets can only be passed together with onsets',
                    )
                data_dict = {
                    'onset': polars.Series([], dtype=polars.Float64),
                    'duration': polars.Series([], dtype=polars.Float64),
                    'name': polars.Series([], dtype=polars.Utf8),
                }
                self.trial_columns = None

        # Build the frame with a numeric millisecond scaffold for the time columns. Their
        # canonical dtype is Duration('us'), but list/int/float input arrives in milliseconds
        # and is converted below. Casting such input straight to Duration('us') here would
        # reinterpret the millisecond values as microseconds. Duration input is left untouched.
        schema_overrides: dict[str, Any] = {'name': polars.Utf8}
        for column in _TIME_COLUMNS:
            if column in data_dict and not isinstance(
                    getattr(data_dict[column], 'dtype', None), polars.Duration,
            ):
                schema_overrides[column] = polars.Float64
        self.frame = polars.DataFrame(data=data_dict, schema_overrides=schema_overrides)

        # Ensure column order: trial columns, then minimal schema, keeping all other columns.
        leading_columns = [*(self.trial_columns or []), *self._minimal_schema]
        self.frame = self.frame.select(
            [
                *leading_columns,
                *[column for column in self.frame.columns if column not in leading_columns],
            ],
        )
        self._additional_columns = [
            column for column in self.frame.columns if column not in self._minimal_schema
        ]

        # Convert onset, offset, and duration to Duration('us').
        # Numeric input values are interpreted according to time_unit (milliseconds by default).
        time_cols = [c for c in _TIME_COLUMNS if c in self.frame.columns]
        numeric_time_cols = [
            c for c in time_cols
            if not isinstance(self.frame.schema[c], polars.Duration)
        ]
        if numeric_time_cols:
            # Reject infinite values early with a clear error (casting them to Duration would
            # otherwise raise a cryptic polars error). NaN is allowed: a missing onset/duration
            # value arrives as NaN and is mapped to null by the conversion below. Nulls stay
            # allowed (is_infinite is null for null, which any() ignores).
            infinite_flags = self.frame.select(
                polars.col(numeric_time_cols).cast(polars.Float64).is_infinite().any(),
            )
            infinite_cols = [c for c in numeric_time_cols if infinite_flags.item(0, c)]
            if infinite_cols:
                raise ValueError(
                    f'time columns {infinite_cols} contain infinite values; '
                    'onset, offset and duration values must be finite or null.',
                )
            self.frame = self.frame.with_columns(
                numeric_to_duration_us(polars.col(numeric_time_cols), time_unit),
            )
        # Normalize Duration input of any other unit to microseconds. Sub-microsecond input
        # (e.g. nanoseconds) is rounded to the nearest microsecond rather than truncated.
        wrong_unit_cols = [
            c for c in time_cols
            if isinstance(self.frame.schema[c], polars.Duration)
            and self.frame.schema[c] != polars.Duration('us')
        ]
        if wrong_unit_cols:
            self.frame = self.frame.with_columns(
                normalize_duration_to_us(polars.col(wrong_unit_cols)),
            )

        entry_inclusive, entry_rate = self._read_entries(offsets_inclusive, sampling_rate)

        self._resolve_durations(
            durations_supplied=durations_supplied,
            offsets_inclusive=offsets_inclusive if offsets_inclusive is not None
            else entry_inclusive,
            offsets_inclusive_source='keyword' if offsets_inclusive is not None else 'entry',
            durations_from_offsets=durations_from_offsets,
            sampling_rate=sampling_rate if sampling_rate is not None else entry_rate,
            sampling_rate_source='keyword' if sampling_rate is not None else 'entry',
            validate=validate,
        )

        self._write_entries(
            offsets_inclusive=offsets_inclusive,
            sampling_rate=sampling_rate,
            warn_missing_columns=validate,
        )

    def _read_entries(
            self,
            offsets_inclusive: bool | None,
            sampling_rate: float | None,
    ) -> tuple[bool | None, float | None]:
        """Read the convention of ``offset`` and the sampling rate from the metadata entries.

        Parameters
        ----------
        offsets_inclusive: bool | None
            The ``offsets_inclusive`` keyword, compared against the ``offset`` entry.
        sampling_rate: float | None
            The ``sampling_rate`` keyword, compared against the sampling-rate entry.

        Returns
        -------
        tuple[bool | None, float | None]
            The convention of the ``offset`` entry and the rate of the sampling-rate entry,
            ``None`` each for an absent entry.

        Raises
        ------
        ValueError
            If a keyword differs from its entry, or if the sampling-rate entry is not a positive
            number.
        """
        # The convention of 'offset' and the sampling rate are declared by keyword or entry.
        entry = self.metadata.get('offset')
        entry_inclusive = entry.get('inclusive') if isinstance(entry, dict) else None
        if (
                offsets_inclusive is not None and entry_inclusive is not None
                and offsets_inclusive != entry_inclusive
        ):
            missing = (
                '' if 'offset' in self.frame.columns
                else ", and the events frame has no 'offset' column"
            )
            raise ValueError(
                f'offsets_inclusive={offsets_inclusive} differs from the metadata entry '
                f"metadata['offset'] = {{'inclusive': {entry_inclusive}}}{missing}. "
                'Correct the entry or drop the keyword.',
            )
        entry_rate = self.metadata.get('sampling_rate')
        rate = None
        if entry_rate is not None:
            if isinstance(entry_rate, bool) or not isinstance(entry_rate, Real):
                raise ValueError(
                    "the metadata entry metadata['sampling_rate'] must be a number, "
                    f'but is {entry_rate!r}.',
                )
            rate = float(entry_rate)
            if rate <= 0:
                raise ValueError(
                    "the metadata entry metadata['sampling_rate'] must be positive, "
                    f'but is {entry_rate}.',
                )
        if sampling_rate is not None and rate is not None and sampling_rate != rate:
            raise ValueError(
                f'sampling_rate={sampling_rate!r} differs from the metadata entry '
                f"metadata['sampling_rate'] = {entry_rate!r}. "
                'Correct the entry or drop the keyword.',
            )
        return entry_inclusive, rate

    def _write_entries(
            self,
            *,
            offsets_inclusive: bool | None,
            sampling_rate: float | None,
            warn_missing_columns: bool,
    ) -> None:
        """Write the entries of the keywords and warn about entries without their column.

        Parameters
        ----------
        offsets_inclusive: bool | None
            The ``offsets_inclusive`` keyword, written to the ``offset`` entry when the frame has
            that column.
        sampling_rate: float | None
            The ``sampling_rate`` keyword, written to the sampling-rate entry.
        warn_missing_columns: bool
            Whether to warn once about convention entries whose column the frame does not have.
        """
        # Entries are written only for what the frame has.
        if offsets_inclusive is not None and 'offset' in self.frame.columns:
            entry = self.metadata.get('offset')
            self.metadata['offset'] = {
                **(entry if isinstance(entry, dict) else {}),
                'inclusive': offsets_inclusive,
            }
        if sampling_rate is not None:
            self.metadata['sampling_rate'] = float(sampling_rate)

        # A convention entry without its column is kept, never removed on construction.
        missing_columns = [
            column for column, entry in self.metadata.items()
            if isinstance(entry, dict) and 'inclusive' in entry
            and column not in self.frame.columns
        ]
        if warn_missing_columns and missing_columns:
            if len(missing_columns) == 1:
                described = f"entry for the column '{missing_columns[0]}'"
            else:
                described = 'entries for the columns ' + ', '.join(
                    f"'{column}'" for column in missing_columns
                )
            warn(
                f'metadata holds offset convention {described}, '
                'which the events frame does not have. The metadata is kept.',
                UserWarning,
                stacklevel=3,
            )

    def _resolve_durations(
            self,
            *,
            durations_supplied: bool,
            offsets_inclusive: bool | None,
            offsets_inclusive_source: Literal['keyword', 'entry'],
            durations_from_offsets: bool,
            sampling_rate: float | None,
            sampling_rate_source: Literal['keyword', 'entry'],
            validate: bool,
    ) -> None:
        """Fill or check the ``duration`` column against the ``offset`` column.

        See the construction rules in the class docstring.
        """
        if 'offset' not in self.frame.columns:
            ignored = []
            if offsets_inclusive_source == 'keyword':
                ignored.append(f'offsets_inclusive={offsets_inclusive}')
            if durations_from_offsets:
                ignored.append('durations_from_offsets=True')
            if ignored:
                warn(
                    f"{' and '.join(ignored)} ignored: the events frame has no 'offset' column.",
                    UserWarning,
                    stacklevel=3,
                )
            return

        if offsets_inclusive is None:
            # Undeclared offsets keep the pre-v0.29.0 behavior until an undeclared offset raises.
            if durations_from_offsets:
                warn(
                    "durations_from_offsets=True ignored: the 'offset' column is undeclared. "
                    'Pass offsets_inclusive to declare its convention.',
                    UserWarning,
                    stacklevel=3,
                )
            if not durations_supplied:
                self.frame = self.frame.with_columns(
                    (polars.col('offset') - polars.col('onset')).alias('duration'),
                )
            return

        sampling_interval = None
        if sampling_rate is not None:
            # Rounded to microseconds like numeric time input, so residuals compare exactly.
            sampling_interval = round(1_000_000 / sampling_rate)

        if not durations_supplied or durations_from_offsets:
            if offsets_inclusive and sampling_interval is None:
                raise ValueError(
                    'deriving durations from inclusive offsets requires a sampling_rate: '
                    'duration = offset - onset + 1 / sampling_rate.',
                )
            derived = polars.col('offset') - polars.col('onset')
            if offsets_inclusive:
                derived = derived + polars.duration(microseconds=sampling_interval)
            self.frame = self.frame.with_columns(derived.alias('duration'))
            return

        if not validate:
            return

        # A null duration beside a non-null offset is reported, never filled.
        null_durations = self.frame.select(
            (polars.col('duration').is_null() & polars.col('offset').is_not_null()).sum(),
        ).item()
        if null_durations:
            row_s = 'row has' if null_durations == 1 else 'rows have'
            warn(
                f"{null_durations} {row_s} a null duration beside a non-null offset. "
                'The durations stay null.',
                UserWarning,
                stacklevel=3,
            )

        self._check_durations_against_offsets(
            offsets_inclusive=offsets_inclusive,
            offsets_inclusive_source=offsets_inclusive_source,
            sampling_rate=sampling_rate,
            sampling_interval=sampling_interval,
            sampling_rate_source=sampling_rate_source,
        )

    def _check_durations_against_offsets(
            self,
            *,
            offsets_inclusive: bool,
            offsets_inclusive_source: Literal['keyword', 'entry'],
            sampling_rate: float | None,
            sampling_interval: int | None,
            sampling_rate_source: Literal['keyword', 'entry'],
    ) -> None:
        """Check the stored durations against the declared offsets.

        Raises if every row with both values non-null contradicts the declaration, warns once if
        some rows do. Every comparison allows 1 µs, because onset, offset and duration are rounded
        to microseconds one by one.
        """
        residuals = self.frame.select(
            (polars.col('duration') - (polars.col('offset') - polars.col('onset')))
            .dt.total_microseconds()
            .drop_nulls(),
        ).to_series()
        if residuals.is_empty():
            return

        declaration = f'offsets_inclusive={offsets_inclusive}'
        if offsets_inclusive_source == 'entry':
            declaration += ' from the metadata entry'
        if not offsets_inclusive:
            expected = 0
            expected_description = '0'
        elif sampling_interval is not None:
            expected = sampling_interval
            expected_description = (
                f'{_format_microseconds(expected)} (sampling_rate={sampling_rate:g})'
            )
        else:
            # Without a sampling rate the positive residuals must agree on one interval.
            positive = residuals.filter(residuals > _RESIDUAL_TOLERANCE_US)
            if positive.is_empty():
                expected = None
                expected_description = 'a common positive residual (no sampling rate)'
            else:
                counts = positive.value_counts(sort=True)
                expected = counts.filter(
                    polars.col('count') == counts['count'].max(),
                )[residuals.name].min()
                expected_description = (
                    'no sampling rate, common positive residual '
                    f'{_format_microseconds(expected)}'
                )

        contradicting = residuals if expected is None else residuals.filter(
            (residuals - expected).abs() > _RESIDUAL_TOLERANCE_US,
        )
        if contradicting.is_empty():
            return

        distinct = sorted(contradicting.unique().to_list())
        legacy_rows = (
            (contradicting.abs() <= _RESIDUAL_TOLERANCE_US).sum() if offsets_inclusive else 0
        )

        if len(contradicting) == len(residuals):
            if len(distinct) == 1:
                seen = f'is {_format_microseconds(distinct[0])} on every row'
            else:
                seen = f'takes the values {_format_residuals(distinct)}'
            message = (
                f'every row contradicts {declaration}: duration - (offset - onset) {seen}, '
                f'expected {expected_description}'
            )
            if offsets_inclusive and all(abs(r) <= _RESIDUAL_TOLERANCE_US for r in distinct):
                message += ', the durations look like pre-v0.29.0 values'
            # The correction is offered where the convention was declared.
            if offsets_inclusive_source == 'keyword':
                correction = f'offsets_inclusive={not offsets_inclusive}'
            else:
                correction = (
                    f"metadata['offset'] = {{'inclusive': {not offsets_inclusive}}}"
                )
            recipes = []
            if offsets_inclusive:
                recipes.append(
                    'durations_from_offsets=True to derive the durations from the offsets',
                )
            other = 'exclusive' if offsets_inclusive else 'inclusive'
            recipes.append(f'{correction} if the offsets are {other}')
            recipes.append('validate=False to keep the durations unchecked')
            message += f". Pass {', '.join(recipes[:-1])}, or {recipes[-1]}."
            raise ValueError(message)

        if sampling_interval is not None and offsets_inclusive:
            rate_source = (
                'the sampling_rate keyword' if sampling_rate_source == 'keyword'
                else 'the metadata entry'
            )
            expected_description = (
                f'{_format_microseconds(sampling_interval)} '
                f'(sampling_rate={sampling_rate:g} from {rate_source})'
            )
        message = (
            f'{len(contradicting)} of {len(residuals)} rows contradict {declaration}: '
            f'expected duration - (offset - onset) to be {expected_description}, '
            f'residuals seen: {_format_residuals(sorted(residuals.unique().to_list()))}.'
        )
        if legacy_rows:
            row_s = 'row looks' if legacy_rows == 1 else 'rows look'
            message += (
                f' {legacy_rows} {row_s} like pre-v0.29.0 durations. Pass '
                'durations_from_offsets=True to derive the durations from the offsets.'
            )
        warn(message, UserWarning, stacklevel=4)

    @property
    def schema(self) -> polars.type_aliases.SchemaDict:
        """Schema of event dataframe."""
        return self.frame.schema

    def __len__(self) -> int:
        """Get number of events in dataframe."""
        return self.frame.__len__()

    def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
        """Get item."""
        return self.frame.__getitem__(*args, **kwargs)

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        return self.frame.columns

    def add_event_properties(
            self,
            event_properties: polars.DataFrame,
            join_on: str | list[str],
    ) -> None:
        """Add new event properties into dataframe.

        Parameters
        ----------
        event_properties: polars.DataFrame
            Dataframe with new event properties.
        join_on: str | list[str]
            Columns to join event properties on.
        """
        self.frame = self.frame.join(event_properties, on=join_on, how='left', nulls_equal=True)

    def drop(
            self,
            columns: str | list[str],
    ) -> None:
        """Remove columns from the events data frame.

        Notes
        -----
        The minimal schema columns ``onset``, ``duration`` and ``name`` cannot be removed.
        Dropping a column also removes the ``metadata`` entry keyed by its name, such as the
        convention entry of an offset column.

        Parameters
        ----------
        columns: str | list[str]
            The columns in the event data frame to remove.

        Raises
        ------
        ValueError
            If ``columns`` do not exist in the event dataframe or it is not allowed to remove them.
        """
        if isinstance(columns, str):
            columns = [columns]
        existing_columns = set(self.frame.columns)
        minimal_schema = set(self._minimal_schema)
        for column in columns:
            available_columns = existing_columns - minimal_schema
            if column not in existing_columns:
                raise ValueError(
                    f"The column '{column}' does not exist and thus cannot be removed. "
                    f'Available columns to remove: {available_columns}.',
                )
            if column in minimal_schema:
                raise ValueError(
                    f"The column '{column}' cannot be removed "
                    'because it belongs to the minimal schema (onset, duration, name). '
                    f'Available columns to remove: {available_columns}.',
                )
        for column in columns:
            self.frame = self.frame.drop(column)
            # A metadata entry keyed by the dropped column describes it and goes with it.
            self.metadata.pop(column, None)

    def add_trial_column(
            self,
            column: str | list[str],
            data: int | float | str | list[int | float | str] | None,
    ) -> None:
        """Add new trial columns with constant values.

        Parameters
        ----------
        column: str | list[str]
            The name(s) of the new trial column(s).
        data: int | float | str | list[int | float | str] | None
            The values to be used for filling the trial column(s). In case multiple columns are
            provided, data must be a list of values matching the provided column order.
        """
        # Create trial column dictionary to iterate over in select().
        if isinstance(column, str):
            trial_columns = {column: data}
        # In case a list of a single column is passed as an explicit value.
        elif len(column) == 1 and (isinstance(data, (int, float, str) or data is None)):
            trial_columns = {column[0]: data}
        else:
            if not isinstance(data, Sequence):
                raise TypeError(
                    'data must be passed as a list of values in case of providing multiple columns',
                )
            checks.check_is_length_matching(column=column, data=data)

            trial_columns = dict(zip(column, data))

        self.frame = self.frame.select(
            [
                polars.lit(column_data).alias(column_name) if not isinstance(column_data, int)
                # Enforce Int64 columns for integers.
                else polars.lit(column_data).alias(column_name).cast(polars.Int64)
                for column_name, column_data in trial_columns.items()
            ] + [polars.all()],
        )

    @property
    def event_property_columns(self) -> list[str]:
        """Event property columns for this dataframe.

        Returns
        -------
        list[str]
            List of event property columns.
        """
        event_property_columns = set(self.frame.columns)
        event_property_columns -= set(list(self._minimal_schema.keys()))
        event_property_columns -= set(self._additional_columns)
        return list(event_property_columns)

    def filter_by_name(self, name: str) -> polars.DataFrame:
        """Filter events by name.

        Parameters
        ----------
        name : str
            Filter events that contain that string in the ``name`` column.
            Supports regular expressions.

        Examples
        --------
        Let's create some events with different names first:

        >>> import pymovements as pm
        >>> events = pm.Events(
        ...     name=[
        ...         'saccade', 'fixation', 'fixation_idt', 'fixation_ivt', 'fixation_eyelink',
        ...         'microsaccade', 'microsaccade', 'saccade',
        ...     ],
        ...     onsets=[90, 99, 99, 100, 101, 115, 145, 175],
        ...     offsets=[100, 176, 175, 178, 175, 124, 157, 199],
        ... )
        >>> events
        shape: (8, 4)
        ┌──────────────┬──────────────┬──────────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name             ┆ offset       │
        │ ---          ┆ ---          ┆ ---              ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str              ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════════╪══════════════╡
        │ 90ms         ┆ 10ms         ┆ saccade          ┆ 100ms        │
        │ 99ms         ┆ 77ms         ┆ fixation         ┆ 176ms        │
        │ 99ms         ┆ 76ms         ┆ fixation_idt     ┆ 175ms        │
        │ 100ms        ┆ 78ms         ┆ fixation_ivt     ┆ 178ms        │
        │ 101ms        ┆ 74ms         ┆ fixation_eyelink ┆ 175ms        │
        │ 115ms        ┆ 9ms          ┆ microsaccade     ┆ 124ms        │
        │ 145ms        ┆ 12ms         ┆ microsaccade     ┆ 157ms        │
        │ 175ms        ┆ 24ms         ┆ saccade          ┆ 199ms        │
        └──────────────┴──────────────┴──────────────────┴──────────────┘

        All fixations:

        >>> events.filter_by_name('fixation')
        shape: (4, 4)
        ┌──────────────┬──────────────┬──────────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name             ┆ offset       │
        │ ---          ┆ ---          ┆ ---              ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str              ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════════╪══════════════╡
        │ 99ms         ┆ 77ms         ┆ fixation         ┆ 176ms        │
        │ 99ms         ┆ 76ms         ┆ fixation_idt     ┆ 175ms        │
        │ 100ms        ┆ 78ms         ┆ fixation_ivt     ┆ 178ms        │
        │ 101ms        ┆ 74ms         ┆ fixation_eyelink ┆ 175ms        │
        └──────────────┴──────────────┴──────────────────┴──────────────┘

        Exact match for fixation:

        >>> events.filter_by_name('^fixation$')
        shape: (1, 4)
        ┌──────────────┬──────────────┬──────────┬──────────────┐
        │ onset        ┆ duration     ┆ name     ┆ offset       │
        │ ---          ┆ ---          ┆ ---      ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str      ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════╪══════════════╡
        │ 99ms         ┆ 77ms         ┆ fixation ┆ 176ms        │
        └──────────────┴──────────────┴──────────┴──────────────┘

        Prefix match:

        >>> events.filter_by_name('^fixation_')
        shape: (3, 4)
        ┌──────────────┬──────────────┬──────────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name             ┆ offset       │
        │ ---          ┆ ---          ┆ ---              ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str              ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════════╪══════════════╡
        │ 99ms         ┆ 76ms         ┆ fixation_idt     ┆ 175ms        │
        │ 100ms        ┆ 78ms         ┆ fixation_ivt     ┆ 178ms        │
        │ 101ms        ┆ 74ms         ┆ fixation_eyelink ┆ 175ms        │
        └──────────────┴──────────────┴──────────────────┴──────────────┘

        Suffix match:

        >>> events.filter_by_name('ivt$')
        shape: (1, 4)
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name         ┆ offset       │
        │ ---          ┆ ---          ┆ ---          ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str          ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════╪══════════════╡
        │ 100ms        ┆ 78ms         ┆ fixation_ivt ┆ 178ms        │
        └──────────────┴──────────────┴──────────────┴──────────────┘

        All saccade variants:

        >>> events.filter_by_name('saccade')
        shape: (4, 4)
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name         ┆ offset       │
        │ ---          ┆ ---          ┆ ---          ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str          ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════╪══════════════╡
        │ 90ms         ┆ 10ms         ┆ saccade      ┆ 100ms        │
        │ 115ms        ┆ 9ms          ┆ microsaccade ┆ 124ms        │
        │ 145ms        ┆ 12ms         ┆ microsaccade ┆ 157ms        │
        │ 175ms        ┆ 24ms         ┆ saccade      ┆ 199ms        │
        └──────────────┴──────────────┴──────────────┴──────────────┘

        Only microsaccades:

        >>> events.filter_by_name('microsaccade')
        shape: (2, 4)
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ onset        ┆ duration     ┆ name         ┆ offset       │
        │ ---          ┆ ---          ┆ ---          ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str          ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════════╪══════════════╡
        │ 115ms        ┆ 9ms          ┆ microsaccade ┆ 124ms        │
        │ 145ms        ┆ 12ms         ┆ microsaccade ┆ 157ms        │
        └──────────────┴──────────────┴──────────────┴──────────────┘

        Exact match for saccade:

        >>> events.filter_by_name('^saccade$')
        shape: (2, 4)
        ┌──────────────┬──────────────┬─────────┬──────────────┐
        │ onset        ┆ duration     ┆ name    ┆ offset       │
        │ ---          ┆ ---          ┆ ---     ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str     ┆ duration[μs] │
        ╞══════════════╪══════════════╪═════════╪══════════════╡
        │ 90ms         ┆ 10ms         ┆ saccade ┆ 100ms        │
        │ 175ms        ┆ 24ms         ┆ saccade ┆ 199ms        │
        └──────────────┴──────────────┴─────────┴──────────────┘

        Returns
        -------
        polars.DataFrame
            DataFrame containing matching events.
        """
        if 'name' not in self.frame.columns:
            raise ValueError("Events frame is missing the 'name' column.")

        return self.frame.filter(polars.col('name').str.contains(name))

    @property
    def fixations(self) -> polars.DataFrame:
        """Fixation events.

        Returns
        -------
        polars.DataFrame
            DataFrame containing all fixation events, i.e., rows where
            ``name`` starts with ``"fixation"`` (e.g., ``"fixation"``, ``"fixation_ivt"``,
            ``"fixation_eyelink"``).
        """
        return self.filter_by_name('fixation')

    @property
    def saccades(self) -> polars.DataFrame:
        """Saccade events.

        Returns
        -------
        polars.DataFrame
            DataFrame containing all saccade events, i.e., rows where
            ``name`` starts with ``"saccade"`` (e.g., ``"saccade"``, ``"saccade_algo"``).
        """
        return self.filter_by_name('saccade')

    @property
    def blinks(self) -> polars.DataFrame:
        """Blink events.

        Returns
        -------
        polars.DataFrame
            DataFrame containing all blink events, i.e., rows where
            ``name`` starts with ``"blink"`` (e.g., ``"blink"``, ``"blink_detectorX"``).
        """
        return self.filter_by_name('blink')

    @property
    def microsaccades(self) -> polars.DataFrame:
        """Microsaccade events.

        Returns
        -------
        polars.DataFrame
            DataFrame containing all microsaccade events, i.e., rows where
            ``name`` starts with ``"microsaccade"`` (e.g., ``"microsaccade"``).
        """
        return self.filter_by_name('microsaccade')

    def clone(self) -> Events:
        """Return a copy of an Events object.

        Returns
        -------
        Events
            A copy of an Events object.
        """
        return Events(
            data=self.frame.clone(),
            trial_columns=self.trial_columns,
            metadata=self.metadata,
            validate=False,
        )

    @overload
    def split(
            self, by: str | Sequence[str] | None = None, *, as_dict: Literal[False],
    ) -> list[Events]:
        ...

    @overload
    def split(
            self, by: str | Sequence[str] | None = None, *, as_dict: Literal[True],
    ) -> dict[tuple[Any, ...], Events]:
        ...

    def split(
            self,
            by: str | Sequence[str] | None = None,
            *,
            as_dict: bool = False,
    ) -> list[Events] | dict[tuple[Any, ...], Events]:
        """Split the Events into multiple frames based on specified column(s).

        Parameters
        ----------
        by: str | Sequence[str] | None
            Column name(s) to split the Events by. If a single string is provided,
            it will be used as a single column name. If a list is provided, the Events
            will be split by unique combinations of values in all specified columns.
            If None, uses trial_columns. (default: None)
        as_dict: bool
            Return a dictionary instead of a list. The dictionary keys are tuples of the distinct
            group values that identify each group split. (default: False)

        Returns
        -------
        list[Events] | dict[tuple[Any, ...], Events]
            A collection of new Events instances, each containing a partition of the original data
            with all metadata and configurations preserved.
        """
        # Use trial_columns if by is None
        if by is None:
            if self.trial_columns is None:
                raise TypeError("Either 'by' or 'Events.trial_columns' must be specified")
            by = self.trial_columns

        event_dfs = self.frame.partition_by(by=by, as_dict=as_dict)

        if as_dict:
            # keys are tuples of the unique values of the columns specified in `by`.
            return {
                key: Events(
                    frame,
                    trial_columns=self.trial_columns,
                    metadata=self.metadata,
                    validate=False,
                )
                for key, frame in event_dfs.items()
            }

        return [
            Events(
                frame,
                trial_columns=self.trial_columns,
                metadata=self.metadata,
                validate=False,
            )
            for frame in event_dfs
        ]

    def drop_nulls(
        self,
        subset: list[str] | None = None,
        how: Literal['all', 'any'] = 'any',
    ) -> None:
        """Drop events with null values.

        Parameters
        ----------
        subset: list[str] | None
            List of column names to check for null values. If None, all columns of the events
            frame are checked. (default: None)
        how: Literal['all', 'any']
            If 'any', drop rows where *any* of the specified columns are null. If 'all', drop rows
            where *all* of the specified columns are null. A nested list column counts as null if
            any of its components is null under 'any', and only if all of its components are null
            under 'all'. (default: 'any')

        Raises
        ------
        ValueError
            If `how` is neither 'any' nor 'all', or if `subset` contains columns that do not
            exist in the events frame.

        Examples
        --------
        Let's create some events with null values in the trial and page columns:

        >>> import polars
        >>> import pymovements as pm
        >>> events = pm.Events(
        ...     polars.DataFrame({
        ...         'name': ['fixation', 'fixation', 'fixation'],
        ...         'onset': [0, 110, 165],
        ...         'offset': [100, 150, 200],
        ...         'trial': [1, None, None],
        ...         'page': [1, 2, None],
        ...     }),
        ... )

        Under ``how='all'``, an event is only dropped if all subset columns are null,
        removing the third fixation:

        >>> events.drop_nulls(subset=['trial', 'page'], how='all')
        >>> events
        shape: (2, 6)
        ┌──────────────┬──────────────┬──────────┬──────────────┬───────┬──────┐
        │ onset        ┆ duration     ┆ name     ┆ offset       ┆ trial ┆ page │
        │ ---          ┆ ---          ┆ ---      ┆ ---          ┆ ---   ┆ ---  │
        │ duration[μs] ┆ duration[μs] ┆ str      ┆ duration[μs] ┆ i64   ┆ i64  │
        ╞══════════════╪══════════════╪══════════╪══════════════╪═══════╪══════╡
        │ 0µs          ┆ 100ms        ┆ fixation ┆ 100ms        ┆ 1     ┆ 1    │
        │ 110ms        ┆ 40ms         ┆ fixation ┆ 150ms        ┆ null  ┆ 2    │
        └──────────────┴──────────────┴──────────┴──────────────┴───────┴──────┘

        Under the default ``how='any'``, a single null value suffices, removing the second
        fixation too:

        >>> events.drop_nulls(subset=['trial', 'page'])
        >>> events
        shape: (1, 6)
        ┌──────────────┬──────────────┬──────────┬──────────────┬───────┬──────┐
        │ onset        ┆ duration     ┆ name     ┆ offset       ┆ trial ┆ page │
        │ ---          ┆ ---          ┆ ---      ┆ ---          ┆ ---   ┆ ---  │
        │ duration[μs] ┆ duration[μs] ┆ str      ┆ duration[μs] ┆ i64   ┆ i64  │
        ╞══════════════╪══════════════╪══════════╪══════════════╪═══════╪══════╡
        │ 0µs          ┆ 100ms        ┆ fixation ┆ 100ms        ┆ 1     ┆ 1    │
        └──────────────┴──────────────┴──────────┴──────────────┴───────┴──────┘
        """
        if subset is None:
            subset = self.frame.columns
        else:
            missing_columns = [column for column in subset if column not in self.frame.columns]
            if missing_columns:
                raise ValueError(
                    f'columns {missing_columns} from subset do not exist in the events frame',
                )

        self.frame = self.frame.remove(row_is_null(self.frame.schema, subset, how))

    def _add_minimal_schema_columns(self, df: polars.DataFrame) -> polars.DataFrame:
        """Add minimal schema columns to :py:class:`polars.DataFrame` if they are missing.

        Parameters
        ----------
        df: polars.DataFrame
            A dataframe to be transformed to a polars dataframe.

        Returns
        -------
        polars.DataFrame
            A dataframe with minimal schema columns added.
        """
        if len(df) == 0:
            return polars.DataFrame(schema={**self._minimal_schema, **df.schema})

        df = df.select(
            [
                polars.lit(None).cast(column_type).alias(column_name)
                for column_name, column_type in self._minimal_schema.items()
                if column_name not in df.columns
            ] + [polars.all()],
        )
        return df

    def unnest(
            self,
            input_columns: list[str] | str | None = None,
            output_suffixes: list[str] | None = None,
            *,
            output_columns: list[str] | None = None,
    ) -> None:
        """Explode columns of type ``polars.List`` into one column for each list component.

        The input columns will be dropped.

        Parameters
        ----------
        input_columns: list[str] | str | None
            Name(s) of input column(s) to be unnested into several component columns.
            If None, all list columns will be unnested if existing. (default: None)
        output_suffixes: list[str] | None
            Suffixes to append to the column names. (default: None)
        output_columns: list[str] | None
            Name of the resulting tuple columns. (default: None)

        Raises
        ------
        ValueError
            If both output_columns and output_suffixes are specified.
            If number of output columns / suffixes does not match number of components.
            If output columns / suffixes are not unique.
            If no columns to unnest exist and none are specified.
            If output columns are specified and more than one input column is specified.
            If a list column to unnest is empty (has no rows).
            If a list column to unnest contains only null values.
            If number of components is not 2, 4 or 6.
        Warning
            If no columns to unnest exist and none are specified.
        """
        self.frame = unnest_list_columns(
            df=self.frame,
            input_columns=input_columns,
            output_suffixes=output_suffixes,
            output_columns=output_columns,
        )

    def map_to_aois(
            self,
            aoi_dataframe: TextStimulus,
            *,
            preserve_structure: bool = True,
            verbose: bool | None = None,
    ) -> None:
        """Map events to AOIs, ignoring non-fixations.

        This function computes AOI membership only for rows whose ``name`` starts with
        ``"fixation"`` (e.g., ``"fixation"``, ``"fixation_ivt"``). Non-fixation rows,
        fixations with invalid coordinates and fixations without a matching AOI receive
        ``None`` values for all appended AOI columns. The original order and number of rows
        are preserved.

        Fixations are matched against the AOI rectangles with a vectorised join and
        half-open rectangle test (``start <= coordinate < end``) instead of one lookup per
        row. Coordinates are taken from numeric ``location_x``/``location_y`` columns, or
        from a numeric ``location`` list column when the component columns are absent.
        Fixations whose coordinates are missing, ``NaN`` or non-numeric receive no match,
        and one ``UserWarning`` reports the number of such fixations per call.

        Trial and page keys:

        - If the stimulus defines ``trial_column``/``page_column`` and the events frame has
          the same columns, matching is restricted to AOIs with equal key values.
        - A null value in a configured key column is a wildcard: that fixation is matched
          against all AOIs for the other key(s), using the per-row
          :meth:`~pymovements.stimulus.TextStimulus.get_aoi` lookup.
        - A non-null key is cast to the AOI key dtype, so the string ``'02'`` matches the
          integer key ``2``. A key that cannot be cast without changing its value (for
          example ``'x'`` or ``1.5`` against an integer column) matches no AOI.

        Overlapping AOIs:

        - If multiple AOIs contain a fixation, the first AOI in stimulus order is selected
          and a ``UserWarning`` is emitted.

        Schema handling:

        - If ``preserve_structure=True`` (default), we mirror legacy behavior when a list
          ``location`` column exists: derive ``location_x``/``location_y`` and drop ``location``.
          This keeps downstream expectations about flat component columns.
        - If ``preserve_structure=False``, no unnesting/derivation occurs and the original
          ``location`` list column is preserved. Matching reads coordinates without altering
          the frame.

        AOI columns used as trial/page keys are only appended when the events frame does not
        already contain a column with that name; existing event columns are never overwritten.

        Parameters
        ----------
        aoi_dataframe: TextStimulus
            Text stimulus defining AOI rectangles.
        preserve_structure: bool
            Control whether to derive component columns and drop the list column as described
            above. Default: True.
        verbose: bool | None
            If ``True``, show a progress bar over the fixations with a null trial or page value,
            the only ones that are still looked up one by one. (default: None)

            .. deprecated:: v0.29.0
                The mapping no longer loops over all events.
                This argument will be removed in v0.34.0.

        Raises
        ------
        ValueError
            If ``aoi_dataframe`` does not have either ``width_column`` or ``end_x_column`` defined.
        ValueError
            If ``width_column`` and ``height_column`` or ``end_x_column`` and ``end_y_column``
            are not both defined or both ``None``.
        ValueError
            If the events frame is empty.
        """
        if verbose is not None:
            warnings.warn(
                DeprecationWarning(
                    "Events.map_to_aois argument 'verbose' is deprecated since version v0.29.0. "
                    'The mapping no longer loops over all events. '
                    'This argument will be removed in v0.34.0.',
                ),
            )

        # Validate AOI configuration early
        if aoi_dataframe.width_column is None and aoi_dataframe.end_x_column is None:
            raise ValueError(
                'either TextStimulus.width or TextStimulus.end_x_column must be defined',
            )
        if aoi_dataframe.width_column is not None:
            checks.check_is_none_is_mutual(
                height_column=aoi_dataframe.width_column,
                width_column=aoi_dataframe.height_column,
            )
        else:
            checks.check_is_none_is_mutual(
                end_x_column=aoi_dataframe.end_x_column,
                end_y_column=aoi_dataframe.end_y_column,
            )
        # Raise when no rows to concat
        if self.frame.height == 0:
            raise ValueError('cannot concat empty list')

        # Backward-compatibility: derive component coordinates if only a list column exists.
        if preserve_structure and 'location' in self.frame.columns and (
            'location_x' not in self.frame.columns or 'location_y' not in self.frame.columns
        ):
            self.frame = self.frame.with_columns(
                [
                    polars.col('location').list.get(0).alias('location_x'),
                    polars.col('location').list.get(1).alias('location_y'),
                ],
            ).drop('location')

        # AOI output columns mirror the stimulus columns, but skip columns that already exist
        # in the Events frame (e.g., trial/page) to avoid duplicate-column errors on concat.
        existing_cols = set(self.frame.columns)
        aoi_columns: list[str] = [c for c in aoi_dataframe.aois.columns if c not in existing_cols]

        # Build a stable dtype schema for the AOI columns to avoid concat SchemaError when
        # mixing empty rows (all None) with real AOI rows (strings/floats, etc.).
        aoi_schema: dict[str, polars.PolarsDataType] = (
            {col: aoi_dataframe.aois.schema[col] for col in aoi_columns}
            if aoi_columns else {}
        )

        if aoi_columns:
            aoi_df = _map_fixations_to_aois(
                self.frame, aoi_dataframe, aoi_columns, aoi_schema, verbose is True,
            )
            self.frame = polars.concat([self.frame, aoi_df], how='horizontal_extend')

        # Backward-compatibility: some pipelines expect that a prior unnest removed the
        # original 'location' list column and kept only component columns. We avoid unnesting,
        # but if component columns already exist, we drop the original list column to preserve
        # legacy schema without altering coordinates.
        if preserve_structure and 'location' in self.frame.columns and (
            'location_x' in self.frame.columns or 'location_y' in self.frame.columns
        ):
            self.frame = self.frame.drop('location')

        # The AOI-mapped events are derived from the stimulus file as well.
        self.metadata = merge_sources(self.metadata, aoi_dataframe.metadata)

    def correct_fixations(
            self,
            aois: TextStimulus,
            algorithm: str | list[str] = 'wisdom_of_the_crowd',
            *,
            directionality: str | None = None,
            word_locations: polars.Series | None = None,
            algorithm_kwargs: dict[str, Any] | None = None,
            fixation_name: str = 'fixation',
            character_level: bool = False,
            inplace: bool = True,
    ) -> Events | None:
        """Correct vertical drift of fixation locations.

        Fixations are corrected per trial according to
        :py:attr:`~pymovements.Events.trial_columns` using the specified drift correction
        algorithm. Fixation locations
        are replaced with their corrected values. Original locations are preserved in a
        ``location_original`` column and the applied algorithm is recorded in a
        ``correction_algorithm`` column. Trials with too few fixations for the requested
        algorithms are skipped with a UserWarning and stay uncorrected. See
        :py:func:`~pymovements.events.correction.correct_fixations` for details.

        Parameters
        ----------
        aois: TextStimulus
            Text stimulus used for line position extraction. Its configured column names
            are mapped to the column names expected by the drift correction algorithms and
            its writing system provides the default reading direction.
        algorithm: str | list[str]
            Name of drift algorithm or list of algorithm names.
            (default: 'wisdom_of_the_crowd')
        directionality: str | None
            Reading direction of the text, either 'left-to-right' or 'right-to-left',
            mirroring the directionality of a text stimulus writing system.
            'top-to-bottom' is not supported and raises a ValueError. If None, the
            reading direction is inferred from the writing system of the text stimulus.
            (default: None)
        word_locations: polars.Series | None
            Series of [x, y] word center coordinates for the DTW-based algorithms
            'compare' and 'warp'. If None, word locations are derived from the aois
            dataframe. A user-supplied series is reused unchanged for every trial, so
            with per-trial AOIs leave it None to derive the word locations of each trial
            separately. (default: None)
        algorithm_kwargs: dict[str, Any] | None
            Additional tuning parameters passed to underlying drift correction algorithms.
            Warning: in ensemble mode an entry fans out to every candidate algorithm whose
            signature accepts the key, even where defaults and semantics differ. For
            example, ``{'x_thresh': 250.0}`` reconfigures 'chain', 'compare' and 'slice'
            at once. (default: None)
        fixation_name: str
            Name of the fixation events to correct. (default: 'fixation')
        character_level: bool
            Set to True when the stimulus AOIs are finer than words, e.g. one row per
            character. The AOIs are then aggregated to one location per word via the
            'word' column, which must be present. (default: False)
        inplace: bool
            If ``True``, mutate this object and return None. If ``False``, return a new
            :py:class:`~pymovements.Events` object with corrected fixation locations,
            leaving this object unchanged. (default: True)

        Returns
        -------
        Events | None
            None if ``inplace`` is True, otherwise a new
            :py:class:`~pymovements.Events` object with corrected fixation locations.

        Raises
        ------
        TypeError
            If ``aois`` is not a :py:class:`~pymovements.stimulus.TextStimulus`.
        ValueError
            If the trial or page column of the stimulus holds multiple unique values
            without being part of :py:attr:`~pymovements.Events.trial_columns`, as the
            AOIs of all trials or pages would be pooled into a single text.

        Examples
        --------
        Let's create fixations that drift away from three lines of text with their
        centers at y = 100, 200 and 300:

        >>> import polars
        >>> import pymovements as pm
        >>> events = pm.Events(
        ...     polars.DataFrame({
        ...         'name': ['fixation', 'fixation', 'fixation'],
        ...         'onset': [0, 200, 400],
        ...         'offset': [100, 300, 500],
        ...         'location': [[100.0, 105.0], [110.0, 195.0], [120.0, 302.0]],
        ...     }),
        ... )
        >>> stimulus = pm.stimulus.TextStimulus(
        ...     aois=polars.DataFrame({
        ...         'word': ['first', 'second', 'third'],
        ...         'start_x': [90.0, 90.0, 90.0],
        ...         'start_y': [80.0, 180.0, 280.0],
        ...         'end_x': [200.0, 200.0, 200.0],
        ...         'end_y': [120.0, 220.0, 320.0],
        ...     }),
        ...     aoi_column='word',
        ...     start_x_column='start_x',
        ...     start_y_column='start_y',
        ...     end_x_column='end_x',
        ...     end_y_column='end_y',
        ... )

        Correcting the fixations snaps each y-coordinate onto its line center and
        preserves the original locations:

        >>> events.correct_fixations(stimulus, algorithm='attach')
        >>> events.frame.select(['name', 'location', 'location_original'])
        shape: (3, 3)
        ┌──────────┬────────────────┬───────────────────┐
        │ name     ┆ location       ┆ location_original │
        │ ---      ┆ ---            ┆ ---               │
        │ str      ┆ list[f64]      ┆ list[f64]         │
        ╞══════════╪════════════════╪═══════════════════╡
        │ fixation ┆ [100.0, 100.0] ┆ [100.0, 105.0]    │
        │ fixation ┆ [110.0, 200.0] ┆ [110.0, 195.0]    │
        │ fixation ┆ [120.0, 300.0] ┆ [120.0, 302.0]    │
        └──────────┴────────────────┴───────────────────┘
        """
        if not isinstance(aois, TextStimulus):
            raise TypeError(
                f'aois must be a TextStimulus, but is of type {type(aois).__name__}.',
            )
        aois_frame = _aois_frame_from_text_stimulus(aois)
        if directionality is None:
            directionality = aois.writing_system.directionality

        for column_kind, column_name in (
                ('trial', aois.trial_column), ('page', aois.page_column),
        ):
            if column_name is None:
                continue
            if self.trial_columns is not None and column_name in self.trial_columns:
                continue
            n_unique = aois.aois[column_name].n_unique()
            if n_unique > 1:
                raise ValueError(
                    f"the stimulus {column_kind} column '{column_name}' holds "
                    f'{n_unique} unique values, but is not part of Events.trial_columns '
                    f'({self.trial_columns}), so the AOIs of all {column_kind}s would '
                    f"be pooled into a single text. Add '{column_name}' to "
                    'Events.trial_columns or pass a stimulus holding a single '
                    f'{column_kind}.',
                )

        corrected_frame = fixation_correction.correct_fixations(
            self.frame,
            aois_frame,
            algorithm=algorithm,
            trial_columns=self.trial_columns,
            directionality=directionality,
            word_locations=word_locations,
            algorithm_kwargs=algorithm_kwargs,
            fixation_name=fixation_name,
            character_level=character_level,
        )
        # The corrected fixations are derived from the stimulus file as well.
        if inplace:
            self.frame = corrected_frame
            self.metadata = merge_sources(self.metadata, aois.metadata)
            return None
        return Events(
            corrected_frame,
            trial_columns=self.trial_columns,
            metadata=merge_sources(self.metadata, aois.metadata),
            validate=False,
        )

    def measure_reading(
            self,
            stimulus: TextStimulus | dict[Any, TextStimulus],
            *,
            group_columns: list[str] | None = None,
            word_index_column: str = 'word_idx',
            word_column: str = 'word',
            event_name: str = 'fixation',
    ) -> ReadingMeasures:
        """Compute word-level reading measures from the fixation events.

        The fixations are mapped onto the AOIs of ``stimulus`` (see
        :py:meth:`~pymovements.Events.map_to_aois`) and word-level reading measures are computed
        from the resulting word indices (see
        :func:`~pymovements.measure.reading.compute_reading_measures`). The events are not
        modified. The mapping happens on an internal copy, and index columns from an earlier
        mapping are replaced on that copy. The ``Sources`` entry of the returned measures lists
        the sources of the events followed by those of the stimuli. Empty events return only
        their own sources.

        Parameters
        ----------
        stimulus : TextStimulus | dict[Any, TextStimulus]
            Text stimulus defining the AOIs. Pass a dict mapping values of the first group column
            to a separate stimulus per value when the events span several stimuli.
        group_columns : list[str] | None
            Column names used to partition the data into independent reading sequences. The first
            column takes the trial role, the remaining columns take the page role. If ``None`` or
            empty, the whole input is treated as a single reading sequence. Required if
            ``stimulus`` is a dict, whose keys are the values of the first group column.
            (default: None)
        word_index_column : str
            Shared column name in the mapped fixations and the stimulus AOIs that corresponds to
            the word index of the text.
            (default: ``'word_idx'``)
        word_column : str
            Column in the stimulus AOIs with the content within each AOI.
            (default: ``'word'``)
        event_name : str
            Name of the fixation events to compute measures from. Rows with a different ``name``
            are dropped. The name must start with ``fixation``, because the AOI mapping only maps
            fixation rows (see :py:meth:`~pymovements.Events.map_to_aois`).
            (default: ``'fixation'``)

        Returns
        -------
        ReadingMeasures
            Reading measures container with one row per word.

        Raises
        ------
        ValueError
            If ``stimulus`` is a dict and ``group_columns`` is empty, the dict is empty, the
            events have no first group column, the keys cannot be cast to that column's dtype,
            an entry's own sequence column differs from its key, or a sequence of the events has
            no entry.

        Warns
        -----
        UserWarning
            If ``stimulus`` is a dict with a key that matches no fixation. Its words are kept in
            the output with all measures zero and ``skipped`` set to 1. Also if a single
            ``stimulus`` has no first group column while the events do, in which case its AOIs
            are broadcast to every value of that column. Also if no fixation row carries
            ``event_name``, for example a name without the ``fixation`` prefix, in which case all
            measures are zero.

        Examples
        --------
        Two fixations on a three-word stimulus, where the third word is never looked at:

        >>> import polars
        >>> import pymovements as pm
        >>> events = pm.Events(
        ...     polars.DataFrame({
        ...         'name': ['fixation', 'fixation'],
        ...         'onset': [0, 200],
        ...         'offset': [200, 400],
        ...         'location': [[15.0, 15.0], [25.0, 15.0]],
        ...     }),
        ... )
        >>> stimulus = pm.stimulus.TextStimulus(
        ...     aois=polars.DataFrame({
        ...         'word': ['The', 'quick', 'brown'],
        ...         'word_idx': [0, 1, 2],
        ...         'start_x': [10.0, 20.0, 30.0],
        ...         'start_y': [10.0, 10.0, 10.0],
        ...         'end_x': [20.0, 30.0, 40.0],
        ...         'end_y': [30.0, 30.0, 30.0],
        ...     }),
        ...     aoi_column='word',
        ...     start_x_column='start_x',
        ...     start_y_column='start_y',
        ...     end_x_column='end_x',
        ...     end_y_column='end_y',
        ... )

        The measures hold one row per word. The skipped word has a total fixation time of 0:

        >>> measures = events.measure_reading(stimulus)
        >>> measures.frame.select(['word_index', 'word', 'TFT', 'TFC', 'skipped'])
        shape: (3, 5)
        ┌────────────┬───────┬─────┬─────┬─────────┐
        │ word_index ┆ word  ┆ TFT ┆ TFC ┆ skipped │
        │ ---        ┆ ---   ┆ --- ┆ --- ┆ ---     │
        │ i64        ┆ str   ┆ i64 ┆ u64 ┆ i64     │
        ╞════════════╪═══════╪═════╪═════╪═════════╡
        │ 0          ┆ The   ┆ 200 ┆ 1   ┆ 0       │
        │ 1          ┆ quick ┆ 200 ┆ 1   ┆ 0       │
        │ 2          ┆ brown ┆ 0   ┆ 0   ┆ 1       │
        └────────────┴───────┴─────┴─────┴─────────┘
        """
        if self.frame.is_empty():
            return ReadingMeasures(metadata=deepcopy(self.metadata))

        # map_to_aois keeps columns the events already have, so a previous mapping would be
        # reused. The index columns the measures consume are dropped from the working copy.
        stale_columns = [
            column for column in (word_index_column, 'char_idx') if column in self.frame.columns
        ]
        working_frame = self.frame.drop(stale_columns)

        if isinstance(stimulus, dict):
            stimuli = list(stimulus.values())
            fixations, aois = self._map_to_reading_stimuli(
                working_frame, stimulus, group_columns, word_index_column,
            )
        else:
            stimuli = [stimulus]
            mapped = Events(
                working_frame,
                trial_columns=self.trial_columns,
                metadata=self.metadata,
                validate=False,
            )
            mapped.map_to_aois(stimulus)
            fixations, aois = mapped.frame, stimulus.aois

        # The measures are derived from the events and the stimulus files.
        metadata = deepcopy(self.metadata)
        for stim in stimuli:
            metadata = merge_sources(metadata, stim.metadata)

        return ReadingMeasures(
            compute_reading_measures(
                fixations,
                aois,
                word_index_column=word_index_column,
                word_column=word_column,
                event_name=event_name,
                group_columns=group_columns,
            ),
            metadata=metadata,
        )

    def _map_to_reading_stimuli(
            self,
            frame: polars.DataFrame,
            stimuli: dict[Any, TextStimulus],
            group_columns: list[str] | None,
            word_index_column: str,
    ) -> tuple[polars.DataFrame, dict[Any, polars.DataFrame]]:
        """Map each reading sequence to its own stimulus and collect the per-key AOI tables.

        The events in ``frame`` are split by the first group column and the subset for each key
        is mapped to the matching stimulus. Rows whose key has no stimulus entry pass through
        unmapped. Returns the concatenated fixations and a dict of the per-key AOI tables for
        :func:`~pymovements.measure.reading.compute_reading_measures`.
        """
        # The dict is keyed by the first group column (the trial role), matching
        # compute_reading_measures.
        if not group_columns:
            raise ValueError(
                'stimulus given as dict requires at least one group column for its keys, e.g. '
                "group_columns=['trial'] when the dict is keyed by the trial column.",
            )
        sequence_column = group_columns[0]
        if sequence_column not in frame.columns:
            raise ValueError(
                f'stimulus is a dict keyed by {sequence_column!r}, but the events have no '
                f'{sequence_column!r} column.',
            )

        # The keys are compared with the column, so they are cast to its dtype once, the same
        # way compute_reading_measures does it.
        sequence_dtype = frame.schema[sequence_column]
        try:
            keys = polars.Series(sequence_column, list(stimuli)).cast(sequence_dtype)
        except polars.exceptions.PolarsError as exception:
            raise ValueError(
                f'stimulus dict keys are not compatible with the fixation {sequence_column} '
                f'dtype {sequence_dtype}: {exception}',
            ) from exception

        mapped_frames = []
        aois = {}
        for key, (original_key, stim) in zip(keys, stimuli.items()):
            aois[original_key] = stim.aois
            if sequence_column in stim.aois.columns:
                # map_to_aois still filters the entry's AOIs by this column, so values other
                # than the key would leave every fixation of the sequence unmatched. AOI rows
                # with a null value are ignored by the mapping and by this check.
                values = (
                    stim.aois[sequence_column].drop_nulls().cast(sequence_dtype, strict=False)
                    .unique().to_list()
                )
                if values != [key]:
                    raise ValueError(
                        f'stimulus for key {original_key!r} has {sequence_column} values '
                        f'{values} that differ from its dict key.',
                    )
                # The dict key defines the sequence column, so it must not also live in the
                # entry handed to compute_reading_measures.
                aois[original_key] = stim.aois.drop(sequence_column)
            subset = Events(
                frame.filter(polars.col(sequence_column) == key),
                trial_columns=self.trial_columns,
                metadata=self.metadata,
                validate=False,
            )
            if subset.frame.is_empty():
                continue
            subset.map_to_aois(stim)
            mapped_frames.append(subset.frame)

        # Rows whose key has no stimulus entry pass through unmapped, so that
        # compute_reading_measures raises its missing-key error instead of silently dropping them.
        # is_in is null for a null key, which filter would drop. A null key is unmatched too.
        unmatched = frame.filter(
            ~polars.col(sequence_column).is_in(keys.implode()).fill_null(False),
        )
        if not unmatched.is_empty():
            mapped_frames.append(unmatched)

        # Different stimuli may contribute different AOI columns, so align on the column union.
        fixations = polars.concat(mapped_frames, how='diagonal_relaxed')
        if word_index_column not in fixations.columns:
            # No key matched any row, so no mapping added the index column that
            # compute_reading_measures reads before it raises its own error.
            fixations = fixations.with_columns(
                polars.lit(None, dtype=polars.Int64).alias(word_index_column),
            )
        return fixations, aois

    def __eq__(self, other: Events) -> bool:
        """Check equality between this and another :py:class:`~pymovements.Events` object."""
        frames_equal = self.frame.equals(other.frame, null_equal=True)
        trial_columns_equal = self.trial_columns == other.trial_columns
        return frames_equal and trial_columns_equal

    def __str__(self: Any) -> str:
        """Return string representation of event dataframe."""
        return self.frame.__str__()

    def __repr__(self) -> str:
        """Return string representation of event dataframe."""
        return self.__str__()

    def summary(self) -> None:
        """Print a summary of the events.

        The summary shows the total number of events, followed by the event counts
        grouped by event name in alphabetical order.

        See Also
        --------
        pymovements.Gaze.summary : Print a summary of a gaze object.
        pymovements.Dataset.summary : Print a summary of a dataset.

        Examples
        --------
        >>> from pymovements import Events
        >>> events = Events(name=['saccade', 'fixation'], onsets=[0, 1], offsets=[1, 2])
        >>> events.summary()
        total events: 2
          fixation: 1
          saccade: 1
        """
        lines = [f'total events: {len(self):,}']

        if len(self):
            name_counts = self.frame['name'].value_counts().sort('name')
            for name, count in name_counts.iter_rows():
                lines.append(f'  {name}: {count:,}')

        print('\n'.join(lines))

    def merge_subsequent_close_events(
            self,
            name: str = 'fixation',
            max_gap: int | float = 50,
            verbose: bool = False,
    ) -> None:
        """Merge subsequent events if they are separated by a gap smaller than a threshold.

        Parameters
        ----------
        name: str
            The name of the events to be merged. (default: 'fixation')

        max_gap: int | float
            The maximum gap (in ms) between subsequent fixation events to be merged. (default: 50)

        verbose: bool
            If ``True``, print the number of events merged and the resulting number of events.

        Examples
        --------
        Let's create some example events first:

        >>> events = Events(
        ...     name='fixation',
        ...     onsets=[0, 2, 5, 13, 21, 22, 30, 40, 53, 73],
        ...     offsets=[1, 3, 10, 20, 22, 29, 35, 49, 70, 90],
        ... )
        >>> events.frame.shape
        (10, 4)

        Merging all events with particular name with a gap smaller than 10 ms:

        >>> events.merge_subsequent_close_events(name='fixation', max_gap=10)
        >>> events.frame
        shape: (1, 4)
        ┌──────────────┬──────────────┬──────────┬──────────────┐
        │ onset        ┆ duration     ┆ name     ┆ offset       │
        │ ---          ┆ ---          ┆ ---      ┆ ---          │
        │ duration[μs] ┆ duration[μs] ┆ str      ┆ duration[μs] │
        ╞══════════════╪══════════════╪══════════╪══════════════╡
        │ 0µs          ┆ 90ms         ┆ fixation ┆ 90ms         │
        └──────────────┴──────────────┴──────────┴──────────────┘

        This combined all the smaller events into a single event with longer duration.

        """
        # Step 1: Filter events of the specified type and sort by onset
        events = self.frame.filter(polars.col('name') == name).sort('onset')
        # set aside other events to merge them back later
        other = self.frame.filter(polars.col('name') != name)

        number_of_events = len(events)

        # Step 2: Calculate the gap between the current onset and the previous offset. Without an
        # offset column the previous event ends at its onset plus its duration.
        has_offset = 'offset' in events.columns
        previous_end = (
            polars.col('offset') if has_offset else polars.col('onset') + polars.col('duration')
        )
        events = events.with_columns(gap=polars.col('onset') - previous_end.shift(1))

        # Step 3: Create a 'group' identifier for merging
        events = events.with_columns(
            # calculate when gap is null or > max_gap (gap is a Duration; compare in
            # fractional milliseconds so sub-millisecond gaps are not truncated away)
            (
                polars.col('gap').is_null()
                | (polars.col('gap').dt.total_milliseconds(fractional=True) > max_gap)
            )
            .cast(polars.Int64)
            # cumulative sum (of ones) in 'group' to assign a unique group number
            # to each sequence of events to be merged
            .cum_sum()
            # the group identifier is the same for events that are close enough to be merged,
            # and different for events that are not close enough to be merged
            .alias('group'),
        )

        # Step 4: Aggregate events by group to merge them
        events = events.group_by('group').agg([
            # all columns from the first group element except offset and duration
            polars.exclude(['offset', 'duration']).first(),
            # the offset of the merged event is the last offset in the group
            *([polars.col('offset').last().alias('offset')] if has_offset else []),
            # the merged event lasts from the first onset to the end of the last event in the group
            (
                polars.col('onset').last() + polars.col('duration').last()
                - polars.col('onset').first()
            ).alias('duration'),
        ]).drop(['group', 'gap'])  # we don't need the group and gap columns anymore

        # Step 5: concatenate new events
        events = events.select(self.frame.columns)  # reorder columns to match original frame
        self.frame = polars.concat([events, other]).sort('onset')

        if verbose:
            print(
                f"Merged {number_of_events} '{name}' events "
                f'into {len(events)} events with max_gap={max_gap} ms.',
            )
