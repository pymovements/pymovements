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
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any
from typing import Literal
from typing import overload
from warnings import warn

import numpy as np
import polars
from tqdm import tqdm

from pymovements._utils import _checks
from pymovements._utils._column_nesting import unnest_list_columns
from pymovements._utils._html import repr_html
from pymovements._utils._nulls import row_is_null
from pymovements._utils._sources import merge_sources
from pymovements._utils._time import normalize_duration_to_us
from pymovements._utils._time import numeric_to_duration_us
from pymovements.events.correction import fixation_correction
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
        if it is the first sample after the event. None leaves the convention undeclared, see
        the notes below. (default: None)
    durations_from_offsets: bool
        If True, durations are derived from declared offsets and replace supplied durations.
        (default: False)
    sampling_rate: float | None
        The sampling rate in Hz. Its interval is the duration of the last sample of an inclusive
        offset. Required to derive durations from inclusive offsets. (default: None)
    validate: bool
        If False, supplied durations are kept unchecked beside declared offsets. (default: True)
    trials: list[int | float | str | None] | np.ndarray | None
        List of trial identifiers. (default: None)
    trial_columns: list[str] | str | None
        List of trial columns in the passed dataframe. (default: None)
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata. A ``Sources`` entry lists the files the
        event data was generated from. (default: None)
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
        ``'us'``, if ``sampling_rate`` is not positive, if durations are derived from
        inclusive offsets without a sampling rate, or if every row contradicts the declared
        offset convention.

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
        self.metadata = metadata if metadata is not None else {}

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
            _checks.check_is_mutual_exclusive(data=data, onsets=onsets)
            _checks.check_is_mutual_exclusive(data=data, durations=durations)
            _checks.check_is_mutual_exclusive(data=data, offsets=offsets)
            _checks.check_is_mutual_exclusive(data=data, name=name)
            _checks.check_is_mutual_exclusive(data=data, name=trials)

            durations_supplied = 'duration' in data.columns
            data = data.clone()
            data = self._add_minimal_schema_columns(data)
            data_dict = data.to_dict()

            if isinstance(trial_columns, str):
                self.trial_columns = [trial_columns]
            else:
                self.trial_columns = trial_columns

        else:
            # Onsets are the anchor: durations and offsets are only accepted beside them.
            if onsets is None and (durations is not None or offsets is not None):
                raise ValueError('durations and offsets can only be passed together with onsets')
            durations_supplied = durations is not None

            if onsets is not None:
                # Make sure that durations, offsets and names match the onsets in length.
                if durations is not None:
                    _checks.check_is_length_matching(onsets=onsets, durations=durations)
                if offsets is not None:
                    _checks.check_is_length_matching(onsets=onsets, offsets=offsets)
                if isinstance(name, Sequence) and not isinstance(name, str):
                    _checks.check_is_length_matching(onsets=onsets, name=name)

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

        self._resolve_durations(
            durations_supplied=durations_supplied,
            offsets_inclusive=offsets_inclusive,
            durations_from_offsets=durations_from_offsets,
            sampling_rate=sampling_rate,
            validate=validate,
        )

    def _resolve_durations(
            self,
            *,
            durations_supplied: bool,
            offsets_inclusive: bool | None,
            durations_from_offsets: bool,
            sampling_rate: float | None,
            validate: bool,
    ) -> None:
        """Fill or check the ``duration`` column against the ``offset`` column.

        See the construction rules in the class docstring.
        """
        if 'offset' not in self.frame.columns:
            ignored = []
            if offsets_inclusive is not None:
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

        if validate:
            self._check_durations_against_offsets(
                offsets_inclusive=offsets_inclusive,
                sampling_rate=sampling_rate,
                sampling_interval=sampling_interval,
            )

    def _check_durations_against_offsets(
            self,
            *,
            offsets_inclusive: bool,
            sampling_rate: float | None,
            sampling_interval: int | None,
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
            if offsets_inclusive:
                message += (
                    '. Pass durations_from_offsets=True to derive the durations from the '
                    'offsets, offsets_inclusive=False if the offsets are exclusive, or '
                    'validate=False to keep the durations unchecked.'
                )
            else:
                message += (
                    '. Pass offsets_inclusive=True if the offsets are inclusive, or '
                    'validate=False to keep the durations unchecked.'
                )
            raise ValueError(message)

        if sampling_interval is not None and offsets_inclusive:
            expected_description = (
                f'{_format_microseconds(sampling_interval)} '
                f'(sampling_rate={sampling_rate:g} from the sampling_rate keyword)'
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
            _checks.check_is_length_matching(column=column, data=data)

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
            metadata=deepcopy(self.metadata),
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
                    metadata=deepcopy(self.metadata),
                    validate=False,
                )
                for key, frame in event_dfs.items()
            }

        return [
            Events(
                frame,
                trial_columns=self.trial_columns,
                metadata=deepcopy(self.metadata),
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
            verbose: bool = True,
    ) -> None:
        """Map events to AOIs, ignoring non-fixations.

        This function computes AOI membership only for rows whose ``name`` starts with
        ``"fixation"`` (e.g., ``"fixation"``, ``"fixation_ivt"``). Rows that are not fixations
        are left unchanged and receive ``None`` values for all AOI columns. The original order
        and number of rows are preserved.

        Schema handling:

        - If ``preserve_structure=True`` (default), we mirror legacy behavior when a list
          ``location`` column exists: derive ``location_x``/``location_y`` and drop ``location``.
          This keeps downstream expectations about flat component columns.
        - If ``preserve_structure=False``, no unnesting/derivation occurs and the original
          ``location`` list column is preserved. Coordinates are extracted per-row without
          altering the frame.

        AOI columns used for trial/page keys in the stimulus (``trial_column``/``page_column``)
        are not appended to the events, as they are dropped by ``TextStimulus.get_aoi`` to avoid
        duplicate columns during concatenation.

        Parameters
        ----------
        aoi_dataframe: TextStimulus
            Text stimulus defining AOI rectangles.
        preserve_structure: bool
            Control whether to derive component columns and drop the list column as described
            above. Default: True.
        verbose : bool
            If ``True``, show a progress bar. (default: True)

        Raises
        ------
        ValueError
            If ``aoi_dataframe`` does not have either ``width_column`` or ``end_x_column`` defined.
        ValueError
            If the events frame is empty.
        """
        # Validate AOI configuration early
        if aoi_dataframe.width_column is None and aoi_dataframe.end_x_column is None:
            raise ValueError(
                'either TextStimulus.width or TextStimulus.end_x_column must be defined',
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

        def _empty_aoi_row() -> polars.DataFrame:
            # Create one row of all-None cast to expected AOI dtypes
            return polars.DataFrame({col: [None] for col in aoi_columns}).cast(aoi_schema)

        out_rows: list[polars.DataFrame] = []
        for row in tqdm(
                self.frame.iter_rows(named=True),
                total=len(self.frame),
                desc='Mapping events to AOIs',
                unit='event',
                ncols=80,
                disable=not verbose,
        ):
            name_val = row.get('name')
            is_fix = isinstance(name_val, str) and name_val.startswith('fixation')
            if not is_fix:
                out_rows.append(_empty_aoi_row())
                continue

            # Extract coordinates - support either pre-existing component columns or list column
            x = row.get('location_x')
            y = row.get('location_y')
            if x is None or y is None:
                loc = row.get('location')
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    x, y = loc[0], loc[1]

            if x is None or y is None:
                out_rows.append(_empty_aoi_row())
                continue

            # Create a shallow copy with temporary keys for AOI lookup
            tmp_row = dict(row)
            tmp_row['__x'] = x
            tmp_row['__y'] = y

            try:
                aoi_row = aoi_dataframe.get_aoi(
                    row=tmp_row, x_eye='__x', y_eye='__y', max_matches=1,
                )
            except (KeyError, TypeError):  # tolerate common lookup/type errors per row
                aoi_row = _empty_aoi_row()
            else:
                # Project to the selected AOI columns and fill any missing ones with None
                if aoi_columns:
                    present = [c for c in aoi_columns if c in aoi_row.columns]
                    missing = [c for c in aoi_columns if c not in aoi_row.columns]
                    aoi_row = aoi_row.select(
                        [polars.col(c) for c in present]
                        + [polars.lit(None).alias(c) for c in missing],
                    )
                    # Cast to the stable AOI schema to ensure consistent dtypes across rows
                    aoi_row = aoi_row.cast(aoi_schema)
                else:
                    # No AOI columns are to be appended (all already exist in the Events frame).
                    # Keep row count but contribute zero columns to avoid duplicate-column errors.
                    aoi_row = aoi_row.select([])
            out_rows.append(aoi_row)

        aoi_df = polars.concat(out_rows)
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
            metadata=merge_sources(deepcopy(self.metadata), aois.metadata),
            validate=False,
        )

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
