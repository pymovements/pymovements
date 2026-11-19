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
"""Module for event processing."""
from __future__ import annotations

from typing import Any
from warnings import warn

import polars as pl

from pymovements._utils.time import duration_to_ms
from pymovements.events._utils.selection import selection_end
from pymovements.exceptions import UnknownMeasure
from pymovements.measure.events.measures import EVENT_MEASURES
from pymovements.measure.samples.library import SampleMeasureLibrary

# Internal column names used to vectorize the event-to-sample assignment.
_EVENT_ID = '__pymovements_event_id'
_ROW_ID = '__pymovements_row_id'
_TRIAL_KEY = '__pymovements_trial_key'
_TIME_MS = '__pymovements_time_ms'
_ONSET_MS = '__pymovements_onset_ms'
_DURATION_MS = '__pymovements_duration_ms'
_END_MS = '__pymovements_end_ms'
_SAMPLE_IDX = '__pymovements_sample_idx'
_START_IDX = '__pymovements_start_idx'
_END_IDX = '__pymovements_end_idx'


def measures_with_kwargs(
        measures: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Normalize a measure specification to a list of names with keyword arguments.

    Parameters
    ----------
    measures: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
        A measure name, a tuple of a name and its keyword arguments, or a list of either.

    Returns
    -------
    list[tuple[str, dict[str, Any]]]
        One tuple per measure, with an empty dict for a bare name.
    """
    _check_measures(measures)

    if isinstance(measures, str):
        return [(measures, {})]
    if isinstance(measures, tuple):
        return [measures]
    # we already validated above, it must be a list of strings and tuples
    return [
        (measure, {}) if isinstance(measure, str) else measure
        for measure in measures
    ]


class EventProcessor:
    """Processes events.

    Parameters
    ----------
    measures: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
        List of event measures, each a name or a tuple of a name and its keyword arguments.

    Raises
    ------
    UnknownMeasure
        If ``measures`` includes an unknown measure. See :ref:`event-measures` for an overview
        of supported measures.
    """

    def __init__(
            self,
            measures: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
    ):
        measures_parsed = measures_with_kwargs(measures)

        for measure_name, _ in measures_parsed:
            if measure_name not in EVENT_MEASURES:
                raise UnknownMeasure(
                    measure_name=measure_name, known_measures=list(EVENT_MEASURES.keys()),
                )

        self.measures = [
            # initialize measure functions to create polars expressions.
            EVENT_MEASURES[measure_name](**measure_kwargs)
            for measure_name, measure_kwargs in measures_parsed
        ]

    def process(self, events: pl.DataFrame) -> pl.DataFrame:
        """Process event dataframe.

        Parameters
        ----------
        events: pl.DataFrame
            Event data to process event properties from.

        Returns
        -------
        pl.DataFrame
            :py:class:`polars.DataFrame` with properties as columns and rows referring to the rows
            in the source dataframe.
        """
        result = events.select(self.measures)
        return result


class EventSamplesProcessor:
    """Processes gaze samples grouped by individual events.

    Parameters
    ----------
    measures: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
        List of sample measures, each a name or a tuple of a name and its keyword arguments.

    Raises
    ------
    UnknownMeasure
        If ``measures`` includes an unknown measure. See :ref:`sample-measures` for an overview
        of supported measures.
    """

    def __init__(
            self,
            measures: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
    ):
        measures_parsed = measures_with_kwargs(measures)

        for measure_name, _ in measures_parsed:
            if measure_name not in SampleMeasureLibrary.measures:
                known_measures = list(SampleMeasureLibrary.measures.keys())
                raise UnknownMeasure(
                    measure_name=measure_name, known_measures=known_measures,
                )

        self.measures: list[pl.Expr] = [
            # initialize measure functions to create polars expressions.
            SampleMeasureLibrary.get(measure_name)(**measure_kwargs)
            for measure_name, measure_kwargs in measures_parsed
        ]

    def process(
            self,
            events: pl.DataFrame,
            samples: pl.DataFrame,
            identifiers: str | list[str] | None = None,
            name: str | None = None,
            sampling_rate: float | None = None,
    ) -> pl.DataFrame:
        """Process event and gaze dataframe.

        The samples of an event are those in the half-open interval from its onset to its end,
        ``onset + duration``. With a sampling rate the end moves half a sampling interval
        earlier, so a sample that arrives slightly early stays out of the preceding event. An
        event with a ``null`` or ``0`` duration selects no samples and keeps its row with null
        properties.

        Parameters
        ----------
        events: pl.DataFrame
            Event data to process event properties from.
        samples: pl.DataFrame
            Samples data to process event properties from.
        identifiers: str | list[str] | None
            Column names to join on events and samples dataframes. (default: None)
        name: str | None
            Process only events that match the name. (default: None)
        sampling_rate: float | None
            Sampling rate in Hz for the tolerance at the end of each event. Without one the
            plain interval applies. (default: None)

        Returns
        -------
        pl.DataFrame
            :py:class:`polars.DataFrame` with properties as columns and rows referring to the rows
            in the source dataframe.

        Raises
        ------
        ValueError
            If list of identifiers is empty.
        RuntimeError
            If specified event name ``name`` is missing from ``events``.
        """
        if identifiers is None:
            _identifiers = []
        elif isinstance(identifiers, str):
            _identifiers = [identifiers]
        else:
            _identifiers = identifiers

        # Each event is uniquely defined by a list of trial identifiers,
        # a name, its onset and its duration.
        event_identifiers = [*_identifiers, 'name', 'onset', 'duration']
        measure_columns = [measure.meta.output_name() for measure in self.measures]

        if name is not None:
            events = events.filter(pl.col('name').str.contains(f'^{name}$'))
            if len(events) == 0:
                warn(f"No events found with name '{name}'.")

        # Sample measures expect the time column in numeric milliseconds. The samples handed to
        # the measures are converted below. Event bounds and sample times are additionally put
        # on a common millisecond scale for the vectorized event membership assignment, so it
        # stays correct whether or not the events and samples time dtypes match (e.g. Duration
        # events against a numeric sample time column, or vice versa). Duration values are
        # converted by their physical unit; a numeric time column follows the codebase
        # convention of already being in milliseconds.
        time_is_duration = (
            'time' in samples.columns and isinstance(samples.schema['time'], pl.Duration)
        )
        samples_work = samples
        if time_is_duration:
            samples_work = samples_work.with_columns(duration_to_ms('time').alias('time'))

        if len(events) == 0:
            warn(
                f'No events available for processing. Creating empty columns for {measure_columns}',
            )

            # run measures on empty samples data frame.
            return pl.LazyFrame(schema=dict(samples_work.schema)).select(
                *[pl.repeat(None, 0).alias(column_name) for column_name in event_identifiers],
                *self.measures,
            ).collect()

        # Give each event a stable id that preserves the original row order, bring the onset and
        # duration onto the common millisecond scale used for the sample assignment and derive
        # the end bound of the half-open selection from them.
        events_work = events.with_columns(
            pl.int_range(pl.len()).alias(_EVENT_ID),
            (
                duration_to_ms('onset')
                if isinstance(events.schema['onset'], pl.Duration)
                else pl.col('onset').cast(pl.Float64)
            ).alias(_ONSET_MS),
            (
                duration_to_ms('duration')
                if isinstance(events.schema['duration'], pl.Duration)
                else pl.col('duration').cast(pl.Float64)
            ).alias(_DURATION_MS),
        ).with_columns(
            selection_end(_ONSET_MS, _DURATION_MS, sampling_rate=sampling_rate).alias(_END_MS),
        )
        samples_work = samples_work.with_columns(
            pl.int_range(pl.len()).alias(_ROW_ID),
            pl.col('time').cast(pl.Float64).alias(_TIME_MS),
        )

        # Assign each sample to the event it belongs to. Grouping by a hashed trial key makes the
        # asof join operate per trial and treats null identifiers as matching values. Overlapping
        # events are expanded to per-event sample ranges when they are detected below.
        if _identifiers:
            events_work = events_work.with_columns(
                pl.struct(_identifiers).hash().alias(_TRIAL_KEY),
            )
            samples_work = samples_work.with_columns(
                pl.struct(_identifiers).hash().alias(_TRIAL_KEY),
            )
            by_columns: list[str] | None = [_TRIAL_KEY]
        else:
            by_columns = None

        # Events without an end bound (a null onset or duration) select no samples and are left
        # out of the assignment, so they neither collect samples nor take them from other events.
        group_columns = by_columns or []
        events_sorted = (
            events_work
            .filter(pl.col(_ONSET_MS).is_not_null() & pl.col(_END_MS).is_not_null())
            .sort([*group_columns, _ONSET_MS, _END_MS])
        )
        samples_sorted = samples_work.sort([*group_columns, _TIME_MS, _ROW_ID])

        # The cheap asof assignment gives every sample to exactly one event, which is only correct
        # when events do not overlap. That holds for the standard detectors, but not necessarily
        # for a whole events frame (e.g. nested events, or a fixation and a saccade sharing
        # samples). In that case every event needs to collect the shared samples, so fall back to
        # expanding each event to its sample range.
        if _events_overlap(events_sorted, by_columns):
            assigned = _assign_samples_by_range(events_sorted, samples_sorted, by_columns)
        else:
            assigned = _assign_samples_asof(events_sorted, samples_sorted, by_columns)

        # Restore the original sample order within each event so order-dependent measures
        # (e.g. disposition) keep observing the samples in their source order, then aggregate
        # all measures in a single pass.
        aggregated = assigned.sort(_ROW_ID).group_by(_EVENT_ID).agg(*self.measures)

        # Left-join back onto the events so events without samples get null properties and the
        # original event order is preserved. Only the identifier columns are carried over to
        # avoid clashing with measure columns that already exist in the events dataframe.
        return (
            events_work
            .select(*event_identifiers, _EVENT_ID)
            .join(aggregated, on=_EVENT_ID, how='left')
            .sort(_EVENT_ID)
            .select(*event_identifiers, *measure_columns)
        )


def _events_overlap(
        events_sorted: pl.DataFrame,
        by_columns: list[str] | None,
) -> bool:
    """Return whether any two events within the same group overlap.

    Expects the events sorted by group, onset and end, with no null onsets or ends. The selection
    is half-open, so an event ending where the next one starts does not overlap with it.
    """
    previous_end = pl.col(_END_MS).shift(1)
    if by_columns is not None:
        previous_end = previous_end.over(by_columns)
    return bool(events_sorted.select((previous_end > pl.col(_ONSET_MS)).any()).item())


def _assign_samples_asof(
        events_sorted: pl.DataFrame,
        samples_sorted: pl.DataFrame,
        by_columns: list[str] | None,
) -> pl.DataFrame:
    """Assign each sample to its event when events do not overlap.

    A backward asof join assigns every sample to the event with the latest onset at or before
    it. This is the cheap path; it requires non-overlapping events. Expects the events sorted by
    group and onset, and the samples sorted by group, time and row id.
    """
    right_columns = [_EVENT_ID, _ONSET_MS, _END_MS, *(by_columns or [])]
    assigned = samples_sorted.join_asof(
        events_sorted.select(right_columns),
        left_on=_TIME_MS,
        right_on=_ONSET_MS,
        by=by_columns,
        strategy='backward',
        check_sortedness=False,
    )
    # Drop samples before the first event and samples at or past the event end.
    return assigned.filter(
        pl.col(_EVENT_ID).is_not_null() & (pl.col(_TIME_MS) < pl.col(_END_MS)),
    )


def _assign_samples_by_range(
        events_sorted: pl.DataFrame,
        samples_sorted: pl.DataFrame,
        by_columns: list[str] | None,
) -> pl.DataFrame:
    """Assign each sample to its events when events may overlap.

    Within a group, the samples spanning an event form a contiguous range once the samples are
    sorted by time. The first sample at or after the onset and the last sample before the end
    are found with forward and backward asof joins; the ranges are then expanded and joined back
    onto the samples, so every overlapping event gets all of its samples. Events without samples
    produce no range and therefore keep null properties. Expects the events sorted by group and
    onset, and the samples sorted by group, time and row id.
    """
    group_columns = by_columns or []
    samples_sorted = samples_sorted.with_columns(pl.int_range(pl.len()).alias(_SAMPLE_IDX))
    sample_positions = samples_sorted.select([*group_columns, _TIME_MS, _SAMPLE_IDX])

    starts = (
        events_sorted
        .join_asof(
            sample_positions,
            left_on=_ONSET_MS,
            right_on=_TIME_MS,
            by=by_columns,
            strategy='forward',
            check_sortedness=False,
        )
        .select(_EVENT_ID, pl.col(_SAMPLE_IDX).alias(_START_IDX))
    )
    ends = (
        events_sorted.sort([*group_columns, _END_MS])
        .join_asof(
            sample_positions,
            left_on=_END_MS,
            right_on=_TIME_MS,
            by=by_columns,
            strategy='backward',
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .select(_EVENT_ID, pl.col(_SAMPLE_IDX).alias(_END_IDX))
    )
    bounds = starts.join(ends, on=_EVENT_ID).filter(
        pl.col(_START_IDX).is_not_null() & pl.col(_END_IDX).is_not_null()
        & (pl.col(_START_IDX) <= pl.col(_END_IDX)),
    )

    return (
        bounds
        .select(
            _EVENT_ID,
            pl.int_ranges(pl.col(_START_IDX), pl.col(_END_IDX) + 1).alias(_SAMPLE_IDX),
        )
        .explode(_SAMPLE_IDX)
        .join(samples_sorted, on=_SAMPLE_IDX, how='left')
    )


def _check_measures(
        measures: str | tuple[str, dict[str, Any]] | list[str]
        | list[str | tuple[str, dict[str, Any]]],
) -> None:
    """Validate event properties."""
    if isinstance(measures, str):
        pass
    elif isinstance(measures, tuple):
        if len(measures) != 2:
            raise ValueError('Tuple must have a length of 2.')
        if not isinstance(measures[0], str):
            raise TypeError(
                f'First item of tuple must be a string, '
                f'but received {type(measures[0])}.',
            )
        if not isinstance(measures[1], dict):
            raise TypeError(
                'Second item of tuple must be a dictionary, '
                f'but received {type(measures[1])}.',
            )
    elif isinstance(measures, list):
        for measure in measures:
            if not isinstance(measure, (str, tuple)):
                raise TypeError(
                    'Each item in the list must be either a string or a tuple, '
                    f'but received {type(measure)}.',
                )
            if isinstance(measure, tuple):
                if len(measure) != 2:
                    raise ValueError('Tuple must have a length of 2.')
                if not isinstance(measure[0], str):
                    raise TypeError(
                        'First item of tuple must be a string, '
                        f'but received {type(measure[0])}.',
                    )
                if not isinstance(measure[1], dict):
                    raise TypeError(
                        'Second item of tuple must be a dictionary, '
                        f'but received {type(measure[1])}.',
                    )
    else:
        raise TypeError(
            'measures must be of type str, tuple, or list, '
            f'but received {type(measures)}.',
        )
