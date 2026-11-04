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

from pymovements._utils._time import duration_to_ms
from pymovements.exceptions import UnknownMeasure
from pymovements.measure.events.measures import EVENT_MEASURES
from pymovements.measure.samples.library import SampleMeasureLibrary

# Internal column names used to vectorize the event-to-sample assignment.
_EVENT_ID = '__pymovements_event_id'
_ROW_ID = '__pymovements_row_id'
_TRIAL_KEY = '__pymovements_trial_key'
_TIME_MS = '__pymovements_time_ms'
_ONSET_MS = '__pymovements_onset_ms'
_OFFSET_MS = '__pymovements_offset_ms'
_SAMPLE_IDX = '__pymovements_sample_idx'
_START_IDX = '__pymovements_start_idx'
_END_IDX = '__pymovements_end_idx'


class EventProcessor:
    """Processes events.

    Parameters
    ----------
    measures: str | list[str]
        List of event measure names.

    Raises
    ------
    UnknownMeasure
        If ``measures`` includes an unknown measure. See :ref:`event-measures` for an overview
        of supported measures.
    """

    def __init__(self, measures: str | list[str]):
        _check_measures(measures)

        if isinstance(measures, str):
            measures = [measures]

        for measure_name in measures:
            if measure_name not in EVENT_MEASURES:
                raise UnknownMeasure(
                    measure_name=measure_name, known_measures=list(EVENT_MEASURES.keys()),
                )

        self.measures = [
            # initialize measure functions to create polars expressions.
            EVENT_MEASURES[measure_name]()
            for measure_name in measures
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
        List of sample measures.

    Raises
    ------
    UnknownMeasure
        If ``event_properties`` includes an unknown measure. See :ref:`sample-measures` and
        :ref:`event-measures` for an overview of supported measures.
    """

    def __init__(
            self,
            measures: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
    ):
        _check_measures(measures)

        measures_with_kwargs: list[tuple[str, dict[str, Any]]]
        if isinstance(measures, str):
            measures_with_kwargs = [(measures, {})]
        elif isinstance(measures, tuple):
            measures_with_kwargs = [measures]
        else:  # we already validated above, it must be a list of strings and tuples
            measures_with_kwargs = [
                (measure, {}) if isinstance(measure, str) else measure
                for measure in measures
            ]

        for measure_name, _ in measures_with_kwargs:
            if measure_name not in SampleMeasureLibrary.measures:
                known_measures = list(SampleMeasureLibrary.measures.keys())
                raise UnknownMeasure(
                    measure_name=measure_name, known_measures=known_measures,
                )

        self.measures: list[pl.Expr] = [
            # initialize measure functions to create polars expressions.
            SampleMeasureLibrary.get(measure_name)(**measure_kwargs)
            for measure_name, measure_kwargs in measures_with_kwargs
        ]

    def process(
            self,
            events: pl.DataFrame,
            samples: pl.DataFrame,
            identifiers: str | list[str] | None = None,
            name: str | None = None,
    ) -> pl.DataFrame:
        """Process event and gaze dataframe.

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
        # a name and its on- and offset.
        event_identifiers = [*_identifiers, 'name', 'onset', 'offset']
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

        # Give each event a stable id that preserves the original row order and bring the event
        # bounds onto the common millisecond scale used for the sample assignment.
        events_work = events.with_columns(
            pl.int_range(pl.len()).alias(_EVENT_ID),
            (
                duration_to_ms('onset')
                if isinstance(events.schema['onset'], pl.Duration)
                else pl.col('onset').cast(pl.Float64)
            ).alias(_ONSET_MS),
            (
                duration_to_ms('offset')
                if isinstance(events.schema['offset'], pl.Duration)
                else pl.col('offset').cast(pl.Float64)
            ).alias(_OFFSET_MS),
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

        # The cheap asof assignment gives every sample to exactly one event, which is only correct
        # when events do not overlap. That holds for the standard detectors, but not necessarily
        # for a whole events frame (e.g. nested events, a fixation and a saccade sharing samples,
        # or two events sharing a boundary sample). In that case every event needs to collect the
        # shared samples, so fall back to expanding each event to its sample range.
        if _events_overlap(events_work, by_columns):
            aggregated = _aggregate_overlapping_events(
                events_work, samples_work, by_columns, self.measures,
            )
        else:
            aggregated = _aggregate_asof_events(
                events_work, samples_work, by_columns, self.measures,
            )

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
        events_work: pl.DataFrame,
        by_columns: list[str] | None,
) -> bool:
    """Return whether any two events within the same group overlap.

    Event bounds are inclusive, so events that only share a boundary sample overlap as well.
    """
    sorted_events = events_work.sort([*(by_columns or []), _ONSET_MS, _OFFSET_MS])
    previous_offset = pl.col(_OFFSET_MS).shift(1)
    if by_columns is not None:
        previous_offset = previous_offset.over(by_columns)
    return bool(sorted_events.select((previous_offset >= pl.col(_ONSET_MS)).any()).item())


def _aggregate_asof_events(
        events_work: pl.DataFrame,
        samples_work: pl.DataFrame,
        by_columns: list[str] | None,
        measures: list[pl.Expr],
) -> pl.DataFrame:
    """Aggregate samples per event when events do not overlap.

    A backward asof join assigns every sample to the event with the latest onset at or before
    it. This is the cheap path; it requires non-overlapping events.
    """
    events_sorted = events_work.sort([*(by_columns or []), _ONSET_MS])
    samples_sorted = samples_work.sort([*(by_columns or []), _TIME_MS, _ROW_ID])

    right_columns = [_EVENT_ID, _ONSET_MS, _OFFSET_MS, *(by_columns or [])]
    assigned = samples_sorted.join_asof(
        events_sorted.select(right_columns),
        left_on=_TIME_MS,
        right_on=_ONSET_MS,
        by=by_columns,
        strategy='backward',
        check_sortedness=False,
    )
    # Drop samples before the first event and samples past the event offset.
    assigned = assigned.filter(
        pl.col(_EVENT_ID).is_not_null() & (pl.col(_TIME_MS) <= pl.col(_OFFSET_MS)),
    )

    # Restore the original sample order within each event so order-dependent measures
    # (e.g. disposition) keep observing the samples in their source order.
    return assigned.sort(_ROW_ID).group_by(_EVENT_ID).agg(*measures)


def _aggregate_overlapping_events(
        events_work: pl.DataFrame,
        samples_work: pl.DataFrame,
        by_columns: list[str] | None,
        measures: list[pl.Expr],
) -> pl.DataFrame:
    """Aggregate samples per event when events may overlap.

    Within a group, the samples spanning an event form a contiguous range once the samples are
    sorted by time. The first sample at or after the onset and the last sample at or before the
    offset are found with forward and backward asof joins; the ranges are then expanded and
    joined back onto the samples, so every overlapping event gets all of its samples. Events
    without samples produce no range and therefore keep null properties.
    """
    group_columns = by_columns or []
    samples_sorted = (
        samples_work.sort([*group_columns, _TIME_MS, _ROW_ID])
        .with_columns(pl.int_range(pl.len()).alias(_SAMPLE_IDX))
    )
    sample_positions = samples_sorted.select([*group_columns, _TIME_MS, _SAMPLE_IDX])

    starts = (
        events_work.sort([*group_columns, _ONSET_MS])
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
        events_work.sort([*group_columns, _OFFSET_MS])
        .join_asof(
            sample_positions,
            left_on=_OFFSET_MS,
            right_on=_TIME_MS,
            by=by_columns,
            strategy='backward',
            check_sortedness=False,
        )
        .select(_EVENT_ID, pl.col(_SAMPLE_IDX).alias(_END_IDX))
    )
    bounds = starts.join(ends, on=_EVENT_ID).filter(
        pl.col(_START_IDX).is_not_null() & pl.col(_END_IDX).is_not_null()
        & (pl.col(_START_IDX) <= pl.col(_END_IDX)),
    )

    assigned = (
        bounds
        .select(
            _EVENT_ID,
            pl.int_ranges(pl.col(_START_IDX), pl.col(_END_IDX) + 1).alias(_SAMPLE_IDX),
        )
        .explode(_SAMPLE_IDX)
        .join(samples_sorted, on=_SAMPLE_IDX, how='left')
    )

    return assigned.sort(_ROW_ID).group_by(_EVENT_ID).agg(*measures)


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
