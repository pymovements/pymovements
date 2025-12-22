# Copyright (c) 2023-2025 The pymovements Project Authors
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

from collections import defaultdict
from collections.abc import Callable
from typing import Any
from warnings import warn

import polars as pl

from pymovements.exceptions import UnknownMeasure
from pymovements.measure.events.measures import EVENT_MEASURES
from pymovements.measure.samples.library import SampleMeasure
from pymovements.measure.samples.library import SampleMeasureLibrary


class EventProcessor:
    """Processes events.

    Parameters
    ----------
    measures: str | list[str]
        List of event measure names.

    Raises
    ------
    UnknownMeasure
        If ``measures`` includes an unknwon measure. See
        :py:mod:`pymovements.measure.events` for an overview of supported measures.
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
            :py:class:`polars.DataFrame` with properties as columns and rows refering to the rows in
            the source dataframe.
        """
        result = events.select(self.measures)
        return result


class EventSamplesProcessor:
    """Processes gaze samples grouped by individual events.

    Parameters
    ----------
    measures: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
        List of sample measures.

    UnknownMeasure
        If ``measures`` includes an unknwon measure. See
        :py:mod:`pymovements.measure.samples` for an overview of supported measures.
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
            :py:class:`polars.DataFrame` with properties as columns and rows refering to the rows in
            the source dataframe.

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

        if name is not None:
            events = events.filter(pl.col('name').str.contains(f'^{name}$'))
            if len(events) == 0:
                warn(f'No events with name "{name}" found in data frame')

        measure_values = defaultdict(list)
        for event in events.iter_rows(named=True):
            # Find samples that belong to the current event.
            event_samples = samples.filter(
                pl.col('time').is_between(event['onset'], event['offset']),
                *[pl.col(identifier) == event[identifier] for identifier in _identifiers],
            )
            # Compute event measure values.
            values = event_samples.select([measure for measure in self.measures])
            # Collect measure values.
            for measure_name in measure_names:
                measure_values[measure_name].append(values[measure_name].item())

        # The resulting DataFrame contains the event identifiers and the computed properties.
        result = events.select(event_identifiers).with_columns(
            *[pl.Series(name, values) for name, values in measure_values.items()],
        )
        return result


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
                f"but received {type(measures[0])}.",
            )
        if not isinstance(measures[1], dict):
            raise TypeError(
                'Second item of tuple must be a dictionary, '
                f"but received {type(measures[1])}.",
            )
    elif isinstance(measures, list):
        for measure in measures:
            if not isinstance(measure, (str, tuple)):
                raise TypeError(
                    'Each item in the list must be either a string or a tuple, '
                    f"but received {type(measure)}.",
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
            f"but received {type(measures)}.",
        )
