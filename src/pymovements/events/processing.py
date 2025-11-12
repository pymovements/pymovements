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

import polars as pl

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements.events.events import Events
from pymovements.events.properties import EVENT_PROPERTIES
from pymovements.exceptions import InvalidProperty


class EventProcessor:
    """Processes events.

    Parameters
    ----------
    event_properties: str | list[str]
        List of event property names.
    """

    def __init__(self, event_properties: str | list[str]):
        _check_event_properties(event_properties)

        if isinstance(event_properties, str):
            event_properties = [event_properties]

        valid_properties = ['duration']  # all other properties need gaze samples.
        for property_name in event_properties:
            if property_name not in valid_properties:
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties = event_properties

    def process(self, events: Events) -> pl.DataFrame:
        """Process event dataframe.

        Parameters
        ----------
        events: Events
            Event data to process event properties from.

        Returns
        -------
        pl.DataFrame
            :py:class:`polars.DataFrame` with properties as columns and rows refering to the rows in
            the source dataframe.

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events` for an overview of supported properties.
        """
        property_expressions: dict[str, Callable[[], pl.Expr]] = {
            property_name: EVENT_PROPERTIES[property_name]
            for property_name in self.event_properties
        }

        expression_list = [
            property_expression().alias(property_name)
            for property_name, property_expression in property_expressions.items()
        ]
        result = events.frame.select(expression_list)
        return result


class EventGazeProcessor:
    """Processes events and gaze samples.

    Parameters
    ----------
    event_properties: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
        List of event property names.
    """

    def __init__(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
    ):
        _check_event_properties(event_properties)

        event_properties_with_kwargs: list[tuple[str, dict[str, Any]]]
        if isinstance(event_properties, str):
            event_properties_with_kwargs = [(event_properties, {})]
        elif isinstance(event_properties, tuple):
            event_properties_with_kwargs = [event_properties]
        else:  # we already validated above, it must be a list of strings and tuples
            event_properties_with_kwargs = [
                (event_property, {}) if isinstance(event_property, str) else event_property
                for event_property in event_properties
            ]

        for property_name, _ in event_properties_with_kwargs:
            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties: list[tuple[str, dict[str, Any]]] = event_properties_with_kwargs

    def process(
            self,
            events: Events,
            gaze: pm.Gaze,
            identifiers: str | list[str],
            name: str | None = None,
    ) -> pl.DataFrame:
        """Process event and gaze dataframe.

        Parameters
        ----------
        events: Events
            Event data to process event properties from.
        gaze: pm.Gaze
            Gaze data to process event properties from.
        identifiers: str | list[str]
            Column names to join on events and gaze dataframes.
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
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events` for an overview of supported properties.
        RuntimeError
            If specified event name ``name`` is missing from ``events``.
        """
        if isinstance(identifiers, str):
            trial_identifiers = [identifiers]
        else:
            trial_identifiers = identifiers

        if len(trial_identifiers) == 0:
            raise ValueError('list of identifiers must not be empty')

        property_expressions: list[Callable[..., pl.Expr]] = [
            EVENT_PROPERTIES[property_name] for property_name, _ in self.event_properties
        ]

        property_names: list[str] = [property_name for property_name, _ in self.event_properties]

        property_kwargs: list[dict[str, Any]] = [
            property_kwargs for _, property_kwargs in self.event_properties
        ]

        # Each event is uniquely defined by a list of trial identifiers,
        # a name and its on- and offset.
        event_identifiers = [*trial_identifiers, 'name', 'onset', 'offset']

        events_frame = events.frame
        if name is not None:
            events_frame = events_frame.filter(pl.col('name').str.contains(f'^{name}$'))
            if len(events_frame) == 0:
                raise RuntimeError(f'No events with name "{name}" found in data frame')

        property_values = defaultdict(list)
        for event in events_frame.iter_rows(named=True):
            # Find gaze samples that belong to the current event.
            filtered_gaze = gaze.samples.filter(
                pl.col('time').is_between(event['onset'], event['offset']),
                *[pl.col(identifier) == event[identifier] for identifier in trial_identifiers],
            )
            # Compute event property values.
            values = filtered_gaze.select(
                [
                    this_property_expression(**this_property_kwargs)
                    .alias(this_property_name)
                    for this_property_name, this_property_expression, this_property_kwargs,
                    in zip(property_names, property_expressions, property_kwargs)
                ],
            )
            # Collect property values.
            for property_name in property_names:
                property_values[property_name].append(values[property_name].item())

        # The resulting DataFrame contains the event identifiers and the computed properties.
        result = events_frame.select(event_identifiers).with_columns(
            *[pl.Series(name, values) for name, values in property_values.items()],
        )
        return result


def _check_event_properties(
        event_properties: str | tuple[str, dict[str, Any]] | list[str]
        | list[str | tuple[str, dict[str, Any]]],
) -> None:
    """Validate event properties."""
    if isinstance(event_properties, str):
        pass
    elif isinstance(event_properties, tuple):
        if len(event_properties) != 2:
            raise ValueError('Tuple must have a length of 2.')
        if not isinstance(event_properties[0], str):
            raise TypeError(
                f'First item of tuple must be a string, '
                f"but received {type(event_properties[0])}.",
            )
        if not isinstance(event_properties[1], dict):
            raise TypeError(
                'Second item of tuple must be a dictionary, '
                f"but received {type(event_properties[1])}.",
            )
    elif isinstance(event_properties, list):
        for event_property in event_properties:
            if not isinstance(event_property, (str, tuple)):
                raise TypeError(
                    'Each item in the list must be either a string or a tuple, '
                    f"but received {type(event_property)}.",
                )
            if isinstance(event_property, tuple):
                if len(event_property) != 2:
                    raise ValueError('Tuple must have a length of 2.')
                if not isinstance(event_property[0], str):
                    raise TypeError(
                        'First item of tuple must be a string, '
                        f'but received {type(event_property[0])}.',
                    )
                if not isinstance(event_property[1], dict):
                    raise TypeError(
                        'Second item of tuple must be a dictionary, '
                        f'but received {type(event_property[1])}.',
                    )
    else:
        raise TypeError(
            'event_properties must be of type str, tuple, or list, '
            f"but received {type(event_properties)}.",
        )
