# Copyright (c) 2022-2025 The pymovements Project Authors
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
from typing import Any
from typing import Literal
from typing import overload

import numpy as np
import polars as pl
from deprecated.sphinx import deprecated
from tqdm import tqdm

from pymovements._utils import _checks
from pymovements._utils._html import repr_html
from pymovements.events.properties import duration
from pymovements.stimulus.text import TextStimulus


@repr_html(['frame', 'trial_columns'])
class Events:
    """A data structure for event data.

    Each row has at least an event name with its onset and offset specified.

    Parameters
    ----------
    data: pl.DataFrame | None
        A dataframe to be transformed to a polars dataframe. This argument is mutually
        exclusive with all the other arguments. (default: None)
    name: str | list[str] | None
        Name of events. (default: None)
    onsets: list[int | float] | np.ndarray | None
        List of onsets. (default: None)
    offsets: list[int | float] | np.ndarray | None
        List of offsets. (default: None)
    trials: list[int | float | str | None] | np.ndarray | None
        List of trial identifiers. (default: None)
    trial_columns: list[str] | str | None
        List of trial columns in passed dataframe.

    Attributes
    ----------
    frame: pl.DataFrame
        A dataframe of events.
    trial_columns: list[str] | None
        The name of the trial columns in the data frame. If not None, processing methods
        will be applied to each trial separately.

    Raises
    ------
    ValueError
        If list of onsets is passed but not a list of offsets, or vice versa, or if length of
        onsets does not match length of offsets.

    Examples
    --------
    We define an event dataframe with given names of events and lists of onsets and offsets.
    Durations are computed automatically.

    >>> event = Events(
    ...    name=['fixation', 'fixation', 'fixation', 'fixation', ],
    ...    onsets=[1988147, 1988351, 1988592, 1988788],
    ...    offsets=[1988322, 1988546, 1988736, 1989013]
    ... )
    >>> event
    shape: (4, 4)
    ┌──────────┬─────────┬─────────┬──────────┐
    │ name     ┆ onset   ┆ offset  ┆ duration │
    │ ---      ┆ ---     ┆ ---     ┆ ---      │
    │ str      ┆ i64     ┆ i64     ┆ i64      │
    ╞══════════╪═════════╪═════════╪══════════╡
    │ fixation ┆ 1988147 ┆ 1988322 ┆ 175      │
    │ fixation ┆ 1988351 ┆ 1988546 ┆ 195      │
    │ fixation ┆ 1988592 ┆ 1988736 ┆ 144      │
    │ fixation ┆ 1988788 ┆ 1989013 ┆ 225      │
    └──────────┴─────────┴─────────┴──────────┘
    """

    frame: pl.DataFrame

    trial_columns: list[str] | None

    _minimal_schema = {'name': pl.Utf8, 'onset': pl.Float64, 'offset': pl.Float64}

    def __init__(
            self,
            data: pl.DataFrame | None = None,
            name: str | list[str] | None = None,
            onsets: list[int | float] | np.ndarray | None = None,
            offsets: list[int | float] | np.ndarray | None = None,
            trials: list[int | float | str | None] | np.ndarray | None = None,
            trial_columns: list[str] | str | None = None,
    ):
        self.trial_columns: list[str] | None  # otherwise mypy gets confused.

        if data is not None:
            _checks.check_is_mutual_exclusive(data=data, onsets=onsets)
            _checks.check_is_mutual_exclusive(data=data, offsets=offsets)
            _checks.check_is_mutual_exclusive(data=data, name=name)
            _checks.check_is_mutual_exclusive(data=data, name=trials)

            data = data.clone()
            data = self._add_minimal_schema_columns(data)
            data_dict = data.to_dict()

            if isinstance(trial_columns, str):
                self.trial_columns = [trial_columns]
            else:
                self.trial_columns = trial_columns

            self._additional_columns = [
                column_name for column_name in data_dict.keys()
                if column_name not in self._minimal_schema
            ]

        else:
            # Make sure that if either onsets or offsets is None, the other one is None too.
            _checks.check_is_none_is_mutual(onsets=onsets, offsets=offsets)

            # Make sure lengths of onsets and offsets are equal.
            if onsets is not None:

                # mypy does not get that offsets cannot be None (l. 87)
                assert offsets is not None

                _checks.check_is_length_matching(onsets=onsets, offsets=offsets)
                # In case name is given as a list, check that too.
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
                    'name': pl.Series(name, dtype=pl.Utf8),
                    'onset': pl.Series(onsets, dtype=pl.Float64),
                    'offset': pl.Series(offsets, dtype=pl.Float64),
                }

                if trials is not None:
                    data_dict['trial'] = pl.Series('trial', trials)
                    self.trial_columns = ['trial']
                else:
                    self.trial_columns = None

            else:
                data_dict = {
                    'name': pl.Series([], dtype=pl.Utf8),
                    'onset': pl.Series([], dtype=pl.Float64),
                    'offset': pl.Series([], dtype=pl.Float64),
                }
                self.trial_columns = None

        self.frame = pl.DataFrame(data=data_dict, schema_overrides=self._minimal_schema)

        # Ensure column order: trial columns, name, onset, offset.
        if self.trial_columns is not None:
            self.frame = self.frame.select([*self.trial_columns, *self._minimal_schema.keys()])

        # Convert to int if possible.
        all_decimals = self.frame.select(
            pl.all_horizontal(
                pl.col('onset', 'offset').round()
                .eq(pl.col('onset', 'offset'))
                .all(),
            ),
        ).item()
        if all_decimals:
            self.frame = self.frame.with_columns(
                pl.col('onset', 'offset').cast(pl.Int64),
            )

        if 'duration' not in self.frame.columns:
            self._add_duration_property()

    @property
    def schema(self) -> pl.type_aliases.SchemaDict:
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

    def _add_duration_property(self) -> None:
        """Add duration property column to dataframe."""
        self.frame = self.frame.select([pl.all(), duration().alias('duration')])

    def add_event_properties(
            self,
            event_properties: pl.DataFrame,
            join_on: str | list[str],
    ) -> None:
        """Add new event properties into dataframe.

        Parameters
        ----------
        event_properties: pl.DataFrame
            Dataframe with new event properties.
        join_on: str | list[str]
            Columns to join event properties on.
        """
        self.frame = self.frame.join(event_properties, on=join_on, how='left')

    def drop(
            self,
            columns: str | list[str],
    ) -> None:
        """Remove columns from the events data frame.

        Notes
        -----
        The minimal schema columns ``name``, ``onset`` and ``offset`` cannot be removed.

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
                    f"Available columns to remove: {available_columns}.",
                )
            if column in minimal_schema:
                raise ValueError(
                    f"The column '{column}' cannot be removed "
                    'because it belongs to the minimal schema (onset, offset, name). '
                    f"Available columns to remove: {available_columns}.",
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
                pl.lit(column_data).alias(column_name) if not isinstance(column_data, int)
                # Enforce Int64 columns for integers.
                else pl.lit(column_data).alias(column_name).cast(pl.Int64)
                for column_name, column_data in trial_columns.items()
            ] + [pl.all()],
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

    def _filter_by_prefix(self, prefix: str) -> pl.DataFrame:
        """Filter events by name prefix.

        Parameters
        ----------
        prefix: str
            Event name prefix used to select rows (e.g., ``"fixation"``, ``"saccade"``).

        Returns
        -------
        pl.DataFrame
            DataFrame containing events whose ``name`` column starts with ``prefix``.
        """
        return self.frame.filter(pl.col('name').str.starts_with(prefix))

    @property
    def fixations(self) -> pl.DataFrame:
        """Fixation events.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all fixation events, i.e., rows where
            ``name`` starts with ``"fixation"`` (e.g., ``"fixation"``, ``"fixation_ivt"``,
            ``"fixation_eyelink"``).
        """
        return self._filter_by_prefix('fixation')

    @property
    def saccades(self) -> pl.DataFrame:
        """Saccade events.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all saccade events, i.e., rows where
            ``name`` starts with ``"saccade"`` (e.g., ``"saccade"``, ``"saccade_algo"``).
        """
        return self._filter_by_prefix('saccade')

    @property
    def blinks(self) -> pl.DataFrame:
        """Blink events.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all blink events, i.e., rows where
            ``name`` starts with ``"blink"`` (e.g., ``"blink"``, ``"blink_detectorX"``).
        """
        return self._filter_by_prefix('blink')

    @property
    def microsaccades(self) -> pl.DataFrame:
        """Microsaccade events.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all microsaccade events, i.e., rows where
            ``name`` starts with ``"microsaccade"`` (e.g., ``"microsaccade"``).
        """
        return self._filter_by_prefix('microsaccade')

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
        )

    @deprecated(
        reason='Please use Events.clone() instead. '
               'This function will be removed in v0.28.0.',
        version='v0.23.0',
    )
    def copy(self) -> Events:
        """Return a copy of an Events object.

        .. deprecated:: v0.23.0
           Please use :py:meth:`~pymovements.events.Events.clone()` instead.
           This function will be removed in v0.28.0.

        Returns
        -------
        Events
            A copy of the Events.
        """
        return self.clone()

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
                key: Events(frame, trial_columns=self.trial_columns)
                for key, frame in event_dfs.items()
            }

        return [
            Events(frame, trial_columns=self.trial_columns)
            for frame in event_dfs
        ]

    def _add_minimal_schema_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add minimal schema columns to :py:class:`polars.DataFrame` if they are missing.

        Parameters
        ----------
        df: pl.DataFrame
            A dataframe to be transformed to a polars dataframe.

        Returns
        -------
        pl.DataFrame
            A dataframe with minimal schema columns added.
        """
        if len(df) == 0:
            return pl.DataFrame(schema={**self._minimal_schema, **df.schema})

        df = df.select(
            [
                pl.lit(None).cast(column_type).alias(column_name)
                for column_name, column_type in self._minimal_schema.items()
                if column_name not in df.columns
            ] + [pl.all()],
        )
        return df

    def unnest(self) -> None:
        """Explode a column of type ``pl.List`` into one column for each list component."""
        cols = ['location']
        input_columns = [col for col in cols if col in self.frame.columns]

        output_suffixes = ['_x', '_y']

        col_names = [
            [f'{input_col}{suffix}' for suffix in output_suffixes]
            for input_col in input_columns
        ]

        for input_col, column_names in zip(input_columns, col_names):
            self.frame = self.frame.with_columns(
                [
                    pl.col(input_col).list.get(component_id).alias(names)
                    for component_id, names in enumerate(column_names)
                ],
            ).drop(input_col)

    def map_to_aois(self, aoi_dataframe: TextStimulus) -> None:
        """Map events to aois.

        Parameters
        ----------
        aoi_dataframe: TextStimulus
            Text dataframe to map fixation to.
        """
        self.unnest()
        aois = [
            aoi_dataframe.get_aoi(row=row, x_eye='location_x', y_eye='location_y')
            for row in tqdm(self.frame.iter_rows(named=True))
        ]
        aoi_df = pl.concat(aois)
        self.frame = pl.concat([self.frame, aoi_df], how='horizontal')

    def __eq__(self, other: Events) -> bool:
        """Check equality between this and another :py:cls:`~pymovements.Events` object."""
        frames_equal = self.frame.equals(other.frame, null_equal=True)
        trial_columns_equal = self.trial_columns == other.trial_columns
        return frames_equal and trial_columns_equal

    def __str__(self: Any) -> str:
        """Return string representation of event dataframe."""
        return self.frame.__str__()

    def __repr__(self) -> str:
        """Return string representation of event dataframe."""
        return self.__str__()
