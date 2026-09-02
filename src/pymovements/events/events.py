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
from typing import Any
from typing import Literal
from typing import overload

import numpy as np
import polars
from tqdm import tqdm

from pymovements._utils import _checks
from pymovements._utils._column_nesting import unnest_list_columns
from pymovements._utils._html import repr_html
from pymovements._utils._nulls import row_is_null
from pymovements.events.correction import fixation_correction
from pymovements.measure.events.measures import duration
from pymovements.stimulus.text import TextStimulus


def _aois_frame_from_text_stimulus(stimulus: TextStimulus) -> pl.DataFrame:
    """Map the configured column names of a TextStimulus to drift correction column names.

    Parameters
    ----------
    stimulus: TextStimulus
        Text stimulus whose AOIs dataframe is extracted.

    Returns
    -------
    pl.DataFrame
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


@repr_html(['frame', 'trial_columns'])
class Events:
    """A data structure for event data.

    Each row has at least an event name with its onset and offset specified.

    Parameters
    ----------
    data: polars.DataFrame | None
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
        List of trial columns in the passed dataframe.

    Attributes
    ----------
    frame: polars.DataFrame
        A dataframe of events.
    trial_columns: list[str] | None
        The name of the trial columns in the data frame. If not None, processing methods
        will be applied to each trial separately.
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

    frame: polars.DataFrame

    trial_columns: list[str] | None

    _minimal_schema = {'name': polars.Utf8, 'onset': polars.Float64, 'offset': polars.Float64}

    def __init__(
            self,
            data: polars.DataFrame | None = None,
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
                    'name': polars.Series(name, dtype=polars.Utf8),
                    'onset': polars.Series(onsets, dtype=polars.Float64),
                    'offset': polars.Series(offsets, dtype=polars.Float64),
                }

                if trials is not None:
                    data_dict['trial'] = polars.Series('trial', trials)
                    self.trial_columns = ['trial']
                else:
                    self.trial_columns = None

            else:
                data_dict = {
                    'name': polars.Series([], dtype=polars.Utf8),
                    'onset': polars.Series([], dtype=polars.Float64),
                    'offset': polars.Series([], dtype=polars.Float64),
                }
                self.trial_columns = None

        self.frame = polars.DataFrame(data=data_dict, schema_overrides=self._minimal_schema)

        # Ensure column order: trial columns, then minimal schema, keeping all other columns.
        if self.trial_columns is not None:
            # Keep any additional columns beyond trial and minimal schema columns
            other_cols = [
                col for col in self.frame.columns
                if col not in self.trial_columns and col not in self._minimal_schema
            ]
            self.frame = self.frame.select(
                [*self.trial_columns, *self._minimal_schema.keys(), *other_cols],
            )

        # Convert to int if possible.
        all_decimals = self.frame.select(
            polars.all_horizontal(
                polars.col('onset', 'offset').round()
                .eq(polars.col('onset', 'offset'))
                .all(),
            ),
        ).item()
        if all_decimals:
            self.frame = self.frame.with_columns(
                polars.col('onset', 'offset').cast(polars.Int64),
            )

        if 'duration' not in self.frame.columns:
            self._add_duration_property()

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

    def _add_duration_property(self) -> None:
        """Add duration property column to dataframe."""
        self.frame = self.frame.select([polars.all(), duration().alias('duration')])

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
                    f'Available columns to remove: {available_columns}.',
                )
            if column in minimal_schema:
                raise ValueError(
                    f"The column '{column}' cannot be removed "
                    'because it belongs to the minimal schema (onset, offset, name). '
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
        ┌──────────────────┬───────┬────────┬──────────┐
        │ name             ┆ onset ┆ offset ┆ duration │
        │ ---              ┆ ---   ┆ ---    ┆ ---      │
        │ str              ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════════╪═══════╪════════╪══════════╡
        │ saccade          ┆ 90    ┆ 100    ┆ 10       │
        │ fixation         ┆ 99    ┆ 176    ┆ 77       │
        │ fixation_idt     ┆ 99    ┆ 175    ┆ 76       │
        │ fixation_ivt     ┆ 100   ┆ 178    ┆ 78       │
        │ fixation_eyelink ┆ 101   ┆ 175    ┆ 74       │
        │ microsaccade     ┆ 115   ┆ 124    ┆ 9        │
        │ microsaccade     ┆ 145   ┆ 157    ┆ 12       │
        │ saccade          ┆ 175   ┆ 199    ┆ 24       │
        └──────────────────┴───────┴────────┴──────────┘

        All fixations:

        >>> events.filter_by_name('fixation')
        shape: (4, 4)
        ┌──────────────────┬───────┬────────┬──────────┐
        │ name             ┆ onset ┆ offset ┆ duration │
        │ ---              ┆ ---   ┆ ---    ┆ ---      │
        │ str              ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════════╪═══════╪════════╪══════════╡
        │ fixation         ┆ 99    ┆ 176    ┆ 77       │
        │ fixation_idt     ┆ 99    ┆ 175    ┆ 76       │
        │ fixation_ivt     ┆ 100   ┆ 178    ┆ 78       │
        │ fixation_eyelink ┆ 101   ┆ 175    ┆ 74       │
        └──────────────────┴───────┴────────┴──────────┘

        Exact match for fixation:

        >>> events.filter_by_name('^fixation$')
        shape: (1, 4)
        ┌──────────┬───────┬────────┬──────────┐
        │ name     ┆ onset ┆ offset ┆ duration │
        │ ---      ┆ ---   ┆ ---    ┆ ---      │
        │ str      ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════╪═══════╪════════╪══════════╡
        │ fixation ┆ 99    ┆ 176    ┆ 77       │
        └──────────┴───────┴────────┴──────────┘

        Prefix match:

        >>> events.filter_by_name('^fixation_')
        shape: (3, 4)
        ┌──────────────────┬───────┬────────┬──────────┐
        │ name             ┆ onset ┆ offset ┆ duration │
        │ ---              ┆ ---   ┆ ---    ┆ ---      │
        │ str              ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════════╪═══════╪════════╪══════════╡
        │ fixation_idt     ┆ 99    ┆ 175    ┆ 76       │
        │ fixation_ivt     ┆ 100   ┆ 178    ┆ 78       │
        │ fixation_eyelink ┆ 101   ┆ 175    ┆ 74       │
        └──────────────────┴───────┴────────┴──────────┘

        Suffix match:

        >>> events.filter_by_name('ivt$')
        shape: (1, 4)
        ┌──────────────┬───────┬────────┬──────────┐
        │ name         ┆ onset ┆ offset ┆ duration │
        │ ---          ┆ ---   ┆ ---    ┆ ---      │
        │ str          ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════╪═══════╪════════╪══════════╡
        │ fixation_ivt ┆ 100   ┆ 178    ┆ 78       │
        └──────────────┴───────┴────────┴──────────┘

        All saccade variants:

        >>> events.filter_by_name('saccade')
        shape: (4, 4)
        ┌──────────────┬───────┬────────┬──────────┐
        │ name         ┆ onset ┆ offset ┆ duration │
        │ ---          ┆ ---   ┆ ---    ┆ ---      │
        │ str          ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════╪═══════╪════════╪══════════╡
        │ saccade      ┆ 90    ┆ 100    ┆ 10       │
        │ microsaccade ┆ 115   ┆ 124    ┆ 9        │
        │ microsaccade ┆ 145   ┆ 157    ┆ 12       │
        │ saccade      ┆ 175   ┆ 199    ┆ 24       │
        └──────────────┴───────┴────────┴──────────┘

        Only microsaccades:

        >>> events.filter_by_name('microsaccade')
        shape: (2, 4)
        ┌──────────────┬───────┬────────┬──────────┐
        │ name         ┆ onset ┆ offset ┆ duration │
        │ ---          ┆ ---   ┆ ---    ┆ ---      │
        │ str          ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════════╪═══════╪════════╪══════════╡
        │ microsaccade ┆ 115   ┆ 124    ┆ 9        │
        │ microsaccade ┆ 145   ┆ 157    ┆ 12       │
        └──────────────┴───────┴────────┴──────────┘

        Exact match for saccade:

        >>> events.filter_by_name('^saccade$')
        shape: (2, 4)
        ┌─────────┬───────┬────────┬──────────┐
        │ name    ┆ onset ┆ offset ┆ duration │
        │ ---     ┆ ---   ┆ ---    ┆ ---      │
        │ str     ┆ i64   ┆ i64    ┆ i64      │
        ╞═════════╪═══════╪════════╪══════════╡
        │ saccade ┆ 90    ┆ 100    ┆ 10       │
        │ saccade ┆ 175   ┆ 199    ┆ 24       │
        └─────────┴───────┴────────┴──────────┘

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
                key: Events(frame, trial_columns=self.trial_columns)
                for key, frame in event_dfs.items()
            }

        return [
            Events(frame, trial_columns=self.trial_columns)
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
        ┌──────────┬───────┬────────┬───────┬──────┬──────────┐
        │ name     ┆ onset ┆ offset ┆ trial ┆ page ┆ duration │
        │ ---      ┆ ---   ┆ ---    ┆ ---   ┆ ---  ┆ ---      │
        │ str      ┆ i64   ┆ i64    ┆ i64   ┆ i64  ┆ i64      │
        ╞══════════╪═══════╪════════╪═══════╪══════╪══════════╡
        │ fixation ┆ 0     ┆ 100    ┆ 1     ┆ 1    ┆ 100      │
        │ fixation ┆ 110   ┆ 150    ┆ null  ┆ 2    ┆ 40       │
        └──────────┴───────┴────────┴───────┴──────┴──────────┘

        Under the default ``how='any'``, a single null value suffices, removing the second
        fixation too:

        >>> events.drop_nulls(subset=['trial', 'page'])
        >>> events
        shape: (1, 6)
        ┌──────────┬───────┬────────┬───────┬──────┬──────────┐
        │ name     ┆ onset ┆ offset ┆ trial ┆ page ┆ duration │
        │ ---      ┆ ---   ┆ ---    ┆ ---   ┆ ---  ┆ ---      │
        │ str      ┆ i64   ┆ i64    ┆ i64   ┆ i64  ┆ i64      │
        ╞══════════╪═══════╪════════╪═══════╪══════╪══════════╡
        │ fixation ┆ 0     ┆ 100    ┆ 1     ┆ 1    ┆ 100      │
        └──────────┴───────┴────────┴───────┴──────┴──────────┘
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

    def correct_fixations(
            self,
            aois: TextStimulus,
            algorithm: str | list[str] = 'wisdom_of_the_crowd',
            *,
            text_right_to_left: bool | None = None,
            word_locations: pl.Series | None = None,
            algorithm_kwargs: dict[str, Any] | None = None,
            fixation_name: str = 'fixation',
            inplace: bool = True,
    ) -> Events | None:
        """Correct vertical drift of fixation locations.

        Fixations are corrected per trial according to :py:attr:`~pymovements.Events.
        trial_columns` using the specified drift correction algorithm. Fixation locations
        are replaced with their corrected values; original locations are preserved in a
        ``location_original`` column and the applied algorithm is recorded in a
        ``correction_algorithm`` column. See
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
        text_right_to_left: bool | None
            Whether the text is read from right to left. If None, the reading direction is
            inferred from the writing system of the text stimulus. (default: None)
        word_locations: pl.Series | None
            Series of [x, y] word center coordinates for the DTW-based algorithms
            'compare' and 'warp'. If None, word locations are derived from the aois
            dataframe. (default: None)
        algorithm_kwargs: dict[str, Any] | None
            Additional tuning parameters passed to underlying drift correction algorithms.
            (default: None)
        fixation_name: str
            Name of the fixation events to correct. (default: 'fixation')
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
        """
        if not isinstance(aois, TextStimulus):
            raise TypeError(
                f'aois must be a TextStimulus, but is of type {type(aois).__name__}.',
            )
        aois_frame = _aois_frame_from_text_stimulus(aois)
        if text_right_to_left is None:
            text_right_to_left = aois.writing_system.directionality == 'right-to-left'

        corrected_frame = fixation_correction.correct_fixations(
            self.frame,
            aois_frame,
            algorithm=algorithm,
            trial_columns=self.trial_columns,
            text_right_to_left=text_right_to_left,
            word_locations=word_locations,
            algorithm_kwargs=algorithm_kwargs,
            fixation_name=fixation_name,
        )
        if inplace:
            self.frame = corrected_frame
            return None
        return Events(corrected_frame, trial_columns=self.trial_columns)

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
            The maximum gap (in ms) between subsequent fixation events to be merged. (default: 75)

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
        ┌──────────┬───────┬────────┬──────────┐
        │ name     ┆ onset ┆ offset ┆ duration │
        │ ---      ┆ ---   ┆ ---    ┆ ---      │
        │ str      ┆ i64   ┆ i64    ┆ i64      │
        ╞══════════╪═══════╪════════╪══════════╡
        │ fixation ┆ 0     ┆ 90     ┆ 90       │
        └──────────┴───────┴────────┴──────────┘

        This combined all the smaller events into a single event with longer duration.

        """
        # Step 1: Filter events of the specified type and sort by onset
        events = self.frame.filter(polars.col('name') == name).sort('onset')
        # set aside other events to merge them back later
        other = self.frame.filter(polars.col('name') != name)

        number_of_events = len(events)

        # Step 2: Calculate the gap between the current onset and the previous offset
        events = events.with_columns(gap=polars.col('onset') - polars.col('offset').shift(1))

        # Step 3: Create a 'group' identifier for merging
        events = events.with_columns(
            # calculate when gap is null or > max_gap
            (polars.col('gap').is_null() | (polars.col('gap') > max_gap))
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
            polars.col('offset').last().alias('offset'),
        ]).drop(['group', 'gap'])  # we don't need the group and gap columns anymore
        # the duration of the merged event is the last offset minus the first onset in the group
        events = events.with_columns(duration().alias('duration'))

        # Step 5: concatenate new events
        events = events.select(self.frame.columns)  # reorder columns to match original frame
        self.frame = polars.concat([events, other]).sort('onset')

        if verbose:
            print(
                f"Merged {number_of_events} '{name}' events "
                f'into {len(events)} events with max_gap={max_gap} ms.',
            )
