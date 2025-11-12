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
"""Provides EventDataFrame class.

.. deprecated:: v0.23.0
   Please use :py:class:`~pymovements.Events` instead.
   This module will be removed in v0.28.0.
"""
from __future__ import annotations

import polars as pl

from pymovements._utils._deprecated import DeprecatedMetaClass
from pymovements.events.events import Events


class EventDataFrame(metaclass=DeprecatedMetaClass):
    """A data structure for event data.

    .. deprecated:: v0.23.0
       Please use :py:class:`~pymovements.Events` instead.
       This module will be removed in v0.28.0.

    Each row has at least an event name with its onset and offset specified.

    Parameters
    ----------
    data: pl.DataFrame | None
        A dataframe to be transformed to a polars dataframe. This argument is mutually
        exclusive with all the other arguments. (default: None)
    name: str | list[str] | None
        Name of events. (default: None)
    onsets: list[int] | np.ndarray | None
        List of onsets. (default: None)
    offsets: list[int] | np.ndarray | None
        List of offsets. (default: None)
    trials: list[int | float | str] | np.ndarray | None
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

    _DeprecatedMetaClass__alias = Events
    _DeprecatedMetaClass__version_deprecated = 'v0.23.0'
    _DeprecatedMetaClass__version_removed = 'v0.28.0'
