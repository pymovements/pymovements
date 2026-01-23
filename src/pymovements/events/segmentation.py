# Copyright (c) 2026 The pymovements Project Authors
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
"""Segmentation utilities for events."""
from __future__ import annotations


import numpy as np
import polars as pl

from ..events.events import Events


def events2segmentation(
    events: Events | pl.DataFrame,
    num_samples: int,
    onset_column: str | None = None,
    offset_column: str | None = None,
) -> np.ndarray:
    """Convert a list of events to a binary segmentation map.

    This function creates a binary array of length num_samples where each
    time point is marked as ``1`` if it falls within any event interval and ``0`` otherwise.
    Events are defined with inclusive onset and exclusive offset, matching the
    convention used by the :py:func:`segmentation2events` function.

    This implementation uses vectorised operations with :py:func:`numpy.bincount` for optimal
    performance.

    Parameters
    ----------
    events : Events | pl.DataFrame
        Event data. Must have onset and offset columns.
    num_samples : int
        The total number of samples in the segmentation map.
    onset_column : str | None
        The name of the column containing the onset of the event (inclusive).
        If None, uses 'onset' column. Default is None.
    offset_column : str | None
        The name of the column containing the offset of the event (exclusive).
        If None, uses 'offset' column. Default is None.

    Returns
    -------
    np.ndarray
        A binary array with ``dtype=np.int32`` where ``1`` indicates an event and
        ``0`` indicates no event.

    Raises
    ------
    ValueError
        If ``num_samples`` is negative.
        If ``onset_column`` or ``offset_column`` is missing from the events.
        If any onset or offset is negative.
        If any onset is greater than or equal to its offset.
        If any offset exceeds ``num_samples``.
        If events overlap, i.e. multiple onsets without corresponding offsets.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.events import events2segmentation
    >>> events_df = pl.DataFrame({'onset': [2, 7], 'offset': [5, 9]})
    >>> events2segmentation(events_df, num_samples=10)
    array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=int32)

    >>> from pymovements.events.events import Events
    >>> events2segmentation(Events(events_df), 10)
    array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=int32)
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, but is {num_samples}")

    onset_column = onset_column or 'onset'
    offset_column = offset_column or 'offset'

    if isinstance(events, Events):
        events_df = events.frame
    else:
        events_df = events

    if onset_column not in events_df.columns:
        raise ValueError(f"Onset column '{onset_column}' not found in events.")
    if offset_column not in events_df.columns:
        raise ValueError(f"Offset column '{offset_column}' not found in events.")

    # Extract event data as numpy arrays
    onsets = events_df[onset_column].to_numpy()
    offsets = events_df[offset_column].to_numpy()

    # Validation of constraints
    if np.any(onsets < 0):
        raise ValueError('Onset must be non-negative, but found negative values')
    if np.any(offsets < 0):
        raise ValueError('Offset must be non-negative, but found negative values')
    if np.any(onsets >= offsets):
        raise ValueError(
            'Onset must be less than offset, but found invalid event(s)',
        )
    if np.any(offsets > num_samples):
        raise ValueError(
            f"Offset exceeds num_samples {num_samples}, but found out-of-bounds values",
        )

    # Empty events case
    if len(onsets) == 0:
        return np.zeros(num_samples, dtype=np.int32)

    # Create indices for all event positions with explicit int32 dtype
    indices_list = [
        np.arange(onset, offset, dtype=np.int32)
        for onset, offset in zip(onsets, offsets)
    ]
    all_indices = np.concatenate(indices_list)

    # Check for overlaps: total index count vs unique count
    if len(indices_list) > 1:
        total_indices = sum(len(indices) for indices in indices_list)
        unique_indices = len(np.unique(all_indices))
        if total_indices > unique_indices:
            raise ValueError('Overlapping events detected')

    # Binary segmentation using bincount
    segmentation = np.bincount(
        all_indices,
        minlength=num_samples,
        weights=np.ones_like(all_indices, dtype=np.int32),
    )
    # Convert counts to binary (handle overlaps by clipping to 1)
    segmentation = np.clip(segmentation, 0, 1)

    return segmentation.astype(dtype=np.int32)


def segmentation2events(
    segmentation: np.ndarray,
) -> Events:
    """Convert a binary segmentation map to a list of events.

    This function identifies continuous regions of ``1``'s in the segmentation array
    and converts them to event onset and offset pairs. The onset is inclusive
    and the offset is exclusive, matching the convention used in :py:func:`events2segmentation`.

    Parameters
    ----------
    segmentation : np.ndarray
        A 1D binary array where ``1`` indicates an event and ``0`` indicates no event.
        Must contain only values ``0`` and ``1`` and have dtype compatible with integers.

    Returns
    -------
    Events
        An Events object containing the onset and offset of each event.
        The onset is inclusive and the offset is exclusive.

    Raises
    ------
    ValueError
        If segmentation is not a 1D array.
        If segmentation contains values other than ``0`` and ``1``.
    TypeError
        If segmentation is not a numpy.ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> from pymovements.events import segmentation2events
    >>> segmentation = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    >>> events = segmentation2events(segmentation)
    >>> events.frame
    shape: (2, 4)
    ┌──────┬───────┬────────┬──────────┐
    │ name ┆ onset ┆ offset ┆ duration │
    │ ---  ┆ ---   ┆ ---    ┆ ---      │
    │ str  ┆ i64   ┆ i64    ┆ i64      │
    ╞══════╪═══════╪════════╪══════════╡
    │ null ┆ 2     ┆ 5      ┆ 3        │
    │ null ┆ 7     ┆ 9      ┆ 2        │
    └──────┴───────┴────────┴──────────┘
    """
    if not isinstance(segmentation, np.ndarray):
        raise TypeError(
            f"segmentation must be a numpy.ndarray, but is {type(segmentation)}",
        )

    if segmentation.ndim != 1:
        raise ValueError(
            f"segmentation must be a 1D array, but has {segmentation.ndim} dimensions",
        )

    if not np.all(np.isin(segmentation, [0, 1])):
        raise ValueError('segmentation must only contain binary values (0 and 1)')

    diff = np.diff(segmentation, prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    df = pl.DataFrame(
        {
            'onset': onsets,
            'offset': offsets,
        },
    )
    return Events(df)
