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

import warnings

import numpy as np
import polars as pl


def events2segmentation(
    events: pl.DataFrame,
    name: str,
    time_column: str = 'time',
    trial_columns: list[str] | None = None,
    onset_column: str = 'onset',
    offset_column: str = 'offset',
) -> pl.Expr:
    """Convert a list of events to a binary segmentation expression.

    This function creates a boolean expression that evaluates to ``True`` if a sample
    falls within any event interval and ``False`` otherwise.
    Events are defined with inclusive onset and inclusive offset, matching the
    convention used by the :py:func:`segmentation2events` function.

    Parameters
    ----------
    events : pl.DataFrame
        Event data. Must have onset and offset columns.
    name : str
        The name of the event type to use for segmentation (e.g. 'blink').
    time_column : str
        The name of the column containing the timestamps. Default is 'time'.
    trial_columns : list[str] | None
        The names of the columns containing trial identifiers. If provided,
        events will only be mapped to samples with matching trial identifiers.
        Default is None.
    onset_column : str
        The name of the column containing the onset of the event (inclusive).
        The values must correspond to the values in ``time_column``.
        Default is 'onset'.
    offset_column : str
        The name of the column containing the offset of the event (inclusive).
        The values must correspond to the values in ``time_column``.
        Default is 'offset'.

    Returns
    -------
    pl.Expr
        A boolean expression aliased to ``name``.

    Raises
    ------
    ValueError
        If ``onset_column`` or ``offset_column`` is missing from the events.
        If any onset is greater than its offset.

    Notes
    -----
    Events are defined with inclusive onset and inclusive offset.
    For example, an event with onset 2 and offset 4 includes samples where the
    ``time_column`` has values 2, 3, and 4.

    The onset and offset values in the ``events`` DataFrame are compared directly
    against the values in the ``time_column`` of the samples DataFrame. If the
    ``time_column`` contains indices, then onsets and offsets are indices. If the
    ``time_column`` contains timestamps, then onsets and offsets are timestamps.

    .. warning::
        The offset is considered inclusive.
        This means that the sample with the offset value in the ``time_column``
        is part of the event.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.events import events2segmentation
    >>>
    >>> events_df = pl.DataFrame(
    ...     {'name': ['blink', 'blink', 'not_blink'], 'onset': [2, 7, 3], 'offset': [5, 9, 6]}
    ... )
    >>> gaze_df = pl.DataFrame({'time': range(10)})
    >>>
    >>> # Create a boolean indicator column for blinks using a Polars expression
    >>> gaze_df.with_columns(
    ...     events2segmentation(events_df, name='blink')
    ... )
    shape: (10, 2)
    ┌──────┬───────┐
    │ time ┆ blink │
    │ ---  ┆ ---   │
    │ i64  ┆ bool  │
    ╞══════╪═══════╡
    │ 0    ┆ false │
    │ 1    ┆ false │
    │ 2    ┆ true  │
    │ 3    ┆ true  │
    │ 4    ┆ true  │
    │ 5    ┆ true │
    │ 6    ┆ false │
    │ 7    ┆ true  │
    │ 8    ┆ true  │
    │ 9    ┆ true │
    └──────┴───────┘
    >>> # With trial columns
    >>> events_df = pl.DataFrame({
    ...     'name': ['blink', 'blink'],
    ...     'onset': [2, 1],
    ...     'offset': [3, 3],
    ...     'trial': [1, 2],
    ... })
    >>> gaze_df = pl.DataFrame({
    ...     'time': pl.Series([0, 1, 2, 0, 1, 2, 3], dtype=pl.Int64),
    ...     'trial': [1, 1, 1, 2, 2, 2, 2],
    ... })
    >>> gaze_df.with_columns(
    ...     events2segmentation(events_df, name='blink', trial_columns=['trial'])
    ... )
    shape: (7, 3)
    ┌──────┬───────┬───────┐
    │ time ┆ trial ┆ blink │
    │ ---  ┆ ---   ┆ ---   │
    │ i64  ┆ i64   ┆ bool  │
    ╞══════╪═══════╪═══════╡
    │ 0    ┆ 1     ┆ false │
    │ 1    ┆ 1     ┆ false │
    │ 2    ┆ 1     ┆ true  │
    │ 0    ┆ 2     ┆ false │
    │ 1    ┆ 2     ┆ true  │
    │ 2    ┆ 2     ┆ true  │
    │ 3    ┆ 2     ┆ true │
    └──────┴───────┴───────┘
    """
    if onset_column not in events.columns:
        raise ValueError(f"Onset column '{onset_column}' not found in events.")
    if offset_column not in events.columns:
        raise ValueError(f"Offset column '{offset_column}' not found in events.")

    # Filter events by name
    if 'name' in events.columns:
        relevant_events = events.filter(pl.col('name') == name)
    else:
        # If no name column, assume all events are relevant
        relevant_events = events

    if relevant_events.is_empty():
        return pl.repeat(False, pl.len()).alias(name)

    onsets = relevant_events[onset_column].to_numpy()
    offsets = relevant_events[offset_column].to_numpy()

    if np.any(onsets > offsets):
        raise ValueError(
            'Onset must be less than or equal to offset, but found invalid event(s)',
        )

    # Check for overlaps
    if len(onsets) > 1:
        # Check for overlaps within each trial
        if trial_columns:
            for trial_group in relevant_events.group_by(trial_columns):
                trial_onsets = trial_group[1][onset_column].to_numpy()
                trial_offsets = trial_group[1][offset_column].to_numpy()
                if _has_overlap(trial_onsets, trial_offsets):
                    warnings.warn(
                        f"Overlapping events detected for trial {trial_group[0]}",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

        # Check for overlaps if no trialised check has been performed
        elif _has_overlap(onsets, offsets):
            # If trial column is present, mention that it might be needed
            if 'trial' in events.columns:
                warnings.warn(
                    'Overlapping events detected. '
                    'Consider providing trial_columns if events are trialized.',
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn('Overlapping events detected', UserWarning, stacklevel=2)

    is_event = pl.repeat(False, pl.len())
    for event in relevant_events.to_dicts():
        is_in_time_range = (
            pl.col(time_column).ge(event[onset_column])
            & pl.col(time_column).le(event[offset_column])
        )

        # Select events matching time and trial criteria
        if trial_columns:
            is_same_trial = pl.lit(True)
            for trial_column in trial_columns:
                is_same_trial = is_same_trial & (pl.col(trial_column) == event[trial_column])
            is_event = is_event | (is_in_time_range & is_same_trial)
        else:
            is_event = is_event | is_in_time_range

    return is_event.alias(name)


def segmentation2events(
    segmentation: np.ndarray,
    name: str,
    time_column: np.ndarray | pl.Series | None = None,
) -> pl.DataFrame:
    """Convert a binary segmentation map to a list of events.

    This function identifies continuous regions of ``1``'s in the segmentation array
    and converts them to event onset and offset pairs. The onset is inclusive,
    and the offset is exclusive, matching the convention used in :py:func:`events2segmentation`.

    Parameters
    ----------
    segmentation : np.ndarray
        A 1D binary array where ``1`` or ``True`` indicates an event and ``0`` or
        ``False`` indicates no event.
        Must contain only values ``0``, ``1``, ``True`` or ``False`` and have dtype
        compatible with integers or booleans.
    name : str
        The name of the event type to use for the 'name' column in the output DataFrame.
    time_column : np.ndarray | pl.Series | None
        The values to use for the onset and offset columns. If provided, the indices
        of the events will be mapped to the values in this column. If None, the
        indices themselves will be used. Default is None.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the onset and offset of each event.
        The onset is inclusive and the offset is exclusive.
        Onsets and offsets correspond to the values in ``time_column`` if provided,
        otherwise they are indices of the input ``segmentation`` array.

    Raises
    ------
    ValueError
        If segmentation is not a 1D array.
        If segmentation contains values other than ``0`` and ``1``.
        If ``time_column`` length does not match ``segmentation`` length.
    TypeError
        If segmentation is not a numpy.ndarray.

    Notes
    -----
    The onset is inclusive and the offset is exclusive.
    For example, a sequence of ones from index 2 to 4 results in an onset of 2 and an
    offset of 5.

    The returned onset and offset values represent the values in the ``time_column``
    (if provided) or the indices of the ``segmentation`` array where an event starts
    and ends.

    .. warning::
        The offset is considered exclusive.
        This means that the sample at the offset value is NOT part of the event.

    Examples
    --------
    >>> import numpy as np
    >>> from pymovements.events import segmentation2events
    >>> segmentation = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    >>> segmentation2events(segmentation, name='blink')
    shape: (2, 3)
    ┌───────┬───────┬────────┐
    │ name  ┆ onset ┆ offset │
    │ ---   ┆ ---   ┆ ---    │
    │ str   ┆ i64   ┆ i64    │
    ╞═══════╪═══════╪════════╡
    │ blink ┆ 2     ┆ 5      │
    │ blink ┆ 7     ┆ 9      │
    └───────┴───────┴────────┘
    >>> # Using a time column:
    >>> time = np.linspace(0.1, 1.0, 10)
    >>> segmentation2events(segmentation, name='blink', time_column=time)
    shape: (2, 3)
    ┌───────┬───────┬────────┐
    │ name  ┆ onset ┆ offset │
    │ ---   ┆ ---   ┆ ---    │
    │ str   ┆ f64   ┆ f64    │
    ╞═══════╪═══════╪════════╡
    │ blink ┆ 0.3   ┆ 0.6    │
    │ blink ┆ 0.8   ┆ 1.0    │
    └───────┴───────┴────────┘
    """
    if not isinstance(segmentation, np.ndarray):
        raise TypeError(
            f"segmentation must be a numpy.ndarray, but is {type(segmentation)}",
        )

    if segmentation.ndim != 1:
        raise ValueError(
            f"segmentation must be a 1D array, but has {segmentation.ndim} dimensions",
        )

    if not np.all(np.isin(segmentation, [0, 1, True, False])):
        raise ValueError('segmentation must only contain binary values (0, 1, True, or False)')

    if time_column is not None and len(time_column) != len(segmentation):
        raise ValueError(
            f"time_column length ({len(time_column)}) must match "
            f"segmentation length ({len(segmentation)})",
        )

    # Ensure segmentation is integer for np.diff
    segmentation_int = segmentation.astype(np.int64)
    diff = np.diff(segmentation_int, prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    # Strategy: if offset == len(segmentation), extrapolate based on the last gap.
    # Reasoning: if an event ends at index i, the offset is i+1.
    # If i is the last index, i+1 is out of bounds.

    if time_column is None:
        return pl.DataFrame(
            {
                'name': pl.Series([name] * len(onsets), dtype=pl.String),
                'onset': pl.Series(onsets, dtype=pl.Int64),
                'offset': pl.Series(offsets, dtype=pl.Int64),
            },
        )

    if isinstance(time_column, pl.Series):
        time_column = time_column.to_numpy()

    onsets_mapped = time_column[onsets]

    offsets_mapped = np.zeros_like(offsets, dtype=time_column.dtype)
    in_bounds = offsets < len(time_column)
    offsets_mapped[in_bounds] = time_column[offsets[in_bounds]]

    out_of_bounds = ~in_bounds
    if np.any(out_of_bounds):
        if len(time_column) >= 2:
            gap = time_column[-1] - time_column[-2]
            offsets_mapped[out_of_bounds] = time_column[-1] + gap
        else:
            # Only one sample, and it's an event.
            # We don't know the gap. Let's just add 1 if it's numeric.
            offsets_mapped[out_of_bounds] = time_column[-1] + 1

    return pl.DataFrame(
        {
            'name': pl.Series([name] * len(onsets), dtype=pl.String),
            'onset': pl.Series(onsets_mapped),
            'offset': pl.Series(offsets_mapped),
        },
    )


def _has_overlap(onsets: np.ndarray, offsets: np.ndarray) -> bool:
    """Check if there are any overlaps between events."""
    if len(onsets) <= 1:
        return False

    # Sort by onset
    sorted_indices = np.argsort(onsets)
    sorted_onsets = onsets[sorted_indices]
    sorted_offsets = offsets[sorted_indices]

    return bool(np.any(sorted_onsets[1:] <= sorted_offsets[:-1]))
