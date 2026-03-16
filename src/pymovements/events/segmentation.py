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

import numbers
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
    padding: float | tuple[float, float] | None = None,
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
    padding : float | tuple[float, float] | None
        Padding to extend each event interval, in the same units as ``time_column``.
        If a single float, the same padding is applied symmetrically before and after
        each event. If a tuple ``(before, after)``, ``before`` is subtracted from the
        onset and ``after`` is added to the offset. Both values must be non-negative.
        Default is None (no padding).

    Returns
    -------
    pl.Expr
        A boolean expression aliased to ``name``.

    Raises
    ------
    TypeError
        If ``padding`` is not None, a tuple, or a number.
    ValueError
        If ``onset_column`` or ``offset_column`` is missing from the events.
        If any onset is greater than its offset.
        If any padding value is negative.

    Notes
    -----
    Events are defined with inclusive onset and inclusive offset.
    For example, an event with onset 2 and offset 4 includes samples where the
    ``time_column`` has values 2, 3, and 4.

    The onset and offset values in the ``events`` DataFrame are compared directly
    against the values in the ``time_column`` of the samples DataFrame. If the
    ``time_column`` contains indices, then onsets and offsets are indices. If the
    ``time_column`` contains timestamps, then onsets and offsets are timestamps.

    When ``padding`` is specified, each event interval is extended by subtracting
    ``pad_before`` from the onset and adding ``pad_after`` to the offset. The padding
    values are in the same units as the ``time_column``.

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
    │ 5    ┆ true  │
    │ 6    ┆ false │
    │ 7    ┆ true  │
    │ 8    ┆ true  │
    │ 9    ┆ true  │
    └──────┴───────┘
    >>> # With padding to extend event intervals
    >>> single_event = pl.DataFrame(
    ...     {'name': ['blink'], 'onset': [3], 'offset': [5]}
    ... )
    >>> gaze_df.with_columns(
    ...     events2segmentation(single_event, name='blink', padding=1)
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
    │ 5    ┆ true  │
    │ 6    ┆ true  │
    │ 7    ┆ false │
    │ 8    ┆ false │
    │ 9    ┆ false │
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
    │ 3    ┆ 2     ┆ true  │
    └──────┴───────┴───────┘
    """
    if onset_column not in events.columns:
        raise ValueError(f"Onset column '{onset_column}' not found in events.")
    if offset_column not in events.columns:
        raise ValueError(f"Offset column '{offset_column}' not found in events.")

    # Parse and validate padding
    if padding is None:
        pad_before, pad_after = 0.0, 0.0
    elif isinstance(padding, tuple):
        pad_before, pad_after = padding
    elif isinstance(padding, numbers.Number):
        pad_before = pad_after = padding
    else:
        raise TypeError(
            'padding should be a number or a two-dimensional tuple'
            f' of numbers, but is {type(padding)}',
        )

    if pad_before < 0 or pad_after < 0:
        raise ValueError(
            f'Padding values must be non-negative, but got ({pad_before}, {pad_after}).',
        )

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

    # Apply padding for overlap checking
    padded_onsets = onsets - pad_before
    padded_offsets = offsets + pad_after

    # Check for overlaps using padded intervals
    if len(onsets) > 1:
        # Check for overlaps within each trial
        if trial_columns:
            for trial_group in relevant_events.group_by(trial_columns):
                trial_onsets = trial_group[1][onset_column].to_numpy() - pad_before
                trial_offsets = trial_group[1][offset_column].to_numpy() + pad_after
                if _has_overlap(trial_onsets, trial_offsets):
                    warnings.warn(
                        f'Overlapping events detected for trial {trial_group[0]}',
                        UserWarning,
                        stacklevel=2,
                    )
                    break

        # Check for overlaps if no trialised check has been performed
        elif _has_overlap(padded_onsets, padded_offsets):
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
        is_in_time_range = pl.col(time_column).ge(event[onset_column] - pad_before) & pl.col(
            time_column
        ).le(event[offset_column] + pad_after)

        # Select events matching time and trial criteria
        if trial_columns:
            is_same_trial = pl.lit(True)
            for trial_column in trial_columns:
                is_same_trial = is_same_trial & (pl.col(trial_column) == event[trial_column])
            is_event = is_event | (is_in_time_range & is_same_trial)
        else:
            is_event = is_event | is_in_time_range

    return is_event.alias(name)


def events2timeratio(
    events: pl.DataFrame,
    samples: pl.DataFrame,
    name: str,
    time_column: str = 'time',
    trial_columns: list[str] | None = None,
    sampling_rate: float | None = None,
    onset_column: str = 'onset',
    offset_column: str = 'offset',
) -> pl.Expr:
    """Create an expression to calculate time-based event ratio.

    This function creates an expression that calculates the ratio of time covered by events
    relative to the total time span.

    Parameters
    ----------
    events : pl.DataFrame
        Event data. Must have onset and offset columns.
    samples : pl.DataFrame
        Sample data containing the time column.
    name : str
        The name of the event type to calculate ratio for (e.g. 'blink').
    time_column : str
        The name of the column containing timestamps in samples.
    trial_columns : list[str] | None
        The names of columns identifying trials. If provided, ratios are computed
        per trial.
    sampling_rate : float | None
        The sampling rate of the gaze data in Hz. If provided, the ratio
        is calculated inclusively by adding the sampling interval to the
        durations and the total time range. If ``None``, the sampling
        interval is estimated as the mode of the time differences.
    onset_column : str
        The name of the column containing event onset times.
    offset_column : str
        The name of the column containing event offset times.

    Returns
    -------
    pl.Expr
        An expression that calculates the event ratio. When aggregated with
        group_by().agg(), computes per-trial ratios.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.events import events2timeratio
    >>> events = pl.DataFrame({
    ...     'name': ['blink', 'blink'],
    ...     'onset': [1.0, 5.0],
    ...     'offset': [3.0, 7.0],
    ... })
    >>> samples = pl.DataFrame({'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    >>> samples.select(events2timeratio(events, samples, 'blink'))
    shape: (1, 1)
    ┌───────────────────┐
    │ event_ratio_blink │
    │ ---               │
    │ f64               │
    ╞═══════════════════╡
    │ 0.75              │
    └───────────────────┘
    >>> # Inclusive ratio using sampling rate
    >>> samples.select(events2timeratio(events, samples, 'blink', sampling_rate=1000.0))
    shape: (1, 1)
    ┌───────────────────┐
    │ event_ratio_blink │
    │ ---               │
    │ f64               │
    ╞═══════════════════╡
    │ 0.75              │
    └───────────────────┘
    """
    if events.is_empty():
        return pl.lit([0.0]).list.sum().alias(f'event_ratio_{name}')

    if onset_column not in events.columns:
        raise ValueError(f'Onset column {onset_column!r} not found in events.')
    if offset_column not in events.columns:
        raise ValueError(f'Offset column {offset_column!r} not found in events.')
    if time_column not in samples.columns:
        raise ValueError(f'Time column {time_column!r} not found in samples.')

    relevant_events = events.filter(pl.col('name') == name)

    if relevant_events.is_empty():
        return pl.lit([0.0]).list.sum().alias(f'event_ratio_{name}')

    if samples.is_empty():
        return pl.lit(None).cast(pl.Float64).alias(f'event_ratio_{name}')

    # Single-sample series: return 1.0 if that sample falls within an event, else 0.0.
    if samples.height == 1:
        sample_time = samples.get_column(time_column).item(0)
        event_filter = (pl.col(onset_column) <= sample_time) & (
            pl.col(offset_column) >= sample_time
        )
        if trial_columns:
            for col in trial_columns:
                sample_val = samples.get_column(col).item(0)
                event_filter = event_filter & (pl.col(col) == sample_val)
        matching = relevant_events.filter(event_filter)
        return pl.lit(1.0 if not matching.is_empty() else 0.0).alias(f'event_ratio_{name}')

    dt_ms = 0.0
    if sampling_rate is not None:
        dt_ms = 1000.0 / sampling_rate
    else:
        # Calculate the mode of time differences as a robust estimate for the sampling interval.
        # At this point, samples is non-empty and has more than one row (single-sample above),
        # so `diff().drop_nulls()` will yield at least one element and `mode()` will be non-empty.
        time_diffs = samples[time_column].diff().drop_nulls()
        dt_ms = float(time_diffs.mode().item())

    # Event ratio considering trial columns
    if trial_columns:
        event_durations = (
            relevant_events.group_by(trial_columns, maintain_order=True)
            .agg((pl.col(offset_column) - pl.col(onset_column) + dt_ms).alias('duration'))
            .with_columns(pl.col('duration').list.sum())
        )

        sample_time_ranges = samples.group_by(trial_columns, maintain_order=True).agg(
            (pl.col(time_column).max() - pl.col(time_column).min() + dt_ms).alias('time_range'),
        )

        trial_ratios = event_durations.join(
            sample_time_ranges,
            on=trial_columns,
            how='full',
        ).with_columns(
            (pl.col('duration') / pl.col('time_range')).alias(f'event_ratio_{name}'),
        )

        ratio_expr: pl.Expr | None = None
        for row in trial_ratios.to_dicts():
            condition: pl.Expr | None = None
            for col in trial_columns:
                trial_val = row.get(col)
                trial_right_val = row.get(f'{col}_right')
                val = trial_val if trial_val is not None else trial_right_val
                if condition is None:
                    condition = pl.col(col) == val
                else:
                    condition = condition & (pl.col(col) == val)

            ratio = row.get(f'event_ratio_{name}')
            if ratio is None:
                ratio = 0.0

            # Build conditional expression for event ratio
            if ratio_expr is None:
                ratio_expr = pl.when(condition).then(pl.lit(ratio))
            else:
                ratio_expr = ratio_expr.when(condition).then(pl.lit(ratio))

        # At this point, trial_columns is guaranteed to be non-empty, so ratio_expr is set
        return ratio_expr.otherwise(pl.lit([0.0]).list.sum()).alias(  # type: ignore[union-attr]
            f'event_ratio_{name}',
        )

    total_duration = (
        relevant_events.select(pl.col(offset_column) - pl.col(onset_column) + dt_ms).sum()
    ).item()

    time_range = samples.select(
        pl.col(time_column).max() - pl.col(time_column).min() + dt_ms,
    ).item()

    return pl.lit(total_duration / time_range).alias(f'event_ratio_{name}')


def segmentation2events(
    segmentation: pl.Series | np.ndarray,
    name: str,
    time_column: pl.Series | np.ndarray | None = None,
    trial_columns: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Convert a binary segmentation map to a list of events.

    This function identifies continuous regions of ``1``'s in the segmentation array
    and converts them to event onset and offset pairs. The onset and offset are
    both inclusive, matching the convention used in :py:func:`events2segmentation`.

    Parameters
    ----------
    segmentation : pl.Series | np.ndarray
        A 1D binary array or Series where ``1`` or ``True`` indicates an event and
        ``0`` or ``False`` indicates no event.
        Must contain only values ``0``, ``1``, ``True`` or ``False``.
    name : str
        The name of the event type to use for the 'name' column in the output DataFrame.
    time_column : pl.Series | np.ndarray | None
        The values to use for the onset and offset columns. If provided, the values
        of the events will be mapped to the values in this column. If None, the
        indices themselves will be used. Default is None.
    trial_columns : pl.DataFrame | None
        A DataFrame containing trial identifiers for each sample. If provided,
        events will be identified within each trial separately.
        Default is None.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the onset and offset of each event.
        The onset and offset are both inclusive.
        Onsets and offsets correspond to the values in ``time_column`` if provided,
        otherwise they are indices of the input ``segmentation`` array.

    Raises
    ------
    ValueError
        If segmentation is not a 1D array.
        If segmentation contains values other than ``0`` and ``1``.
        If ``time_column`` length does not match ``segmentation`` length.
        If ``trial_columns`` length does not match ``segmentation`` length.
    TypeError
        If segmentation is not a polars.Series or numpy.ndarray.

    Notes
    -----
    The onset and offset are both inclusive.
    For example, a sequence of ones from index 2 to 4 results in an onset of 2 and an
    offset of 4.

    The returned onset and offset values represent the values in the ``time_column``
    (if provided) or the indices of the ``segmentation`` array where an event starts
    and ends.

    .. warning::
        The offset is considered inclusive.
        This means that the sample at the offset value is part of the event.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.events import segmentation2events
    >>> segmentation = pl.Series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    >>> segmentation2events(segmentation, name='blink')
    shape: (2, 3)
    ┌───────┬───────┬────────┐
    │ name  ┆ onset ┆ offset │
    │ ---   ┆ ---   ┆ ---    │
    │ str   ┆ i64   ┆ i64    │
    ╞═══════╪═══════╪════════╡
    │ blink ┆ 2     ┆ 4      │
    │ blink ┆ 7     ┆ 8      │
    └───────┴───────┴────────┘
    >>> # Using a time column:
    >>> time = pl.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    >>> segmentation2events(segmentation, name='blink', time_column=time)
    shape: (2, 3)
    ┌───────┬───────┬────────┐
    │ name  ┆ onset ┆ offset │
    │ ---   ┆ ---   ┆ ---    │
    │ str   ┆ f64   ┆ f64    │
    ╞═══════╪═══════╪════════╡
    │ blink ┆ 0.3   ┆ 0.5    │
    │ blink ┆ 0.8   ┆ 0.9    │
    └───────┴───────┴────────┘
    """
    if isinstance(segmentation, np.ndarray):
        if segmentation.ndim != 1:
            raise ValueError(
                f'segmentation must be a 1D array, but has {segmentation.ndim} dimensions',
            )
        segmentation = pl.Series('__segmentation__', segmentation)
    elif isinstance(segmentation, pl.Series):
        segmentation = segmentation.alias('__segmentation__')
    else:
        raise TypeError(
            f'segmentation must be a polars.Series or numpy.ndarray, but is {type(segmentation)}',
        )

    if segmentation.dtype == pl.Boolean:
        pass
    elif not segmentation.is_in([0, 1]).all():
        raise ValueError('segmentation must only contain binary values (0, 1, True, or False)')

    df_dict = {'__segmentation__': segmentation}

    if time_column is not None:
        if len(time_column) != len(segmentation):
            raise ValueError(
                f'time_column length ({len(time_column)}) must match '
                f'segmentation length ({len(segmentation)})',
            )
        if isinstance(time_column, np.ndarray):
            time_column = pl.Series('__time__', time_column)
        elif isinstance(time_column, pl.Series):
            time_column = time_column.alias('__time__')
        else:
            raise TypeError(
                f'time_column must be a polars.Series or numpy.ndarray, but is {type(time_column)},'
                f' alternatively leave it at None to use indices instead of timestamps.',
            )
        df_dict['__time__'] = time_column
    else:
        df_dict['__time__'] = pl.int_range(0, len(segmentation), dtype=pl.Int64, eager=True)

    df = pl.DataFrame(df_dict)

    group_cols = []
    if trial_columns is not None:
        if len(trial_columns) != len(segmentation):
            raise ValueError(
                f'trial_columns length ({len(trial_columns)}) must match '
                f'segmentation length ({len(segmentation)})',
            )
        df = pl.concat([df, trial_columns], how='horizontal')
        group_cols.extend(trial_columns.columns)

    # Use rle_id to identify contiguous segments
    df = df.with_columns(__event_id__=pl.col('__segmentation__').rle_id())
    group_cols.append('__event_id__')

    events_df = (
        df.filter(pl.col('__segmentation__').cast(pl.Boolean))
        .group_by(group_cols, maintain_order=True)
        .agg(
            pl.lit(name).alias('name'),
            pl.col('__time__').min().alias('onset'),
            pl.col('__time__').max().alias('offset'),
        )
    )

    # Final clean-up: reorder columns and drop internal ones
    # Result should have: name, onset, offset, then trial columns
    final_cols = ['name', 'onset', 'offset']
    if trial_columns is not None:
        final_cols.extend(trial_columns.columns)

    if events_df.is_empty():
        schema = {
            'name': pl.String,
            'onset': df.schema['__time__'],
            'offset': df.schema['__time__'],
        }
        if trial_columns is not None:
            schema.update(trial_columns.schema)
        return pl.DataFrame(None, schema=schema)

    return events_df.select(final_cols)


def _has_overlap(onsets: np.ndarray, offsets: np.ndarray) -> bool:
    """Check if there are any overlaps between events."""
    if len(onsets) <= 1:
        return False

    # Sort by onset
    sorted_indices = np.argsort(onsets)
    sorted_onsets = onsets[sorted_indices]
    sorted_offsets = offsets[sorted_indices]

    return bool(np.any(sorted_onsets[1:] <= sorted_offsets[:-1]))
