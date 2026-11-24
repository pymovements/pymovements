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

from pymovements._utils.expressions import as_expr
from pymovements._utils.time import duration_to_ms
from pymovements._utils.time import sampling_interval_ms
from pymovements._utils.time import sampling_interval_us
from pymovements.events._utils.selection import selection_end


def _to_num(series: pl.Series) -> np.ndarray:
    """Convert a time series to millisecond values as numpy array."""
    if isinstance(series.dtype, pl.Duration):
        return duration_to_ms(series).to_numpy()
    return series.to_numpy()


def _check_column(frame: pl.DataFrame, column: str | pl.Expr, label: str) -> None:
    """Raise if a column given by name is missing from the frame."""
    if isinstance(column, str) and column not in frame.columns:
        raise ValueError(f'{label} column {column!r} not found in events.')


def _warn_deprecated_column(function_name: str, argument: str, replacement: str) -> None:
    """Warn that a column argument is deprecated in favour of its replacement."""
    warnings.warn(
        DeprecationWarning(
            f"{function_name} argument '{argument}' is deprecated since version v0.29.0. "
            f"Use '{replacement}' instead. "
            'This argument will be removed in v0.34.0.',
        ),
        stacklevel=4,
    )


def _warn_deprecated_columns(
        function_name: str,
        onset_column: str | None,
        offset_column: str | None,
) -> None:
    """Warn for each deprecated column argument that is given."""
    if onset_column is not None:
        _warn_deprecated_column(function_name, 'onset_column', 'onset')
    if offset_column is not None:
        _warn_deprecated_column(function_name, 'offset_column', 'duration')


def _check_event_columns(
        events: pl.DataFrame,
        onset: str | pl.Expr,
        duration: str | pl.Expr,
        offset_column: str | None,
) -> None:
    """Raise if the onset column or the duration or deprecated offset column is missing."""
    _check_column(events, onset, 'Onset')
    if offset_column is not None:
        _check_column(events, offset_column, 'Offset')
    else:
        _check_column(events, duration, 'Duration')


def events2segmentation(
    events: pl.DataFrame,
    name: str,
    time_column: str = 'time',
    trial_columns: list[str] | None = None,
    onset_column: str | None = None,
    offset_column: str | None = None,
    padding: float | tuple[float, float] | None = None,
    *,
    onset: str | pl.Expr = 'onset',
    duration: str | pl.Expr = 'duration',
    sampling_rate: float | None = None,
) -> pl.Expr:
    """Convert a list of events to a binary segmentation expression.

    This function creates a boolean expression that evaluates to ``True`` if a sample
    falls within any event interval and ``False`` otherwise.
    Each event covers the samples with ``onset <= time < onset + duration``, the interval
    that :py:func:`segmentation2events` produces.

    Parameters
    ----------
    events : pl.DataFrame
        Event data. Must have onset and duration columns.
    name : str
        The name of the event type to use for segmentation (e.g. 'blink').
    time_column : str
        The name of the column containing the timestamps. Default is 'time'.
    trial_columns : list[str] | None
        The names of the columns containing trial identifiers. If provided,
        events will only be mapped to samples with matching trial identifiers.
        Default is None.
    onset_column : str | None
        The name of the column containing the onset of the event (inclusive).
        Default is None.

        .. deprecated:: v0.29.0
            Please use ``onset`` instead.
            This argument will be removed in v0.34.0.
    offset_column : str | None
        The name of the column containing the offset of the event (inclusive). If given,
        each event covers the samples with ``onset <= time <= offset``. Default is None.

        .. deprecated:: v0.29.0
            Please use ``duration`` instead.
            This argument will be removed in v0.34.0.
    padding : float | tuple[float, float] | None
        Padding to extend each event interval, in the same units as ``time_column``.
        If the onset and duration columns hold ``polars.Duration`` values, the padding
        values are interpreted as milliseconds.
        If a single float, the same padding is applied symmetrically before and after
        each event. If a tuple ``(before, after)``, ``before`` is subtracted from the
        onset and ``after`` is added to the end of the event. The padded event covers the
        samples with ``onset - before <= time < onset + duration + after``. With a
        ``sampling_rate`` the padded end moves half a sampling interval early as well.
        Both values must be non-negative. Default is None (no padding).
    onset : str | pl.Expr
        The onset of the event (inclusive), as a column name or an expression.
        The values must correspond to the values in ``time_column``.
        Default is 'onset'.
    duration : str | pl.Expr
        The duration of the event, as a column name or an expression, in the unit of
        ``onset``. An event with a ``null`` duration selects no samples, an event with a ``0``
        duration selects only the samples of its padding. Default is 'duration'.
    sampling_rate : float | None
        The sampling rate of the samples in Hz. If given, each event ends half a sampling
        interval early, ``onset <= time < onset + duration - Δ/2`` with ``Δ = 1000 /
        sampling_rate`` milliseconds, so a following sample that arrives slightly early stays
        out of the event. This requires a ``time_column`` in milliseconds or
        ``polars.Duration``. An event whose end falls exactly on a sample timestamp less than
        half an interval after its last sample can lose that last sample. If ``None``, the
        plain interval applies. Default is None.

    Returns
    -------
    pl.Expr
        A boolean expression aliased to ``name``.

    Raises
    ------
    TypeError
        If ``padding`` is not None, a tuple, or a number.
    ValueError
        If the onset or duration column is missing from the events.
        If any duration is negative.
        If any padding value is negative.
        If the deprecated ``offset_column`` holds an offset before its onset.

    Notes
    -----
    Events cover the half-open interval ``[onset, onset + duration)``.
    For example, an event with onset 2 and duration 3 includes samples where the
    ``time_column`` has values 2, 3, and 4.

    The onset and duration values in the ``events`` DataFrame are compared directly
    against the values in the ``time_column`` of the samples DataFrame. If the
    ``time_column`` contains indices, then onsets are indices and durations count samples.
    If the ``time_column`` contains timestamps, then onsets are timestamps and durations are
    spans in the same unit.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.transforms import events2segmentation
    >>>
    >>> events_df = pl.DataFrame({
    ...     'name': ['blink', 'blink', 'not_blink'],
    ...     'onset': [2, 7, 3],
    ...     'duration': [4, 3, 4],
    ... })
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
    ...     {'name': ['blink'], 'onset': [3], 'duration': [3]}
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
    ...     'duration': [2, 3],
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

    With a sampling rate, a sample that arrives slightly before the end of an event stays
    out of it:

    >>> events_df = pl.DataFrame({'name': ['blink'], 'onset': [0.0], 'duration': [8.0]})
    >>> gaze_df = pl.DataFrame({'time': [0.0, 4.0, 7.9, 12.0]})
    >>> gaze_df.with_columns(
    ...     events2segmentation(events_df, name='blink', sampling_rate=250)
    ... )
    shape: (4, 2)
    ┌──────┬───────┐
    │ time ┆ blink │
    │ ---  ┆ ---   │
    │ f64  ┆ bool  │
    ╞══════╪═══════╡
    │ 0.0  ┆ true  │
    │ 4.0  ┆ true  │
    │ 7.9  ┆ false │
    │ 12.0 ┆ false │
    └──────┴───────┘
    """
    _warn_deprecated_columns('events2segmentation', onset_column, offset_column)
    onset = onset_column or onset
    _check_event_columns(events, onset, duration, offset_column)

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

    if offset_column is not None:
        relevant_events, starts, ends = _closed_event_bounds(
            relevant_events, onset=onset, offset_column=offset_column,
            padding=(pad_before, pad_after),
        )
    else:
        relevant_events, starts, ends = _half_open_event_bounds(
            relevant_events, onset=onset, duration=duration, sampling_rate=sampling_rate,
            padding=(pad_before, pad_after),
        )

    # An empty selection, such as a 0 duration without padding, selects no samples.
    selects_samples = ends > starts
    relevant_events = relevant_events.filter(pl.Series(selects_samples))
    starts, ends = starts[selects_samples], ends[selects_samples]

    if relevant_events.is_empty():
        return pl.repeat(False, pl.len()).alias(name)

    # Check for overlaps using padded intervals
    if len(starts) > 1:
        # Check for overlaps within each trial
        if trial_columns:
            bounds = relevant_events.with_columns(
                pl.Series('__start__', starts), pl.Series('__end__', ends),
            )
            for trial_group in bounds.group_by(trial_columns):
                if _has_overlap(
                        trial_group[1]['__start__'].to_numpy(),
                        trial_group[1]['__end__'].to_numpy(),
                ):
                    warnings.warn(
                        f'Overlapping events detected for trial {trial_group[0]}',
                        UserWarning,
                        stacklevel=2,
                    )
                    break

        # Check for overlaps if no trialised check has been performed
        elif _has_overlap(starts, ends):
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

    # Compare in numeric milliseconds so the mask is correct whether the event bounds and the
    # sample time column are numeric or ``polars.Duration``, and whether or not their dtypes
    # match. Duration values are converted by their physical unit; numeric values follow the
    # codebase convention that a numeric time column is already in milliseconds. The event
    # bounds are already numeric milliseconds via ``_to_num``; the time column is coerced to
    # milliseconds at evaluation time, dispatching on its runtime dtype.
    time_ms = pl.col(time_column).map_batches(
        lambda series: duration_to_ms(series)
        if isinstance(series.dtype, pl.Duration) else series.cast(pl.Float64),
        return_dtype=pl.Float64,
    )

    is_event = pl.repeat(False, pl.len())
    for index, event in enumerate(relevant_events.to_dicts()):
        is_in_time_range = time_ms.ge(starts[index]) & time_ms.lt(ends[index])

        # Select events matching time and trial criteria
        if trial_columns:
            is_same_trial = pl.lit(True)
            for trial_column in trial_columns:
                is_same_trial = is_same_trial & (pl.col(trial_column) == event[trial_column])
            is_event = is_event | (is_in_time_range & is_same_trial)
        else:
            is_event = is_event | is_in_time_range

    return is_event.alias(name)


def _half_open_event_bounds(
    events: pl.DataFrame,
    *,
    onset: str | pl.Expr,
    duration: str | pl.Expr,
    sampling_rate: float | None,
    padding: tuple[float, float],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the events that select samples with the padded bounds of their selection.

    Events with a ``null`` onset or a ``null`` duration select no samples and are left out. The end
    bound is exclusive and moves half a sampling interval early with a sampling rate.
    """
    bounds = events.select(
        as_expr(onset).alias('onset'),
        as_expr(duration).alias('duration'),
    )
    bounds = pl.DataFrame({
        'onset': _to_num(bounds['onset']).astype(np.float64),
        'duration': _to_num(bounds['duration']).astype(np.float64),
    }).fill_nan(None)

    if (bounds['duration'] < 0).any():
        raise ValueError(
            'Duration must be non-negative, but found invalid event(s)',
        )

    selects_samples = bounds['onset'].is_not_null() & bounds['duration'].is_not_null()
    bounds = bounds.filter(selects_samples)
    # The selection end works in microseconds, the bounds here are milliseconds.
    ends = bounds.select(
        selection_end(
            pl.col('onset') * 1000, pl.col('duration') * 1000, sampling_rate=sampling_rate,
        ) / 1000,
    )
    pad_before, pad_after = padding
    return (
        events.filter(selects_samples),
        bounds['onset'].to_numpy() - pad_before,
        ends.to_series().to_numpy() + pad_after,
    )


def _closed_event_bounds(
    events: pl.DataFrame,
    *,
    onset: str | pl.Expr,
    offset_column: str,
    padding: tuple[float, float],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the events with the padded bounds of a deprecated inclusive offset column.

    The end bound is the next float above the padded offset, so the exclusive comparison against
    it keeps a sample on the offset.
    """
    onsets = _to_num(events.select(as_expr(onset)).to_series())
    offsets = _to_num(events[offset_column])

    if np.any(onsets > offsets):
        raise ValueError(
            'Onset must be less than or equal to offset, but found invalid event(s)',
        )
    pad_before, pad_after = padding
    return events, onsets - pad_before, np.nextafter(offsets + pad_after, np.inf)


def events2timeratio(
    events: pl.DataFrame,
    samples: pl.DataFrame,
    name: str,
    time_column: str = 'time',
    trial_columns: list[str] | None = None,
    sampling_rate: float | None = None,
    onset_column: str | None = None,
    offset_column: str | None = None,
    *,
    onset: str | pl.Expr = 'onset',
    duration: str | pl.Expr = 'duration',
) -> pl.Expr:
    """Create an expression to calculate time-based event ratio.

    This function creates an expression that calculates the ratio of time covered by events
    relative to the total time span. Overlapping intervals of events with the matching
    ``name`` are merged before their durations are summed, so each time point is counted
    only once and the resulting ratio never exceeds 1.0. Events with a ``null`` duration
    contribute nothing.

    Parameters
    ----------
    events : pl.DataFrame
        Event data. Must have onset and duration columns.
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
        The sampling rate of the gaze data in Hz. The total time range spans
        ``t_last - t_first + Δ`` over the samples, with ``Δ = 1000 / sampling_rate``
        milliseconds. If ``None``, ``Δ`` is estimated as the mode of the time differences.
    onset_column : str | None
        The name of the column containing event onset times.

        .. deprecated:: v0.29.0
            Please use ``onset`` instead.
            This argument will be removed in v0.34.0.
    offset_column : str | None
        The name of the column containing inclusive event offset times. If given, each event
        lasts ``offset - onset + Δ``.

        .. deprecated:: v0.29.0
            Please use ``duration`` instead.
            This argument will be removed in v0.34.0.
    onset : str | pl.Expr
        The event onset, as a column name or an expression. Default is 'onset'.
    duration : str | pl.Expr
        The event duration, as a column name or an expression. Default is 'duration'.

    Returns
    -------
    pl.Expr
        An expression that calculates the event ratio. When aggregated with
        group_by().agg(), computes per-trial ratios.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.transforms import events2timeratio
    >>> events = pl.DataFrame({
    ...     'name': ['blink', 'blink'],
    ...     'onset': [1.0, 5.0],
    ...     'duration': [3.0, 3.0],
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
    >>> # Time range from the sampling rate
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
    _warn_deprecated_columns('events2timeratio', onset_column, offset_column)
    onset = onset_column or onset

    if events.is_empty():
        return pl.lit([0.0]).list.sum().alias(f'event_ratio_{name}')

    _check_event_columns(events, onset, duration, offset_column)
    if time_column not in samples.columns:
        raise ValueError(f'Time column {time_column!r} not found in samples.')

    relevant_events = events.filter(pl.col('name') == name)

    if relevant_events.is_empty():
        return pl.lit([0.0]).list.sum().alias(f'event_ratio_{name}')

    if samples.is_empty():
        return pl.lit(None).cast(pl.Float64).alias(f'event_ratio_{name}')

    # Work in numeric milliseconds so event durations and sample time ranges combine correctly
    # whether either side is numeric or polars.Duration, and whether or not their dtypes match.
    # Duration values are converted by their physical unit; numeric values follow the codebase
    # convention that a numeric time column is already in milliseconds. Each column is coerced to
    # fractional milliseconds based on its own dtype.
    def _ms(frame: pl.DataFrame, column: str | pl.Expr) -> pl.Expr:
        expr = as_expr(column)
        if isinstance(frame.select(expr).dtypes[0], pl.Duration):
            return duration_to_ms(expr)
        return expr.cast(pl.Float64)

    onset_ms = _ms(relevant_events, onset)
    time_ms = _ms(samples, time_column)

    # Single-sample series: return 1.0 if that sample falls within an event, else 0.0.
    if samples.height == 1:
        sample_time = _to_num(samples.get_column(time_column))[0]
        if offset_column is not None:
            event_filter = (
                (onset_ms <= sample_time) & (_ms(relevant_events, offset_column) >= sample_time)
            )
        else:
            end_ms = onset_ms + _ms(relevant_events, duration)
            event_filter = (onset_ms <= sample_time) & (end_ms > sample_time)
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
        time_diffs = samples.select(time_ms.alias(time_column))[time_column].diff().drop_nulls()
        dt_ms = float(time_diffs.mode().item())

    # A deprecated inclusive offset column lasts one sampling interval past the offset.
    if offset_column is not None:
        duration_ms = _ms(relevant_events, offset_column) - onset_ms + dt_ms
    else:
        duration_ms = _ms(relevant_events, duration)

    # Coerce event onsets and durations to fractional milliseconds for the merge helper. Events
    # without a duration cover no time and are left out.
    events_ms = relevant_events.with_columns(
        onset_ms.alias('__onset__'),
        duration_ms.alias('__duration__'),
    ).filter(pl.col('__onset__').is_not_null() & pl.col('__duration__').is_not_null())

    # Event ratio considering trial columns
    if trial_columns:
        # Merge overlapping same-name event intervals within each trial before
        # summing durations, so that overlapping intervals are not double-counted.
        event_durations = _merged_event_durations(
            events=events_ms,
            trial_columns=trial_columns,
        )

        sample_time_ranges = samples.group_by(trial_columns, maintain_order=True).agg(
            (time_ms.max() - time_ms.min() + dt_ms).alias('time_range'),
        )

        trial_ratios = event_durations.join(
            sample_time_ranges,
            on=trial_columns,
            how='full',
            nulls_equal=True,
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

    # Merge overlapping same-name event intervals before summing durations, so that
    # overlapping intervals are not double-counted.
    total_duration = _merged_event_durations(events=events_ms).item()

    time_range = samples.select(
        time_ms.max() - time_ms.min() + dt_ms,
    ).item()

    return pl.lit(total_duration / time_range).alias(f'event_ratio_{name}')


def segmentation2events(
    segmentation: pl.Series | np.ndarray,
    name: str,
    time_column: pl.Series | np.ndarray | None = None,
    trial_columns: pl.DataFrame | None = None,
    *,
    sampling_rate: float,
) -> pl.DataFrame:
    """Convert a binary segmentation map to a list of events.

    This function identifies continuous regions of ``1``'s in the segmentation array
    and converts them to events with an onset and a duration. Each event lasts from its first
    sample to one sampling interval past its last sample, so :py:func:`events2segmentation`
    selects exactly its samples again.

    Parameters
    ----------
    segmentation : pl.Series | np.ndarray
        A 1D binary array or Series where ``1`` or ``True`` indicates an event and
        ``0`` or ``False`` indicates no event.
        Must contain only values ``0``, ``1``, ``True`` or ``False``.
    name : str
        The name of the event type to use for the 'name' column in the output DataFrame.
    time_column : pl.Series | np.ndarray | None
        The timestamps of the samples. If provided, the onsets are the timestamps of the first
        samples. If None, the indices themselves will be used. Default is None.
    trial_columns : pl.DataFrame | None
        A DataFrame containing trial identifiers for each sample. If provided,
        events will be identified within each trial separately.
        Default is None.
    sampling_rate : float
        The sampling rate of the samples in Hz. The duration of each event is
        ``t_last - t_first + Δ`` with the nominal sampling interval ``Δ = 1000 / sampling_rate``
        milliseconds. Without ``time_column`` the onsets are indices and the durations count
        samples, so ``Δ`` is one index.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the name, onset and duration of each event.
        Onsets correspond to the values in ``time_column`` if provided,
        otherwise they are indices of the input ``segmentation`` array.

    Raises
    ------
    ValueError
        If segmentation is not a 1D array.
        If segmentation contains values other than ``0`` and ``1``.
        If ``time_column`` length does not match ``segmentation`` length.
        If ``trial_columns`` length does not match ``segmentation`` length.
        If ``sampling_rate`` is not positive.
    TypeError
        If segmentation is not a polars.Series or numpy.ndarray.

    Notes
    -----
    For example, a sequence of ones from index 2 to 4 results in an onset of 2 and a
    duration of 3.

    Examples
    --------
    >>> import polars as pl
    >>> from pymovements.transforms import segmentation2events
    >>> segmentation = pl.Series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    >>> segmentation2events(segmentation, name='blink', sampling_rate=1000)
    shape: (2, 3)
    ┌───────┬───────┬──────────┐
    │ name  ┆ onset ┆ duration │
    │ ---   ┆ ---   ┆ ---      │
    │ str   ┆ i64   ┆ i64      │
    ╞═══════╪═══════╪══════════╡
    │ blink ┆ 2     ┆ 3        │
    │ blink ┆ 7     ┆ 2        │
    └───────┴───────┴──────────┘
    >>> # Using a time column in milliseconds, sampled at 10 Hz:
    >>> time = pl.Series([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    >>> segmentation2events(segmentation, name='blink', time_column=time, sampling_rate=10)
    shape: (2, 3)
    ┌───────┬───────┬──────────┐
    │ name  ┆ onset ┆ duration │
    │ ---   ┆ ---   ┆ ---      │
    │ str   ┆ i64   ┆ f64      │
    ╞═══════╪═══════╪══════════╡
    │ blink ┆ 200   ┆ 300.0    │
    │ blink ┆ 700   ┆ 200.0    │
    └───────┴───────┴──────────┘
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

    if not _has_binary_values(segmentation):
        raise ValueError('segmentation must only contain binary values (0, 1, True, or False)')

    # Validates the sampling rate in every mode.
    interval_ms = sampling_interval_ms(sampling_rate)

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
        if isinstance(time_column.dtype, pl.Duration):
            interval = pl.duration(microseconds=sampling_interval_us(sampling_rate))
        else:
            interval = pl.lit(interval_ms)
    else:
        df_dict['__time__'] = pl.int_range(0, len(segmentation), dtype=pl.Int64, eager=True)
        # The durations count samples, so one sample lasts one index.
        interval = pl.lit(1)

    df = pl.DataFrame(df_dict)

    group_cols = []
    if trial_columns is not None:
        if len(trial_columns) != len(segmentation):
            raise ValueError(
                f'trial_columns length ({len(trial_columns)}) must match '
                f'segmentation length ({len(segmentation)})',
            )
        df = pl.concat([df, trial_columns], how='horizontal_extend')
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
            (pl.col('__time__').max() - pl.col('__time__').min() + interval).alias('duration'),
        )
    )

    # Final clean-up: reorder columns and drop internal ones
    # Result should have: name, onset, duration, then trial columns
    final_cols = ['name', 'onset', 'duration']
    if trial_columns is not None:
        final_cols.extend(trial_columns.columns)

    if events_df.is_empty():
        schema = {
            'name': pl.String,
            'onset': df.schema['__time__'],
            'duration': df.select(pl.col('__time__') + interval).dtypes[0],
        }
        if trial_columns is not None:
            schema.update(trial_columns.schema)
        return pl.DataFrame(None, schema=schema)

    return events_df.select(final_cols)


def _merged_event_durations(
    events: pl.DataFrame,
    trial_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Sum merged durations of overlapping intervals, optionally per trial.

    Each event covers ``[__onset__, __onset__ + __duration__)``. Overlapping intervals are merged
    before their durations are summed, so that each time point is counted only once.

    Merging is done by sorting events by onset and tracking the running maximum end: a new
    merged interval starts whenever an onset exceeds the running end of the previous event.
    When ``trial_columns`` is provided, merging and duration sums are computed within each
    trial group.
    """
    sort_columns = (trial_columns or []) + ['__onset__']
    group_columns = (trial_columns or []) + ['_interval_id']

    running_end = (pl.col('__onset__') + pl.col('__duration__')).cum_max()
    shifted_end = pl.col('_running_end').shift(1)
    interval_id = pl.col('_new_interval').cum_sum()

    if trial_columns:
        running_end = running_end.over(trial_columns)
        shifted_end = shifted_end.over(trial_columns)
        interval_id = interval_id.over(trial_columns)

    merged = (
        events.sort(sort_columns)
        .with_columns(running_end.alias('_running_end'))
        .with_columns(
            (pl.col('__onset__') > shifted_end).fill_null(False).alias('_new_interval'),
        )
        .with_columns(interval_id.alias('_interval_id'))
        .group_by(group_columns, maintain_order=True)
        .agg(
            pl.col('__onset__').min().alias('_merged_onset'),
            (pl.col('__onset__') + pl.col('__duration__')).max().alias('_merged_end'),
        )
    )

    durations = (pl.col('_merged_end') - pl.col('_merged_onset')).sum().alias('duration')

    if trial_columns:
        return merged.group_by(trial_columns, maintain_order=True).agg(durations)
    return merged.select(durations)


def _has_overlap(starts: np.ndarray, ends: np.ndarray) -> bool:
    """Check if any two half-open event intervals ``[start, end)`` overlap.

    An event ending where the next one starts only touches it.
    """
    if len(starts) <= 1:
        return False

    # Sort by start
    sorted_indices = np.argsort(starts)
    sorted_starts = starts[sorted_indices]
    sorted_ends = ends[sorted_indices]

    return bool(np.any(sorted_starts[1:] < sorted_ends[:-1]))


def _has_binary_values(series: pl.Series) -> bool:
    """Check if a series holds only 0, 1, True or False."""
    if series.dtype == pl.Boolean:
        return True
    if not series.dtype.is_numeric():
        return False
    return bool(((series == 0) | (series == 1)).all())
