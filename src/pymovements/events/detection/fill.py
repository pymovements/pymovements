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
"""Provides the implementation of the event fill function."""
from __future__ import annotations

import numpy
import polars

from pymovements._utils.time import event_times_to_ms
from pymovements._utils.time import sampling_interval_ms
from pymovements._utils.time import timesteps_to_numpy
from pymovements.events._utils.selection import selection_end
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events
from pymovements.transforms.numpy import consecutive


@register_event_detection
def fill(
        events: Events,
        timesteps: list[int] | numpy.ndarray | polars.Series,
        minimum_duration: int = 1,
        name: str = 'unclassified',
        *,
        sampling_rate: float,
) -> Events:
    """Classify all previously unclassified timesteps as events.

    Parameters
    ----------
    events: Events
        The already detected events.
    timesteps: list[int] | numpy.ndarray | polars.Series
        shape (N, )
        Continuous 1D timestep time series. Each event covers the timesteps with
        ``onset <= t < onset + duration - Δ/2``, with ``Δ`` the nominal sampling interval of
        ``sampling_rate``. An event whose end falls exactly on a timestep less than half an
        interval after its last timestep can lose that last timestep. The event ``onset`` and
        ``duration`` values must be in the same unit as ``timesteps``. When ``events`` stores
        ``polars.Duration`` time columns, ``timesteps`` are expected to be in milliseconds.
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
        (default: 1)
    name: str
        Name for detected events in Events. (default: 'unclassified')
    sampling_rate: float
        Sampling rate of the signal in Hz. Each sample covers the nominal interval
        ``1000 / sampling_rate`` milliseconds, which is added to the duration of every event
        and written to the sampling-rate entry of the events metadata.

    Returns
    -------
    Events
        A dataframe with detected fixations as rows.

    Raises
    ------
    ValueError
        If sampling_rate is not positive.
    """
    if isinstance(timesteps, polars.Series):
        timesteps = timesteps_to_numpy(timesteps)
    timesteps = numpy.array(timesteps)

    sampling_interval = sampling_interval_ms(sampling_rate)

    # Create binary mask where each existing event is marked.
    events_mask = numpy.zeros(len(timesteps), dtype=bool)

    # Compare in microseconds, the scale of the selection end. The timesteps and the event times
    # are milliseconds. Events without an end select no timesteps.
    timesteps_us = timesteps * 1000
    events_frame = event_times_to_ms(events.frame).select(
        (polars.col('onset') * 1000).alias('onset'),
        (polars.col('duration') * 1000).alias('duration'),
    ).select(
        'onset',
        selection_end('onset', 'duration', sampling_rate=sampling_rate).alias('end'),
    ).drop_nulls()

    for onset, end in events_frame.iter_rows():
        # Mark every timestep inside the event selection ``[onset, end)``.
        # Matching by interval membership instead of exact timestep equality keeps this
        # robust to floating-point timesteps and to sample spacings other than one.
        events_mask |= (timesteps_us >= onset) & (timesteps_us < end)

    # Mask all indices where there is no event.
    candidate_mask = ~events_mask

    # Get indices of true values in candidate mask.
    candidate_indices = numpy.where(candidate_mask)[0]

    # Get all fixation candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    if len(candidates) == 1 and numpy.array_equal(
            candidates[0], numpy.array([], dtype=numpy.int64),
    ):
        return Events(
            name=name, onsets=[], durations=[],
            offsets=[], offsets_inclusive=True, sampling_rate=sampling_rate,
        )

    # Filter all candidates by minimum duration, each sample covering the nominal interval.
    candidates = [
        candidate for candidate in candidates
        if timesteps[candidate[-1]] - timesteps[candidate[0]] + sampling_interval
        >= minimum_duration
    ]

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate_indices[0] for candidate_indices in candidates]].flatten()
    # Offset of each event candidate is last event in candidate indices.
    offsets = timesteps[[candidate_indices[-1] for candidate_indices in candidates]].flatten()
    # Each sample covers the nominal interval, so the duration reaches past the last sample.
    durations = offsets - onsets + sampling_interval

    events = Events(
        name=name, onsets=onsets, durations=durations,
        offsets=offsets, offsets_inclusive=True, sampling_rate=sampling_rate,
    )
    return events
