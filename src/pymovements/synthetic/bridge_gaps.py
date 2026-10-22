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
"""Provides gap bridging for event sequences."""
from __future__ import annotations

from copy import deepcopy

import polars as pl

from pymovements.events.events import Events
from pymovements.synthetic._utils import check_not_negative
from pymovements.synthetic._utils import ordered_frame
from pymovements.synthetic._utils import previous_offset


def bridge_gaps(events: Events, *, max_gap: float | None = None) -> Events:
    """Return the events with each gap between two consecutive events filled by a saccade.

    A detected fixation sequence has gaps, since the saccades between the fixations are not part
    of it. Bridging is opt-in, because not every gap is a saccade: a long gap is more likely a
    blink or data loss, and ``max_gap`` leaves such gaps untouched. Events with trial columns are
    bridged per trial, so no saccade spans two trials.

    Parameters
    ----------
    events: Events
        The events to bridge. They are not modified.
    max_gap: float | None
        The longest gap in milliseconds that is bridged, or ``None`` to bridge every gap.
        (default: None)

    Returns
    -------
    Events
        The events sorted by trial and onset, with a ``'saccade'`` for each bridged gap. Columns
        beyond name, onset, offset, duration and trial are null for the inserted saccades.

    Raises
    ------
    TypeError
        If ``max_gap`` is not a number.
    ValueError
        If ``max_gap`` is negative, or two events of a trial overlap.

    Examples
    --------
    >>> import pymovements as pm
    >>> fixations = pm.Events(
    ...     name=['fixation', 'fixation', 'fixation'],
    ...     onsets=[0, 300, 2400],
    ...     offsets=[250, 560, 2700],
    ... )
    >>> bridge_gaps(fixations, max_gap=100).frame['name'].to_list()
    ['fixation', 'saccade', 'fixation', 'fixation']
    """
    if max_gap is not None:
        max_gap = check_not_negative(max_gap, 'max_gap')

    trial_columns = events.trial_columns or []
    frame = ordered_frame(events)

    gap = pl.col('onset') - pl.col('previous_offset')
    bridged = gap > pl.duration(microseconds=0)
    if max_gap is not None:
        bridged = bridged & (gap <= pl.duration(microseconds=round(max_gap * 1000)))

    frame_with_previous = frame.with_columns(previous_offset=previous_offset(trial_columns))
    saccades = frame_with_previous.filter(bridged).select(
        *trial_columns,
        name=pl.lit('saccade'),
        onset=pl.col('previous_offset'),
        offset=pl.col('onset'),
        duration=gap,
    )

    combined = pl.concat([frame, saccades], how='diagonal_relaxed').sort(
        [*trial_columns, 'onset'], maintain_order=True,
    )

    return Events(
        combined,
        trial_columns=events.trial_columns,
        metadata=deepcopy(events.metadata),
    )
