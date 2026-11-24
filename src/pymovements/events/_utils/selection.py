# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Select the samples of an event by its onset and duration."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pymovements._utils.time import sampling_interval_us

if TYPE_CHECKING:
    from pymovements.events.events import Events
    from pymovements.gaze.experiment import Experiment


def selection_end(
        onset: str | pl.Expr,
        duration: str | pl.Expr,
        *,
        sampling_rate: float | None,
) -> pl.Expr:
    """Return the end bound of the half-open sample selection of an event.

    The samples of an event are those with ``onset <= time < end``. Without a sampling rate the
    end is ``onset + duration``. With one, the end moves half a sampling interval earlier, so a
    sample that arrives slightly early stays out of the preceding event. A null duration gives a
    null end, a duration of ``0`` an end at or before the onset, so both select no samples.

    Parameters
    ----------
    onset: str | pl.Expr
        The onset column in microseconds, as a name or an expression.
    duration: str | pl.Expr
        The duration column in microseconds, as a name or an expression.
    sampling_rate: float | None
        The sampling rate in Hz, or ``None`` to apply the interval as is.

    Returns
    -------
    pl.Expr
        The end bound in microseconds.
    """
    onset_expr = pl.col(onset) if isinstance(onset, str) else onset
    duration_expr = pl.col(duration) if isinstance(duration, str) else duration
    end = onset_expr + duration_expr
    if sampling_rate is None:
        return end
    # Half the interval the time columns carry.
    return end - sampling_interval_us(sampling_rate) / 2


def resolve_sampling_rate(events: Events, experiment: Experiment | None) -> float | None:
    """Return the sampling rate of the events, from their entry first and the experiment second.

    Parameters
    ----------
    events: Events
        The events, whose ``metadata`` may hold a sampling-rate entry.
    experiment: Experiment | None
        The experiment, whose sampling rate applies when the events have no entry.

    Returns
    -------
    float | None
        The sampling rate in Hz, or ``None`` when neither source has one.
    """
    entry_rate = events.metadata.get('sampling_rate')
    if entry_rate is not None:
        return entry_rate
    if experiment is not None:
        return experiment.sampling_rate
    return None
