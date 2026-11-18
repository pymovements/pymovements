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
"""Provides functions for calculating event properties."""
from __future__ import annotations

from collections.abc import Callable

import polars as pl

EVENT_MEASURES: dict[str, Callable] = {}


def register_event_measure(function: Callable) -> Callable:
    """Register an event measure.

    Parameters
    ----------
    function: Callable
        Function to be registered as a valid property.

    Returns
    -------
    Callable
        The function that was passed as an argument.
    """
    EVENT_MEASURES[function.__name__] = function
    return function


@register_event_measure
def duration() -> pl.Expr:
    """Duration of an event.

    The duration is defined as the difference between offset time and onset time.

    Returns
    -------
    pl.Expr
        The duration of the event.
    """
    result = pl.col('offset') - pl.col('onset')
    return result.alias('duration')


@register_event_measure
def offset(*, inclusive: bool, sampling_rate: float | None = None) -> pl.Expr:
    """Offset of an event, derived from its onset and duration.

    The offset is the timestamp of the last sample of the event, ``onset + duration`` minus one
    sampling interval, or the end of the event, ``onset + duration``, one past the last sample.
    Both are written as a ``polars.Duration('us')`` column from the onset and duration columns
    of an events frame.

    Parameters
    ----------
    inclusive: bool
        If ``True``, the offset is the timestamp of the last sample, which needs the sampling
        rate. An event with a duration of ``0`` or ``null`` has no last sample, so its offset
        is ``null``. If ``False``, the offset is the end of the event, which is the onset
        itself for a duration of ``0`` and needs no sampling rate. Use ``inclusive=False``
        for exact-boundary events, whose start and end times are exact and not tied to
        samples.
    sampling_rate: float | None
        The sampling rate in Hz. Required with ``inclusive=True``, ignored otherwise.
        (default: None)

    Returns
    -------
    pl.Expr
        The offset of the event.

    Raises
    ------
    ValueError
        If ``inclusive`` is ``True`` and no ``sampling_rate`` is given.
    """
    end = pl.col('onset') + pl.col('duration')
    if not inclusive:
        return end.alias('offset')
    if sampling_rate is None:
        raise ValueError(
            'inclusive=True requires a sampling_rate: the inclusive offset is '
            'onset + duration - 1 / sampling_rate.',
        )
    # Rounded to microseconds like the durations the detectors derive, so the round trip
    # through the duration measure is exact.
    sampling_interval = pl.duration(microseconds=round(1_000_000 / sampling_rate))
    result = (
        pl.when(pl.col('duration') > pl.duration(microseconds=0))
        .then(end - sampling_interval)
        .otherwise(None)
    )
    return result.alias('offset')
