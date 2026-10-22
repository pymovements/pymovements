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
"""Provides the rules for reading an event sequence as one signal per trial.

Shared by the sample generators: each trial is generated as its own continuous signal, so the
time between trials is never filled.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from pymovements.events.events import Events


def milliseconds(durations: pl.Series) -> pl.Series:
    """Return durations in fractional milliseconds; ``total_milliseconds`` drops the fraction."""
    return durations.dt.total_microseconds() / 1000.0


def previous_offset(trial_columns: list[str]) -> pl.Expr:
    """Return the offset of the event before each one in its trial, null for the first."""
    expression = pl.col('offset').shift(1)
    if trial_columns:
        expression = expression.over(trial_columns)
    return expression


def ordered_frame(events: Events) -> pl.DataFrame:
    """Return the events sorted by trial and onset, rejecting overlaps within a trial."""
    trial_columns = events.trial_columns or []
    frame = events.frame.sort([*trial_columns, 'onset'], maintain_order=True)

    overlapping = frame.with_columns(previous_offset=previous_offset(trial_columns)).filter(
        pl.col('onset') < pl.col('previous_offset'),
    )
    if overlapping.height:
        first = overlapping.row(0, named=True)
        raise ValueError(
            f"events overlap, a {first['name']} starts at "
            f"{_milliseconds(first['onset'])} ms before the previous event ends "
            f"at {_milliseconds(first['previous_offset'])} ms",
        )

    return frame


def check_contiguous(events: Events) -> None:
    """Reject a gap between two events of a trial, for generators that cannot render one."""
    trial_columns = events.trial_columns or []
    gaps = ordered_frame(events).with_columns(
        previous_offset=previous_offset(trial_columns),
    ).filter(pl.col('onset') > pl.col('previous_offset'))

    if gaps.height:
        first = gaps.row(0, named=True)
        raise ValueError(
            f"events leave a gap from {_milliseconds(first['previous_offset'])} ms to "
            f"{_milliseconds(first['onset'])} ms before a {first['name']}; fill it first, "
            'for example with pymovements.synthetic.bridge_gaps',
        )


def per_trial(
        events: Events,
        synthesize: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    """Return the samples synthesized for each trial, with the trial columns in front.

    ``synthesize`` receives one trial's events, ordered by onset, and returns its samples.
    """
    trial_columns = events.trial_columns or []
    frame = ordered_frame(events)

    if frame.is_empty():
        raise ValueError('events are empty, so there is nothing to generate samples for')

    if not trial_columns:
        return synthesize(frame)

    return pl.concat([
        trial.select(trial_columns).head(1).join(synthesize(trial), how='cross')
        for trial in frame.partition_by(trial_columns, maintain_order=True)
    ])


def sample_times(onset: float, offset: float, sampling_rate: float) -> np.ndarray:
    """Return the sample times in ms from onset up to offset, with at least one sample."""
    n_samples = max(round((offset - onset) * sampling_rate / 1000.0), 1)
    return onset + np.arange(n_samples) * (1000.0 / sampling_rate)


def _milliseconds(duration: object) -> float:
    """Return one duration in fractional milliseconds."""
    return float(milliseconds(pl.Series([duration]))[0])
