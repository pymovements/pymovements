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
"""Provides helpers shared by the synthetic generators and event operations.

Two kinds: the rules for reading an event sequence as one continuous signal per trial, and the
checks of hyperparameters and run parameters.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any

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


def ordered_frame(events: Events, *, generator: str | None = None) -> pl.DataFrame:
    """Return the events sorted by trial and onset, rejecting overlaps within a trial.

    One signal cannot follow two events at once, so an overlap is an error rather than something
    to resolve silently. Events of different trials may overlap, since each trial is its own
    signal.
    """
    trial_columns = events.trial_columns or []
    frame = events.frame.sort([*trial_columns, 'onset'], maintain_order=True)

    overlapping = frame.with_columns(previous_offset=previous_offset(trial_columns)).filter(
        pl.col('onset') < pl.col('previous_offset'),
    )
    if overlapping.height:
        first = overlapping.row(0, named=True)
        raise ValueError(
            f"{_prefix(generator)}events overlap, a {first['name']} starts at "
            f"{_milliseconds(first['onset'])} ms before the previous event ends "
            f"at {_milliseconds(first['previous_offset'])} ms",
        )

    return frame


def check_contiguous(events: Events, *, generator: str | None = None) -> None:
    """Reject a gap between two events of a trial, for generators that cannot render one.

    A gap is not necessarily a saccade; it may be a blink or data loss. Filling it is left to
    the user, who knows which it is.
    """
    trial_columns = events.trial_columns or []
    gaps = ordered_frame(events, generator=generator).with_columns(
        previous_offset=previous_offset(trial_columns),
    ).filter(pl.col('onset') > pl.col('previous_offset'))

    if gaps.height:
        first = gaps.row(0, named=True)
        raise ValueError(
            f"{_prefix(generator)}events leave a gap from "
            f"{_milliseconds(first['previous_offset'])} ms to {_milliseconds(first['onset'])} ms "
            f"before a {first['name']}; fill it first, for example with "
            'pymovements.synthetic.bridge_gaps',
        )


def per_trial(
        events: Events,
        synthesize: Callable[[pl.DataFrame], pl.DataFrame],
        *,
        generator: str | None = None,
) -> pl.DataFrame:
    """Return the samples synthesized for each trial, with the trial columns in front.

    Each trial is generated as its own continuous signal, so the time between trials is never
    filled. ``synthesize`` receives one trial's events, ordered by onset, and returns its
    samples.
    """
    trial_columns = events.trial_columns or []
    frame = ordered_frame(events, generator=generator)

    if frame.is_empty():
        raise ValueError(
            f'{_prefix(generator)}events are empty, so there is nothing to generate samples for',
        )

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


def check_number(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers, including booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f'{name} must be a number but is of type {type(value).__name__}: {value}',
        )

    return float(value)


def check_not_negative(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers and negative values."""
    number = check_number(value, name)

    if number < 0:
        raise ValueError(f'{name} must not be negative but is: {number}')

    return number


def check_positive(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers and values of zero or less."""
    number = check_number(value, name)

    if number <= 0:
        raise ValueError(f'{name} must be greater than zero but is: {number}')

    return number


def check_name(value: Any, name: str, allowed: tuple[str, ...]) -> None:
    """Reject a value outside a fixed set of names."""
    if value not in allowed:
        raise ValueError(f'{name} must be one of {list(allowed)} but is: {value!r}')


def _prefix(generator: str | None) -> str:
    """Return the start of an error message naming the generator, if there is one."""
    return f"generator '{generator}': " if generator else ''


def _milliseconds(duration: object) -> float:
    """Return one duration in fractional milliseconds."""
    return float(milliseconds(pl.Series([duration]))[0])
