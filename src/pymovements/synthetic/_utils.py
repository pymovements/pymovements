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
"""Provides the rules for reading an event sequence as one signal per trial."""
from __future__ import annotations

import polars as pl

from pymovements.events.events import Events


def milliseconds(durations: pl.Series) -> pl.Series:
    """Return durations in fractional milliseconds; ``total_milliseconds`` drops the fraction."""
    return durations.dt.total_microseconds() / 1000.0


def ordered_frame(events: Events) -> pl.DataFrame:
    """Return the events sorted by trial and onset, rejecting overlaps within a trial."""
    trial_columns = events.trial_columns or []
    frame = events.frame.sort([*trial_columns, 'onset'], maintain_order=True)

    previous_offset = pl.col('offset').shift(1)
    if trial_columns:
        previous_offset = previous_offset.over(trial_columns)

    overlapping = frame.with_columns(previous_offset=previous_offset).filter(
        pl.col('onset') < pl.col('previous_offset'),
    )
    if overlapping.height:
        first = overlapping.row(0, named=True)
        raise ValueError(
            f"events overlap, a {first['name']} starts at "
            f"{milliseconds(pl.Series([first['onset']]))[0]} ms before the previous event ends "
            f"at {milliseconds(pl.Series([first['previous_offset']]))[0]} ms",
        )

    return frame
