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
"""Provides helpers for handling time columns backed by ``polars.Duration``."""
from __future__ import annotations

import polars as pl


def durations_to_ms(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert all Duration columns of a dataframe to numeric milliseconds.

    Duration columns are converted to Float64 millisecond values. Columns
    holding only whole milliseconds are narrowed to Int64 afterwards.

    Parameters
    ----------
    frame: pl.DataFrame
        DataFrame whose Duration columns should be converted.

    Returns
    -------
    pl.DataFrame
        DataFrame with Duration columns converted to numeric milliseconds.
    """
    duration_columns = [
        column for column in frame.columns
        if isinstance(frame.schema[column], pl.Duration)
    ]
    if not duration_columns:
        return frame

    frame = frame.with_columns(
        [
            (pl.col(column) / pl.duration(milliseconds=1)).alias(column)
            for column in duration_columns
        ],
    )

    # Convert to int if possible.
    for column in duration_columns:
        all_whole = frame.select(
            pl.col(column).round().eq(pl.col(column)).all(),
        ).item()
        if all_whole:
            frame = frame.with_columns(pl.col(column).cast(pl.Int64))

    return frame
