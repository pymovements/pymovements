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
"""Shared test helpers."""
from __future__ import annotations

import polars as pl


def to_duration(df: pl.DataFrame) -> pl.DataFrame:
    """Cast time-related columns to ``pl.Duration('ms')`` for comparison.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame whose columns should be cast.

    Returns
    -------
    pl.DataFrame
        DataFrame with applicable columns cast to ``pl.Duration('ms')``.
    """
    cols = [
        c for c in ['time', 'onset', 'offset', 'duration']
        if c in df.columns and df.schema[c] not in (pl.Duration('ms'),)
    ]
    if cols:
        return df.with_columns(pl.col(cols).round().cast(pl.Duration('ms')))
    return df
