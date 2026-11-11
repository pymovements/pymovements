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
"""Provides the wisdom of the crowd ensemble correction method."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import polars as pl

from pymovements.events.correction._utils import woc_core


def wisdom_of_the_crowd(assignment_columns: Sequence[str]) -> pl.Expr:
    """Ensemble correction choosing line assignment with most votes across algorithms.

    In the event of a tie, the left-most column in ``assignment_columns`` is given
    priority, following the reference implementation.

    Reference: :cite:p:`Mercier2024b`.

    Parameters
    ----------
    assignment_columns: Sequence[str]
        Names of the columns holding the corrected y-coordinates or line assignments of
        the candidate algorithms, in order of tie-breaking priority.

    Returns
    -------
    pl.Expr
        Expression computing the ensemble-corrected values.
    """
    column_priority = {column: priority for priority, column in enumerate(assignment_columns)}
    return (
        pl.struct(list(assignment_columns))
        .map_batches(partial(woc_core, column_priority=column_priority))
        .alias('y_wisdom_of_the_crowd')
    )
