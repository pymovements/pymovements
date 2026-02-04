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
"""Provides filter specific functions.

.. deprecated:: v0.21.1
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

import numpy as np
from deprecated.sphinx import deprecated

from pymovements.events._utils._filters import events_split_nans as _events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans \
    as _filter_candidates_remove_nans


@deprecated(
    reason='This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def filter_candidates_remove_nans(
        candidates: list[np.ndarray],
        values: np.ndarray,
) -> list[np.ndarray]:
    """Filter a list of candidates for an event-detection algorithm.

    Removes leading and ending np.nans for all candidates in candidates

    .. deprecated:: v0.21.1
       This module will be removed in v0.26.0.

    Parameters
    ----------
    candidates: list[np.ndarray]
        List of candidates; each candidate consists of a list of indices
    values: np.ndarray
        shape (N, 1) or shape (N, 2)
        Corresponding continuous 1D/2D values time series.

    Returns
    -------
    list[np.ndarray]
        Returns a filtered list of candidates.
    """
    return _filter_candidates_remove_nans(
        candidates=candidates,
        values=values,
    )


@deprecated(
    reason='This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def events_split_nans(
        candidates: list[np.ndarray],
        values: np.ndarray,
) -> list[np.ndarray]:
    """Filter a list of candidates for an event-detection algorithm.

    Splits events if np.nans are within an event

    .. deprecated:: v0.21.1
       This module will be removed in v0.26.0.

    Parameters
    ----------
    candidates: list[np.ndarray]
        List of candidates; each candidate consists of a list of indices
    values: np.ndarray
        shape (N, 1) or shape (N, 2)
        Corresponding continuous 1D/2D values time series.

    Returns
    -------
    list[np.ndarray]
        Returns a filtered list of candidates.
    """
    return _events_split_nans(
        candidates=candidates,
        values=values,
    )
