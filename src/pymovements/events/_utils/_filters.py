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
"""Provides filter specific funtions."""
from __future__ import annotations

import numpy as np

from pymovements.gaze.transforms_numpy import consecutive


def filter_candidates_remove_nans(
        candidates: list[np.ndarray],
        values: np.ndarray,
) -> list[np.ndarray]:
    """Filter a list of candidates for an event-detection algorithm.

    Removes leading and ending np.nans for all candidates in candidates

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
    values = np.array(values)
    return_candidates = []
    for candidate in candidates:
        if len(candidate) == 0:
            continue
        cand_values = values[np.array(candidate)]
        start_id = 0
        while np.sum(np.isnan(cand_values[start_id, :])) > 0:
            start_id += 1
        end_id = len(cand_values) - 1
        while np.sum(np.isnan(cand_values[end_id, :])) > 0:
            end_id -= 1
        cur_candidate = list(candidate[start_id:end_id + 1])
        return_candidates.append(np.array(cur_candidate))
    return return_candidates


def events_split_nans(
        candidates: list[np.ndarray],
        values: np.ndarray,
) -> list[np.ndarray]:
    """Filter a list of candidates for an event-detection algorithm.

    Splits events if np.nans are within an event

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
    values = np.array(values)
    return_candidates = []
    for candidate in candidates:
        if len(candidate) == 0:
            continue
        cur_values = values[np.array(candidate)]
        nan_candidates = consecutive(arr=np.where(~np.isnan(np.sum(cur_values, axis=1)))[0])
        cand_list = [
            np.array(candidate[candidate_indices[0]:candidate_indices[-1] + 1])
            for candidate_indices in nan_candidates
        ]
        return_candidates += cand_list
    return return_candidates
