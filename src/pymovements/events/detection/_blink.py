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
"""Provides detection of blinks from the pupil signal."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pymovements._utils import _checks
from pymovements.events.detection._library import register_event_detection
from pymovements.events.events import Events
from pymovements.gaze.transforms_numpy import consecutive


@register_event_detection
def blink(
        pupil: list[float] | np.ndarray,
        *,
        timesteps: list[int] | np.ndarray | None = None,
        delta: float | None = None,
        minimum_duration: int = 50,
        maximum_duration: int | None = 500,
        minimum_gap: int = 5,
        minimum_candidate_duration_to_absorb_gap: tuple[int, int] | int = 2,
        name: str = 'blink',
) -> Events:
    """Detect blinks from the pupil signal.

    The blink detection algorithm consists of three main stages. The implementation is inspired by
    :cite:p:`PupilPre`, which adapts the velocity-based method described in :cite:p:`Hershman2018`.

    **Stage 1 — Flag pupil loss:** All samples where the pupil value is NaN or zero (common
    blink indicators in eye-tracking data) are candidate_mask as blink candidates.

    **Stage 2 — Flag abrupt pupil changes:**: Each sample i+1 is masked as blink candidate if the
    absolute difference between consecutive pupil samples i, i+1 exceeds a ``delta`` threshold.
    If ``delta`` is ``None``, it is automatically estimated as 5 times the 95th percentile of valid
    absolute differences (non-zero, not NaN) .

    **Stage 3 — Combine blink candidates:** Combine consecutive blinks with a time gap lower than
    the specified ``minimum_gap``. Blinks are only combined if the candidates before and after the
    short gap have a ``minimum_candidate_duration_to_absorb_gap``.

    Blink events shorter than ``minimum_duration`` and longer than maximum_duration are discarded.
    Following :cite:p:`Nystrom2024` typical blinks last 50–500 ms.

    Parameters
    ----------
    pupil: list[float] | np.ndarray
        shape (N,)
        Continuous 1D pupil size time series (e.g., diameter or area).
    timesteps: list[int] | np.ndarray | None
        shape (N,)
        Corresponding continuous 1D timestep time series. If None, sample-based timesteps are
        assumed. (default: None)
    delta: float | None
        Threshold on absolute pupil difference for flagging rapid changes. If None, it is
        auto-estimated as ``5 * np.nanpercentile(abs_diff, 95)`` from valid absolute
        differences. (default: None)
    minimum_duration: int
        Minimum blink duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
        (default: 50)
    maximum_duration: int | None
        Maximum blink duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``maximum_duration`` is specified in numbers of samples.
        Set to None to disable the upper bound. (default: 500)
    minimum_gap: int
        Minimum time gap in-between two blinks. Blinks that have a smaller time gap are combined
        into a single event. The duration is specified in the units used in ``timesteps``. If
        ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
        (default: 100)
    minimum_candidate_duration_to_absorb_gap: tuple[int, int] | int
        Minimum number of candidate_mask samples required on each side of an unflagged run
        for it to be absorbed. (default: 2)
    name: str
        Name for detected events in Events. (default: 'blink')

    Returns
    -------
    Events
        A dataframe with detected blink events as rows.

    Raises
    ------
    TypeError
        If minimum_candidate_duration_to_absorb_gap neither int nor tuple[int].
    ValueError
        If pupil is not 1D.
        If pupil and timesteps have different lengths.
        If delta is not positive.
        If minimum_duration is not positive.
        If maximum_duration is not positive or less than minimum_duration.
    """
    pupil = np.array(pupil, dtype=float)

    if pupil.ndim != 1:
        raise ValueError(
            f'pupil must be a 1D array, but got array with shape {pupil.shape}',
        )

    if timesteps is not None:
        timesteps = np.array(timesteps)
        _checks.check_is_length_matching(pupil=pupil, timesteps=timesteps)
    else:
        timesteps = np.arange(len(pupil), dtype=np.int64)

    if delta is not None and delta <= 0:
        raise ValueError(
            f'delta must be positive, but got {delta}',
        )

    if minimum_duration < 0:
        raise ValueError(
            f'minimum_duration must not be negative, but got {minimum_duration}',
        )

    if maximum_duration is not None:
        if maximum_duration < 1:
            raise ValueError(
                f'maximum_duration must be positive or None, but got {maximum_duration}',
            )
        if maximum_duration < minimum_duration:
            raise ValueError(
                f'maximum_duration must be >= minimum_duration, but got '
                f'maximum_duration={maximum_duration} < minimum_duration={minimum_duration}',
            )

    if isinstance(minimum_candidate_duration_to_absorb_gap, int):
        minimum_candidate_duration_to_absorb_gap = (
            minimum_candidate_duration_to_absorb_gap, minimum_candidate_duration_to_absorb_gap,
        )
    elif isinstance(minimum_candidate_duration_to_absorb_gap, Sequence):
        if not all(isinstance(d, int) for d in minimum_candidate_duration_to_absorb_gap):
            raise TypeError(
                'minimum_candidate_duration_to_absorb_gap must be an int or a sequence of int'
                f' but is {repr(minimum_candidate_duration_to_absorb_gap)}',
            )
        if len(minimum_candidate_duration_to_absorb_gap) != 2:
            raise ValueError(
                'minimum_candidate_duration_to_absorb_gap must be an int or a sequence of length 2'
                f' but is {len(minimum_candidate_duration_to_absorb_gap)}',
            )

    if len(pupil) == 0:
        return Events(name=name, onsets=[], offsets=[])

    # Stage 1: Flag all samples with pupil loss as blink candidates.
    candidate_mask = np.isnan(pupil) | (pupil == 0)

    # Stage 2: Flag pupil changes that exceed delta threshold.
    # Compute absolute sample differences. Prepend nan sample to preserve array shape.
    abs_diff = np.abs(np.diff(pupil, prepend=np.nan))

    # NaN diff values (from NaN pupil values) are ignored for calculating delta.
    valid_diff = abs_diff[~np.isnan(abs_diff)]
    if delta is None and len(valid_diff) > 0:
        delta = 5.0 * np.nanpercentile(valid_diff, 95)

    if delta is not None and len(valid_diff) > 0:
        # Flag sample i+1 if abs(pupil[i+1] - pupil[i]) > delta
        exceeds_mask = abs_diff > delta
        candidate_mask = candidate_mask | exceeds_mask

    # Stage 3: Combine blinks with less than minimum time gap in-between.
    if minimum_gap > 0:
        candidate_mask = _merge_blink_candidates(
            candidate_mask, minimum_gap, minimum_candidate_duration_to_absorb_gap,
        )

    # Group consecutive candidate_mask samples into events
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return Events(name=name, onsets=[], offsets=[])

    candidates = consecutive(arr=candidate_indices)

    # Filter all candidates by duration (in unit of timesteps array).
    if minimum_duration:
        candidates = [
            c_indices for c_indices in candidates
            if minimum_duration <= timesteps[c_indices[-1]] - timesteps[c_indices[0]]
        ]
    if maximum_duration:
        candidates = [
            c_indices for c_indices in candidates
            if timesteps[c_indices[-1]] - timesteps[c_indices[0]] <= maximum_duration
        ]

    if len(candidates) == 0:
        return Events(name=name, onsets=[], offsets=[])

    onsets = timesteps[[c[0] for c in candidates]].flatten()
    offsets = timesteps[[c[-1] for c in candidates]].flatten()

    return Events(name=name, onsets=onsets, offsets=offsets)


def _merge_blink_candidates(
        candidate_mask: np.ndarray,
        minimum_gap: int,
        minimum_candidate_duration_to_absorb_gap: tuple[int, int],
) -> np.ndarray:
    """Absorb short unflagged runs surrounded by masked samples.

    Parameters
    ----------
    candidate_mask: np.ndarray
        Boolean array of flagged samples.
    minimum_gap: int
        Maximum length of an unflagged run to absorb.
    minimum_candidate_duration_to_absorb_gap: tuple[int, int]
        Minimum flagged samples required on each side.

    Returns
    -------
    np.ndarray
        Updated blink candidate mask where minimum gaps are absorbed by surrounding blinks.
    """
    candidate_mask = candidate_mask.copy()
    n = len(candidate_mask)

    # Find runs of unflagged samples
    unflagged_indices = np.where(~candidate_mask)[0]
    if len(unflagged_indices) == 0:
        return candidate_mask

    runs = consecutive(arr=unflagged_indices)

    for run in runs:
        run_len = len(run)
        if run_len > minimum_gap:
            continue

        start = run[0]
        end = run[-1]

        # Count candidate_mask samples before this run
        before_start = max(0, start - minimum_candidate_duration_to_absorb_gap[0])
        flagged_before = np.sum(candidate_mask[before_start:start])

        # Count candidate_mask samples after this run
        after_end = min(n, end + 1 + minimum_candidate_duration_to_absorb_gap[1])
        flagged_after = np.sum(candidate_mask[end + 1:after_end])

        if (
                flagged_before >= minimum_candidate_duration_to_absorb_gap[0]
                and flagged_after >= minimum_candidate_duration_to_absorb_gap[1]
        ):
            candidate_mask[run] = True

    return candidate_mask
