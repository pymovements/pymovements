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
        max_value_run: int = 3,
        nas_around_run: int = 2,
        minimum_duration: int = 50,
        maximum_duration: int | None = 500,
        name: str = 'blink',
) -> Events:
    """Detect blinks from the pupil signal using a two-stage algorithm.

    The detection algorithm adapts the differential approach of Hershman et al. (2018) [1]_
    as implemented in PupilPre (Kyröläinen et al., 2019) [2]_.

    **Stage 1 — Flagging:** Samples are flagged when the pupil value is NaN or zero (common
    blink indicators in eye-tracking data), or when the absolute difference between consecutive
    pupil samples exceeds a ``delta`` threshold. If ``delta`` is ``None``, it is automatically
    estimated as 5 times the 95th percentile of valid absolute differences.

    **Stage 2 — Island absorption:** Short runs of unflagged samples that are surrounded by
    flagged samples are absorbed into the flagged region. A run is absorbed if its length is
    at most ``max_value_run`` and there are at least ``nas_around_run`` flagged samples on
    each side.

    Consecutive flagged samples are then grouped into blink events. Events with duration
    outside the ``[minimum_duration, maximum_duration]`` range are discarded. Following
    Nyström et al. (2024) [3]_, typical blinks last 50–500 ms.

    References
    ----------
    .. [1] Hershman, R., Henik, A., & Cohen, N. (2018). A novel blink detection method based
       on pupillometry noise. *Behavior Research Methods*, 50(1), 107–114.
       https://doi.org/10.3758/s13428-017-1008-1
    .. [2] Kyröläinen, A.-J., Porretta, V., van Rij, J., & Järvikivi, J. (2019).
       PupilPre: Tools for Preprocessing Pupil Size Data (R package).
       https://CRAN.R-project.org/package=PupilPre
    .. [3] Nyström, M., Andersson, R., Niehorster, D. C., Hessels, R. S., & Hooge, I. T. C.
       (2024). What is a blink? Classifying and characterizing blinks in eye openness signals.
       *Behavior Research Methods*, 56, 3280–3299.
       https://doi.org/10.3758/s13428-023-02333-9

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
    max_value_run: int
        Maximum length of an unflagged run to be absorbed during island absorption.
        Set to 0 to disable absorption. (default: 3)
    nas_around_run: int
        Minimum number of flagged samples required on each side of an unflagged run
        for it to be absorbed. (default: 2)
    minimum_duration: int
        Minimum blink duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
        (default: 50)
    maximum_duration: int | None
        Maximum blink duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``maximum_duration`` is specified in numbers of samples.
        Set to None to disable the upper bound. (default: 500)
    name: str
        Name for detected events in Events. (default: 'blink')

    Returns
    -------
    Events
        A dataframe with detected blink events as rows.

    Raises
    ------
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

    if minimum_duration < 1:
        raise ValueError(
            f'minimum_duration must be at least 1, but got {minimum_duration}',
        )

    if maximum_duration is not None:
        if maximum_duration < 1:
            raise ValueError(
                f'maximum_duration must be at least 1, but got {maximum_duration}',
            )
        if maximum_duration < minimum_duration:
            raise ValueError(
                f'maximum_duration must be >= minimum_duration, but got '
                f'maximum_duration={maximum_duration} < minimum_duration={minimum_duration}',
            )

    if len(pupil) == 0:
        return Events(name=name, onsets=[], offsets=[])

    # Stage 1: Flagging
    flagged = np.isnan(pupil) | (pupil == 0)

    abs_diff = np.abs(np.diff(pupil))
    # NaN diffs (from NaN pupil values) should not trigger delta flagging on their own;
    # those samples are already flagged above.
    valid_diffs = abs_diff[~np.isnan(abs_diff)]

    if delta is None and len(valid_diffs) > 0:
        delta = 5.0 * np.nanpercentile(valid_diffs, 95)

    if delta is not None and len(valid_diffs) > 0:
        # Flag sample i+1 if abs(pupil[i+1] - pupil[i]) > delta
        diff_flag = np.zeros(len(pupil), dtype=bool)
        exceeds = abs_diff > delta
        # Replace NaN comparisons with False
        exceeds = np.where(np.isnan(abs_diff), False, exceeds)
        diff_flag[1:] |= exceeds
        diff_flag[:-1] |= exceeds
        flagged |= diff_flag

    # Stage 2: Island absorption
    if max_value_run > 0:
        flagged = _absorb_islands(flagged, max_value_run, nas_around_run)

    # Group consecutive flagged samples into events
    candidate_indices = np.where(flagged)[0]

    if len(candidate_indices) == 0:
        return Events(name=name, onsets=[], offsets=[])

    candidates = consecutive(arr=candidate_indices)

    # Filter by duration using timestep units (ms when timesteps are in ms)
    filtered = []
    for c in candidates:
        duration = timesteps[c[-1]] - timesteps[c[0]]
        if duration < minimum_duration:
            continue
        if maximum_duration is not None and duration > maximum_duration:
            continue
        filtered.append(c)
    candidates = filtered

    if len(candidates) == 0:
        return Events(name=name, onsets=[], offsets=[])

    onsets = timesteps[[c[0] for c in candidates]].flatten()
    offsets = timesteps[[c[-1] for c in candidates]].flatten()

    return Events(name=name, onsets=onsets, offsets=offsets)


def _absorb_islands(
        flagged: np.ndarray,
        max_value_run: int,
        nas_around_run: int,
) -> np.ndarray:
    """Absorb short unflagged runs surrounded by flagged samples.

    Parameters
    ----------
    flagged: np.ndarray
        Boolean array of flagged samples.
    max_value_run: int
        Maximum length of an unflagged run to absorb.
    nas_around_run: int
        Minimum flagged samples required on each side.

    Returns
    -------
    np.ndarray
        Updated boolean flagged array with absorbed islands.
    """
    flagged = flagged.copy()
    n = len(flagged)

    # Find runs of unflagged samples
    unflagged_indices = np.where(~flagged)[0]
    if len(unflagged_indices) == 0:
        return flagged

    runs = consecutive(arr=unflagged_indices)

    for run in runs:
        run_len = len(run)
        if run_len > max_value_run:
            continue

        start = run[0]
        end = run[-1]

        # Count flagged samples before this run
        before_start = max(0, start - nas_around_run)
        flagged_before = np.sum(flagged[before_start:start])

        # Count flagged samples after this run
        after_end = min(n, end + 1 + nas_around_run)
        flagged_after = np.sum(flagged[end + 1:after_end])

        if flagged_before >= nas_around_run and flagged_after >= nas_around_run:
            flagged[run] = True

    return flagged
