# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Provides the implementation for I-DT algorithm."""
from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events._utils._filters import events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection._library import register_event_detection
from pymovements.events.events import Events


def dispersion(positions: list[list[float]] | np.ndarray) -> float:
    """Compute the dispersion of a group of consecutive points in a 2D position time series.

    The dispersion is defined as the sum of the differences between
    the points' maximum and minimum x and y values

    Parameters
    ----------
    positions: list[list[float]] | np.ndarray
        Continuous 2D position time series.

    Returns
    -------
    float
        Dispersion of the group of points.
    """
    return sum(np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0))


@register_event_detection
def idt(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        minimum_duration: int = 100,
        dispersion_threshold: float = 1.0,
        include_nan: bool = False,
        name: str = 'fixation',
) -> Events:
    """Fixation identification based on dispersion threshold (I-DT).

    The algorithm identifies fixations by grouping consecutive points
    within a maximum separation (dispersion) threshold and a minimum duration threshold.
    The algorithm uses a moving window to check the dispersion of the points in the window.
    If the dispersion is below the threshold, the window represents a fixation,
    and the window is expanded until the dispersion is above threshold.

    The implementation and its default parameter values are based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: list[list[float]] | list[tuple[float, float]] | np.ndarray
        shape (N, 2)
        Continuous 2D position time series
    timesteps: list[int] | np.ndarray | None
        shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed. (default: None)
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
        (default: 100)
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive samples to be identified as fixation.
        (default: 1.0)
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (np.nan).
        (default: False)
    name: str
        Name for detected events in Events. (default: 'fixation')

    Returns
    -------
    Events
        A dataframe with detected fixations as rows.

    Raises
    ------
    TypeError
        If minimum_duration is not of type ``int`` or timesteps
    ValueError
        If positions is not shaped (N, 2)
        If dispersion_threshold is not greater than 0
        If duration_threshold is not greater than 0
    """
    positions = np.array(positions)

    _checks.check_shapes(positions=positions)

    if timesteps is None:
        timesteps = np.arange(len(positions), dtype=np.int64)
    timesteps = np.array(timesteps).flatten()

    # Check that timesteps are integers or are floats without a fractional part.
    timesteps_int = timesteps.astype(int)
    if np.any((timesteps - timesteps_int) != 0):
        raise TypeError('timesteps must be of type int')
    timesteps = timesteps_int

    _checks.check_is_length_matching(positions=positions, timesteps=timesteps)

    if dispersion_threshold <= 0:
        raise ValueError('dispersion_threshold must be greater than 0')
    if minimum_duration <= 0:
        raise ValueError('minimum_duration must be greater than 0')
    if not isinstance(minimum_duration, int):
        raise TypeError(
            'minimum_duration must be of type int'
            f' but is of type {type(minimum_duration)}',
        )

    onsets = []
    offsets = []

    # Infer minimum duration in number of samples.
    # This implementation is currently very restrictive.
    # It requires that the interval between timesteps is constant.
    # It requires that the minimum duration is divisible by the constant interval between timesteps.
    timesteps_diff = np.diff(timesteps)
    if not np.all(timesteps_diff == timesteps_diff[0]):
        raise ValueError('interval between timesteps must be constant')
    if not minimum_duration % timesteps_diff[0] == 0:
        raise ValueError(
            'minimum_duration must be divisible by the constant interval between timesteps',
        )
    if (minimum_sample_duration := int(minimum_duration // timesteps_diff[0])) < 2:
        raise ValueError('minimum_duration must be longer than the equivalent of 2 samples')

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = minimum_sample_duration

    while win_start < len(timesteps) and win_end <= len(timesteps):

        # Initialize window over first points to cover the duration threshold.
        # This automatically extends the window to the specified minimum event duration.
        win_end = max(win_start + minimum_sample_duration, win_end)
        win_end = min(win_end, len(timesteps))
        if win_end - win_start < minimum_sample_duration:
            break

        if dispersion(positions[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold.
            while dispersion(positions[win_start:win_end]) < dispersion_threshold:
                # break if we reach end of input data
                if win_end == len(timesteps):
                    break

                win_end += 1

            # check for np.nan values
            if np.sum(np.isnan(positions[win_start:win_end - 1])) > 0:
                tmp_candidates = [np.arange(win_start, win_end - 1, 1)]
                tmp_candidates = filter_candidates_remove_nans(
                    candidates=tmp_candidates,
                    values=positions,
                )
                # split events if include_nan == False
                if not include_nan:
                    tmp_candidates = events_split_nans(
                        candidates=tmp_candidates,
                        values=positions,
                    )

                # Filter all candidates by minimum duration.
                tmp_candidates = [
                    candidate for candidate in tmp_candidates
                    if len(candidate) >= minimum_sample_duration
                ]
                for candidate in tmp_candidates:
                    onsets.append(timesteps[candidate[0]])
                    offsets.append(timesteps[candidate[-1]])

            else:
                # Note a fixation at the centroid of the window points.

                onsets.append(timesteps[win_start])
                offsets.append(timesteps[win_end - 1])

            # Remove window points from points.
            # Initialize new window excluding the previous window
            win_start = win_end
        else:
            # Remove first point from points.
            # Move window start one step further without modifying window end.
            win_start += 1

    # Create proper flat numpy arrays.
    onsets_arr = np.array(onsets).flatten()
    offsets_arr = np.array(offsets).flatten()

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)
    return events
