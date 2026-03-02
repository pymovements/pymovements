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
"""Provides detection of out-of-screen gaze samples."""

from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events.detection._library import register_event_detection
from pymovements.events.events import Events
from pymovements.gaze.transforms_numpy import consecutive


@register_event_detection
def out_of_screen(
    pixels: list[list[float]] | list[tuple[float, float]] | np.ndarray,
    *,
    x_max: float,
    y_max: float,
    x_min: float = 0,
    y_min: float = 0,
    timesteps: list[int] | np.ndarray | None = None,
    name: str = 'out_of_screen',
) -> Events:
    """Detect gaze samples with pixel coordinates outside of screen boundaries.

    The algorithm classifies each gaze sample as out-of-screen if the x or y pixel coordinate
    falls outside the given screen boundaries. Consecutive out-of-screen samples are merged
    into single events.

    Parameters
    ----------
    pixels: list[list[float]] | list[tuple[float, float]] | np.ndarray
        shape (N, 2)
        Continuous 2D pixel coordinate time series. The first column is the x coordinate
        and the second column is the y coordinate.
    x_max: float
        Maximum valid x pixel coordinate (exclusive). For a 1920px wide screen, set to 1920.
    y_max: float
        Maximum valid y pixel coordinate (exclusive). For a 1080px tall screen, set to 1080.
    x_min: float
        Minimum valid x pixel coordinate (inclusive). (default: 0)
    y_min: float
        Minimum valid y pixel coordinate (inclusive). (default: 0)
    timesteps: list[int] | np.ndarray | None
        shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed. (default: None)
    name: str
        Name for detected events in Events. (default: 'out_of_screen')

    Returns
    -------
    Events
        A dataframe with detected out-of-screen events as rows.

    Raises
    ------
    ValueError
        If pixels is None or does not have shape (N, 2).
        If x_min >= x_max.
        If y_min >= y_max.
    """
    pixels = np.array(pixels)

    _checks.check_shapes(pixels=pixels)

    if x_min >= x_max:
        raise ValueError(
            f'x_min must be less than x_max, but got x_min={x_min} and x_max={x_max}',
        )
    if y_min >= y_max:
        raise ValueError(
            f'y_min must be less than y_max, but got y_min={y_min} and y_max={y_max}',
        )

    if timesteps is None:
        timesteps = np.arange(len(pixels), dtype=np.int64)
    timesteps = np.array(timesteps)
    _checks.check_is_length_matching(pixels=pixels, timesteps=timesteps)

    # A sample is out-of-screen if x or y is outside the screen boundaries.
    x = pixels[:, 0]
    y = pixels[:, 1]

    out_of_screen_mask = (x < x_min) | (x >= x_max) | (y < y_min) | (y >= y_max)

    # Get indices of out-of-screen samples.
    candidate_indices = np.where(out_of_screen_mask)[0]

    if len(candidate_indices) == 0:
        return Events(name=name, onsets=[], offsets=[])

    # Group consecutive indices into events.
    candidates = consecutive(arr=candidate_indices)

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate[0] for candidate in candidates]].flatten()
    # Offset of each event candidate is last index in candidate indices.
    offsets = timesteps[[candidate[-1] for candidate in candidates]].flatten()

    return Events(name=name, onsets=onsets, offsets=offsets)
