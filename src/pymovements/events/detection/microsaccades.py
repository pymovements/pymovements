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
"""Provides the implementation for the Engbert microsaccades algorithm."""
from __future__ import annotations

from collections.abc import Sized

import numpy as np

from pymovements._utils import _checks
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events
from pymovements.gaze.transforms_numpy import consecutive


@register_event_detection
def microsaccades(
        velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        minimum_duration: int = 6,
        threshold: np.ndarray | tuple[float, float] | str = 'engbert2015',
        threshold_factor: float = 6,
        minimum_threshold: float = 1e-10,
        include_nan: bool = False,
        name: str = 'saccade',
) -> Events:
    """Detect micro-saccades from velocity gaze sequence.

    This algorithm has a noise-adaptive velocity threshold parameter, which can also be set
    explicitly.

    The implementation and its default parameter values are based on the description from
    Engbert & Kliegl :cite:p:`EngbertKliegl2003` and is adopted from the Microsaccade Toolbox 0.9
    originally implemented in R :cite:p:`Engbert2015`.

    Parameters
    ----------
    velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray
        shape (N, 2)
        x and y velocities of N samples in chronological order
    timesteps: list[int] | np.ndarray | None
        shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed. (default: None)
    minimum_duration: int
        Minimum saccade duration. The duration is specified in the units used in ``timesteps``.
         If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
         (default: 6)
    threshold: np.ndarray | tuple[float, float] | str
        If tuple of floats then use this as explicit elliptic threshold. If str, then use
        a data-driven velocity threshold method. See :func:`~events.engbert.compute_threshold` for
        a reference of valid methods. (default: 'engbert2015')
    threshold_factor: float
        factor for relative velocity threshold computation. (default: 6)
    minimum_threshold: float
        minimal threshold value. Raises ValueError if calculated threshold is too low.
        (default: 1e-10)
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (np.nan)
        (default: False)
    name: str
        Name for detected events in Events. (default: 'saccade')

    Returns
    -------
    Events
        A dataframe with detected saccades as rows.

    Raises
    ------
    ValueError
        If `threshold` value is below `min_threshold` value.
        If passed `threshold` is either not two-dimensional or not a supported method.

    Examples
    --------
    Create a synthetic velocity signal representing micro-saccades.

    >>> import numpy as np
    >>> from pymovements.synthetic import step_function
    >>> from pymovements.gaze import from_numpy
    >>> velocities = step_function(
    ...     length=200, steps=[2, 5, 9, 111, 150],
    ...     values=[(0.5, 0.5), (11.0, 12.0), (0.2, 0.2), (10.0, 20.0), (0.1, 0.1)],
    ...     start_value=(0., 0.),
    ...     noise=0.001,
    ... )
    >>> velocities.shape
    (200, 2)

    Apply event detection algorithm on numpy array:

    >>> microsaccades(velocities)
    shape: (1, 4)
    ┌─────────┬───────┬────────┬──────────┐
    │ name    ┆ onset ┆ offset ┆ duration │
    │ ---     ┆ ---   ┆ ---    ┆ ---      │
    │ str     ┆ i64   ┆ i64    ┆ i64      │
    ╞═════════╪═══════╪════════╪══════════╡
    │ saccade ┆ 2     ┆ 199    ┆ 197      │
    └─────────┴───────┴────────┴──────────┘

    Run fixation detection with custom parameters:

    >>> microsaccades(velocities, minimum_duration=10, threshold=0.1)
    shape: (1, 4)
    ┌─────────┬───────┬────────┬──────────┐
    │ name    ┆ onset ┆ offset ┆ duration │
    │ ---     ┆ ---   ┆ ---    ┆ ---      │
    │ str     ┆ i64   ┆ i64    ┆ i64      │
    ╞═════════╪═══════╪════════╪══════════╡
    │ saccade ┆ 111   ┆ 149    ┆ 38       │
    └─────────┴───────┴────────┴──────────┘

    We can also apply the detection on a :py:class:`~pymovements.Gaze` object.

    >>> from pymovements import Experiment
    >>> gaze = from_numpy(
    ...    velocity=velocities.T,
    ...    time=np.arange(len(velocities)),
    ... )
    >>> gaze  # doctest: +SKIP
    shape: (200, 2)
    ┌──────┬────────────────────────┐
    │ time ┆ velocity               │
    │ ---  ┆ ---                    │
    │ i64  ┆ list[f64]              │
    ╞══════╪════════════════════════╡
    │ 0    ┆ [-0.000628, 0.000055]  │
    │ 1    ┆ [-0.000818, -0.000694] │
    │ 2    ┆ [0.50097, 0.498064]    │
    │ 3    ┆ [0.500475, 0.500865]   │
    │ 4    ┆ [0.498559, 0.499092]   │
    │ …    ┆ …                      │
    │ 195  ┆ [0.100042, 0.100217]   │
    │ 196  ┆ [0.099627, 0.099812]   │
    │ 197  ┆ [0.101374, 0.098537]   │
    │ 198  ┆ [0.099669, 0.100079]   │
    │ 199  ┆ [0.098491, 0.101448]   │
    └──────┴────────────────────────┘

    Run saccade detection by using the :py:meth:`~pymovements.Gaze.detect` method.

    >>> gaze.detect('microsaccades')
    >>> gaze.events
    shape: (1, 4)
    ┌─────────┬───────┬────────┬──────────┐
    │ name    ┆ onset ┆ offset ┆ duration │
    │ ---     ┆ ---   ┆ ---    ┆ ---      │
    │ str     ┆ i64   ┆ i64    ┆ i64      │
    ╞═════════╪═══════╪════════╪══════════╡
    │ saccade ┆ 2     ┆ 199    ┆ 197      │
    └─────────┴───────┴────────┴──────────┘

    Passing parameters to :py:meth:`~pymovements.Gaze.detect`:

    >>> gaze.detect('microsaccades', minimum_duration=10, threshold=0.1, name='microsaccade')
    >>> gaze.events.filter_by_name('microsaccade')
    shape: (1, 4)
    ┌──────────────┬───────┬────────┬──────────┐
    │ name         ┆ onset ┆ offset ┆ duration │
    │ ---          ┆ ---   ┆ ---    ┆ ---      │
    │ str          ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════════╪═══════╪════════╪══════════╡
    │ microsaccade ┆ 111   ┆ 149    ┆ 38       │
    └──────────────┴───────┴────────┴──────────┘

    """
    velocities = np.array(velocities)

    if timesteps is None:
        timesteps = np.arange(len(velocities), dtype=np.int64)
    timesteps = np.array(timesteps)
    _checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    if isinstance(threshold, str):
        threshold = compute_threshold(velocities, method=threshold)
    else:
        if isinstance(threshold, Sized) and len(threshold) != 2:
            raise ValueError('threshold must be either string or two-dimensional')
        threshold = np.array(threshold)

    if (threshold < minimum_threshold).any():
        raise ValueError(
            'threshold does not provide enough variance as required by min_threshold'
            f' ({threshold} < {minimum_threshold})',
        )

    # Radius of elliptic threshold.
    radius = threshold * threshold_factor

    # If value is greater than 1, point lies outside the ellipse.
    candidate_mask = np.greater(np.sum(np.power(velocities / radius, 2), axis=1), 1)

    # Add nans to candidates if desired.
    if include_nan:
        candidate_mask = np.logical_or(candidate_mask, np.isnan(velocities).any(axis=1))

    # Get indices of true values in candidate mask.
    candidate_indices = np.where(candidate_mask)[0]

    # Get all saccade candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    # Remove leading and trailing nan values from candidates.
    if include_nan:
        candidates = filter_candidates_remove_nans(candidates=candidates, values=velocities)

    # Filter all candidates by minimum duration.
    candidates = [
        candidate for candidate in candidates
        if len(candidate) > 0
        and timesteps[candidate[-1]] - timesteps[candidate[0]] >= minimum_duration
    ]

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate_indices[0] for candidate_indices in candidates]].flatten()
    # Offset of each event candidate is last event in candidate indices.
    offsets = timesteps[[candidate_indices[-1] for candidate_indices in candidates]].flatten()

    # Create event dataframe from onsets and offsets.
    events = Events(name=name, onsets=onsets, offsets=offsets)
    return events


def compute_threshold(arr: np.ndarray, method: str = 'engbert2015') -> np.ndarray:
    """Determine threshold by computing variation.

    The following methods are supported:

    - `std`: This is the channel-wise standard deviation.
    - `mad`: This is the channel-wise median absolute deviation.
    - `engbert2003`: This is the threshold method as described in :cite:p:`EngbertKliegl2003`.
    - `engbert2015`: This is the threshold method as described in :cite:p:`Engbert2015`.

    Parameters
    ----------
    arr : np.ndarray
        Array for which threshold is to be computed.
    method : str
        Method for threshold computation. (default: 'engbert2015')

    Returns
    -------
    np.ndarray
        Threshold values for horizontal and vertical direction.

    Raises
    ------
    ValueError
        If passed method is not supported.
    """
    if method == 'std':
        thx = np.nanstd(arr[:, 0])
        thy = np.nanstd(arr[:, 1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(arr[:, 0] - np.nanmedian(arr[:, 0])))
        thy = np.nanmedian(np.absolute(arr[:, 1] - np.nanmedian(arr[:, 1])))

    elif method == 'engbert2003':
        thx = np.sqrt(
            np.nanmedian(np.power(arr[:, 0], 2)) - np.power(np.nanmedian(arr[:, 0]), 2),
        )
        thy = np.sqrt(
            np.nanmedian(np.power(arr[:, 1], 2)) - np.power(np.nanmedian(arr[:, 1]), 2),
        )

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(arr[:, 0] - np.nanmedian(arr[:, 0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(arr[:, 1] - np.nanmedian(arr[:, 1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(f'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])
