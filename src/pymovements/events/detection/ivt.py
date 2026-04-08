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
"""Provides the implementation of the I-VT algorithm."""
from __future__ import annotations

import numpy
import polars

from pymovements._utils import _checks
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events
from pymovements.gaze.transforms_numpy import consecutive
from pymovements.gaze.transforms_numpy import norm


@register_event_detection
def ivt(
        velocities: list[list[float]] | list[tuple[float, float]] | numpy.ndarray | polars.Series,
        timesteps: list[int] | numpy.ndarray | polars.Series | None = None,
        minimum_duration: int = 100,
        velocity_threshold: float = 20.0,
        include_nan: bool = False,
        name: str = 'fixation',
) -> Events:
    """Identification of fixations based on a velocity-threshold (I-VT).

    The algorithm classifies each point as a fixation if the velocity is below
    the given velocity threshold. Consecutive fixation points are merged into
    one fixation.

    The implementation and its default parameter values are based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    velocities: list[list[float]] | list[tuple[float, float]] | numpy.ndarray | polars.Series
        shape (N, 2)
        Corresponding continuous 2D velocity time series.
    timesteps: list[int] | numpy.ndarray | polars.Series | None
        shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed. (default: None)
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
        If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
        (default: 100)
    velocity_threshold: float
        Threshold for a point to be classified as a fixation. If the
        velocity is below the threshold, the point is classified as a fixation. (default: 20.0)
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (numpy.nan)
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
        If velocities is a polars Series and dtype not List
    ValueError
        If velocities is None
        If velocities does not have shape (N, 2)
        If velocity threshold is None.
        If velocity threshold is not greater than 0.

    Examples
    --------
    Create a synthetic velocity signal representing low-velocity fixations.

    >>> import numpy as np
    >>> from pymovements.synthetic import step_function
    >>> from pymovements.gaze import from_numpy
    >>> velocities = step_function(
    ...     length=200, steps=[2, 5, 9, 111, 150],
    ...     values=[(10., 20.), (20., 30.), (0., 0.), (20., 20.), (0., 0.)],
    ...     start_value=(0., 0.),
    ... )
    >>> velocities.shape
    (200, 2)

    Apply event detection algorithm on numpy array:

    >>> ivt(velocities)
    shape: (1, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 110    ┆ 101      │
    └──────────┴───────┴────────┴──────────┘

    Run fixation detection with custom parameters:

    >>> ivt(velocities, minimum_duration = 50, velocity_threshold=30)
    shape: (1, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 199    ┆ 190      │
    └──────────┴───────┴────────┴──────────┘

    Polars series are also supported as input. Let's create a nested position series from our numpy
    array:

    >>> df = polars.from_numpy(velocities, schema=['x', 'y'])
    >>> velocity_series = df.select(polars.concat_list(('x', 'y')).alias('velocity'))['velocity']
    >>> velocity_series
    shape: (200,)
    Series: 'velocity' [list[f64]]
    [
        [0.0, 0.0]
        [0.0, 0.0]
        [10.0, 20.0]
        [10.0, 20.0]
        [10.0, 20.0]
        …
        [0.0, 0.0]
        [0.0, 0.0]
        [0.0, 0.0]
        [0.0, 0.0]
        [0.0, 0.0]
    ]

    Apply event detection algorithm on polars series:

    >>> ivt(velocity_series)
    shape: (1, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 110    ┆ 101      │
    └──────────┴───────┴────────┴──────────┘

    We can also apply the detection on a :py:class:`~pymovements.Gaze` object.

    >>> from pymovements import Experiment
    >>> gaze = from_numpy(
    ...    velocity=velocities.T,
    ...    time=np.arange(len(velocities)),
    ... )
    >>> gaze
    shape: (200, 2)
    ┌──────┬──────────────┐
    │ time ┆ velocity     │
    │ ---  ┆ ---          │
    │ i64  ┆ list[f64]    │
    ╞══════╪══════════════╡
    │ 0    ┆ [0.0, 0.0]   │
    │ 1    ┆ [0.0, 0.0]   │
    │ 2    ┆ [10.0, 20.0] │
    │ 3    ┆ [10.0, 20.0] │
    │ 4    ┆ [10.0, 20.0] │
    │ …    ┆ …            │
    │ 195  ┆ [0.0, 0.0]   │
    │ 196  ┆ [0.0, 0.0]   │
    │ 197  ┆ [0.0, 0.0]   │
    │ 198  ┆ [0.0, 0.0]   │
    │ 199  ┆ [0.0, 0.0]   │
    └──────┴──────────────┘

    Run fixation detection by using the :py:meth:`~pymovements.Gaze.detect` method.

    >>> gaze.detect('ivt')
    >>> gaze.events
    shape: (1, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 110    ┆ 101      │
    └──────────┴───────┴────────┴──────────┘

    Passing parameters to :py:meth:`~pymovements.Gaze.detect`:

    >>> gaze.detect('ivt', minimum_duration = 50, velocity_threshold=30, name='fixation_ivt')
    >>> gaze.events.filter_by_name('fixation_ivt')
    shape: (1, 4)
    ┌──────────────┬───────┬────────┬──────────┐
    │ name         ┆ onset ┆ offset ┆ duration │
    │ ---          ┆ ---   ┆ ---    ┆ ---      │
    │ str          ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════════╪═══════╪════════╪══════════╡
    │ fixation_ivt ┆ 9     ┆ 199    ┆ 190      │
    └──────────────┴───────┴────────┴──────────┘
    """
    numeric_dtypes = polars.datatypes.FloatType, polars.datatypes.IntegerType
    if isinstance(velocities, polars.Series):
        if not isinstance(velocities.dtype, polars.List):
            raise TypeError(f'velocities dtype must be List but is {velocities.dtype}')
        if not (velocities.list.len() == 2).all():
            list_lengths = velocities.list.len().unique().to_list()
            raise ValueError(f'velocities must be 2D list but list lengths are: {list_lengths}')
        velocities = numpy.vstack([velocities.list.get(0), velocities.list.get(1)]).transpose()
    velocities = numpy.array(velocities)
    _checks.check_shapes(velocities=velocities)

    if velocity_threshold is None:
        raise ValueError('velocity threshold must not be None')
    if velocity_threshold <= 0:
        raise ValueError('velocity threshold must be greater than 0')

    if isinstance(timesteps, polars.Series):
        if not isinstance(timesteps.dtype, numeric_dtypes):
            raise TypeError(f'timesteps dtype must be float or int but is {timesteps.dtype}')
        timesteps = timesteps.to_numpy()
    elif timesteps is not None:
        timesteps = numpy.array(timesteps)
    else:
        timesteps = numpy.arange(len(velocities), dtype=numpy.int64)
    timesteps = numpy.array(timesteps).flatten()
    _checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    # Get all indices with norm-velocities below threshold.
    velocity_norm = norm(velocities, axis=1)
    candidate_mask = velocity_norm < velocity_threshold

    # Add nans to candidates if desired.
    if include_nan:
        candidate_mask = numpy.logical_or(candidate_mask, numpy.isnan(velocities).any(axis=1))

    # Get indices of true values in candidate mask.
    candidate_indices = numpy.where(candidate_mask)[0]

    # Get all fixation candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    # Remove leading and trailing nan values from candidates.
    if include_nan:
        candidates = filter_candidates_remove_nans(candidates=candidates, values=velocities)

    # Remove empty candidates.
    candidates = [candidate for candidate in candidates if len(candidate) > 0]

    # Filter all candidates by minimum duration.
    candidates = [
        candidate for candidate in candidates
        if timesteps[candidate[-1]] - timesteps[candidate[0]] >= minimum_duration
    ]

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate_indices[0] for candidate_indices in candidates]].flatten()
    # Offset of each event candidate is last event in candidate indices.
    offsets = timesteps[[candidate_indices[-1] for candidate_indices in candidates]].flatten()

    # Create event dataframe from onsets and offsets.
    events = Events(name=name, onsets=onsets, offsets=offsets)
    return events
