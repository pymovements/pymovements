# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Module to create a Gaze from a numpy array."""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pymovements._utils import _checks
from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze


def from_numpy(
        samples: np.ndarray | None = None,
        experiment: Experiment | None = None,
        events: Events | None = None,
        *,
        trial: np.ndarray | None = None,
        time: np.ndarray | None = None,
        pixel: np.ndarray | None = None,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        distance: np.ndarray | None = None,
        schema: list[str] | None = None,
        orient: Literal['col', 'row'] = 'col',
        trial_columns: str | list[str] | None = None,
        time_column: str | None = None,
        time_unit: str | None = None,
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
        distance_column: str | None = None,
        data: np.ndarray | None = None,
) -> Gaze:
    """Get a :py:class:`~pymovements.Gaze` from a numpy array.

    There are two mutually exclusive ways of conversion.

    **Single data array**: Pass a single numpy array via `data` and specify its schema and
    orientation. You can then additionally pass column specifiers, e.g. `time_column` and
    `position_columns`.

    **Column specific arrays**: For each type of signal, you can pass the numpy array explicitly,
    e.g. `position` or `velocity`. You must not pass `samples` or any column list specifiers using
    this method.

    Parameters
    ----------
    samples: np.ndarray | None
        Two-dimensional samples data represented as a numpy ndarray. (default: None)
    experiment: Experiment | None
        The experiment definition. (default: None)
    events: Events | None
        A dataframe of events in the gaze signal. (default: None)
    trial: np.ndarray | None
        Array of trial identifiers for each timestep. (default: None)
    time: np.ndarray | None
        Array of timestamps. (default: None)
    pixel: np.ndarray | None
        Array of gaze pixel positions. (default: None)
    position: np.ndarray | None
        Array of gaze positions in degrees of visual angle. (default: None)
    velocity: np.ndarray | None
        Array of gaze velocities in degrees of visual angle per second. (default: None)
    acceleration: np.ndarray | None
        Array of gaze accelerations in degrees of visual angle per square second. (default: None)
    distance: np.ndarray | None
        Array of eye-to-screen distances in millimiters. (default: None)
    schema: list[str] | None
        A list of column names. (default: None)
    orient: Literal['col', 'row']
        Whether to interpret the two-dimensional samples data as columns or as rows.
        (default: 'col')
    trial_columns: str | list[str] | None
        The name of the trial columns in the samples data frame. If the list is empty or None,
        the samples data frame is assumed to contain only one trial. If the list is not empty,
        the samples data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the samples data frame. (default: None)
    time_unit: str | None
        The unit of the timestamps in the timestamp column in the samples data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. If time_unit is None, milliseconds are assumed. (default: None)
    pixel_columns: list[str] | None
        The name of the pixel position columns in the samples data frame. (default: None)
    position_columns: list[str] | None
        The name of the dva position columns in the samples data frame. (default: None)
    velocity_columns: list[str] | None
        The name of the dva velocity columns in the samples data frame. (default: None)
    acceleration_columns: list[str] | None
        The name of the dva acceleration columns in the samples data frame. (default: None)
    distance_column: str | None
        The name of the column containing eye-to-screen distance in millimiters for each sample
        in the samples data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken from
        the experiment definition. (default: None)
    data: np.ndarray | None
        Two-dimensional samples data represented as a numpy ndarray. (default: None)
        .. deprecated:: v0.23.0
        Please use ``samples`` instead. This field will be removed in v0.28.0.

    Returns
    -------
    Gaze
        Returns Gaze object with data read from numpy array.

    Examples
    --------
    Creating an example numpy array with 4 columns and 100 rows. We call this layout column
    orientation.
    >>> import numpy as np
    >>> import pymovements as pm
    >>>
    >>> arr = np.zeros((3, 100))
    >>> arr.shape
    (3, 100)

    Specifying the underlying schema:
    >>> schema = ['t', 'x', 'y']

    Pass the array as ``samples`` to ``pm.gaze.from_numpy()``, by specifying schema and components.
    >>> gaze = pm.gaze.from_numpy(
    ...     samples=arr,
    ...     schema=schema,
    ...     time_column='t',
    ...     time_unit='ms',
    ...     position_columns=['x', 'y'],
    ...     orient='col',
    ... )
    >>> gaze.samples
    shape: (100, 2)
    ┌──────┬────────────┐
    │ time ┆ position   │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ …    ┆ …          │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    └──────┴────────────┘

    Use the ``orient`` keyword argument to specify the layout of your array.
    >>> arr.T.shape
    (100, 3)

    >>> gaze = pm.gaze.from_numpy(
    ...     samples=arr.T,
    ...     schema=schema,
    ...     time_column='t',
    ...     time_unit='ms',
    ...     position_columns=['x', 'y'],
    ...     orient='row',
    ... )
    >>> gaze.samples
    shape: (100, 2)
    ┌──────┬────────────┐
    │ time ┆ position   │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ …    ┆ …          │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    └──────┴────────────┘

    Pass the samples explicitly via the specific keyword arguments, without having to specify a
    schema.
    >>> gaze = pm.gaze.from_numpy(
    ...     time=arr[0],
    ...     time_unit='ms',
    ...     position=arr[[1, 2]],
    ...     orient='col',
    ... )
    >>> gaze.samples
    shape: (100, 2)
    ┌──────┬────────────┐
    │ time ┆ position   │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ …    ┆ …          │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    │ 0    ┆ [0.0, 0.0] │
    └──────┴────────────┘
    """
    if data is not None:
        warnings.warn(
            DeprecationWarning(
                "from_numpy() argument 'data' is deprecated since version v0.23.0. "
                "Please use argument 'samples' instead. "
                'This argument will be removed in v0.28.0.',
            ),
        )
        _checks.check_is_mutual_exclusive(samples=samples, data=data)
        samples = data

    # Either samples or {time, pixel, position, velocity, acceleration} must be None.
    _checks.check_is_mutual_exclusive(samples=samples, time=time)
    _checks.check_is_mutual_exclusive(samples=samples, pixel=pixel)
    _checks.check_is_mutual_exclusive(samples=samples, position=position)
    _checks.check_is_mutual_exclusive(samples=samples, velocity=velocity)
    _checks.check_is_mutual_exclusive(samples=samples, acceleration=acceleration)
    _checks.check_is_mutual_exclusive(samples=samples, distance=distance)

    if samples is not None:
        return Gaze(
            samples=pl.from_numpy(data=samples, schema=schema, orient=orient),
            experiment=experiment,
            events=events,
            trial_columns=trial_columns,
            time_column=time_column,
            time_unit=time_unit,
            pixel_columns=pixel_columns,
            position_columns=position_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
            distance_column=distance_column,
        )

    # Initialize with an empty DataFrame, as every column specifier could be None.
    sample_components: list[pl.DataFrame] = [pl.DataFrame()]

    trial_columns = None
    if trial is not None:
        sample_component = pl.from_numpy(data=trial, schema=['trial'], orient=orient)
        sample_components.append(sample_component)
        trial_columns = 'trial'

    time_column = None
    if time is not None:
        sample_component = pl.from_numpy(data=time, schema=['time'], orient=orient)
        sample_components.append(sample_component)
        time_column = 'time'

    pixel_columns = None
    if pixel is not None:
        sample_component = pl.from_numpy(
            data=pixel, orient=orient,
        ).select(
            pl.all().name.prefix('pixel_'),
        )
        sample_components.append(sample_component)
        pixel_columns = sample_component.columns

    position_columns = None
    if position is not None:
        sample_component = pl.from_numpy(
            data=position, orient=orient,
        ).select(
            pl.all().name.prefix('position_'),
        )
        sample_components.append(sample_component)
        position_columns = sample_component.columns

    velocity_columns = None
    if velocity is not None:
        sample_component = pl.from_numpy(
            data=velocity, orient=orient,
        ).select(
            pl.all().name.prefix('velocity_'),
        )
        sample_components.append(sample_component)
        velocity_columns = sample_component.columns

    acceleration_columns = None
    if acceleration is not None:
        sample_component = pl.from_numpy(data=acceleration, orient=orient)
        sample_component = sample_component.select(pl.all().name.prefix('acceleration_'))
        sample_components.append(sample_component)
        acceleration_columns = sample_component.columns

    distance_column = None
    if distance is not None:
        sample_component = pl.from_numpy(data=distance, schema=['distance'], orient=orient)
        sample_components.append(sample_component)
        distance_column = 'distance'

    samples = pl.concat(sample_components, how='horizontal')
    return Gaze(
        samples=samples,
        experiment=experiment,
        events=events,
        time_column=time_column,
        time_unit=time_unit,
        trial_columns=trial_columns,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        distance_column=distance_column,
    )


def from_pandas(
        samples: pd.DataFrame,
        experiment: Experiment | None = None,
        events: Events | None = None,
        *,
        trial_columns: str | list[str] | None = None,
        time_column: str | None = None,
        time_unit: str | None = None,
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
        distance_column: str | None = None,
        data: pd.DataFrame | None = None,
) -> Gaze:
    """Get a :py:class:`~pymovements.Gaze` from a pandas DataFrame.

    Parameters
    ----------
    samples: pd.DataFrame
        Gaze samples represented as a pandas DataFrame.
    experiment : Experiment | None
        The experiment definition. (default: None)
    events: Events | None
        A dataframe of events in the gaze signal. (default: None)
    trial_columns: str | list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. (default: None)
    time_unit: str | None
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. If time_unit is None, milliseconds are assumed. (default: None)
    pixel_columns: list[str] | None
        The name of the pixel position columns in the input data frame. (default: None)
    position_columns: list[str] | None
        The name of the dva position columns in the input data frame. (default: None)
    velocity_columns: list[str] | None
        The name of the dva velocity columns in the input data frame. (default: None)
    acceleration_columns: list[str] | None
        The name of the dva acceleration columns in the input data frame. (default: None)
    distance_column: str | None
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken from
        the experiment definition. (default: None)
    data: pd.DataFrame | None
        Gaze samples represented as a pandas DataFrame.
        .. deprecated:: v0.23.0
        Please use ``samples`` instead. This field will be removed in v0.28.0. (default: None)

    Returns
    -------
    Gaze
        Returns initialized gaze object with data read from pandas data frame.
    """
    if data is not None:
        warnings.warn(
            DeprecationWarning(
                "from_pandas() argument 'data' is deprecated since version v0.23.0. "
                "Please use argument 'samples' instead. "
                'This argument will be removed in v0.28.0.',
            ),
        )
        _checks.check_is_mutual_exclusive(samples=samples, data=data)
        samples = data

    return Gaze(
        samples=pl.from_pandas(data=samples),
        experiment=experiment,
        events=events,
        trial_columns=trial_columns,
        time_column=time_column,
        time_unit=time_unit,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        distance_column=distance_column,
    )
