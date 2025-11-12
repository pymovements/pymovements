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
"""GazeDataFrame implementation.

.. deprecated:: v0.23.0
   Please use :py:class:`~pymovements.Gaze` instead.
   This module will be removed in v0.28.0.
"""
from __future__ import annotations

import polars as pl

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements._utils._deprecated import DeprecatedMetaClass
from pymovements._utils._html import repr_html
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze


@repr_html(['samples', 'events', 'trial_columns', 'experiment'])
class GazeDataFrame(metaclass=DeprecatedMetaClass):
    """Self-contained data structure containing gaze represented as samples or events.

    .. deprecated:: v0.22.3
       Please use :py:class:`~pymovements.Gaze` instead.
       This module will be removed in v0.28.0.

    Includes metadata on the experiment and recording setup.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.

    Parameters
    ----------
    samples: pl.DataFrame | None
        A dataframe that contains gaze samples. (default: None)
    experiment : Experiment | None
        The experiment definition. (default: None)
    events: pm.Events | None
        A dataframe of events in the gaze signal. (default: None)
    trial_columns: str | list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``. (default: None)
    time_unit: str | None
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. If time_unit is None, milliseconds are assumed. (default: None)
    pixel_columns:list[str] | None
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created. (default: None)
    position_columns: list[str] | None
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created. (default: None)
    velocity_columns: list[str] | None
        The name of the velocity columns in the input data frame. These columns will be nested
        into the column ``velocity``. If the list is empty or None, the nested ``velocity``
        column will not be created. (default: None)
    acceleration_columns: list[str] | None
        The name of the acceleration columns in the input data frame. These columns will be
        nested into the column ``acceleration``. If the list is empty or None, the nested
        ``acceleration`` column will not be created. (default: None)
    distance_column: str | None
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken
        from the experiment definition. This column will be renamed to ``distance``. (default: None)
    auto_column_detect: bool
        Flag indicating if the column names should be inferred automatically. (default: False)
    definition: pm.DatasetDefinition | None
        A dataset definition. Explicitly passed arguments take precedence over definition.
        (default: None)
    data: pl.DataFrame | None
        A dataframe that contains gaze samples. (default: None)

    Attributes
    ----------
    samples: pl.DataFrame
        A dataframe to be transformed to a polars dataframe.
    events: pm.Events
        A dataframe of events in the gaze signal.
    experiment : Experiment | None
        The experiment definition.
    trial_columns: list[str] | None
        The name of the trial columns in the data frame. If not None, the transformation methods
        will be applied to each trial separately.
    n_components: int | None
        The number of components in the pixel, position, velocity and acceleration columns.

    Notes
    -----
    About using the arguments ``pixel_columns``, ``position_columns``, ``velocity_columns``,
    and ``acceleration_columns``:

    By passing a list of columns as any of these arguments, these columns will be merged into a
    single column with the corresponding name , e.g. using `pixel_columns` will merge the
    respective columns into the column `pixel`.

    The supported number of component columns with the expected order are:

    * zero columns: No nested component column will be created.
    * two columns: monocular data; expected order: x-component, y-component
    * four columns: binocular data; expected order: x-component left eye, y-component left eye,
        x-component right eye, y-component right eye,
    * six columns: binocular data with additional cyclopian data; expected order: x-component
        left eye, y-component left eye, x-component right eye, y-component right eye,
        x-component cyclopian eye, y-component cyclopian eye,


    Examples
    --------
    First let's create an example `DataFrame` with three columns:
    the timestamp ``t`` and ``x`` and ``y`` for the pixel position.

    >>> df = pl.from_dict(
    ...     data={'t': [1000, 1001, 1002], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]},
    ... )
    >>> df
    shape: (3, 3)
    ┌──────┬─────┬─────┐
    │ t    ┆ x   ┆ y   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ f64 ┆ f64 │
    ╞══════╪═════╪═════╡
    │ 1000 ┆ 0.1 ┆ 0.1 │
    │ 1001 ┆ 0.2 ┆ 0.2 │
    │ 1002 ┆ 0.3 ┆ 0.3 │
    └──────┴─────┴─────┘

    We can now initialize our ``Gaze`` by specyfing the names of the pixel position
    columns, the timestamp column and the unit of the timestamps.

    >>> gaze = Gaze(samples=df, pixel_columns=['x', 'y'], time_column='t', time_unit='ms')
    >>> gaze
    shape: (3, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 1000 ┆ [0.1, 0.1] │
    │ 1001 ┆ [0.2, 0.2] │
    │ 1002 ┆ [0.3, 0.3] │
    └──────┴────────────┘

    In case your data has no time column available, you can pass an
    :py:class:`~pymovements.gaze.Experiment` to create a time column with the correct sampling rate
    during initialization. The time column will be represented in millisecond units.

    >>> df_no_time = df.select(pl.exclude('t'))
    >>> df_no_time
    shape: (3, 2)
    ┌─────┬─────┐
    │ x   ┆ y   │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 0.1 ┆ 0.1 │
    │ 0.2 ┆ 0.2 │
    │ 0.3 ┆ 0.3 │
    └─────┴─────┘

    >>> experiment = Experiment(1024, 768, 38, 30, 60, 'center', sampling_rate=100)
    >>> gaze = Gaze(samples=df_no_time, experiment=experiment, pixel_columns=['x', 'y'])
    >>> gaze
    Experiment(screen=Screen(width_px=1024, height_px=768, width_cm=38, height_cm=30,
     distance_cm=60, origin='center'), eyetracker=EyeTracker(sampling_rate=100, left=None,
      right=None, model=None, version=None, vendor=None, mount=None))
    shape: (3, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.1, 0.1] │
    │ 10   ┆ [0.2, 0.2] │
    │ 20   ┆ [0.3, 0.3] │
    └──────┴────────────┘
    """

    _DeprecatedMetaClass__alias = Gaze
    _DeprecatedMetaClass__version_deprecated = 'v0.23.0'
    _DeprecatedMetaClass__version_removed = 'v0.28.0'

    samples: pl.DataFrame

    events: pm.Events

    experiment: Experiment | None

    trial_columns: list[str] | None

    n_components: int | None
