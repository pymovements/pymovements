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
"""Gaze implementation."""
# pylint: disable=too-many-lines
from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Literal
from typing import overload

import numpy as np
import polars
from deprecated.sphinx import deprecated
from tqdm import tqdm

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements._utils._checks import check_is_mutual_exclusive
from pymovements._utils._html import repr_html
from pymovements.events.processing import EventGazeProcessor
from pymovements.gaze import transforms
from pymovements.gaze.experiment import Experiment


@repr_html(['samples', 'events', 'trial_columns', 'experiment'])
class Gaze:
    """Self-contained data structure containing gaze represented as samples or events.

    Includes metadata on the experiment and recording setup.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.

    Parameters
    ----------
    samples: polars.DataFrame | None
        A dataframe that contains gaze samples. (default: None)
    experiment : Experiment | None
        The experiment definition. (default: None)
    events: pm.Events | None
        A dataframe of events in the gaze signal. (default: None)
    messages: polars.DataFrame | None
        DataFrame containing messages from the experiment.
        The required columns are 'time' and 'content'. (default: None)
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
    data: polars.DataFrame | None
        A dataframe that contains gaze samples. (default: None)
        .. deprecated:: v0.23.0
        Please use ``samples`` instead. This field will be removed in v0.28.0.

    Attributes
    ----------
    samples: polars.DataFrame
        A dataframe of recorded gaze samples.
    events: pm.Events
        A dataframe of events in the gaze signal.
    experiment : Experiment | None
        The experiment definition.
    trial_columns: list[str] | None
        The name of the trial columns in the samples data frame. If not None, the transformation
        methods will be applied to each trial separately.
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

    >>> df = polars.from_dict(
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

    >>> df_no_time = df.select(polars.exclude('t'))
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

    samples: polars.DataFrame

    events: pm.Events

    experiment: Experiment | None

    trial_columns: list[str] | None

    n_components: int | None

    def __init__(
            self,
            samples: polars.DataFrame | None = None,
            experiment: Experiment | None = None,
            events: pm.Events | None = None,
            *,
            messages: polars.DataFrame | None = None,
            trial_columns: str | list[str] | None = None,
            time_column: str | None = None,
            time_unit: str | None = None,
            pixel_columns: list[str] | None = None,
            position_columns: list[str] | None = None,
            velocity_columns: list[str] | None = None,
            acceleration_columns: list[str] | None = None,
            distance_column: str | None = None,
            auto_column_detect: bool = False,
            definition: pm.DatasetDefinition | None = None,
            data: polars.DataFrame | None = None,
    ):
        if data is not None:
            warnings.warn(
                DeprecationWarning(
                    "Gaze.__init__() argument 'data' is deprecated since version v0.23.0. "
                    "Please use argument 'samples' instead. "
                    'This argument will be removed in v0.28.0.',
                ),
            )
            check_is_mutual_exclusive(samples=samples, data=data)
            samples = data

        if samples is None:
            samples = polars.DataFrame()
        else:
            samples = samples.clone()
        self.samples = samples

        # Set nan values to null.
        self.samples = self.samples.fill_nan(None)

        self._init_experiment(experiment, definition)

        self._init_columns(
            trial_columns=trial_columns,
            time_column=time_column,
            time_unit=time_unit,
            pixel_columns=pixel_columns,
            position_columns=position_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
            distance_column=distance_column,
            auto_column_detect=auto_column_detect,
            definition=definition,
        )

        if events is None:
            if self.trial_columns is None:
                self.events = pm.Events()
            else:  # Ensure that trial columns with correct dtype are present in event dataframe.
                self.events = pm.Events(
                    data=polars.DataFrame(
                        schema={
                            column: self.samples.schema[column] for column in self.trial_columns
                        },
                    ),
                    trial_columns=self.trial_columns,
                )
        else:
            self.events = events.clone()

        _check_messages(messages)
        self.messages = messages

        # Remove this attribute once #893 is fixed
        self._metadata: dict[str, Any] | None = None

    def apply(
            self,
            function: str,
            **kwargs: Any,
    ) -> None:
        """Apply preprocessing method to Gaze.

        Parameters
        ----------
        function: str
            Name of the preprocessing function to apply.
        **kwargs: Any
            kwargs that will be forwarded when calling the preprocessing method.
        """
        if transforms.TransformLibrary.__contains__(function):
            self.transform(function, **kwargs)
        elif pm.events.EventDetectionLibrary.__contains__(function):
            self.detect(function, **kwargs)
        else:
            raise ValueError(f"unsupported method '{function}'")

    @overload
    def split(
            self, by: str | Sequence[str] | None = None, *, as_dict: Literal[False],
    ) -> list[Gaze]:
        ...

    @overload
    def split(
            self, by: Sequence[str] | None = None, *, as_dict: Literal[True],
    ) -> dict[tuple[Any, ...], Gaze]:
        ...

    def split(
            self,
            by: str | Sequence[str] | None = None,
            *,
            as_dict: bool = False,
    ) -> list[Gaze] | dict[tuple[Any, ...], Gaze]:
        """Split a single Gaze object into multiple Gaze objects based on specified column(s).

        Parameters
        ----------
        by: str | Sequence[str] | None
            Column name(s) to split the DataFrame by. If a single string is provided,
            it will be used as a single column name. If a sequence is provided, the DataFrame
            will be split by unique combinations of values in all specified columns.
            If None, uses trial_columns. (default=None)
        as_dict: bool
            Return a dictionary instead of a list. The dictionary keys are tuples of the distinct
            group values that identify each group split. (default: False)

        Returns
        -------
        list[Gaze] | dict[tuple[Any, ...], Gaze]
            A collection of new Gaze instances, each containing a partition of the
            original data with all metadata and configurations preserved.

        Notes
        -----
            The original gaze data and metadata are not modified; the method returns new
            Gaze objects.

        Examples
        --------
        First let's create a simple samples dataframe:

        >>> import numpy as np
        >>> import polars as pl
        >>> import pymovements as pm
        >>> samples = polars.from_dict(
        ...     {'x': range(100), 'y': range(100), 'trial': np.repeat([1, 2, 3, 4, 5], 20)},
        ... )
        >>> samples
        shape: (100, 3)
        ┌─────┬─────┬───────┐
        │ x   ┆ y   ┆ trial │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═══════╡
        │ 0   ┆ 0   ┆ 1     │
        │ 1   ┆ 1   ┆ 1     │
        │ 2   ┆ 2   ┆ 1     │
        │ 3   ┆ 3   ┆ 1     │
        │ 4   ┆ 4   ┆ 1     │
        │ …   ┆ …   ┆ …     │
        │ 95  ┆ 95  ┆ 5     │
        │ 96  ┆ 96  ┆ 5     │
        │ 97  ┆ 97  ┆ 5     │
        │ 98  ┆ 98  ┆ 5     │
        │ 99  ┆ 99  ┆ 5     │
        └─────┴─────┴───────┘

        Then let's initialize our `Gaze` object:

        >>> gaze = pm.Gaze(samples=samples, pixel_columns=['x', 'y'], trial_columns='trial')
        >>> gaze
        shape: (100, 2)
        ┌───────┬───────────┐
        │ trial ┆ pixel     │
        │ ---   ┆ ---       │
        │ i64   ┆ list[i64] │
        ╞═══════╪═══════════╡
        │ 1     ┆ [0, 0]    │
        │ 1     ┆ [1, 1]    │
        │ 1     ┆ [2, 2]    │
        │ 1     ┆ [3, 3]    │
        │ 1     ┆ [4, 4]    │
        │ …     ┆ …         │
        │ 5     ┆ [95, 95]  │
        │ 5     ┆ [96, 96]  │
        │ 5     ┆ [97, 97]  │
        │ 5     ┆ [98, 98]  │
        │ 5     ┆ [99, 99]  │
        └───────┴───────────┘

        Now we can split the gaze by the 5 unique trial column values into 5 separate objects:

        >>> gazes = gaze.split(by='trial')
        >>> len(gazes)
        5
        """
        # Use trial_columns if by is None
        if by is None:
            if self.trial_columns is None:
                raise TypeError("Either 'by' or 'Gaze.trial_columns' must be specified")
            by = self.trial_columns

        # Convert single string to list for consistent handling
        by = [by] if isinstance(by, str) else by

        if not self.samples.is_empty():
            # We use as_dict=True here to make sure to map samples to the correct events.
            grouped_samples = self.samples.partition_by(by=by, as_dict=True)
            sample_key_dtypes = [dtype.to_python() for dtype in self.samples[by].dtypes]
        else:
            grouped_samples = {}
            sample_key_dtypes = []

        if self.events:
            # We use as_dict=True here to make sure to map events to the correct samples.
            grouped_events = self.events.split(by=by, as_dict=True)
            events_key_dtypes = [dtype.to_python() for dtype in self.events.frame[by].dtypes]
        else:
            grouped_events = {}
            events_key_dtypes = []

        keys = sorted(
            set(grouped_samples.keys()) | set(grouped_events.keys()),
            key=_replace_nones_in_split_keys(sample_key_dtypes, events_key_dtypes),
        )

        gazes = {
            key: Gaze(
                samples=grouped_samples.get(key, polars.DataFrame(schema=self.samples.schema)),
                events=grouped_events.get(key, None),
                experiment=self.experiment,
                trial_columns=self.trial_columns,
            )
            for key in keys
        }

        if as_dict:
            return gazes
        return list(gazes.values())

    def transform(
            self,
            transform_method: str | Callable[..., polars.Expr],
            **kwargs: Any,
    ) -> None:
        """Apply transformation method.

        Parameters
        ----------
        transform_method: str | Callable[..., polars.Expr]
            The transformation method to be applied.
        **kwargs: Any
            Additional keyword arguments to be passed to the transformation method.
        """
        if isinstance(transform_method, str):
            transform_method = transforms.TransformLibrary.get(transform_method)

        if transform_method.__name__ == 'downsample':
            downsample_factor = kwargs.pop('factor')
            self.samples = self.samples.select(
                transforms.downsample(
                    factor=downsample_factor, **kwargs,
                ),
            )

            # sampling rate
        elif transform_method.__name__ == 'resample':
            resample_rate = kwargs.pop('resampling_rate')

            if self.trial_columns is None:
                self.samples = transforms.resample(
                    samples=self.samples,
                    resampling_rate=resample_rate,
                    n_components=self.n_components,
                    **kwargs,
                )
            else:
                # Manipulate columns to exclude trial columns
                resample_columns = kwargs.pop('columns', 'all')

                if resample_columns == 'all':
                    resample_columns = self.samples.columns
                elif isinstance(resample_columns, str):
                    resample_columns = [resample_columns]

                if resample_columns is not None:
                    resample_columns = [
                        col for col in resample_columns if col not in self.trial_columns
                    ]

                self.samples = polars.concat(
                    [
                        transforms.resample(
                            samples=df,
                            resampling_rate=resample_rate,
                            n_components=self.n_components,
                            columns=resample_columns,
                            **kwargs,
                        )
                        for group, df in
                        self.samples.group_by(self.trial_columns, maintain_order=True)
                    ],
                )

                # forward fill trial columns
                self.samples = self.samples.with_columns(
                    polars.col(self.trial_columns).fill_null(strategy='forward'),
                )

            # set new sampling rate in experiment
            if self.experiment is not None:
                self.experiment.sampling_rate = resample_rate

        else:
            method_kwargs = inspect.getfullargspec(transform_method).kwonlyargs
            if 'origin' in method_kwargs and 'origin' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                if self.experiment.screen.origin is not None:
                    kwargs['origin'] = self.experiment.screen.origin

            if 'screen_resolution' in method_kwargs and 'screen_resolution' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['screen_resolution'] = (
                    self.experiment.screen.width_px, self.experiment.screen.height_px,
                )

            if 'screen_size' in method_kwargs and 'screen_size' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['screen_size'] = (
                    self.experiment.screen.width_cm, self.experiment.screen.height_cm,
                )

            if 'distance' in method_kwargs and 'distance' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None

                if 'distance' in self.samples.columns:
                    kwargs['distance'] = 'distance'

                    if self.experiment.screen.distance_cm:
                        warnings.warn(
                            "Both a distance column and experiment's "
                            'eye-to-screen distance are specified. '
                            'Using eye-to-screen distances from column '
                            "'distance' in the samples dataframe.",
                        )
                elif self.experiment.screen.distance_cm:
                    kwargs['distance'] = self.experiment.screen.distance_cm
                else:
                    raise AttributeError(
                        'Neither eye-to-screen distance is in the columns of the samples dataframe '
                        'nor experiment eye-to-screen distance is specified.',
                    )

            if 'sampling_rate' in method_kwargs and 'sampling_rate' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['sampling_rate'] = self.experiment.sampling_rate

            if 'n_components' in method_kwargs and 'n_components' not in kwargs:
                self._check_n_components()
                kwargs['n_components'] = self.n_components

            if transform_method.__name__ in {'pos2vel', 'pos2acc'}:
                if 'position' not in self.samples.columns and 'position_column' not in kwargs:
                    if 'pixel' in self.samples.columns:
                        raise polars.exceptions.ColumnNotFoundError(
                            "Neither is 'position' in the samples dataframe columns, "
                            'nor is a position column explicitly specified. '
                            "Since the samples dataframe has a 'pixel' column, consider running "
                            f'pix2deg() before {transform_method.__name__}(). If you want '
                            'to run transformations in pixel units, you can do so by using '
                            f"{transform_method.__name__}(position_column='pixel'). "
                            f'Available columns in samples dataframe are: {self.samples.columns}',
                        )
                    raise polars.exceptions.ColumnNotFoundError(
                        "Neither is 'position' in the samples dataframe columns, "
                        'nor is a position column explicitly specified. '
                        'You can specify the position column via: '
                        f'{transform_method.__name__}(position_column="your_position_column"). '
                        f'Available columns in samples dataframe are: {self.samples.columns}',
                    )

            if transform_method.__name__ in {'pix2deg'}:
                if 'pixel' not in self.samples.columns and 'pixel_column' not in kwargs:
                    raise polars.exceptions.ColumnNotFoundError(
                        "Neither is 'pixel' in the samples dataframe columns, "
                        'nor is a pixel column explicitly specified. '
                        'You can specify the pixel column via: '
                        f'{transform_method.__name__}(pixel_column="name_of_your_pixel_column"). '
                        f'Available columns in samples dataframe are: {self.samples.columns}',
                    )

            if transform_method.__name__ in {'deg2pix'}:
                if (
                    'position_column' in kwargs and
                    kwargs.get('position_column') not in self.samples.columns
                ):
                    raise polars.exceptions.ColumnNotFoundError(
                        f"The specified 'position_column' ({kwargs.get('position_column')}) "
                        'is not found in the samples dataframe columns. '
                        'You can specify the position column via: '
                        f'{transform_method.__name__}'
                        f'(position_column="name_of_your_position_column"). '
                        f'Available columns in samples dataframe are: {self.samples.columns}',
                    )

            if self.trial_columns is None:
                self.samples = self.samples.with_columns(transform_method(**kwargs))
            else:
                self.samples = polars.concat(
                    [
                        df.with_columns(transform_method(**kwargs))
                        for group, df in
                        self.samples.group_by(self.trial_columns, maintain_order=True)
                    ],
                )

    def clip(
            self,
            lower_bound: int | float | None,
            upper_bound: int | float | None,
            *,
            input_column: str,
            output_column: str,
            **kwargs: Any,
    ) -> None:
        """Clip gaze signal values.

        This method requires a properly initialized :py:attr:`~.Gaze.experiment` attribute.

        After success, the values in :py:attr:`~.Gaze.samples` are clipped.

        Parameters
        ----------
        lower_bound : int | float | None
            Lower bound of the clipped column.
        upper_bound : int | float | None
            Upper bound of the clipped column.
        input_column : str
            Name of the input column.
        output_column : str
            Name of the output column.
        **kwargs: Any
            Additional keyword arguments to be passed to the :func:`~transforms.clip()` method.

        Raises
        ------
        AttributeError
            If :py:attr:`~.Gaze.samples` is None, or if :py:attr:`~.Gaze.experiment` is None.
        """
        self.transform(
            'clip',
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            input_column=input_column,
            output_column=output_column,
            **kwargs,
        )

    def pix2deg(self) -> None:
        """Compute gaze positions in degrees of visual angle from pixel position coordinates.

        This method requires a properly initialized :py:attr:`~.Gaze.experiment` attribute.

        After success, :py:attr:`~.Gaze.samples` is extended by the resulting dva position columns.

        Raises
        ------
        AttributeError
            If :py:attr:`~.Gaze.samples` is None, or if :py:attr:`~.Gaze.experiment` is None.
        """
        self.transform('pix2deg')

    def deg2pix(
            self,
            pixel_origin: str = 'upper left',
            position_column: str = 'position',
            pixel_column: str = 'pixel',
    ) -> None:
        """Compute gaze positions in pixel position coordinates from degrees of visual angle.

        This method requires a properly initialized :py:attr:`~.Gaze.experiment` attribute.

        After success, :py:attr:`~.Gaze.samples` is extended by the resulting dva position columns.

        Parameters
        ----------
        pixel_origin: str
            The desired location of the pixel origin. (default: 'upper left')
            Supported values: ``center``, ``upper left``.
        position_column: str
            The input position column name. (default: 'position')
        pixel_column: str
            The output pixel column name. (default: 'pixel')

        Raises
        ------
        AttributeError
            If :py:attr:`~.Gaze.samples` is None, or if :py:attr:`~.Gaze.experiment` is None.
        """
        self.transform(
            'deg2pix',
            pixel_origin=pixel_origin,
            position_column=position_column,
            pixel_column=pixel_column,
        )

    def pos2acc(
            self,
            *,
            degree: int = 2,
            window_length: int = 7,
            padding: str | float | int | None = 'nearest',
    ) -> None:
        """Compute gaze acceleration in dva/s^2 from dva position coordinates.

        This method requires a properly initialized :py:attr:`~.Gaze.experiment` attribute.

        After success, :py:attr:`~.Gaze.samples` is extended by the resulting velocity columns.

        Parameters
        ----------
        degree: int
            The degree of the polynomial to use. (default: 2)
        window_length: int
            The window size to use. (default: 7)
        padding: str | float | int | None
            The padding method to use. See ``savitzky_golay`` for details. (default: 'nearest')

        Raises
        ------
        AttributeError
            If :py:attr:`~.Gaze.samples` is None, or if :py:attr:`~.Gaze.experiment` is None.
        """
        self.transform('pos2acc', window_length=window_length, degree=degree, padding=padding)

    def pos2vel(
            self,
            method: str = 'fivepoint',
            **kwargs: int | float | str,
    ) -> None:
        """Compute gaze velocity in dva/s from dva position coordinates.

        This method requires a properly initialized :py:attr:`~.Gaze.experiment` attribute.

        After success, :py:attr:`~.Gaze.samples` is extended by the resulting velocity columns.

        Parameters
        ----------
        method: str
            Computation method. See :func:`~transforms.pos2vel()` for details, default: fivepoint.
            (default: 'fivepoint')
        **kwargs: int | float | str
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Raises
        ------
        AttributeError
            If :py:attr:`~.Gaze.samples` is None, or if :py:attr:`~.Gaze.experiment` is None.
        """
        self.transform('pos2vel', method=method, **kwargs)

    def resample(
            self,
            resampling_rate: float,
            columns: str | list[str] = 'all',
            fill_null_strategy: str = 'interpolate_linear',
    ) -> None:
        """Resample :py:attr:`~.Gaze.samples` to a new sampling rate by timestamps in time column.

        :py:attr:`~.Gaze.samples` is resampled by upsampling or downsampling the data to the new
        sampling rate. Can also be used to achieve a constant sampling rate for inconsistent data.

        Parameters
        ----------
        resampling_rate: float
            The new sampling rate.
        columns: str | list[str]
            The columns to apply the fill null strategy. Specify a single column name or a list of
            column names. If 'all' is specified, the fill null strategy is applied to all columns.
            (default: 'all')
        fill_null_strategy: str
            The strategy to fill null values of the resampled DataFrame. Supported strategies
            are: 'forward', 'backward', 'interpolate_linear', 'interpolate_nearest'.
            (default: 'interpolate_linear')

        Examples
        --------
        Lets create an example Gaze of 1000Hz with a time column and a position column.
        Please note that time is always stored in milliseconds in the Gaze.

        >>> df = polars.DataFrame({
        ...     'time': [0, 1, 2, 3, 4],
        ...     'x': [1, 2, 3, 4, 5],
        ...     'y': [1, 2, 3, 4, 5],
        ... })
        >>> gaze = Gaze(samples=df, time_column='time', pixel_columns=['x', 'y'])
        >>> gaze.samples
        shape: (5, 2)
        ┌──────┬───────────┐
        │ time ┆ pixel     │
        │ ---  ┆ ---       │
        │ i64  ┆ list[i64] │
        ╞══════╪═══════════╡
        │ 0    ┆ [1, 1]    │
        │ 1    ┆ [2, 2]    │
        │ 2    ┆ [3, 3]    │
        │ 3    ┆ [4, 4]    │
        │ 4    ┆ [5, 5]    │
        └──────┴───────────┘

        We can now upsample the Gaze to 2000Hz with interpolating the values in
        the pixel column.

        >>> gaze.resample(
        ...     resampling_rate=2000,
        ...     fill_null_strategy='interpolate_linear',
        ...     columns=['pixel'],
        ... )
        >>> gaze.samples
        shape: (9, 2)
        ┌──────┬────────────┐
        │ time ┆ pixel      │
        │ ---  ┆ ---        │
        │ f64  ┆ list[f64]  │
        ╞══════╪════════════╡
        │ 0.0  ┆ [1.0, 1.0] │
        │ 0.5  ┆ [1.5, 1.5] │
        │ 1.0  ┆ [2.0, 2.0] │
        │ 1.5  ┆ [2.5, 2.5] │
        │ 2.0  ┆ [3.0, 3.0] │
        │ 2.5  ┆ [3.5, 3.5] │
        │ 3.0  ┆ [4.0, 4.0] │
        │ 3.5  ┆ [4.5, 4.5] │
        │ 4.0  ┆ [5.0, 5.0] │
        └──────┴────────────┘

        Downsample the Gaze to 500Hz results in the following DataFrame.

        >>> gaze.resample(resampling_rate=500)
        >>> gaze.samples
        shape: (3, 2)
        ┌──────┬────────────┐
        │ time ┆ pixel      │
        │ ---  ┆ ---        │
        │ i64  ┆ list[f64]  │
        ╞══════╪════════════╡
        │ 0    ┆ [1.0, 1.0] │
        │ 2    ┆ [3.0, 3.0] │
        │ 4    ┆ [5.0, 5.0] │
        └──────┴────────────┘
        """
        self.transform(
            'resample',
            resampling_rate=resampling_rate,
            columns=columns,
            fill_null_strategy=fill_null_strategy,
        )

    def smooth(
            self,
            method: str = 'savitzky_golay',
            window_length: int = 7,
            degree: int = 2,
            column: str = 'position',
            padding: str | float | int | None = 'nearest',
            **kwargs: int | float | str,
    ) -> None:
        """Smooth column values in :py:attr:`~.Gaze.samples`.

        Parameters
        ----------
        method: str
            The method to use for smoothing. Choose from ``savitzky_golay``, ``moving_average``,
            ``exponential_moving_average``. See :func:`~transforms.smooth()` for details.
            (default: 'savitzky_golay')
        window_length: int
            For ``moving_average`` this is the window size to calculate the mean of the subsequent
            samples. For ``savitzky_golay`` this is the window size to use for the polynomial fit.
            For ``exponential_moving_average`` this is the span parameter. (default: 7)
        degree: int
            The degree of the polynomial to use. This has only an effect if using
            ``savitzky_golay`` as smoothing method. `degree` must be less than `window_length`.
            (default: 2)
        column: str
            The input column name to which the smoothing is applied. (default: 'position')
        padding: str | float | int | None
            Must be either ``None``, a scalar or one of the strings
            ``mirror``, ``nearest`` or ``wrap``.
            This determines the type of extension to use for the padded signal to
            which the filter is applied.
            When passing ``None``, no extension padding is used.
            When passing a scalar value, sample series will be padded using the passed value.
            See :func:`~transforms.smooth()` for details on the padding methods.
            (default: 'nearest')
        **kwargs: int | float | str
            Additional keyword arguments to be passed to the :func:`~transforms.smooth()` method.
        """
        self.transform(
            'smooth',
            column=column,
            method=method,
            degree=degree,
            window_length=window_length,
            padding=padding,
            **kwargs,
        )

    def detect(
            self,
            method: Callable[..., pm.Events] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            **kwargs: Any,
    ) -> None:
        """Detect events by applying a specific event detection method.

        Parameters
        ----------
        method: Callable[..., pm.Events] | str
            The event detection method to be applied.
        eye: str
            Select which eye to choose. Valid options are ``auto``, ``left``, ``right`` or ``None``.
            If ``auto`` is passed, eye is inferred in the order ``['right', 'left', 'eye']`` from
            the available columns in :py:attr:`~.Gaze.samples`. (default: 'auto')
        clear: bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
            merged into the existing one. (default: False)
        **kwargs: Any
            Additional keyword arguments to be passed to the event detection method.
        """
        if self.events is None or clear:
            if self.trial_columns is None:
                self.events = pm.Events()
            else:  # Ensure that trial columns with correct dtype are present in event dataframe.
                self.events = pm.Events(
                    data=polars.DataFrame(
                        schema={
                            column: self.samples.schema[column] for column in self.trial_columns
                        },
                    ),
                    trial_columns=self.trial_columns,
                )

        if isinstance(method, str):
            method = pm.events.EventDetectionLibrary.get(method)

        if self.n_components is not None:
            eye_components = self._infer_eye_components(eye)
        else:
            eye_components = None

        if self.trial_columns is None:
            method_kwargs = self._fill_event_detection_kwargs(
                method,
                samples=self.samples,
                events=self.events,
                eye_components=eye_components,
                **kwargs,
            )

            new_events = method(**method_kwargs)

            self.events.frame = polars.concat(
                [self.events.frame, new_events.frame],
                how='diagonal_relaxed',
            )
        else:
            grouped_samples = self.samples.partition_by(
                self.trial_columns, maintain_order=True, include_key=True, as_dict=True,
            )

            missing_trial_columns = [
                trial_column for trial_column in self.trial_columns
                if trial_column not in self.events.frame.columns
            ]
            if missing_trial_columns:
                raise polars.exceptions.ColumnNotFoundError(
                    f'trial columns {missing_trial_columns} missing from events, '
                    f'available columns: {self.events.frame.columns}',
                )

            new_events_grouped: list[polars.DataFrame] = []

            for group_identifier, group_gaze in grouped_samples.items():
                # Create filter expression for selecting respective group rows.
                if len(self.trial_columns) == 1:
                    group_filter_expression = polars.col(
                        self.trial_columns[0],
                    ) == group_identifier[0]
                else:
                    group_filter_expression = polars.col(
                        self.trial_columns[0],
                    ) == group_identifier[0]
                    for name, value in zip(self.trial_columns[1:], group_identifier[1:]):
                        group_filter_expression = group_filter_expression & (
                            polars.col(name) == value
                        )

                # Select group events
                group_events = pm.Events(self.events.frame.filter(group_filter_expression))

                method_kwargs = self._fill_event_detection_kwargs(
                    method,
                    samples=group_gaze,
                    events=group_events,
                    eye_components=eye_components,
                    **kwargs,
                )

                new_events = method(**method_kwargs)
                # add group identifiers as new columns
                new_events.add_trial_column(self.trial_columns, group_identifier)

                new_events_grouped.append(new_events.frame)

            self.events.frame = polars.concat(
                [self.events.frame, *new_events_grouped],
                how='diagonal',
            )

    def drop_event_properties(
            self,
            event_properties: str | list[str],
    ) -> None:
        """Remove event properties from the event dataframe.

        Parameters
        ----------
        event_properties: str | list[str]
            The event properties to remove.

        Raises
        ------
        ValueError
            If ``event_properties`` do not exist in the event dataframe
            or it is not allowed to remove them
        """
        self.events.drop(event_properties)

    def compute_event_properties(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
            name: str | None = None,
    ) -> None:
        """Calculate event properties for given events.

        The calculated event properties are added as columns to
        :py:attr:`~pymovements.gaze.Gaze.events`.

        Parameters
        ----------
        event_properties: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
            The event properties to compute.
        name: str | None
            Process only events that match the name. (default: None)

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events.event_properties` for an overview of supported properties.
        RuntimeError
            If specified event name ``name`` is missing from ``events``.
        ValueError
            If the computed property already exists as a column in ``events``.
        """
        if len(self.events) == 0:
            warnings.warn(
                'No events available to compute event properties. '
                'Did you forget to use detect()?',
            )

        identifiers = self.trial_columns if self.trial_columns is not None else []
        processor = EventGazeProcessor(event_properties)

        event_property_names = [property[0] for property in processor.event_properties]
        existing_columns = set(self.events.columns) & set(event_property_names)
        if existing_columns:
            raise ValueError(
                f"The following event properties already exist and cannot be recomputed: "
                f"{existing_columns}. Please remove them first.",
            )

        new_properties = processor.process(
            self.events, self, identifiers=identifiers, name=name,
        )
        join_on = identifiers + ['name', 'onset', 'offset']
        self.events.add_event_properties(new_properties, join_on=join_on)

    def measure_samples(
            self,
            method: str | Callable[..., polars.Expr],
            **kwargs: Any,
    ) -> polars.DataFrame:
        """Calculate eye movement measure on :py:attr:`~.Gaze.samples`.

        If :py:class:``Gaze`` has :py:attr:``trial_columns``, measures will be grouped by
        trials.

        Parameters
        ----------
        method: str | Callable[..., polars.Expr]
            Measure to be calculated.
        **kwargs: Any
            Keyword arguments to be passed to the respective measure function.

        Returns
        -------
        polars.DataFrame
            Measure results.

        Examples
        --------
        Let's initialize an example Gaze first:
        >>> gaze = pm.gaze.from_numpy(
        ...     pixel=np.concatenate(
        ...         [np.zeros((2, 40)), np.full((2, 10), np.nan), np.ones((2, 50))],
        ...         axis=1,
        ...     ),
        ... )

        You can calculate measures, for example the null ratio like this:
        >>> gaze.measure_samples('null_ratio', column='pixel')
        shape: (1, 1)
        ┌────────────┐
        │ null_ratio │
        │ ---        │
        │ f64        │
        ╞════════════╡
        │ 0.1        │
        └────────────┘
        """
        if isinstance(method, str):
            method = pm.measure.SampleMeasureLibrary.get(method)

        if 'column_dtype' in inspect.getfullargspec(method).args:
            kwargs['column_dtype'] = self.samples[kwargs['column']].dtype

        if self.trial_columns is None:
            return self.samples.select(method(**kwargs))

        # Group measure values by trial columns.
        return polars.concat(
            [
                df.select(
                    [  # add trial columns first, then add column for measure.
                        polars.lit(value).cast(self.samples.schema[name]).alias(name)
                        for name, value in zip(self.trial_columns, trial_values)
                    ] + [method(**kwargs)],
                )
                for trial_values, df in
                self.samples.group_by(self.trial_columns, maintain_order=True)
            ],
        )

    @property
    def schema(self) -> polars.type_aliases.SchemaDict:
        """Schema of samples dataframe."""
        return self.samples.schema

    @property
    def columns(self) -> list[str]:
        """List of column names in samples dataframe."""
        return self.samples.columns

    @property
    @deprecated(
        reason='Please use Gaze.samples instead. '
               'This property will be removed in v0.28.0.',
        version='v0.23.0',
    )
    def frame(self) -> polars.DataFrame:
        """Gaze samples dataframe.

        .. deprecated:: v0.23.0
        Please use Gaze.samples instead.
        This property will be removed in v0.28.0.

        Returns
        -------
        polars.DataFrame
            Gaze samples dataframe.

        """
        return self.samples

    @frame.setter
    @deprecated(
        reason='Please use Gaze.samples instead. '
               'This property will be removed in v0.28.0.',
        version='v0.23.0',
    )
    def frame(self, data: polars.DataFrame) -> None:
        self.samples = data

    def map_to_aois(
            self,
            aoi_dataframe: pm.stimulus.TextStimulus,
            *,
            eye: str = 'auto',
            gaze_type: str = 'pixel',
    ) -> None:
        """Map gaze data to aois.

        We map each gaze point to an aoi, considering the boundary still part of the
        area of interest.

        Parameters
        ----------
        aoi_dataframe: pm.stimulus.TextStimulus
            Area of interest dataframe.
        eye: str
            String specificer for inferring eye components. Supported values are: auto, mono, left
            right, cyclops. Default: auto.
        gaze_type: str
            String specificer for whether to use position or pixel coordinates for
            mapping. Default: pixel.
        """
        component_suffixes = ['x', 'y', 'xl', 'yl', 'xr', 'yr', 'xa', 'ya']
        self.unnest()

        pix_column_canditates = ['pixel_' + suffix for suffix in component_suffixes]
        pixel_columns = [c for c in pix_column_canditates if c in self.samples.columns]
        pos_column_canditates = ['position_' + suffix for suffix in component_suffixes]
        position_columns = [
            c
            for c in pos_column_canditates
            if c in self.samples.columns
        ]

        if gaze_type == 'pixel':
            if eye == 'left':
                x_eye = [col for col in pixel_columns if col.endswith('xl')][0]
                y_eye = [col for col in pixel_columns if col.endswith('yl')][0]
            elif eye == 'right':
                x_eye = [col for col in pixel_columns if col.endswith('xr')][0]
                y_eye = [col for col in pixel_columns if col.endswith('yr')][0]
            elif eye == 'auto':
                x_eye = [col for col in pixel_columns if col.endswith('xr')][0]
                y_eye = [col for col in pixel_columns if col.endswith('yr')][0]
            else:
                x_eye = [col for col in pixel_columns if col.endswith('xr')][0]
                y_eye = [col for col in pixel_columns if col.endswith('yr')][0]
        elif gaze_type == 'position':
            if eye == 'left':
                x_eye = [col for col in position_columns if col.endswith('xl')][0]
                y_eye = [col for col in position_columns if col.endswith('yl')][0]
            elif eye == 'right':
                x_eye = [col for col in position_columns if col.endswith('xr')][0]
                y_eye = [col for col in position_columns if col.endswith('yr')][0]
            elif eye == 'auto':
                x_eye = [col for col in position_columns if col.endswith('xr')][0]
                y_eye = [col for col in position_columns if col.endswith('yr')][0]
            else:
                x_eye = [col for col in position_columns if col.endswith('xr')][0]
                y_eye = [col for col in position_columns if col.endswith('yr')][0]
        else:
            raise ValueError(
                'neither position nor pixel column in samples dataframe, '
                'at least one needed for mapping',
            )

        aois = [
            aoi_dataframe.get_aoi(row=row, x_eye=x_eye, y_eye=y_eye)
            for row in tqdm(self.samples.iter_rows(named=True))
        ]
        aoi_df = polars.concat(aois)
        self.samples = polars.concat([self.samples, aoi_df], how='horizontal')

    def nest(
            self,
            input_columns: list[str],
            output_column: str,
    ) -> None:
        """Nest component columns into a single tuple column.

        Input component columns will be dropped.

        Parameters
        ----------
        input_columns: list[str]
            Names of input columns to be merged into a single tuple column.
        output_column: str
            Name of the resulting tuple column.
        """
        self._check_component_columns(**{output_column: input_columns})

        self.samples = self.samples.with_columns(
            polars.concat_list([polars.col(component) for component in input_columns])
            .alias(output_column),
        ).drop(input_columns)

    def unnest(
            self,
            input_columns: list[str] | str | None = None,
            output_suffixes: list[str] | None = None,
            *,
            output_columns: list[str] | None = None,
    ) -> None:
        """Explode a column of type ``polars.List`` into one column for each list component.

        The input column will be dropped.

        Parameters
        ----------
        input_columns: list[str] | str | None
            Name(s) of input column(s) to be unnested into several component columns.
            If None all list columns 'pixel', 'position', 'velocity' and
            'acceleration' will be unnested if existing. (default: None)
        output_suffixes: list[str] | None
            Suffixes to append to the column names. (default: None)
        output_columns: list[str] | None
            Name of the resulting tuple columns. (default: None)

        Raises
        ------
        ValueError
            If both output_columns and output_suffixes are specified.
            If number of output columns / suffixes does not match number of components.
            If output columns / suffixes are not unique.
            If no columns to unnest exist and none are specified.
            If output columns are specified and more than one input column is specified.
        AttributeError
            If number of components is not 2, 4 or 6.
        Warning
            If no columns to unnest exist and none are specified.
        """
        if input_columns is None:
            cols = ['pixel', 'position', 'velocity', 'acceleration']
            input_columns = [col for col in cols if col in self.samples.columns]

            if len(input_columns) == 0:
                raise Warning(
                    'No columns to unnest. '
                    'Please specify columns to unnest via the "input_columns" argument.',
                )

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # no support for custom output columns if more than one input column will be unnested
        if output_columns is not None and not len(input_columns) == 1:
            raise ValueError(
                'You cannot specify output columns if you want to unnest more than '
                'one input column. Please specify output suffixes or use a single '
                'input column instead.',
            )

        check_is_mutual_exclusive(
            output_columns=output_columns,
            output_suffixes=output_suffixes,
        )

        self._check_n_components()
        assert self.n_components in {2, 4, 6}

        col_names = [output_columns] if output_columns is not None else []

        if output_columns is None and output_suffixes is None:
            if self.n_components == 2:
                output_suffixes = ['_x', '_y']
            elif self.n_components == 4:
                output_suffixes = ['_xl', '_yl', '_xr', '_yr']
            else:  # This must be 6 as we already have checked our n_components.
                output_suffixes = ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya']

        if output_suffixes:
            col_names = [
                [f'{input_col}{suffix}' for suffix in output_suffixes]
                for input_col in input_columns
            ]

        if len([
            name for name_list in col_names for name in name_list
        ]) != self.n_components * len(input_columns):
            raise ValueError(
                f'Number of output columns / suffixes ({len(col_names[0])}) '
                f'must match number of components ({self.n_components})',
            )

        if len({name for name_list in col_names for name in name_list}) != len(
                [name for name_list in col_names for name in name_list],
        ):
            raise ValueError('Output columns / suffixes must be unique')

        for input_col, column_names in zip(input_columns, col_names):
            self.samples = self.samples.with_columns(
                [
                    polars.col(input_col).list.get(component_id).alias(names)
                    for component_id, names in enumerate(column_names)
                ],
            ).drop(input_col)

    def clone(self) -> Gaze:
        """Return a copy of the Gaze.

        Returns
        -------
        Gaze
            A copy of the Gaze.
        """
        gaze = Gaze(
            samples=self.samples.clone(),
            experiment=deepcopy(self.experiment),
            events=self.events.clone(),
        )
        gaze.n_components = self.n_components
        return gaze

    def _check_experiment(self) -> None:
        """Check if experiment attribute has been set.

        Raises
        ------
        AttributeError
            If experiment is None.
        """
        if self.experiment is None:
            raise AttributeError('experiment must not be None for this method to work')

    def _check_n_components(self) -> None:
        """Check that n_components is either 2, 4 or 6.

        Ensure that the number of gaze components is valid.

        Valid configurations are:
            - 2 components: monocular data (e.g., x and y)
            - 4 components: binocular data (e.g., x/y for left and right eye)
            - 6 components: binocular + cyclopean data (x/y for left, right, and cyclopean eye)

        If no valid gaze columns were specified (pixel, position, etc.), raise an error
        with a helpful message to guide proper initialization.

        Raises
        ------
        AttributeError
            If n_components is not 2, 4 or 6.
        """
        if self.n_components not in {2, 4, 6}:
            raise AttributeError(
                'Number of components required but no gaze components could be inferred.\n'
                'This usually happens if you did not specify any column content'
                ' and the content could not be autodetected from the column names. \n'
                "Please specify 'pixel_columns', 'position_columns', 'velocity_columns'"
                " or 'acceleration_columns' explicitly during initialization.",
            )

    def _check_component_columns(self, **kwargs: list[str]) -> None:
        """Check if component columns are in valid format.

        Parameters
        ----------
        **kwargs: list[str]
            Keyword arguments of component columns.
        """
        for component_type, columns in kwargs.items():
            if not isinstance(columns, list):
                raise TypeError(
                    f'{component_type} must be of type list, '
                    f'but is of type {type(columns).__name__}',
                )

            for column in columns:
                if not isinstance(column, str):
                    raise TypeError(
                        f'all elements in {component_type} must be of type str, '
                        f'but one of the elements is of type {type(column).__name__}',
                    )

            if len(columns) not in [2, 4, 6]:
                raise ValueError(
                    f'{component_type} must contain either 2, 4 or 6 columns, '
                    f'but has {len(columns)}',
                )

            for column in columns:
                if column not in self.samples.columns:
                    raise polars.exceptions.ColumnNotFoundError(
                        f'column {column} from {component_type}'
                        ' is not available in samples dataframe',
                    )

            if len(set(self.samples[columns].dtypes)) != 1:
                types_list = sorted([str(t) for t in set(self.samples[columns].dtypes)])
                raise ValueError(
                    f'all columns in {component_type} must be of same type, '
                    f'but types are {types_list}',
                )

    def _infer_n_components(self, column_specifiers: list[list[str]]) -> int | None:
        """Infer number of components from DataFrame.

        Method checks nested columns `pixel`, `position`, `velocity` and `acceleration` for number
        of components by getting their list lenghts, which must be equal for all else a ValueError
        is raised. Additionally, a list of list of column specifiers is checked for consistency.

        Parameters
        ----------
        column_specifiers: list[list[str]]
            List of list of column specifiers.

        Returns
        -------
        int | None
            Number of components

        Raises
        ------
        ValueError
            If number of components is not equal for all considered columns and rows.
        """
        all_considered_columns = ['pixel', 'position', 'velocity', 'acceleration']
        considered_columns = [
            column for column in all_considered_columns if column in self.samples.columns
        ]

        list_lengths = {
            list_length
            for column in considered_columns
            for list_length in self.samples.get_column(column).list.len().unique().to_list()
        }

        for column_specifier_list in column_specifiers:
            list_lengths.add(len(column_specifier_list))

        if len(list_lengths) > 1:
            raise ValueError(f'inconsistent number of components inferred: {list_lengths}')

        if len(list_lengths) == 0:
            return None

        return next(iter(list_lengths))

    def _infer_eye_components(self, eye: str) -> tuple[int, int]:
        """Infer eye components from eye string.

        Parameters
        ----------
        eye: str
            String specificer for inferring eye components. Supported values are: auto, mono, left
            right, cyclops. Default: auto.

        Returns
        -------
        tuple[int, int]
            Tuple of eye component indices.
        """
        self._check_n_components()

        if eye == 'auto':
            # Order of inference: cyclops, right, left.
            if self.n_components == 6:
                eye_components = 4, 5
            elif self.n_components == 4:
                eye_components = 2, 3
            else:  # We already checked number of components, must be 2.
                eye_components = 0, 1
        elif eye == 'left':
            if isinstance(self.n_components, int) and self.n_components < 4:
                # Left only makes sense if there are at least two eyes.
                raise AttributeError(
                    'left eye is only supported for data with at least 4 components',
                )
            eye_components = 0, 1
        elif eye == 'right':
            if isinstance(self.n_components, int) and self.n_components < 4:
                # Right only makes sense if there are at least two eyes.
                raise AttributeError(
                    'right eye is only supported for data with at least 4 components',
                )
            eye_components = 2, 3
        elif eye == 'cyclops':
            if isinstance(self.n_components, int) and self.n_components < 6:
                raise AttributeError(
                    'cyclops eye is only supported for data with at least 6 components',
                )
            eye_components = 4, 5
        else:
            raise ValueError(
                f"unknown eye '{eye}'. Supported values are: ['auto', 'left', 'right', 'cyclops']",
            )

        return eye_components

    def _fill_event_detection_kwargs(
            self,
            method: Callable[..., pm.Events],
            samples: polars.DataFrame,
            events: pm.Events,
            eye_components: tuple[int, int] | None,
            **kwargs: Any,
    ) -> dict[str, Any]:
        """Fill event detection method kwargs with gaze attributes.

        Parameters
        ----------
        method: Callable[..., pm.Events]
            The method for which the keyword argument dictionary will be filled.
        samples: polars.DataFrame
            The samples to be used for filling event detection keyword arguments.
        events: pm.Events
            The event dataframe to be used for filling event detection keyword arguments.
        eye_components: tuple[int, int] | None
            The eye components to be used for filling event detection keyword arguments.
        **kwargs: Any
            The source keyword arguments passed to the `Gaze.detect()` method.

        Returns
        -------
        dict[str, Any]
            The filled keyword argument dictionary.
        """
        # Automatically infer eye to use for event detection.
        method_args = inspect.getfullargspec(method).args

        if 'positions' in method_args:
            if 'position' not in samples.columns:
                raise polars.exceptions.ColumnNotFoundError(
                    f'Column \'position\' not found.'
                    f' Available columns are: {samples.columns}',
                )

            if eye_components is None:
                raise ValueError(
                    'eye_components must not be None if passing position to event detection',
                )

            kwargs['positions'] = np.vstack(
                [
                    samples.get_column('position').list.get(eye_component)
                    for eye_component in eye_components
                ],
            ).transpose()

        if 'velocities' in method_args:
            if 'velocity' not in samples.columns:
                raise polars.exceptions.ColumnNotFoundError(
                    f'Column \'velocity\' not found.'
                    f' Available columns are: {samples.columns}',
                )

            if eye_components is None:
                raise ValueError(
                    'eye_components must not be None if passing velocity to event detection',
                )

            kwargs['velocities'] = np.vstack(
                [
                    samples.get_column('velocity').list.get(eye_component)
                    for eye_component in eye_components
                ],
            ).transpose()

        if 'events' in method_args:
            kwargs['events'] = events

        if 'timesteps' in method_args and 'time' in samples.columns:
            kwargs['timesteps'] = samples.get_column('time').to_numpy()

        return kwargs

    def _init_columns(
            self,
            trial_columns: str | list[str] | None = None,
            time_column: str | None = None,
            time_unit: str | None = None,
            pixel_columns: list[str] | None = None,
            position_columns: list[str] | None = None,
            velocity_columns: list[str] | None = None,
            acceleration_columns: list[str] | None = None,
            distance_column: str | None = None,
            auto_column_detect: bool = False,
            definition: pm.DatasetDefinition | None = None,
    ) -> None:
        """Initialize columns of :py:attr:`~.Gaze.samples`."""
        # Explicit arguments take precedence over definition.
        if definition:
            if trial_columns is None:
                trial_columns = definition.trial_columns

            if time_column is None:
                time_column = definition.time_column

            if time_unit is None:
                time_unit = definition.time_unit

            if pixel_columns is None:
                pixel_columns = definition.pixel_columns

            if position_columns is None:
                position_columns = definition.position_columns

            if velocity_columns is None:
                velocity_columns = definition.velocity_columns

            if acceleration_columns is None:
                acceleration_columns = definition.acceleration_columns

            if distance_column is None:
                distance_column = definition.distance_column

        # Initialize trial_columns.
        trial_columns = [trial_columns] if isinstance(trial_columns, str) else trial_columns
        if trial_columns is not None and len(trial_columns) == 0:
            trial_columns = None
        _check_trial_columns(trial_columns, self.samples)
        self.trial_columns = trial_columns

        # Initialize time column.
        self._init_time_column(time_column, time_unit)

        # Rename distance column if necessary.
        if distance_column is not None and distance_column != 'distance':
            self.samples = self.samples.rename({distance_column: 'distance'})

        # Autodetect column names.
        component_suffixes = ['x', 'y', 'xl', 'yl', 'xr', 'yr', 'xa', 'ya']

        if auto_column_detect and pixel_columns is None:
            column_canditates = ['pixel_' + suffix for suffix in component_suffixes]
            pixel_columns = [c for c in column_canditates if c in self.samples.columns]

        if auto_column_detect and position_columns is None:
            column_canditates = ['position_' + suffix for suffix in component_suffixes]
            position_columns = [c for c in column_canditates if c in self.samples.columns]

        if auto_column_detect and velocity_columns is None:
            column_canditates = ['velocity_' + suffix for suffix in component_suffixes]
            velocity_columns = [c for c in column_canditates if c in self.samples.columns]

        if auto_column_detect and acceleration_columns is None:
            column_canditates = ['acceleration_' + suffix for suffix in component_suffixes]
            acceleration_columns = [c for c in column_canditates if c in self.samples.columns]

        # List of passed not-None column specifier lists.
        # The list will be used for inferring n_components.
        column_specifiers: list[list[str]] = []

        # Nest multi-component columns.
        if pixel_columns:
            self._check_component_columns(pixel_columns=pixel_columns)
            self.nest(pixel_columns, output_column='pixel')
            column_specifiers.append(pixel_columns)

        if position_columns:
            self._check_component_columns(position_columns=position_columns)
            self.nest(position_columns, output_column='position')
            column_specifiers.append(position_columns)

        if velocity_columns:
            self._check_component_columns(velocity_columns=velocity_columns)
            self.nest(velocity_columns, output_column='velocity')
            column_specifiers.append(velocity_columns)

        if acceleration_columns:
            self._check_component_columns(acceleration_columns=acceleration_columns)
            self.nest(acceleration_columns, output_column='acceleration')
            column_specifiers.append(acceleration_columns)

        self.n_components = self._infer_n_components(column_specifiers)
        # Warning if contains samples but no gaze-related columns were provided.
        # This can lead to failure in downstream methods that rely on those columns
        # (e.g., transformations).
        if len(self.samples) > 0 and not self.n_components:
            warnings.warn(
                'Gaze contains samples but no components could be inferred. \n'
                'This usually happens if you did not specify any column content'
                ' and the content could not be autodetected from the column names. \n'
                "Please specify 'pixel_columns', 'position_columns', 'velocity_columns'"
                " or 'acceleration_columns' explicitly during initialization."
                ' Otherwise, transformation methods may fail.',
            )

    def _init_time_column(
            self,
            time_column: str | None = None,
            time_unit: str | None = None,
    ) -> None:
        """Initialize time column."""
        # If no time column exists, create a new one starting with zero and set time unit to steps.
        if time_column is None and 'time' not in self.samples.columns:
            # In case we have an experiment with sampling rate given, we create a time
            if self.experiment is not None and self.experiment.sampling_rate is not None:
                self.samples = self.samples.with_columns(
                    time=polars.arange(0, len(self.samples)),
                )

                time_column = 'time'
                time_unit = 'step'

        # If no time_unit specified, assume milliseconds.
        if time_unit is None:
            time_unit = 'ms'

        # Rename time_column to 'time'.
        if time_column is not None and time_column != 'time':
            self.samples = self.samples.rename({time_column: 'time'})

        # Convert time column to milliseconds.
        if 'time' in self.samples.columns:
            self._convert_time_units(time_unit)

    def _convert_time_units(self, time_unit: str | None) -> None:
        """Convert the time column to milliseconds based on the specified time unit."""
        if time_unit == 's':
            self.samples = self.samples.with_columns(polars.col('time').mul(1000))

        elif time_unit == 'step':
            if self.experiment is not None:
                self.samples = self.samples.with_columns(
                    polars.col('time').mul(1000).truediv(self.experiment.sampling_rate),
                )
            else:
                raise ValueError(
                    "experiment with sampling rate must be specified if time_unit is 'step'",
                )

        elif time_unit != 'ms':
            raise ValueError(
                f"unsupported time unit '{time_unit}'. "
                "Supported units are 's' for seconds, 'ms' for milliseconds and "
                "'step' for steps.",
            )

        # Convert to int if possible.
        if self.samples.schema['time'] == polars.Float64:
            all_decimals = self.samples.select(
                polars.col('time').round().eq(polars.col('time')).all(),
            ).item()

            if all_decimals:
                self.samples = self.samples.with_columns(
                    polars.col('time').cast(polars.Int64),
                )

    def _init_experiment(
            self, experiment: Experiment | None, definition: pm.DatasetDefinition | None,
    ) -> None:
        """Explicitly passed experiment takes precedence over definition."""
        if definition is not None and experiment is None:
            self.experiment = definition.experiment
        else:
            self.experiment = experiment

    def __eq__(self, other: Gaze) -> bool:
        """Check equality between this and another :py:cls:`~pymovements.Gaze` object."""
        samples_equal = self.samples.equals(other.samples, null_equal=True)
        events_equal = self.events == other.events
        experiment_equal = self.experiment == other.experiment
        trial_columns_equal = self.trial_columns == other.trial_columns
        return samples_equal and events_equal and experiment_equal and trial_columns_equal

    def __str__(self) -> str:
        """Return string representation of Gaze.

        If :py:attr:`~.Gaze.messages` is not ``None``, includes ``messages=<N> rows``,
        where ``N`` is the number of rows.
        """
        fields = []

        if self.experiment is not None:
            fields.append(self.experiment.__str__())

        if self.samples is not None:
            fields.append(self.samples.__str__())

        if self.messages is not None:
            fields.append(f'messages={self.messages.height} rows')

        return '\n'.join(fields)

    def __repr__(self) -> str:
        """Return string representation of Gaze.

        If :py:attr:`~.Gaze.messages` is not ``None``, includes ``messages=<N> rows``,
        where ``N`` is the number of rows.
        """
        return self.__str__()

    def save(
            self,
            dirpath: str | Path,
            *,
            save_events: bool | None = None,
            save_samples: bool | None = None,
            save_experiment: bool | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Gaze:
        """Save data from the Gaze object in the provided directory.

        Depending on parameters it may save three files:
        * preprocessed gaze in samples (samples)
        * calculated gaze events (events)
        * metadatata experiment in YAML file (experiment).

        Data will be saved as feather or csv files.

        Returns
        -------
        Gaze
            Returns self, useful for method cascading.

        Parameters
        ----------
        dirpath: str | Path
            Absloute directory name to save data.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        save_events: bool | None
            Save events in events.{extension} file
        save_samples: bool | None
            Save samples in sample.{extension} file
        save_experiment: bool | None
            Save experiment metadata in experiment.yaml file
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)
        extension: str
            Extension specifies the fileformat to store the data. (default: 'feather')

        Raises
        ------
        ValueError
            If save_events is True and self.events is None or empty
        ValueError
            If save_experiment is True and self.experiment is None

        """
        # Create dir if does not exist
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        if save_events is None or save_events:
            if save_events and (self.events is None or len(self.events) == 0):
                raise ValueError('there are no events in the Gaze object')
            self.save_events(Path(f'{dirpath}/events.{extension}'), verbose=verbose)

        if save_samples is None or save_samples:
            self.save_samples(Path(f'{dirpath}/samples.{extension}'), verbose=verbose)

        if save_experiment is None or save_experiment:
            if verbose >= 2:
                print('Saving experiment file to', dirpath)
            if self.experiment is not None:
                self.experiment.to_yaml(Path(f"{dirpath}/experiment.yaml"))
            elif save_experiment is not None:
                raise ValueError('no experiment data in the Gaze object')
        return self

    def save_events(
            self,
            path: Path,
            *,
            verbose: int = 1,

    ) -> None:
        """Save gaze events to file.

        Data will be saved as the given file.

        Parameters
        ----------
        path: Path
            File to save data.
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)

        Raises
        ------
        ValueError
            If file extension in path is not in list of valid extensions.
        """
        events_out = self.events.frame.clone()
        extension = path.suffix[1:]

        if verbose >= 2:
            print('Saving events to ', path)

        if extension == 'feather':
            events_out.write_ipc(path)
        elif extension == 'csv':
            events_out.write_csv(path)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )

    def save_samples(
            self,
            path: Path,
            *,
            verbose: int = 1,
    ) -> None:
        """Save preprocessed gaze files.

        Samples will be saved to the file given by path.

        Parameters
        ----------
        path: Path
            File to save data.
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)

        Raises
        ------
        ValueError
            If file extension in path is not in list of valid extensions.
        """
        gaze = self.clone()
        extension = path.suffix[1:]

        if extension == 'csv':
            gaze.unnest()

        if verbose >= 2:
            print('Saving samples to', path)

        if extension == 'feather':
            gaze.samples.write_ipc(path)
        elif extension == 'csv':
            gaze.samples.write_csv(path)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def _check_trial_columns(trial_columns: list[str] | None, samples: polars.DataFrame) -> None:
    """Check trial_columns for integrity.

    Parameters
    ----------
    trial_columns: list[str] | None
        The name of the trial columns in the samples data frame.
    samples: polars.DataFrame
        The samples dataframe that is checked for columns.
    """
    if trial_columns:
        # Make sure there are no duplicates in trial_columns, else polars raises DuplicateError.
        if len(set(trial_columns)) != len(trial_columns):
            seen = set()
            dupes = []
            for column in trial_columns:
                if column in seen:
                    dupes.append(column)
                else:
                    seen.add(column)

            raise ValueError(f'duplicates in trial_columns: {", ".join(dupes)}')

        # Make sure all trial_columns exist in samples.
        if len(set(trial_columns).intersection(samples.columns)) != len(trial_columns):
            missing = set(trial_columns) - set(samples.columns)
            raise KeyError(f'trial_columns missing in samples: {", ".join(missing)}')


def _replace_nones_in_split_keys(
        sample_key_dtypes: list[type], events_key_dtypes: list[type],
) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    """Replace None values with comparable surrogates according to specified datatypes."""
    def _surrogate_none(dtype: type) -> float | str:
        """Return a comparable surrogate value for a particular datatype."""
        if dtype in {float, int, bool}:
            return -math.inf
        if dtype == str:
            return ''
        raise TypeError(
            f'dtype {dtype.__name__} not supported as "by" column dtype in split(). '
            f'supported dtypes are str, float, int and bool',
        )

    def _replace_nones_in_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        """Replace None values with comparable surrogates in a particular key."""
        if None in key:
            return tuple(
                element if element is not None else _surrogate_none(dtype)
                for element, dtype in zip(key, key_dtypes)
            )
        return key

    # Create single list of key dtypes.
    if sample_key_dtypes and events_key_dtypes:
        if sample_key_dtypes != events_key_dtypes:
            raise TypeError(
                '"by" column dtypes do not match between samples and events:'
                f'{[c.__name__ for c in sample_key_dtypes]}'
                f' != {[c.__name__ for c in events_key_dtypes]}',
            )
        key_dtypes = sample_key_dtypes
    elif sample_key_dtypes and not events_key_dtypes:
        key_dtypes = sample_key_dtypes
    else:
        key_dtypes = events_key_dtypes

    return _replace_nones_in_key


def _check_messages(messages: polars.DataFrame) -> None:
    """Check that messages is a polars.DataFrame with the two columns time and content."""
    if messages is not None:
        if not isinstance(messages, polars.DataFrame):
            raise TypeError(
                "The `messages` must be a polars DataFrame with columns ['time', 'content'], "
                f"not {type(messages)}.",
            )
        required_cols = {'time', 'content'}
        if not required_cols.issubset(set(messages.columns)):
            raise TypeError(
                "The `messages` polars DataFrame must contain the columns ['time', 'content'].",
            )
