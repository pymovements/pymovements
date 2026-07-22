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
"""DatasetDefinition module."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import KW_ONLY
from pathlib import Path
from typing import Any
from warnings import warn

import yaml

from pymovements._utils._html import repr_html
from pymovements.dataset._utils._yaml import reverse_substitute_types
from pymovements.dataset._utils._yaml import substitute_types
from pymovements.dataset._utils._yaml import type_constructor
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


ResourcesLike = Sequence[dict[str, Any]] | dict[str, Sequence[dict[str, Any]]]

yaml.add_multi_constructor('!', type_constructor, Loader=yaml.SafeLoader)


@repr_html(['name', 'long_name', 'description', 'experiment', 'resources'])
@dataclass
class DatasetDefinition:
    """Definition to initialize a :py:class:`~Dataset`.

    Attributes
    ----------
    name: str
        The name of the dataset. (default: '.')
    long_name: str | None
        The entire name of the dataset. (default: None)
    description: str | None
        A fulltext description of the dataset.
        (default: None)
    resources: ResourceDefinitions
        A list of dataset resources. Each list entry must be a dictionary with the following keys:

        - `resource`: The url suffix of the resource.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

        (default: ResourceDefinitions())
    experiment: Experiment | None
        The experiment definition. (default: None)
    custom_read_kwargs: dict[str, dict[str, Any]] | None
        If specified, these keyword arguments will be passed to the file reading function. The
        behavior of this argument depends on the file extension of the dataset files.
        If the file extension is `.csv`, the keyword arguments will be passed
        to :py:func:`polars.read_csv`. If the file extension is `.asc`, the keyword arguments
        will be passed to :py:func:`pymovements.gaze.from_asc`.
        See Notes for more details on how to use this argument.
        (default: field(default_factory=dict))

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    column_map : dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    trial_columns: list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials, and the transformation
        methods will be applied to each trial separately. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    time_column: str | None
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    time_unit: str | None
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. (default: 'ms')

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    pixel_columns: list[str] | None
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    position_columns: list[str] | None
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    velocity_columns: list[str] | None
        The name of the velocity columns in the input data frame. These columns will be nested
        into the column ``velocity``. If the list is empty or None, the nested ``velocity``
        column will not be created. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    acceleration_columns: list[str] | None
        The name of the acceleration columns in the input data frame. These columns will be
        nested into the column ``acceleration``. If the list is empty or None, the nested
        ``acceleration`` column will not be created. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    distance_column : str | None
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken from
        the experiment definition. This column will be renamed to ``distance``. (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.

    Parameters
    ----------
    name: str
        The name of the dataset. (default: '.')
    long_name: str | None
        The entire name of the dataset. (default: None)
    description: str | None
        A fulltext description of the dataset.
        (default: None)
    resources: ResourceDefinitions | ResourcesLike | None
        A list of dataset resources. Each list entry must be a dictionary with the following keys:

        - `source`: The url suffix of the resource.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

        (default: None)
    experiment: Experiment | None
        The experiment definition. (default: None)
    custom_read_kwargs: dict[str, dict[str, Any]] | None
        If specified, these keyword arguments will be passed to the file reading function. The
        behavior of this argument depends on the file extension of the dataset files.
        If the file extension is `.csv`, the keyword arguments will be passed
        to :py:func:`polars.read_csv`. If the file extension is `.asc`, the keyword arguments
        will be passed to :py:func:`pymovements.gaze.from_asc`.
        See Notes for more details on how to use this argument.
        (default: None)

        .. deprecated:: v0.25.0
           Please use :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
           This field will be removed in v0.30.0.
    column_map : dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)
    trial_columns: list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials, and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``. (default: None)
    time_unit: str | None
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step,' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. (default: 'ms')
    pixel_columns: list[str] | None
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
    distance_column : str | None
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken from
        the experiment definition. This column will be renamed to ``distance``. (default: None)

    Notes
    -----
    .. deprecated:: v0.25.0
       The ``custom_read_kwargs`` attribute is deprecated.
       Please specify :py:attr:`~pymovements.ResourceDefinition.load_kwargs` instead.
       This field will be removed in v0.30.0.

    When working with the ``custom_read_kwargs`` attribute, there are specific use cases and
    considerations to keep in mind, especially for reading csv files:

    1. Custom separator:
       To read a csv file with a custom separator, you can pass the `separator` keyword argument to
       ``custom_read_kwargs``. For example pass ``custom_read_kwargs={'separator': ';'}`` to
       read a semicolon-separated csv file.

    2. Reading subset of columns:
       To read only specific columns, specify them in ``custom_read_kwargs``. For example:
       ``custom_read_kwargs={'columns': ['col1', 'col2']}``

    3. Specifying column datatypes:
       :py:func:`polars.read_csv` infers data types from a fixed number of rows,
       which might not be accurate for the entire dataset.
       To ensure correct data types, you can pass a dictionary to the
       ``schema_overrides`` keyword argument in ``custom_read_kwargs``.
       Use data types from the :py:mod:`polars` library.
       For instance:
       ``custom_read_kwargs={'schema_overrides': {'col1': polars.Int64, 'col2': polars.Float64}}``
    """

    # pylint: disable=too-many-instance-attributes
    name: str = '.'

    _: KW_ONLY  # all fields below can only be passed as a positional argument.

    long_name: str | None = None

    description: str | None = None

    resources: ResourceDefinitions = field(default_factory=ResourceDefinitions)

    experiment: Experiment | None = field(default_factory=Experiment)

    custom_read_kwargs: dict[str, dict[str, Any]] | None = None

    column_map: dict[str, str] | None = None

    trial_columns: list[str] | None = None
    time_column: str | None = None
    time_unit: str | None = None
    pixel_columns: list[str] | None = None
    position_columns: list[str] | None = None
    velocity_columns: list[str] | None = None
    acceleration_columns: list[str] | None = None
    distance_column: str | None = None

    def __init__(
            self,
            name: str = '.',
            *,
            long_name: str | None = None,
            description: str | None = None,
            resources: ResourceDefinitions | ResourcesLike | None = None,
            experiment: Experiment | None = None,
            custom_read_kwargs: dict[str, dict[str, Any]] | None = None,
            column_map: dict[str, str] | None = None,
            trial_columns: list[str] | None = None,
            time_column: str | None = None,
            time_unit: str | None = None,
            pixel_columns: list[str] | None = None,
            position_columns: list[str] | None = None,
            velocity_columns: list[str] | None = None,
            acceleration_columns: list[str] | None = None,
            distance_column: str | None = None,
    ) -> None:
        self.name = name
        self.long_name = long_name
        self.description = description

        self.experiment = experiment

        self.resources = self._initialize_resources(resources=resources)

        if trial_columns is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.trial_columns is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.trial_columns = trial_columns

        if time_column is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.time_column is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.time_column = time_column

        if time_unit is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.time_unit is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.time_unit = time_unit

        if pixel_columns is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.pixel_columns is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.pixel_columns = pixel_columns

        if position_columns is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.position_columns is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.position_columns = position_columns

        if velocity_columns is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.velocity_columns is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.velocity_columns = velocity_columns

        if acceleration_columns is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.acceleration_columns is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.acceleration_columns = acceleration_columns

        if distance_column is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.distance_column is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.distance_column = distance_column

        if column_map is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.column_map is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.column_map = column_map

        if custom_read_kwargs is not None:
            warn(
                DeprecationWarning(
                    'DatasetDefinition.custom_read_kwargs is deprecated since version v0.25.0. '
                    'Please specify ResourceDefinition.load_kwargs instead. '
                    'This field will be removed in v0.30.0.',
                ),
            )
            self.custom_read_kwargs = custom_read_kwargs

    @staticmethod
    def from_yaml(path: str | Path) -> DatasetDefinition:
        """Load a dataset definition from a YAML file.

        Parameters
        ----------
        path: str | Path
            Path to the YAML definition file

        Returns
        -------
        DatasetDefinition
            Initialized dataset definition
        """
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Convert experiment dict to Experiment object if present
        if 'experiment' in data:
            data['experiment'] = Experiment.from_dict(data['experiment'])

        data = reverse_substitute_types(data)
        # Initialize DatasetDefinition with YAML data
        return DatasetDefinition(**data)

    def to_dict(
        self,
        *,
        exclude_private: bool = True,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """Return dictionary representation.

        Parameters
        ----------
        exclude_private: bool
            Exclude attributes that start with ``_``.
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of dataset definition.
        """
        data = asdict(self)

        # Delete private fields from dictionary.
        if exclude_private:
            # we need a separate list of keys here or else we get a
            # RuntimeError: dictionary changed size during iteration
            for key in list(data.keys()):
                if key.startswith('_'):
                    del data[key]

        # Delete fields that evaluate to False (False, None, [], {})
        if exclude_none:
            for key, value in list(data.items()):
                if not isinstance(value, (bool, int, float)) and not value:
                    del data[key]

        # Convert those object fields to dictionaries.
        if 'experiment' in data and data['experiment'] is not None:
            data['experiment'] = data['experiment'].to_dict(exclude_none=exclude_none)
        if 'resources' in data and data['resources'] is not None:
            data['resources'] = self.resources.to_dicts(exclude_none=exclude_none)

        return data

    def to_yaml(
        self,
        path: str | Path,
        *,
        exclude_private: bool = True,
        exclude_none: bool = True,
    ) -> None:
        """Save a dataset definition to a YAML file.

        Parameters
        ----------
        path: str | Path
            Path where to save the YAML file to.
        exclude_private: bool
            Exclude attributes that start with ``_``.
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.
        """
        data = self.to_dict(exclude_private=exclude_private, exclude_none=exclude_none)

        data = substitute_types(data)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)

    def _initialize_resources(
            self,
            resources: ResourceDefinitions | ResourcesLike | None,
    ) -> ResourceDefinitions:
        """Initialize ``ResourceDefinitions`` instance if necessary."""
        if isinstance(resources, ResourceDefinitions):
            return resources

        if resources is None:
            return ResourceDefinitions()

        if isinstance(resources, Sequence):
            return ResourceDefinitions(resources)

        if isinstance(resources, dict):
            # this calls a deprecated method and will be removed in the future.
            return ResourceDefinitions.from_dict(resources)

        raise TypeError(
            f'resources is of type {type(resources).__name__} but must be of type'
            ' ResourceDefinitions, list, or dict.',
        )
