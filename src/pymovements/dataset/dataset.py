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
"""Provides the Dataset class."""
from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pymovements._utils._html import repr_html
from pymovements.dataset import dataset_download
from pymovements.dataset import dataset_files
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import DatasetLibrary
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.events import Events
from pymovements.events.precomputed import PrecomputedEventDataFrame
from pymovements.gaze import Gaze
from pymovements.reading_measures import ReadingMeasures


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@repr_html()
class Dataset:
    """Dataset base class.

    Initialize the dataset object.

    Parameters
    ----------
    definition: str | Path | DatasetDefinition | type[DatasetDefinition]
        Dataset definition to initialize dataset with.
    path : str | Path | DatasetPaths
        Path to the dataset directory. You can set up a custom directory structure by passing a
        :py:class:`~pymovements.dataset.DatasetPaths` instance.
    """

    def __init__(
            self,
            definition: str | Path | DatasetDefinition | type[DatasetDefinition],
            path: str | Path | DatasetPaths,
    ):
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.gaze: list[Gaze] = []
        self.precomputed_events: list[PrecomputedEventDataFrame] = []
        self.precomputed_reading_measures: list[ReadingMeasures] = []

        # Handle different definition input types
        if isinstance(definition, (str, Path)):
            # Check if it's a path to a YAML file
            if isinstance(definition, Path) or str(definition).endswith('.yaml'):
                self.definition = DatasetDefinition.from_yaml(definition)
            else:
                # Try to load from registered datasets
                self.definition = DatasetLibrary.get(definition)

        elif isinstance(definition, type):
            self.definition = definition()
        else:
            self.definition = deepcopy(definition)

        # Handle path setup
        if isinstance(path, (str, Path)):
            self.paths = DatasetPaths(root=path, dataset='.')
        else:
            self.paths = deepcopy(path)

        # Fill dataset directory name with dataset definition name if specified
        self.paths.fill_name(self.definition.name)

    def load(
            self,
            *,
            events: bool | None = None,
            preprocessed: bool = False,
            subset: dict[str, float | int | str | list[float | int | str]] | None = None,
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Parse file information and load all gaze files.

        The parsed file information is assigned to the `fileinfo` attribute.
        All gaze files will be loaded as dataframes and assigned to the `gaze` attribute.

        Parameters
        ----------
        events: bool | None
            If ``True``, load previously saved event data. (default: None)
        preprocessed: bool
            If ``True``, load previously saved preprocessed data, otherwise load raw data.
            (default: False)
        subset:  dict[str, float | int | str | list[float | int | str]] | None
            If specified, load only a subset of the dataset. All keys in the dictionary must be
            present in the fileinfo dataframe inferred by `scan()`. Values can be either
            float, int , str or a list of these. (default: None)
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`. (default: None)
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`,
            `tsv`, `txt`, `asc`.
            (default: 'feather')

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        self.scan()
        self.fileinfo = dataset_files.take_subset(fileinfo=self.fileinfo, subset=subset)

        if self.definition.resources.has_content('gaze'):
            self.load_gaze_files(
                preprocessed=preprocessed,
                preprocessed_dirname=preprocessed_dirname,
                extension=extension,
            )

        # Event files precomputed by authors of the dataset
        if self.definition.resources.has_content('precomputed_events'):
            self.load_precomputed_events()

        # Reading measures files precomputed by authors of the dataset
        if self.definition.resources.has_content('precomputed_reading_measures'):
            self.load_precomputed_reading_measures()

        # Events detected previously by pymovements
        if events:
            self.load_event_files(
                events_dirname=events_dirname,
                extension=extension,
            )

        return self

    @property
    def events(self) -> tuple[Events, ...]:
        """Return ``Events`` for all ``Gaze`` objects in the ``Dataset``.

        Each element in the returned tuple references :py:attr:`~pymovements.Gaze.events` of the
        corresponding :py:class:`~pymovements.Gaze` in :py:attr:`~pymovements.Dataset.gaze`.

        Returns
        -------
        tuple[Events, ...]
            Tuple mapping ``Dataset.events[i]`` to ``Dataset.gaze[i].events``.

        Notes
        -----
        Changes to ``Dataset.events[i]`` are also reflected in ``Dataset.gaze[i].events`` and vice
        versa as they both reference the same :py:class:`~pymovements.Events` object.
        """
        return tuple(gaze.events for gaze in self.gaze)

    @events.setter
    def events(self, data: Sequence[Events]) -> None:
        """Assign ``Events`` to each ``Gaze`` object in the ``Dataset``.

        Each :py:class:`~pymovements.Gaze` in :py:attr:`~pymovements.Dataset.gaze` is updated with
        the corresponding :py:class:`~pymovements.Events` of the input.

        Parameters
        ----------
        data: Sequence[Events]
            Must have the same length as :py:attr:`~pymovements.Dataset.gaze`.

        Raises
        ------
        ValueError
            If the lengths of ``data`` and :py:attr:`~pymovements.Dataset.gaze` do not match.

        Notes
        -----
        Assigning to a single element of :py:attr:`~pymovements.Dataset.events` raises a
        ``TypeError`` as :py:attr:`~pymovements.Dataset.events` returns an immutable tuple.

        To assign to a single element, assign directly via ``Dataset.gaze[i].events = new_events``
        instead.
        """
        if len(data) != len(self.gaze):
            raise ValueError(
                f"Number of events ({len(data)}) does not match "
                f"number of gazes ({len(self.gaze)}).",
            )
        for gaze, ev in zip(self.gaze, data):
            gaze.events = ev

    def scan(self) -> Dataset:
        """Infer information from filepaths and filenames.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If no regular expression for parsing filenames is defined.
        RuntimeError
            If an error occurred during matching filenames or no files have been found.
        """
        self.fileinfo = dataset_files.scan_dataset(definition=self.definition, paths=self.paths)
        return self

    def load_gaze_files(
            self,
            preprocessed: bool = False,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Load all available gaze data files.

        Parameters
        ----------
        preprocessed: bool
            If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
            (default: False)
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`,
            `tsv`, `txt`, `asc`.
            (default: 'feather')

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        RuntimeError
            If file type of gaze file is not supported.
        """
        self._check_fileinfo()
        self.gaze = dataset_files.load_gaze_files(
            definition=self.definition,
            fileinfo=self.fileinfo['gaze'],
            paths=self.paths,
            preprocessed=preprocessed,
            preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        return self

    def load_precomputed_events(self) -> None:
        """Load precomputed events.

        This method checks that the file information for precomputed events is available,
        then loads each event file listed in `self.fileinfo['precomputed_events']` using
        the dataset definition and path settings. The resulting list of
        `PrecomputedEventDataFrame` objects is assigned to `self.precomputed_events`.

        Supported file extensions:
        - CSV-like: .csv, .tsv, .txt
        - JSON like: .jsonl, .ndjson
        - RDA like: .rda

        Raises
        ------
        ValueError
            If the file info is missing or improperly formatted.
        """
        self._check_fileinfo()
        self.precomputed_events = dataset_files.load_precomputed_event_files(
            self.definition,
            self.fileinfo['precomputed_events'],
            self.paths,
        )

    def load_precomputed_reading_measures(self) -> None:
        """Load precomputed reading measures.

        This method checks that the file information for precomputed reading measures are
        available, then loads each event file listed in
        `self.fileinfo['precomputed_reading_measures']` using the dataset definition and
        path settings. The resulting list of `ReadingMeasures` objects is assigned to
        `self.reading_measures`.

        Supported file extensions:
        - CSV-like: .csv, .tsv, .txt
        - Excel-like: .xlsx
        - RDA like: .rda

        Raises
        ------
        ValueError
            If the file info is missing or improperly formatted.
        """
        self._check_fileinfo()
        self.precomputed_reading_measures = dataset_files.load_precomputed_reading_measures(
            self.definition,
            self.fileinfo['precomputed_reading_measures'],
            self.paths,
        )

    def split_gaze_data(
            self,
            by: Sequence[str],
    ) -> None:
        """Split gaze data into separated Gaze objects.

        Parameters
        ----------
        by: Sequence[str]
            Column(s) to split dataframe by.
        """
        fileinfo_dicts = self.fileinfo['gaze'].to_dicts()

        all_gaze_frames = []
        all_fileinfo_rows = []

        for frame, fileinfo_row in zip(self.gaze, fileinfo_dicts):
            split_frames = frame.split(by=by, as_dict=False)
            all_gaze_frames.extend(split_frames)
            all_fileinfo_rows.extend([fileinfo_row] * len(split_frames))

        self.gaze = all_gaze_frames
        self.fileinfo['gaze'] = pl.concat([pl.from_dict(row) for row in all_fileinfo_rows])

    def split_precomputed_events(
            self,
            by: list[str] | str,
    ) -> None:
        """Split precomputed event data into separated PrecomputedEventDataFrame's.

        Parameters
        ----------
        by: list[str] | str
            Column's to split dataframe by.
        """
        if isinstance(by, str):
            by = [by]
        self.precomputed_events = [
            PrecomputedEventDataFrame(new_frame) for _frame in self.precomputed_events
            for new_frame in _frame.frame.partition_by(by=by)
        ]

    def load_event_files(
            self,
            events_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Load all available event files.

        Parameters
        ----------
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`. (default: None)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            (default: 'feather')

        Returns
        -------
        Dataset
            List of event dataframes.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        ValueError
            If extension is not in list of valid extensions.
        """
        self._check_fileinfo()
        events = dataset_files.load_event_files(
            definition=self.definition,
            fileinfo=self.fileinfo['gaze'],
            paths=self.paths,
            events_dirname=events_dirname,
            extension=extension,
        )
        self.events = events
        return self

    def apply(
            self,
            function: str,
            *,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Apply preprocessing method to all Gazes in Dataset.

        Parameters
        ----------
        function: str
            Name of the preprocessing function to apply.
        verbose : bool
            If True, show progress bar of computation. (default: True)
        **kwargs: Any
            kwargs that will be forwarded when calling the preprocessing method.

        Returns
        -------
        Dataset
            Returns preprocessed dataset.

        Examples
        --------
        Let's load in our dataset first,
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='toy_dataset')
        >>> dataset.download()# doctest:+ELLIPSIS
        Downloading ... to toy_dataset...downloads...
        Checking integrity of ...
        Extracting ... to toy_dataset...raw
        <pymovements.dataset.dataset.Dataset object at ...>
        >>> dataset.load()# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        Use apply for your gaze transformations:
        >>> dataset.apply('pix2deg')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        >>> dataset.apply('pos2vel', method='neighbors')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        Use apply for your event detection:
        >>> dataset.apply('ivt')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        >>> dataset.apply('microsaccades', minimum_duration=8)# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        Use apply for upsampling, downsampling or making the sampling rate constant
        using resample:
        >>> dataset.apply('resample', resampling_rate=2000)# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>
        """
        self._check_gaze()

        disable_progressbar = not verbose
        for gaze in tqdm(self.gaze, disable=disable_progressbar):
            gaze.apply(function, **kwargs)

        return self

    def clip(
            self,
            lower_bound: int | float | None,
            upper_bound: int | float | None,
            *,
            input_column: str,
            output_column: str,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Clip gaze signal values.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is clipped.

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
        verbose : bool
            If True, show progress of computation. (default: True)
        **kwargs: Any
            Additional keyword arguments to be passed to the :func:`~transforms.clip()` method.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        return self.apply(
            'clip',
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            input_column=input_column,
            output_column=output_column,
            verbose=verbose,
            **kwargs,
        )

    def resample(
            self,
            resampling_rate: float,
            columns: str | list[str] = 'all',
            fill_null_strategy: str = 'interpolate_linear',
            verbose: bool = True,
    ) -> Dataset:
        """Resample a DataFrame to a new sampling rate by timestamps in time column.

        The DataFrame is resampled by upsampling or downsampling the data to the new sampling rate.
        Can also be used to achieve a constant sampling rate for inconsistent data.

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
        verbose: bool
            If True, show progress of computation. (default: True)

        Returns
        -------
        Dataset
        """
        return self.apply(
            'resample',
            resampling_rate=resampling_rate,
            fill_null_strategy=fill_null_strategy,
            columns=columns,
            verbose=verbose,
        )

    def pix2deg(self, verbose: bool = True) -> Dataset:
        """Compute gaze positions in degrees of visual angle from pixel coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting dva columns.

        Parameters
        ----------
        verbose : bool
            If True, show progress of computation. (default: True)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        return self.apply('pix2deg', verbose=verbose)

    def deg2pix(
            self,
            pixel_origin: str = 'upper left',
            position_column: str = 'position',
            pixel_column: str = 'pixel',
            verbose: bool = True,
    ) -> Dataset:
        """Compute gaze positions in pixel coordinates from degrees of visual angle.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting dva columns.

        Parameters
        ----------
        pixel_origin: str
            The desired location of the pixel origin. (default: 'upper left')
            Supported values: ``center``, ``upper left``.
        position_column: str
            The input position column name. (default: 'position')
        pixel_column: str
            The output pixel column name. (default: 'pixel')
        verbose : bool
            If True, show progress of computation. (default: True)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        return self.apply(
            'deg2pix',
            pixel_origin=pixel_origin,
            position_column=position_column,
            pixel_column=pixel_column,
            verbose=verbose,
        )

    def pos2acc(
            self,
            *,
            degree: int = 2,
            window_length: int = 7,
            padding: str | float | int | None = 'nearest',
            verbose: bool = True,
    ) -> Dataset:
        """Compute gaze accelerations in dva/s^2 from dva coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting acceleration columns.

        Parameters
        ----------
        degree: int
            The degree of the polynomial to use. (default: 2)
        window_length: int
            The window size to use. (default: 7)
        padding: str | float | int | None
            The padding method to use. See ``savitzky_golay`` for details. (default: 'nearest')
        verbose: bool
            If True, show progress of computation. (default: True)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        return self.apply(
            'pos2acc',
            window_length=window_length,
            degree=degree,
            padding=padding,
            verbose=verbose,
        )

    def pos2vel(
            self,
            method: str = 'fivepoint',
            *,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Compute gaze velocities in dva/s from dva coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting velocity columns.

        Parameters
        ----------
        method: str
            Computation method. See :func:`~transforms.pos2vel()` for details.
            (default: 'fivepoint')
        verbose: bool
            If True, show progress of computation. (default: True)
        **kwargs: Any
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        return self.apply('pos2vel', method=method, verbose=verbose, **kwargs)

    def detect_events(
            self,
            method: Callable[..., Events] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Detect events by applying a specific event detection method.

        Parameters
        ----------
        method : Callable[..., Events] | str
            The event detection method to be applied.
        eye: str
            Select which eye to choose. Valid options are ``auto``, ``left``, ``right`` or ``None``.
            If ``auto`` is passed, eye is inferred in the order ``['right', 'left', 'eye']`` from
            the available :py:attr:`~.Dataset.gaze` dataframe columns. (default: 'auto')
        clear: bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
             merged into the existing one. (default: False)
        verbose: bool
            If ``True``, show progress bar. (default: True)
        **kwargs: Any
            Additional keyword arguments to be passed to the event detection method.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If gaze files have not been loaded yet or gaze files do not contain the right columns.
        """
        return self.detect(
            method=method,
            eye=eye,
            clear=clear,
            verbose=verbose,
            **kwargs,
        )

    def detect(
            self,
            method: Callable[..., Events] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Detect events by applying a specific event detection method.

        Alias for :py:meth:`pymovements.Dataset.detect_events`

        Parameters
        ----------
        method: Callable[..., Events] | str
            The event detection method to be applied.
        eye: str
            Select which eye to choose. Valid options are ``auto``, ``left``, ``right`` or ``None``.
            If ``auto`` is passed, eye is inferred in the order ``['right', 'left', 'eye']`` from
            the available :py:attr:`~.Dataset.gaze` dataframe columns. (default: 'auto')
        clear: bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
             merged into the existing one. (default: False)
        verbose: bool
            If ``True``, show progress bar. (default: True)
        **kwargs: Any
            Additional keyword arguments to be passed to the event detection method.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If gaze files have not been loaded yet or gaze files do not contain the right columns.
        """
        self._check_gaze()

        disable_progressbar = not verbose
        for gaze, fileinfo_row in tqdm(
                zip(self.gaze, self.fileinfo['gaze'].to_dicts()),
                disable=disable_progressbar,
        ):
            gaze.detect(method, eye=eye, clear=clear, **kwargs)
            # workaround until events are fully part of the Gaze
            gaze.events.frame = dataset_files.add_fileinfo(
                definition=self.definition,
                df=gaze.events.frame,
                fileinfo=fileinfo_row,
            )

        return self

    def drop_event_properties(
            self,
            event_properties: str | list[str],
    ) -> Dataset:
        """Remove event properties from the event dataframe.

        Parameters
        ----------
        event_properties: str | list[str]
            The event properties to remove.

        Raises
        ------
        InvalidProperty
            If ``event_properties`` does not exist in the event dataframe

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        for gaze in self.gaze:
            gaze.drop_event_properties(event_properties)
        return self

    def compute_event_properties(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
            name: str | None = None,
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property and add it as a column to the event dataframe.

        Parameters
        ----------
        event_properties: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
            The event properties to compute.
        name: str | None
            Process only events that match the name. (default: None)
        verbose : bool
            If ``True``, show progress bar. (default: True)

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events` for an overview of supported properties.
        RuntimeError
            If specified event name ``name`` is missing from ``events``.
        ValueError
            If the computed property already exists in the event dataframe.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        for gaze in tqdm(self.gaze, disable=not verbose):
            gaze.compute_event_properties(event_properties, name=name)
        return self

    def compute_properties(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
            name: str | None = None,
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property for and add it as a column to the event dataframe.

        Alias for :py:meth:`pymovements.Dataset.compute_event_properties`

        Parameters
        ----------
        event_properties: str | tuple[str, dict[str, Any]] | list[str | tuple[str, dict[str, Any]]]
            The event properties to compute.
        name: str | None
            Process only events that match the name. (default: None)
        verbose: bool
            If ``True``, show progress bar. (default: True)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events` for an overview of supported properties.
        """
        return self.compute_event_properties(
            event_properties=event_properties,
            name=name,
            verbose=verbose,
        )

    def clear_events(self) -> Dataset:
        """Clear event DataFrame.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        if len(self.events) == 0:
            return self

        for gaze in self.gaze:
            gaze.events = Events()

        return self

    def save(
            self,
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze and event files.

        Data will be saved as feather/csv files to ``Dataset.preprocessed_roothpath`` or
        ``Dataset.events_roothpath`` with the same directory structure as the raw data.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Parameters
        ----------
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`. (default: None)
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)
        extension: str
            Extension specifies the fileformat to store the data. (default: 'feather')
        """
        self.save_events(events_dirname, verbose=verbose, extension=extension)
        self.save_preprocessed(preprocessed_dirname, verbose=verbose, extension=extension)
        return self

    def save_events(
            self,
            events_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save events to files.

        Data will be saved as feather files to ``Dataset.events_roothpath`` with the same directory
        structure as the raw data.

        Parameters
        ----------
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`. (default: None)
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            (default: 'feather')

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        ValueError
            If extension is not in list of valid extensions.
        """
        dataset_files.save_events(
            events=self.events,
            fileinfo=self.fileinfo['gaze'],
            paths=self.paths,
            events_dirname=events_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def save_preprocessed(
            self,
            preprocessed_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze files.

        Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` with the same
        directory structure as the raw data.

        Parameters
        ----------
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            (default: 'feather')

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        ValueError
            If extension is not in list of valid extensions.
        """
        dataset_files.save_preprocessed(
            gazes=self.gaze,
            fileinfo=self.fileinfo['gaze'],
            paths=self.paths,
            preprocessed_dirname=preprocessed_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def download(
            self,
            *,
            extract: bool = True,
            remove_finished: bool = False,
            resume: bool = True,
            verbose: int = 1,
    ) -> Dataset:
        """Download dataset resources.

        This downloads all resources of the dataset. Per default this also extracts all archives
        into :py:meth:`Dataset.paths.raw`,
        To save space on your device you can remove the archive files after
        successful extraction with ``remove_finished=True``.

        If a corresponding file already exists in the local system, its checksum is calculated and
        checked against the expected checksum.
        Downloading will be evaded if the integrity of the existing file can be verified.
        If the existing file does not match the expected checksum it is overwritten with the
        downloaded new file.

        Parameters
        ----------
        extract: bool
            Extract dataset archive files. (default: True)
        remove_finished: bool
            Remove archive files after extraction. (default: False)
        resume: bool
            Resume previous extraction by skipping existing files.
            Checks for correct size of existing files but not integrity. (default: True)
        verbose: int
            Verbosity levels: (1) Show download progress bar and print info messages on downloading
            and extracting archive files without printing messages for recursive archive extraction.
            (2) Print additional messages for each recursive archive extract. (default: 1)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If number of mirrors or number of resources specified for dataset is zero.
        RuntimeError
            If downloading a resource failed for all given mirrors.
        """
        logger.info(self._disclaimer())

        dataset_download.download_dataset(
            definition=self.definition,
            paths=self.paths,
            extract=extract,
            remove_finished=remove_finished,
            resume=resume,
            verbose=bool(verbose),
        )
        return self

    def extract(
            self,
            *,
            remove_finished: bool = False,
            remove_top_level: bool = True,
            resume: bool = True,
            verbose: int = 1,
    ) -> Dataset:
        """Extract downloaded dataset archive files.

        Parameters
        ----------
        remove_finished: bool
            Remove archive files after extraction. (default: False)
        remove_top_level: bool
            If ``True``, remove the top-level directory if it has only one child. (default: True)
        resume: bool
            Resume previous extraction by skipping existing files.
            Checks for correct size of existing files but not integrity. (default: True)
        verbose: int
            Verbosity levels: (1) Print messages for extracting each dataset resource without
            printing messages for recursive archives. (2) Print additional messages for each
            recursive archive extract. (default: 1)

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        dataset_download.extract_dataset(
            definition=self.definition,
            paths=self.paths,
            remove_finished=remove_finished,
            remove_top_level=remove_top_level,
            resume=resume,
            verbose=verbose,
        )
        return self

    @property
    def path(self) -> Path:
        """The path to the dataset directory.

        The dataset path points to the dataset directory under the root path. Per default the
        dataset path points to the exact same directory as the root path. Add ``dataset_dirname``
        to your initialization call to specify an explicit dataset directory in your root path.

        Returns
        -------
        Path
            Path to the dataset directory.

        Example
        -------
        By passing a `str` or a `Path` as `path` during initialization you can explicitly set the
        directory path of the dataset:
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/dataset')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/ToyDataset')

        You can also specify an alternative dataset directory name:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/', dataset='my_dataset')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/my_dataset')
        """
        return self.paths.dataset

    def _check_fileinfo(self) -> None:
        """Check if fileinfo attribute is set and there is at least one row present."""
        if self.fileinfo is None:
            raise AttributeError(
                'fileinfo was not loaded yet. please run load() or scan() beforehand',
            )
        if len(self.fileinfo) == 0:
            raise AttributeError('no files present in fileinfo attribute')

    def _check_gaze(self) -> None:
        """Check if gaze attribute is set and there is at least one gaze dataframe available."""
        if self.gaze is None:
            raise AttributeError('gaze files were not loaded yet. please run load() beforehand')
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')

    def _disclaimer(self) -> str:
        """Return string for dataset download disclaimer."""
        if self.definition.long_name is not None:
            dataset_name = self.definition.long_name
        else:
            dataset_name = self.definition.name + ' dataset'

        return f"""
        You are downloading the {dataset_name}. Please be aware that pymovements does not
        host or distribute any dataset resources and only provides a convenient interface to
        download the public dataset resources that were published by their respective authors.

        Please cite the referenced publication if you intend to use the dataset in your research.
        """

    def report(self):
        """Generate a comprehensive overview report of the dataset.
        
        Prints a detailed overview of the dataset including definition details, 
        eyetracker configuration, screen setup, resource files, and data structure.
        
        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        print("=" * 80)
        print(f"DATASET OVERVIEW")
        print("=" * 80)

        definition = self.definition

        # Basic information
        print("\nDEFINITION:")
        # DatasetDefinition explicit fields printed if present
        if hasattr(definition, 'name') and definition.name is not None:
            print(f"  Name: {definition.name}")
        if hasattr(definition, 'long_name') and definition.long_name:
            print(f"  Long Name: {definition.long_name}")
        if hasattr(self, 'path') and self.path is not None:
            print(f"  Path: {self.path}")
        if hasattr(definition, 'resources') and definition.resources:
            print(f"  Number of Resources: {len(getattr(definition, 'resources', []))}")
        if hasattr(definition, 'mirrors') and definition.mirrors:
            print(f"  Mirrors: {', '.join([f'{k}({len(v)})' for k, v in definition.mirrors.items()])}")
        if hasattr(definition, 'extract') and definition.extract:
            print(f"  Extract: {definition.extract}")
        if hasattr(definition, 'column_map') and definition.column_map:
            print(f"  Column Map: {definition.column_map}")
        if hasattr(definition, 'trial_columns') and definition.trial_columns:
            print(f"  Trial Columns: {definition.trial_columns}")
        if hasattr(definition, 'time_column') and definition.time_column:
            print(f"  Time Column: {definition.time_column}")
        if hasattr(definition, 'time_unit') and definition.time_unit:
            print(f"  Time Unit: {definition.time_unit}")
        if hasattr(definition, 'pixel_columns') and definition.pixel_columns:
            print(f"  Pixel Columns: {definition.pixel_columns}")
        if hasattr(definition, 'position_columns') and definition.position_columns:
            print(f"  Position Columns: {definition.position_columns}")
        if hasattr(definition, 'velocity_columns') and definition.velocity_columns:
            print(f"  Velocity Columns: {definition.velocity_columns}")
        if hasattr(definition, 'acceleration_columns') and definition.acceleration_columns:
            print(f"  Acceleration Columns: {definition.acceleration_columns}")
        if hasattr(definition, 'distance_column') and definition.distance_column:
            print(f"  Distance Column: {definition.distance_column}")
        if hasattr(definition, 'custom_read_kwargs') and definition.custom_read_kwargs:
            print(f"  Custom Read Kwargs keys: {list(definition.custom_read_kwargs.keys())}")
      
        # Experiment information
        if hasattr(definition, 'experiment') and definition.experiment:
            exp = definition.experiment
            if hasattr(exp, 'name') and exp.name is not None:
                print(f"  Name: {exp.name}")
            if hasattr(exp, 'description') and exp.description is not None:
                print(f"  Description: {exp.description}")
            if hasattr(exp, 'sampling_rate_hz') and exp.sampling_rate_hz is not None:
                print(f"  Sampling Rate: {exp.sampling_rate_hz} Hz")
            if hasattr(exp, 'coordinates') and exp.coordinates is not None:
                print(f"  Coordinates: {exp.coordinates}")
            if hasattr(exp, 'origin') and exp.origin is not None:
                print(f"  Origin: {exp.origin}")
            if hasattr(exp, 'units') and exp.units is not None:
                print(f"  Units: {exp.units}")

            # Eyetracker (check each attribute from EyeTracker class)
            if hasattr(exp, 'eyetracker') and exp.eyetracker is not None:
                tracker = exp.eyetracker
                print("\nEYETRACKER:")
                if getattr(tracker, 'sampling_rate', None) is not None:
                    print(f"  Sampling Rate: {tracker.sampling_rate} Hz")
                if getattr(tracker, 'left', None) is not None:
                    print(f"  Left Eye Tracked: {tracker.left}")
                if getattr(tracker, 'right', None) is not None:
                    print(f"  Right Eye Tracked: {tracker.right}")
                if getattr(tracker, 'model', None) is not None:
                    print(f"  Model: {tracker.model}")
                if getattr(tracker, 'version', None) is not None:
                    print(f"  Version: {tracker.version}")
                if getattr(tracker, 'vendor', None) is not None:
                    print(f"  Vendor: {tracker.vendor}")
                if getattr(tracker, 'mount', None) is not None:
                    print(f"  Mount: {tracker.mount}")

            # Screen configuration
            if hasattr(exp, 'screen') and exp.screen:
                screen = exp.screen
                print("\nSCREEN CONFIGURATION:")
                if hasattr(screen, 'width_px') and screen.width_px is not None:
                    print(f"  Width: {screen.width_px} px")
                if hasattr(screen, 'height_px') and screen.height_px is not None:
                    print(f"  Height: {screen.height_px} px")
                if hasattr(screen, 'width_cm') and screen.width_cm is not None:
                    print(f"  Width (physical): {screen.width_cm} cm")
                if hasattr(screen, 'height_cm') and screen.height_cm is not None:
                    print(f"  Height (physical): {screen.height_cm} cm")
                if hasattr(screen, 'distance_cm') and screen.distance_cm is not None:
                    print(f"  Distance: {screen.distance_cm} cm")
                if hasattr(screen, 'framerate_hz') and screen.framerate_hz is not None:
                    print(f"  Framerate: {screen.framerate_hz} Hz")
                # Derived metrics
                if (hasattr(screen, 'width_px') and screen.width_px is not None and
                    hasattr(screen, 'height_px') and screen.height_px not in (None, 0)):
                    aspect_ratio = screen.width_px / screen.height_px
                    print(f"   Aspect Ratio: {aspect_ratio:.2f}")
                if (hasattr(screen, 'width_px') and screen.width_px is not None and
                    hasattr(screen, 'width_cm') and screen.width_cm not in (None, 0)):
                    ppcm = screen.width_px / screen.width_cm
                    print(f"   Pixels per cm: {ppcm:.1f}")
                    
        # Data structure snapshot
        if hasattr(self, 'gaze') and self.gaze:
            print("\nDATA STRUCTURE:")
            print(f"  Gaze Recordings: {len(self.gaze)}")
            sample_gaze = self.gaze[0]
            print(f"  Sample columns: {list(sample_gaze.samples.columns)}")
            print(f"  Sample data shape: {sample_gaze.samples.shape}")
            try:
                mem_mb = sample_gaze.samples.estimated_size() / (1024 * 1024)
                print(f"  Estimated memory usage: {mem_mb:.1f} MB")
            except Exception:
                pass

        if hasattr(self, 'events') and self.events:
            print(f"  Event Recordings: {len(self.events)}")

            # Aggregate metrics across all recordings
        total_samples_all = 0
        null_counts_all = {}
        
        # First pass to collect all column names
        all_columns = set()
        for gaze_recording in self.gaze:
            all_columns.update(gaze_recording.samples.columns)
            total_samples_all += len(gaze_recording.samples)
        
        # Initialize null counters for all columns
        for col in all_columns:
            null_counts_all[col] = 0
        
        # Count nulls in each recording
        for gaze_recording in self.gaze:
            samples = gaze_recording.samples
            for col in samples.columns:
                null_counts_all[col] += samples[col].is_null().sum()
        
        # Print overall statistics
        print("\nDATA QUALITY:")
        print(f"Total samples across all recordings: {total_samples_all}")
        
        # Missing data summary
        print("\nMissing Data:")
        has_missing = False
        for col, null_count in sorted(null_counts_all.items(), key=lambda x: x[1], reverse=True):
            if null_count > 0:
                has_missing = True
                percent = (null_count / total_samples_all) * 100
                print(f"  '{col}': {null_count} missing values ({percent:.2f}%)")
        
        if not has_missing:
            print("  No missing values found in any column")
    
        # References (if present)
        if hasattr(definition, 'references') and definition.references:
            print("\nREFERENCES:")
            for i, ref in enumerate(definition.references, 1):
                print(f"  {i}.")
                for attr_name in dir(ref):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(ref, attr_name)
                        if attr_value is not None and not callable(attr_value):
                            if isinstance(attr_value, list):
                                print(f"     {attr_name.replace('_', ' ').title()}: {', '.join(attr_value)}")
                            else:
                                print(f"     {attr_name.replace('_', ' ').title()}: {attr_value}")

        # Participants (if present)
        if hasattr(definition, 'participants') and definition.participants:
            print("\nPARTICIPANTS:")
            for i, participant in enumerate(definition.participants):
                print(f"  Participant {i + 1}:")
                for attr_name in dir(participant):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(participant, attr_name)
                        if attr_value is not None and not callable(attr_value):
                            print(f"    {attr_name.replace('_', ' ').title()}: {attr_value}")

        # Resource details
        print("\nRESOURCE FILES:")
        for i, resource in enumerate(getattr(definition, 'resources', []), 1):
            print(f"  {i}. {resource.filename}")
            for attr_name in dir(resource):
                if not attr_name.startswith('_') and attr_name != 'filename':
                    attr_value = getattr(resource, attr_name)
                    if attr_value is not None and not callable(attr_value):
                        if attr_name == 'columns' and hasattr(attr_value, '__iter__'):
                            column_names = [col.name for col in attr_value if hasattr(col, 'name')]
                            if column_names:
                                print(f"     Columns: {', '.join(column_names[:5])}")
                                if len(column_names) > 5:
                                    print(f"     ... and {len(column_names) - 5} more columns")
                        else:
                            print(f"     {attr_name.replace('_', ' ').title()}: {attr_value}")

        print("=" * 80)
        return self
