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
"""Functionality to scan, load and save dataset files."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from warnings import warn

import polars as pl
import pyreadr
from tqdm.auto import tqdm

from pymovements._utils._paths import match_filepaths
from pymovements._utils._strings import curly_to_regex
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.dataset.resources import ResourceDefinition
from pymovements.events import Events
from pymovements.events.precomputed import PrecomputedEventDataFrame
from pymovements.gaze.gaze import Gaze
from pymovements.gaze.io import from_asc
from pymovements.gaze.io import from_begaze
from pymovements.gaze.io import from_csv
from pymovements.gaze.io import from_ipc
from pymovements.reading_measures import ReadingMeasures


@dataclass
class DatasetFile:
    """A file of a dataset.

    Attributes
    ----------
    path: Path
        Absolute path of the dataset file.
    definition: ResourceDefinition
        Associated :py:class:`~pymovements.ResourceDefinition`.
    metadata: dict[str, Any]
        Additional metadata parsed via `:py:attr:`~pymovements.ResourceDefinition.filename_pattern`.

    Parameters
    ----------
    path: Path
        Absolute path of the dataset file.
    definition: ResourceDefinition
        Associated :py:class:`~pymovements.ResourceDefinition`.
    metadata: dict[str, Any]
        Additional metadata parsed via `:py:attr:`~pymovements.ResourceDefinition.filename_pattern`.
    """

    path: Path
    definition: ResourceDefinition
    metadata: dict[str, Any]


def scan_dataset(
        definition: DatasetDefinition, paths: DatasetPaths,
) -> tuple[dict[str, pl.DataFrame], list[DatasetFile]]:
    """Infer information from filepaths and filenames.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition.
    paths: DatasetPaths
        The dataset paths.

    Returns
    -------
    dict[str, pl.DataFrame]
        File information dataframe for each content type.
    list[DatasetFile]
        List of scanned dataset files.

    Raises
    ------
    AttributeError
        If no regular expression for parsing filenames is defined.
    RuntimeError
        If an error occurred during matching filenames or no files have been found.
    """
    # Get all filepaths that match regular expression.
    _fileinfo_dicts: dict[str, pl.DataFrame] = {}
    _files: list[DatasetFile] = []

    for resource_definition in definition.resources:
        content_type = resource_definition.content

        if content_type == 'gaze':
            resource_dirpath = paths.raw
        elif content_type == 'precomputed_events':
            resource_dirpath = paths.precomputed_events
        elif content_type == 'precomputed_reading_measures':
            resource_dirpath = paths.precomputed_reading_measures
        else:
            warn(
                f'content type {content_type} is not supported. '
                'supported contents are: gaze, precomputed_events, precomputed_reading_measures. '
                'skipping this resource definition during scan.',
            )
            continue

        filepaths = match_filepaths(
            path=resource_dirpath,
            regex=curly_to_regex(resource_definition.filename_pattern),
            relative=True,
        )

        if not filepaths:
            raise RuntimeError(f'no matching files found in {resource_dirpath}')

        fileinfo_df = pl.from_dicts(data=filepaths, infer_schema_length=1)
        fileinfo_df = fileinfo_df.sort(by='filepath')

        if resource_definition.filename_pattern_schema_overrides:
            items = resource_definition.filename_pattern_schema_overrides.items()
            fileinfo_df = fileinfo_df.with_columns([
                pl.col(fileinfo_key).cast(fileinfo_dtype)
                for fileinfo_key, fileinfo_dtype in items
            ])

        if resource_definition.content in _fileinfo_dicts:
            _fileinfo_dicts[content_type] = pl.concat([_fileinfo_dicts[content_type], fileinfo_df])
        else:
            _fileinfo_dicts[content_type] = fileinfo_df

        content_files = [
            DatasetFile(
                path=resource_dirpath / file['filepath'],  # absolute path
                definition=resource_definition,
                metadata={key: value for key, value in file.items() if key != 'filepath'},
            )
            for file in fileinfo_df.to_dicts()
        ]
        _files.extend(content_files)

    return _fileinfo_dicts, _files


def load_event_files(
        files: list[DatasetFile],
        paths: DatasetPaths,
        events_dirname: str | None = None,
        extension: str = 'feather',
        verbose: bool = True,
) -> list[Events]:
    """Load all event files associated to a gaze sample file.

    Parameters
    ----------
    files: list[DatasetFile]
        Load these files using the associated :py:class:`pymovements.ResourceDefinition`.
    paths: DatasetPaths
        Path of directory containing event files.
    events_dirname: str | None
        One-time usage of an alternative directory name to save data relative to dataset path.
        This argument is used only for this single call and does not alter
        :py:meth:`pymovements.Dataset.events_rootpath`.
    extension: str
        Specifies the file format for loading data. Valid options are: `csv`, `feather`,
        `tsv`, `txt`.
        (default: 'feather')
    verbose : bool
        If ``True``, show progress bar. (default: True)

    Returns
    -------
    list[Events]
        List of event dataframes.

    Raises
    ------
    AttributeError
        If `fileinfo` is None or the `fileinfo` dataframe is empty.
    ValueError
        If extension is not in list of valid extensions.
    """
    list_of_events: list[Events] = []

    # read and preprocess input files
    for file in tqdm(
            files, total=len(files), desc='Loading event files', unit='file', disable=not verbose,
    ):
        filepath = paths.raw_to_event_filepath(
            file.path,
            events_dirname=events_dirname,
            extension=extension,
        )

        if extension == 'feather':
            events = pl.read_ipc(filepath)
        elif extension in {'csv', 'tsv', 'txt'}:
            events = pl.read_csv(filepath)
        else:
            valid_extensions = ['csv', 'txt', 'tsv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )

        list_of_events.append(Events(events))

    return list_of_events


def load_gaze_files(
        definition: DatasetDefinition,
        files: list[DatasetFile],
        paths: DatasetPaths,
        preprocessed: bool = False,
        preprocessed_dirname: str | None = None,
        extension: str = 'feather',
) -> list[Gaze]:
    """Load all available gaze data files.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition.
    files: list[DatasetFile]
        Load these files using the associated :py:class:`pymovements.ResourceDefinition`.
    paths: DatasetPaths
        Path of directory containing event files.
    preprocessed : bool
        If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
        (default: False)
    preprocessed_dirname : str | None
        One-time usage of an alternative directory name to save data relative to
        :py:meth:`pymovements.Dataset.path`.
        This argument is used only for this single call and does not alter
        :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
    extension: str
        Specifies the file format for loading data. Valid options are: `csv`, `feather`,
        `txt`, `tsv`.
        (default: 'feather')

    Returns
    -------
    list[Gaze]
        Returns self, useful for method cascading.

    Raises
    ------
    AttributeError
        If `fileinfo` is None or the `fileinfo` dataframe is empty.
    RuntimeError
        If file type of gaze file is not supported.
    """
    gazes: list[Gaze] = []

    for file in tqdm(files, total=len(files), desc='Loading gaze files', unit='file'):
        # Preprocessed files are in a separate directory.
        if preprocessed:
            file = replace(
                file,
                path=paths.get_preprocessed_filepath(
                    file.path, preprocessed_dirname=preprocessed_dirname,
                    extension=extension,
                ),
            )

        gaze = load_gaze_file(
            filepath=file.path,
            resource_definition=file.definition,
            dataset_definition=deepcopy(definition),
            preprocessed=preprocessed,
        )
        gazes.append(gaze)

    return gazes


def load_gaze_file(
        filepath: Path,
        resource_definition: ResourceDefinition,
        dataset_definition: DatasetDefinition,
        preprocessed: bool = False,
) -> Gaze:
    """Load a gaze data file as Gaze.

    Parameters
    ----------
    filepath: Path
        Path of gaze file.
    resource_definition: ResourceDefinition
        Use this ResourceDefinition to get the correct load function and keyword arguments.
    dataset_definition: DatasetDefinition
        The dataset definition.
    preprocessed: bool
        If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
        (default: False)

    Returns
    -------
    Gaze
        The resulting Gaze

    Raises
    ------
    RuntimeError
        If file type of gaze file is not supported.
    ValueError
        If extension is not in list of valid extensions.
    """
    load_function_name = resource_definition.load_function
    if load_function_name is None:
        if filepath.suffix in {'.csv', '.txt', '.tsv'}:
            load_function_name = 'from_csv'
        elif filepath.suffix == '.feather':
            load_function_name = 'from_ipc'
        elif filepath.suffix == '.asc':
            load_function_name = 'from_asc'
        else:
            valid_extensions = ['csv', 'tsv', 'txt', 'feather', 'asc']
            raise ValueError(
                f'Unknown file extension "{filepath.suffix}". '
                f'Known extensions are: {valid_extensions}\n'
                f'Otherwise, specify load_function in the resource definition.',
            )

    load_function_kwargs = resource_definition.load_kwargs
    if load_function_kwargs is None:
        load_function_kwargs = {}

    if load_function_name == 'from_csv':
        if preprocessed:
            # Time unit is always milliseconds for preprocessed data if a time column is present.
            time_unit = 'ms'

            gaze = from_csv(
                filepath,
                time_unit=time_unit,
                auto_column_detect=True,
            )
        else:
            if dataset_definition.trial_columns is not None:
                load_function_kwargs['trial_columns'] = dataset_definition.trial_columns
            if dataset_definition.time_column is not None:
                load_function_kwargs['time_column'] = dataset_definition.time_column
            if dataset_definition.time_unit is not None:
                load_function_kwargs['time_unit'] = dataset_definition.time_unit
            if dataset_definition.pixel_columns is not None:
                load_function_kwargs['pixel_columns'] = dataset_definition.pixel_columns
            if dataset_definition.position_columns is not None:
                load_function_kwargs['position_columns'] = dataset_definition.position_columns
            if dataset_definition.velocity_columns is not None:
                load_function_kwargs['velocity_columns'] = dataset_definition.velocity_columns
            if dataset_definition.acceleration_columns is not None:
                acceleration_columns = dataset_definition.acceleration_columns
                load_function_kwargs['acceleration_columns'] = acceleration_columns
            if dataset_definition.distance_column is not None:
                load_function_kwargs['distance_column'] = dataset_definition.distance_column
            if dataset_definition.column_map:
                load_function_kwargs['column_map'] = dataset_definition.column_map
            if dataset_definition.custom_read_kwargs:
                read_csv_kwargs = dataset_definition.custom_read_kwargs.get('gaze', {})
                load_function_kwargs['read_csv_kwargs'] = {
                    **load_function_kwargs.get('read_csv_kwargs', {}), **read_csv_kwargs,
                }

            gaze = from_csv(
                filepath,
                experiment=dataset_definition.experiment,
                **load_function_kwargs,
            )
    elif load_function_name == 'from_ipc':
        gaze = from_ipc(
            filepath,
            experiment=dataset_definition.experiment,
        )
    elif load_function_name == 'from_asc':
        if dataset_definition.trial_columns is not None:
            load_function_kwargs['trial_columns'] = dataset_definition.trial_columns
        if dataset_definition.custom_read_kwargs:
            custom_read_kwargs = dataset_definition.custom_read_kwargs.get('gaze', {})
            load_function_kwargs = {**load_function_kwargs, **custom_read_kwargs}

        gaze = from_asc(
            filepath,
            experiment=dataset_definition.experiment,
            **load_function_kwargs,
        )
    elif load_function_name == 'from_begaze':
        if dataset_definition.trial_columns is not None:
            load_function_kwargs['trial_columns'] = dataset_definition.trial_columns
        if dataset_definition.custom_read_kwargs:
            custom_read_kwargs = dataset_definition.custom_read_kwargs.get('gaze', {})
            load_function_kwargs = {**load_function_kwargs, **custom_read_kwargs}

        gaze = from_begaze(
            filepath,
            experiment=dataset_definition.experiment,
            **load_function_kwargs,
        )
    else:
        valid_load_functions = ['from_csv', 'from_ipc', 'from_asc', 'from_begaze']
        raise ValueError(
            f'Unsupported load_function "{load_function_name}". '
            f'Available options are: {valid_load_functions}',
        )

    return gaze


def load_precomputed_reading_measures(
        definition: DatasetDefinition,
        files: list[DatasetFile],
) -> list[ReadingMeasures]:
    """Load reading measures files.

    Parameters
    ----------
    definition: DatasetDefinition
        Dataset definition to load precomputed events.
    files: list[DatasetFile]
        Load these files using the associated :py:class:`pymovements.ResourceDefinition`.

    Returns
    -------
    list[ReadingMeasures]
        Return list of precomputed event dataframes.
    """
    precomputed_reading_measures = []
    for file in files:
        load_function_kwargs = file.definition.load_kwargs
        if load_function_kwargs is None:
            load_function_kwargs = {}
        if definition.custom_read_kwargs is not None:
            custom_read_kwargs = definition.custom_read_kwargs.get(
                'precomputed_reading_measures', {},
            )
            load_function_kwargs.update(custom_read_kwargs)

        precomputed_reading_measures.append(
            load_precomputed_reading_measure_file(file.path, load_function_kwargs),
        )
    return precomputed_reading_measures


def load_precomputed_reading_measure_file(
        data_path: str | Path,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> ReadingMeasures:
    """Load precomputed reading measure from file.

    This function supports both CSV-based (.csv, .tsv, .txt) and Excel (.xlsx) formats for
    reading preprocessed eye-tracking or behavioral data related to reading. File reading
    is customized via keyword arguments passed to Polars' reading functions. If an unsupported
    file format is encountered, a `ValueError` is raised.

    Parameters
    ----------
    data_path:  str | Path
        Path to file to be read.
    custom_read_kwargs: dict[str, Any] | None
        Custom read keyword arguments for polars. (default: None)

    Returns
    -------
    ReadingMeasures
        Returns the text stimulus file.

    Raises
    ------
    ValueError
        Raises ValueError if unsupported file type is encountered.
    """
    data_path = Path(data_path)
    if custom_read_kwargs is None:
        custom_read_kwargs = {}

    csv_extensions = {'.csv', '.tsv', '.txt'}
    r_extensions = {'.rda'}
    excel_extensions = {'.xlsx'}
    valid_extensions = csv_extensions | r_extensions | excel_extensions
    if data_path.suffix in csv_extensions:
        precomputed_reading_measure_df = pl.read_csv(data_path, **custom_read_kwargs)
    elif data_path.suffix in r_extensions:
        if 'r_dataframe_key' in custom_read_kwargs:
            precomputed_r = pyreadr.read_r(data_path)
            # convert to polars DataFrame because read_r has no .clone().
            precomputed_reading_measure_df = pl.DataFrame(
                precomputed_r[custom_read_kwargs['r_dataframe_key']],
            )
        else:
            raise ValueError('please specify r_dataframe_key in custom_read_kwargs')
    elif data_path.suffix in excel_extensions:
        precomputed_reading_measure_df = pl.read_excel(
            data_path,
            sheet_name=custom_read_kwargs['sheet_name'],
        )
    else:
        raise ValueError(
            f'unsupported file format "{data_path.suffix}". '
            f'Supported formats are: {", ".join(sorted(valid_extensions))}',
        )

    return ReadingMeasures(precomputed_reading_measure_df)


def load_precomputed_event_files(
        definition: DatasetDefinition,
        files: list[DatasetFile],
) -> list[PrecomputedEventDataFrame]:
    """Load precomputed event dataframes from files.

    For each file listed in `fileinfo`, construct the full path using `paths.precomputed_events`,
    and load it with `load_precomputed_event_file` using any custom read arguments defined
    in `definition.custom_read_kwargs['precomputed_events']`.

    Parameters
    ----------
    definition:  DatasetDefinition
        Dataset definition to load precomputed events.
    files: list[DatasetFile]
        Load these files using the associated :py:class:`pymovements.ResourceDefinition`.
        Valid extensions: .csv, .tsv, .txt, .jsonl, and .ndjson.

    Returns
    -------
    list[PrecomputedEventDataFrame]
        Return list of precomputed event dataframes.
    """
    precomputed_events = []
    for file in files:
        load_function_kwargs = file.definition.load_kwargs
        if load_function_kwargs is None:
            load_function_kwargs = {}
        if definition.custom_read_kwargs is not None:
            custom_read_kwargs = definition.custom_read_kwargs.get('precomputed_events', {})
            load_function_kwargs.update(custom_read_kwargs)

        precomputed_events.append(
            load_precomputed_event_file(file.path, load_function_kwargs),
        )
    return precomputed_events


def load_precomputed_event_file(
        data_path: str | Path,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> PrecomputedEventDataFrame:
    """Load precomputed events from a single file.

    File format is inferred from the extension:
        - CSV-like: .csv, .tsv, .txt
        - JSON-like: jsonl, .ndjson

    Raises a ValueError for unsupported formats.

    Parameters
    ----------
    data_path:  str | Path
        Path to file to be read.

    custom_read_kwargs: dict[str, Any] | None
        Custom read keyword arguments for polars. (default: None)

    Returns
    -------
    PrecomputedEventDataFrame
        Returns the precomputed event dataframe.

    Raises
    ------
    ValueError
        If the file format is unsupported based on its extension.
    """
    data_path = Path(data_path)
    if custom_read_kwargs is None:
        custom_read_kwargs = {}

    csv_extensions = {'.csv', '.tsv', '.txt'}
    r_extensions = {'.rda'}
    json_extensions = {'.jsonl', '.ndjson'}
    valid_extensions = csv_extensions | r_extensions | json_extensions
    if data_path.suffix in csv_extensions:
        precomputed_event_df = pl.read_csv(data_path, **custom_read_kwargs)
    elif data_path.suffix in r_extensions:
        if 'r_dataframe_key' in custom_read_kwargs:
            precomputed_r = pyreadr.read_r(data_path)
            # convert to polars DataFrame because read_r has no .clone().
            precomputed_event_df = pl.DataFrame(
                precomputed_r[custom_read_kwargs['r_dataframe_key']],
            )
        else:
            raise ValueError('please specify r_dataframe_key in custom_read_kwargs')
    elif data_path.suffix in json_extensions:
        precomputed_event_df = pl.read_ndjson(data_path, **custom_read_kwargs)
    else:
        raise ValueError(
            f'unsupported file format "{data_path.suffix}". '
            f'Supported formats are: {", ".join(sorted(valid_extensions))}',
        )

    return PrecomputedEventDataFrame(data=precomputed_event_df)


def save_events(
        events: Sequence[Events],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        events_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save events to files.

    Data will be saved as feather files to ``Dataset.events_roothpath`` with the same directory
    structure as the raw data.

    Parameters
    ----------
    events: Sequence[Events]
        A sequence of :py:class:`pymovements.Events` objects to save.
    fileinfo: pl.DataFrame
        A dataframe holding file information.
    paths: DatasetPaths
        Path of directory containing event files.
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

    Raises
    ------
    ValueError
        If extension is not in list of valid extensions.
    """
    disable_progressbar = not verbose

    for file_id, events_instance in enumerate(
        tqdm(
            events,
            total=len(events),
            desc='Saving event files',
            unit='file',
            disable=disable_progressbar,
        ),
    ):
        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        events_filepath = paths.raw_to_event_filepath(
            raw_filepath, events_dirname=events_dirname,
            extension=extension,
        )

        if verbose >= 2:
            print('Save file to', events_filepath)

        events_filepath.parent.mkdir(parents=True, exist_ok=True)
        if extension == 'feather':
            events_instance.frame.write_ipc(events_filepath)
        elif extension == 'csv':
            events_instance.frame.write_csv(events_filepath)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def save_preprocessed(
        gazes: list[Gaze],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        preprocessed_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save preprocessed gaze files.

    Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` with the same
    directory structure as the raw data.

    Parameters
    ----------
    gazes: list[Gaze]
        The gaze objects to save.
    fileinfo: pl.DataFrame
        A dataframe holding file information.
    paths: DatasetPaths
        Path of directory containing event files.
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

    Raises
    ------
    ValueError
        If extension is not in list of valid extensions.
    """
    disable_progressbar = not verbose

    for file_id, gaze in enumerate(
        tqdm(
            gazes,
            total=len(gazes),
            desc='Saving preprocessed files',
            unit='file',
            disable=disable_progressbar,
        ),
    ):
        gaze = gaze.clone()

        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        preprocessed_filepath = paths.get_preprocessed_filepath(
            raw_filepath, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        if extension == 'csv':
            gaze.unnest()

        if verbose >= 2:
            print('Save file to', preprocessed_filepath)

        preprocessed_filepath.parent.mkdir(parents=True, exist_ok=True)
        if extension == 'feather':
            gaze.samples.write_ipc(preprocessed_filepath)
        elif extension == 'csv':
            gaze.samples.write_csv(preprocessed_filepath)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def take_subset(
        fileinfo: pl.DataFrame,
        files: list[DatasetFile],
        subset: dict[
            str, bool | float | int | str | list[bool | float | int | str],
        ] | None = None,
) -> tuple[pl.DataFrame, list[DatasetFile]]:
    """Take a subset of the fileinfo dataframe and dataset file list.

    Parameters
    ----------
    fileinfo: pl.DataFrame
        File information dataframe.
    files: list[DatasetFile]
        Filter this list of dataset files for values specified by subset.
    subset: dict[str, bool | float | int | str | list[bool | float | int | str]] | None
        If specified, take a subset of the dataset. All keys in the dictionary must be
        present in the fileinfo dataframe inferred by `scan_dataset()`. Values can be either
        bool, float, int , str or a list of these. (default: None)

    Returns
    -------
    pl.DataFrame
        Subset of file information dataframe.
    list[DatasetFile]
        Subset of dataset files.

    Raises
    ------
    ValueError
        If dictionary key is not a column in the fileinfo dataframe.
    TypeError
        If dictionary key or value is not of valid type.
    """
    if subset is None:
        return fileinfo, files

    if not isinstance(subset, dict):
        raise TypeError(f'subset must be of type dict but is of type {type(subset)}')

    for metadata_key, metadata_value in subset.items():
        if not isinstance(metadata_key, str):
            raise TypeError(
                f'subset keys must be of type str but key {metadata_key} is of type'
                f' {type(metadata_key)}',
            )

        if metadata_key not in fileinfo['gaze'].columns:
            raise ValueError(
                f'subset key {metadata_key} must be a column in the fileinfo attribute.'
                f" Available columns are: {fileinfo['gaze'].columns}",
            )

        for file in files:
            if metadata_key not in file.metadata:  # pragma: no cover
                # This code is currently unreachable via public interfaces.
                # The pragma directive should be removed after the removal of fileinfo from Dataset.
                raise ValueError(
                    f'subset key {metadata_key} must exist as metadata key in DatasetFile. '
                    f"Available metadata: {file.metadata}",
                )

        if isinstance(metadata_value, (bool, float, int, str)):
            metadata_values = [metadata_value]
        elif isinstance(metadata_value, (list, tuple, range)):
            metadata_values = metadata_value
        else:
            raise TypeError(
                f'subset values must be of type bool, float, int, str, range, or list, '
                f'but value of pair {metadata_key}: {metadata_value} is of type: '
                f'{type(metadata_value)}',
            )

        fileinfo['gaze'] = fileinfo['gaze'].filter(pl.col(metadata_key).is_in(metadata_values))
        files = [file for file in files if file.metadata[metadata_key] in metadata_values]
    return fileinfo, files
