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
"""Functionality to load Gaze from a csv file."""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import polars as pl

from pymovements.events.frame import Events
from pymovements.gaze._utils._parsing_begaze import parse_begaze
from pymovements.gaze._utils._parsing_eyelink import parse_eyelink
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze


def from_csv(
        file: str | Path,
        experiment: Experiment | None = None,
        *,
        trial_columns: str | list[str] | None = None,
        time_column: str | None = None,
        time_unit: str | None = None,
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
        distance_column: str | None = None,
        auto_column_detect: bool = False,
        column_map: dict[str, str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, type] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
) -> Gaze:
    """Initialize a :py:class:`~pymovements.Gaze`.

    Parameters
    ----------
    file: str | Path
        Path of gaze file.
    experiment : Experiment | None
        The experiment definition. (default: None)
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
        The name of the eye-to-screen distance column in the input data frame. If specified,
        the column will be used for pixel to dva transformations. If not specified, the
        constant eye-to-screen distance will be taken from the experiment definition.
        (default: None)
    auto_column_detect: bool
        Flag indicating if the column names should be inferred automatically. (default: False)
    column_map: dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides:  dict[str, type] | None
        Dictionary containing types for columns.
        (default: None)
    read_csv_kwargs: dict[str, Any] | None
        Additional keyword arguments to be passed to :py:func:`polars.read_csv` to read in the csv.
        These can include custom separators, a subset of columns, or specific data types
        for columns. (default: None)
    **kwargs: Any
        Additional keyword arguments to be passed to :py:func:`polars.read_csv` to read in the csv.
        These can include custom separators, a subset of columns, or specific data types
        for columns.

    Returns
    -------
    Gaze
        The initialized gaze object read from the csv file.

    Notes
    -----
    About using the arguments ``pixel_columns``, ``position_columns``, ``velocity_columns``,
    and ``acceleration_columns``:

    By passing a list of columns as any of these arguments, these columns will be merged into a
    single column with the corresponding name , e.g. using `pixel_columns` will merge the
    respective columns into the column `pixel`.

    The supported number of component columns with the expected order are:

    - **zero columns**: No nested component column will be created.
    - **two columns**: monocular data; expected order: x-component, y-component
    - **four columns**: binocular data; expected order: x-component left eye, y-component left eye,
      x-component right eye, y-component right eye
    - **six columns**: binocular data with additional cyclopian data; expected order: x-component
      left eye, y-component left eye, x-component right eye, y-component right eye,
      x-component cyclopian eye, y-component cyclopian eye


    Examples
    --------
    First let's assume a CSV file stored `tests/files/monocular_example.csv`
    with the following content:
    shape: (10, 3)
    ┌──────┬────────────┬────────────┐
    │ time ┆ x_left_pix ┆ y_left_pix │
    │ ---  ┆ ---        ┆ ---        │
    │ i64  ┆ i64        ┆ i64        │
    ╞══════╪════════════╪════════════╡
    │ 0    ┆ 0          ┆ 0          │
    │ 1    ┆ 0          ┆ 0          │
    │ 2    ┆ 0          ┆ 0          │
    │ 3    ┆ 0          ┆ 0          │
    │ 4    ┆ 0          ┆ 0          │
    │ 5    ┆ 0          ┆ 0          │
    │ 6    ┆ 0          ┆ 0          │
    │ 7    ┆ 0          ┆ 0          │
    │ 8    ┆ 0          ┆ 0          │
    │ 9    ┆ 0          ┆ 0          │
    └──────┴────────────┴────────────┘

    We can now load the data into a ``Gaze`` by specyfing the experimental setting
    and the names of the pixel position columns. We can specify a custom separator for the csv
    file by passing it as a keyword argument to :py:func:`polars.read_csv`:

    >>> from pymovements.gaze.io import from_csv
    >>> gaze = from_csv(
    ...     file='tests/files/monocular_example.csv',
    ...     time_column='time',
    ...     time_unit='ms',
    ...     pixel_columns=['x_left_pix','y_left_pix'],
    ...     read_csv_kwargs={'separator': ','},
    ... )
    >>> gaze.samples
    shape: (10, 2)
    ┌──────┬───────────┐
    │ time ┆ pixel     │
    │ ---  ┆ ---       │
    │ i64  ┆ list[i64] │
    ╞══════╪═══════════╡
    │ 0    ┆ [0, 0]    │
    │ 1    ┆ [0, 0]    │
    │ 2    ┆ [0, 0]    │
    │ 3    ┆ [0, 0]    │
    │ 4    ┆ [0, 0]    │
    │ 5    ┆ [0, 0]    │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    Please be aware that data types are inferred from a fixed number of rows. To ensure
    correct data types, you can pass a dictionary of column names and data types to the
    `schema_overrides` keyword argument of :py:func:`polars.read_csv`:

    >>> from pymovements.gaze.io import from_csv
    >>> import polars as pl
    >>> gaze = from_csv(
    ...     file='tests/files/monocular_example.csv',
    ...     time_column='time',
    ...     time_unit='ms',
    ...     pixel_columns=['x_left_pix','y_left_pix'],
    ...     read_csv_kwargs={
    ...         'schema_overrides': {'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64},
    ...     },
    ... )
    >>> gaze.samples
    shape: (10, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 1    ┆ [0.0, 0.0] │
    │ 2    ┆ [0.0, 0.0] │
    │ 3    ┆ [0.0, 0.0] │
    │ 4    ┆ [0.0, 0.0] │
    │ 5    ┆ [0.0, 0.0] │
    │ 6    ┆ [0.0, 0.0] │
    │ 7    ┆ [0.0, 0.0] │
    │ 8    ┆ [0.0, 0.0] │
    │ 9    ┆ [0.0, 0.0] │
    └──────┴────────────┘

    """
    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    if kwargs:
        warnings.warn(
            DeprecationWarning(
                "from_csv() argument '**kwargs' is deprecated since version v0.24.0. "
                'This argument will be removed in v0.29.0.'
                "Please use argument 'read_csv_kwargs' instead. ",
            ),
        )
        # merge dictionaries, **kwargs takes precedence
        read_csv_kwargs = {**read_csv_kwargs, **kwargs}

    # Read data.
    samples = pl.read_csv(file, **read_csv_kwargs)
    if column_map is not None:
        samples = samples.rename({
            key: column_map[key] for key in
            [
                key for key in column_map.keys()
                if key in samples.columns
            ]
        })

    if add_columns is not None:
        samples = samples.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in samples.columns
        ])

    # Cast numerical columns to Float64 if they were incorrectly inferred to be Utf8.
    # This can happen if the column only has missing values in the top 100 rows.
    numerical_columns = (
        (pixel_columns or [])
        + (position_columns or [])
        + (velocity_columns or [])
        + (acceleration_columns or [])
        + ([distance_column] if distance_column else [])
    )
    for column in numerical_columns:
        if samples[column].dtype == pl.Utf8:
            samples = samples.with_columns([
                pl.col(column).cast(pl.Float64),
            ])

    if column_schema_overrides is not None:
        # Apply overrides as provided - callers should pass concrete pl.DataType instances
        samples = samples.with_columns([
            pl.col(column_key).cast(column_dtype)
            for column_key, column_dtype in column_schema_overrides.items()
        ])

    # Create gaze object.
    gaze = Gaze(
        samples=samples,
        experiment=experiment,
        trial_columns=trial_columns,
        time_column=time_column,
        time_unit=time_unit,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        distance_column=distance_column,
        auto_column_detect=auto_column_detect,
    )
    return gaze


def metadata_to_cal_frame(metadata: dict[str, Any]) -> pl.DataFrame:
    """Convert and consume EyeLink calibration metadata to a DataFrame.

    Pops the 'calibrations' key from the metadata dict and returns a DataFrame with schema:
    time(f64), num_points(i64), eye(utf8), tracking_mode(utf8).
    """
    cal_items = metadata.pop('calibrations', []) or []
    if cal_items:
        return pl.from_dicts([
            {
                'time': float(item.get('timestamp')) if item.get('timestamp') not in (None, '')
                else None,
                'num_points': int(item.get('num_points')) if item.get('num_points')
                not in (None, '') else None,
                'eye': (
                    'left' if (item.get('tracked_eye') or '').upper() == 'LEFT' else
                    'right' if (item.get('tracked_eye') or '').upper() == 'RIGHT' else None
                ),
                'tracking_mode': item.get('type') if item.get('type') not in (None, '') else None,
            }
            for item in cal_items
        ]).with_columns([
            pl.col('time').cast(pl.Float64),
            pl.col('num_points').cast(pl.Int64),
            pl.col('eye').cast(pl.Utf8),
            pl.col('tracking_mode').cast(pl.Utf8),
        ])
    return pl.DataFrame(
        schema={
            'time': pl.Float64,
            'num_points': pl.Int64,
            'eye': pl.Utf8,
            'tracking_mode': pl.Utf8,
        },
    )


def metadata_to_val_frame(metadata: dict[str, Any]) -> pl.DataFrame:
    """Convert and consume EyeLink validation metadata to a DataFrame.

    Pops the 'validations' key from the metadata dict and returns a DataFrame with schema:
    time(f64), num_points(i64), eye(utf8), accuracy_avg(f64), accuracy_max(f64).
    """
    val_items = metadata.pop('validations', []) or []
    if val_items:
        return pl.from_dicts([
            {
                'time': float(item.get('timestamp')) if item.get('timestamp') not in (None, '')
                else None,
                'num_points': int(item.get('num_points')) if item.get('num_points')
                not in (None, '') else None,
                'eye': (
                    'left' if (item.get('tracked_eye') or '').upper() == 'LEFT' else
                    'right' if (item.get('tracked_eye') or '').upper() == 'RIGHT' else None
                ),
                'accuracy_avg': float(item.get('validation_score_avg')) if
                item.get('validation_score_avg') not in (None, '') else None,
                'accuracy_max': float(item.get('validation_score_max')) if
                item.get('validation_score_max') not in (None, '') else None,
            }
            for item in val_items
        ]).with_columns([
            pl.col('time').cast(pl.Float64),
            pl.col('num_points').cast(pl.Int64),
            pl.col('eye').cast(pl.Utf8),
            pl.col('accuracy_avg').cast(pl.Float64),
            pl.col('accuracy_max').cast(pl.Float64),
        ])
    return pl.DataFrame(
        schema={
            'time': pl.Float64,
            'num_points': pl.Int64,
            'eye': pl.Utf8,
            'accuracy_avg': pl.Float64,
            'accuracy_max': pl.Float64,
        },
    )


def from_asc(
        file: str | Path,
        *,
        patterns: str | list[dict[str, Any] | str] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        experiment: Experiment | None = None,
        trial_columns: str | list[str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, Any] | None = None,
        encoding: str | None = None,
        events: bool = False,
        messages: bool | list[str] = False,
) -> Gaze:
    """Initialize a :py:class:`~pymovements.Gaze`.

    Parameters
    ----------
    file: str | Path
        Path of ASC file.
    patterns: str | list[dict[str, Any] | str] | None
        List of patterns to match for additional columns or a key identifier of eye tracker specific
        default patterns. Supported values are: `'eyelink'`. If `None` is passed, `'eyelink'` is
        assumed. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        List of patterns to match for extracting metadata from custom logged messages.
        (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    experiment: Experiment | None
        The experiment definition. (default: None)
    trial_columns: str | list[str] | None
        The names of the columns (extracted by patterns) to use as trial columns.
        If the list is empty or None, the asc file is assumed to contain only one trial.
        If the list is not empty, the asc file is assumed to contain multiple trials and
        the transformation methods will be applied to each trial separately. (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides: dict[str, Any] | None
        Dictionary containing types for columns.
        (default: None)
    encoding: str | None
        Text encoding of the file. If None, the locale encoding is used. (default: None)
    events: bool
        Flag indicating if events should be parsed from the asc file. (default: False)
    messages: bool | list[str]
        Flag indicating if any additional messages should be parsed from the asc file
        and stored in :py:class:`pymovements.gaze.experiment.Experiment`.
        The message format is 'MSG <timestamp> <content>'.
        If True, all available messages will be parsed from the asc,
        alternatively, a list of regular expressions can be passed and only the
        messages that match any of the regular expressions will be kept.
        Regular expressions are only applied to the message content,
        implicitly parsing the `MSG <timestamp>` prefix.
        (default: False)

    Returns
    -------
    Gaze
        The initialized gaze object read from the asc file.

    Notes
    -----
    ASC files are created from EyeLink EDF files using the ``edf2asc`` tool
    (can be downloaded from the SR Research Support website).
    ASC files contain gaze samples, events, and metadata about
    the experiment in a text (ASCII) format.
    For example, if you have an Eyelink EDF file stored at
    ``tests/files/eyelink_monocular_example.edf``,
    you can convert it to an ASC file using the following command:
    ``edf2asc tests/files/eyelink_monocular_example.edf``.
    This will create an ASC file named ``tests/files/eyelink_monocular_example.asc``.

    Running ``edf2asc`` with the default settings (no flags/parameters) will always produce
    an ASC file that can be read by ``from_asc()``.

    Moreover, the following optional ``edf2asc`` parameters are safe to use
    and will also result in an ASC file that is compatible with ``from_asc()``:

    - ``-input``: include the status of the Host PC parallel port (although the values will not be
      read by `pymovements`).

    - ``-ftime``: format timestamps as floating point values.

    - ``-t``: use only tabs as delimiters.

    - ``-utf8``: force UTF-8 encoding.

    - ``-buttons``: include button events (although the values will not be read by `pymovements`).

    - ``-vel`` and ``-fvel``: include velocity values (although the values will not be read by
      `pymovements`).

    - ``-l`` or ``-nr``: only include left eye data.

    - ``-r`` or ``-nl``: only include right eye data.

    - ``-avg``: include average values of the left and right eye data in case of a binocular file
      (although the values will not be read by `pymovements`).

    Using other ``edf2asc`` parameters may lead to errors or unexpected behavior. For example, using
    ``-e`` or ``-ns`` to output only events or ``-s`` or ``-ne`` to only output samples will not
    work with this function, as it expects both samples and events to be present in the ASC file.


    Examples
    --------
    We can load an asc file stored at `tests/files/eyelink_monocular_example.asc` into a ``Gaze``:

    >>> from pymovements.gaze.io import from_asc
    >>> gaze = from_asc(file='tests/files/eyelink_monocular_example.asc')
    >>> gaze.samples
    shape: (16, 3)
    ┌─────────┬───────┬────────────────┐
    │ time    ┆ pupil ┆ pixel          │
    │ ---     ┆ ---   ┆ ---            │
    │ i64     ┆ f64   ┆ list[f64]      │
    ╞═════════╪═══════╪════════════════╡
    │ 2154556 ┆ 778.0 ┆ [138.1, 132.8] │
    │ 2154557 ┆ 778.0 ┆ [138.2, 132.7] │
    │ 2154560 ┆ 777.0 ┆ [137.9, 131.6] │
    │ 2154564 ┆ 778.0 ┆ [138.1, 131.0] │
    │ 2154596 ┆ 784.0 ┆ [139.6, 132.1] │
    │ …       ┆ …     ┆ …              │
    │ 2339246 ┆ 622.0 ┆ [629.9, 531.9] │
    │ 2339271 ┆ 617.0 ┆ [639.4, 531.9] │
    │ 2339272 ┆ 617.0 ┆ [639.0, 531.9] │
    │ 2339290 ┆ 618.0 ┆ [637.6, 531.4] │
    │ 2339291 ┆ 618.0 ┆ [637.3, 531.2] │
    └─────────┴───────┴────────────────┘
    >>> gaze.experiment.eyetracker.sampling_rate
    1000.0
    """
    if isinstance(patterns, str):
        if patterns == 'eyelink':
            # We use the default patterns of parse_eyelink then.
            _patterns = None
        else:
            raise ValueError(f"unknown pattern key '{patterns}'. Supported keys are: eyelink")
    else:
        _patterns = patterns

    # Read data.
    samples, event_data, metadata, messages_df = parse_eyelink(
        file,
        patterns=_patterns,
        schema=schema,
        metadata_patterns=metadata_patterns,
        encoding=encoding,
        messages=messages,
    )

    if add_columns is not None:
        samples = samples.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in samples.columns
        ])

    if column_schema_overrides is not None:
        samples = samples.with_columns([
            pl.col(column_key).cast(column_dtype)
            for column_key, column_dtype in column_schema_overrides.items()
        ])

    # Fill experiment with parsed metadata.
    experiment = _fill_experiment_from_parsing_eyelink_metadata(experiment, metadata)

    # Detect pixel / position column names (monocular or binocular) and pass them to Gaze
    # Note: column detection for ASC files now uses simple substring matching
    # for 'pix' and 'pos' further down in `from_asc`. The older helper-based
    # detection was removed to avoid duplication and simplify the logic.

    cols = set(samples.columns)

    # Simpler detection: pick any columns that contain the substrings 'pix' or 'pos'
    # and pass them directly to Gaze. This covers monocular/binocular naming
    # produced by parse_eyelink without complex subset checks.
    # Annotate as optional so mypy knows these variables may be None.
    detected_pixel_columns: list[str] | None = [c for c in samples.columns if '_pix' in c]

    # Instantiate Gaze with parsed data using detected column names
    # If binocular pupils exist, create a nested 'pupil' column [left, right]
    if 'pupil_left' in cols and 'pupil_right' in cols:
        samples = samples.with_columns(
            pl.concat_list([pl.col('pupil_left'), pl.col('pupil_right')]).alias('pupil'),
        ).drop(['pupil_left', 'pupil_right'])

    gaze = Gaze(
        samples=samples,
        experiment=experiment,
        events=Events(event_data) if events else None,
        messages=messages_df,
        trial_columns=trial_columns,
        time_column='time',
        time_unit='ms',
        pixel_columns=detected_pixel_columns,
    )
    # Build cal/val frames and consume them from metadata dict
    gaze.calibrations = metadata_to_cal_frame(metadata)
    gaze.validations = metadata_to_val_frame(metadata)
    # Keep remaining metadata privately on Gaze
    gaze._metadata = dict(metadata)  # pylint: disable=protected-access
    return gaze


def from_ipc(
        file: str | Path,
        experiment: Experiment | None = None,
        *,
        trial_columns: str | list[str] | None = None,
        column_map: dict[str, str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, type] | None = None,
        read_ipc_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
) -> Gaze:
    """Initialize a :py:class:`~pymovements.Gaze`.

    Parameters
    ----------
    file: str | Path
        Path of IPC/feather file.
    experiment : Experiment | None
        The experiment definition.
        (default: None)
    trial_columns: str | list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    column_map: dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides:  dict[str, type] | None
        Dictionary containing types for columns.
        (default: None)
    read_ipc_kwargs: dict[str, Any] | None
            Additional keyword arguments to be passed to :py:func:`polars.read_ipc`. (default: None)
    **kwargs: Any
            Additional keyword arguments to be passed to :py:func:`polars.read_ipc`.

    Returns
    -------
    Gaze
        The initialized gaze object read from the ipc file.

    Examples
    --------
    Let's assume we have an IPC file stored at `tests/files/monocular_example.feather`.
    We can then load the data into a ``Gaze``:

    >>> from pymovements.gaze.io import from_ipc
    >>> gaze = from_ipc(file='tests/files/monocular_example.feather')
    >>> gaze.samples
    shape: (10, 2)
    ┌──────┬───────────┐
    │ time ┆ pixel     │
    │ ---  ┆ ---       │
    │ i64  ┆ list[i64] │
    ╞══════╪═══════════╡
    │ 0    ┆ [0, 0]    │
    │ 1    ┆ [0, 0]    │
    │ 2    ┆ [0, 0]    │
    │ 3    ┆ [0, 0]    │
    │ 4    ┆ [0, 0]    │
    │ 5    ┆ [0, 0]    │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    """
    if read_ipc_kwargs is None:
        read_ipc_kwargs = {}

    if kwargs:
        warnings.warn(
            DeprecationWarning(
                "from_ipc() argument '**kwargs' is deprecated since version v0.24.0. "
                'This argument will be removed in v0.29.0.',
                "Please use argument 'read_ipc_kwargs' instead. ",
            ),
        )
        # merge dictionaries, **kwargs takes precedence
        read_ipc_kwargs = {**read_ipc_kwargs, **kwargs}

    # Read data.
    samples = pl.read_ipc(file, **read_ipc_kwargs)

    if column_map is not None:
        samples = samples.rename({
            key: column_map[key] for key in
            [
                key for key in column_map.keys()
                if key in samples.columns
            ]
        })

    if add_columns is not None:
        samples = samples.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in samples.columns
        ])

    if column_schema_overrides is not None:
        samples = samples.with_columns([
            pl.col(column_key).cast(column_dtype)
            for column_key, column_dtype in column_schema_overrides.items()
        ])

    # Create gaze object.
    gaze = Gaze(
        samples=samples,
        experiment=experiment,
        trial_columns=trial_columns,
    )
    return gaze


def _fill_experiment_from_parsing_eyelink_metadata(
        experiment: Experiment | None,
        metadata: dict[str, Any],
) -> Experiment:
    """Fill Experiment with metadata gained from parsing."""
    if experiment is None:
        experiment = Experiment(sampling_rate=metadata['sampling_rate'])

    # Compare metadata from experiment definition with metadata from ASC file.
    # Fill in missing metadata in experiment definition and raise an error if there are conflicts
    issues = []

    # Screen resolution (assuming that width and height will always be missing or set together)
    experiment_resolution = (experiment.screen.width_px, experiment.screen.height_px)
    if experiment_resolution == (None, None):
        try:
            width, height = metadata['resolution']
            experiment.screen.width_px = math.ceil(width)
            experiment.screen.height_px = math.ceil(height)
        except TypeError:
            warnings.warn('No screen resolution found.')
    elif experiment_resolution != metadata['resolution']:
        issues.append(f"Screen resolution: {experiment_resolution} != {metadata['resolution']}")

    # Sampling rate
    if experiment.eyetracker.sampling_rate != metadata['sampling_rate']:
        issues.append(
            f"Sampling rate: {experiment.eyetracker.sampling_rate} != {metadata['sampling_rate']}",
        )

    # Tracked eye
    asc_left_eye = 'L' in (metadata['tracked_eye'] or '')
    asc_right_eye = 'R' in (metadata['tracked_eye'] or '')
    if experiment.eyetracker.left is None:
        experiment.eyetracker.left = asc_left_eye
    elif experiment.eyetracker.left != asc_left_eye:
        issues.append(f'Left eye tracked: {experiment.eyetracker.left} != {asc_left_eye}')
    if experiment.eyetracker.right is None:
        experiment.eyetracker.right = asc_right_eye
    elif experiment.eyetracker.right != asc_right_eye:
        issues.append(f'Right eye tracked: {experiment.eyetracker.right} != {asc_right_eye}')

    # Mount configuration
    if experiment.eyetracker.mount is None:
        try:
            experiment.eyetracker.mount = metadata['mount_configuration']['mount_type']
        except KeyError:
            warnings.warn('No mount configuration found.')
    elif experiment.eyetracker.mount != metadata['mount_configuration']['mount_type']:
        issues.append(
            f'Mount configuration: {experiment.eyetracker.mount} != '
            f"{metadata['mount_configuration']['mount_type']}",
        )

    # Eye tracker vendor
    asc_vendor = 'EyeLink' if 'EyeLink' in metadata['model'] else None
    if experiment.eyetracker.vendor is None:
        experiment.eyetracker.vendor = asc_vendor
    elif experiment.eyetracker.vendor != asc_vendor:
        issues.append(f'Eye tracker vendor: {experiment.eyetracker.vendor} != {asc_vendor}')

    # Eye tracker model
    if experiment.eyetracker.model is None:
        experiment.eyetracker.model = metadata['model']
    elif experiment.eyetracker.model != metadata['model']:
        issues.append(f"Eye tracker model: {experiment.eyetracker.model} != {metadata['model']}")

    # Eye tracker software version
    if experiment.eyetracker.version is None:
        experiment.eyetracker.version = metadata['version_number']
    elif experiment.eyetracker.version != metadata['version_number']:
        issues.append(
            f'Eye tracker software version: {experiment.eyetracker.version} != '
            f"{metadata['version_number']}",
        )

    if issues:
        raise ValueError(
            'Experiment metadata does not match the metadata in the ASC file:\n'
            + '\n'.join(f'- {issue}' for issue in issues),
        )

    return experiment


def _fill_experiment_from_parsing_begaze_metadata(
        experiment: Experiment | None,
        metadata: dict[str, Any],
) -> Experiment:
    """Fill Experiment with BeGaze metadata.

    Behavior:
    - Create a new ``Experiment`` if none is provided, using the parsed sampling rate.
    - If a field on ``experiment`` is None, set it from parsed metadata when available.
    - If a field is already set and differs from parsed metadata, emit a warning and keep the
      existing value (do not raise).
    """
    # Ensure an Experiment exists
    if experiment is None:
        experiment = Experiment()

    # Only set the sampling rate if not set
    if experiment.eyetracker.sampling_rate is None:
        experiment.eyetracker.sampling_rate = metadata['sampling_rate']

    # Tracked eye flags if present (metadata may provide 'L'/'R')
    tracked = (metadata.get('tracked_eye') or '')
    left_parsed = 'L' in tracked
    right_parsed = 'R' in tracked

    if experiment.eyetracker.left is None:
        experiment.eyetracker.left = left_parsed
    elif experiment.eyetracker.left != left_parsed:
        warnings.warn(
            f"BeGaze metadata suggests left tracked={left_parsed} but experiment has "
            f"{experiment.eyetracker.left}; keeping experiment value.",
        )

    if experiment.eyetracker.right is None:
        experiment.eyetracker.right = right_parsed
    elif experiment.eyetracker.right != right_parsed:
        warnings.warn(
            f"BeGaze metadata suggests right tracked={right_parsed} but experiment has "
            f"{experiment.eyetracker.right}; keeping experiment value.",
        )

    # BeGaze headers typically do not include screen resolution in a standard way; if present as
    # 'resolution' in metadata, only set when experiment fields are None; warn on mismatch
    if 'resolution' in metadata:
        res = metadata['resolution']
        try:
            width, height = res
        except (TypeError, ValueError):
            width = height = None
        if experiment.screen.width_px is None and width is not None:
            experiment.screen.width_px = int(width)
        elif width is not None and experiment.screen.width_px not in (None, int(width)):
            warnings.warn(
                f"BeGaze metadata screen width={width} differs from experiment value "
                f"{experiment.screen.width_px}; keeping experiment value.",
            )
        if experiment.screen.height_px is None and height is not None:
            experiment.screen.height_px = int(height)
        elif height is not None and experiment.screen.height_px not in (None, int(height)):
            warnings.warn(
                f"BeGaze metadata screen height={height} differs from experiment value "
                f"{experiment.screen.height_px}; keeping experiment value.",
            )

    return experiment


def from_begaze(
        file: str | Path,
        *,
        patterns: list[dict[str, Any] | str] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        experiment: Experiment | None = None,
        trial_columns: str | list[str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, Any] | None = None,
        encoding: str | None = 'ascii',
        prefer_eye: str = 'L',
) -> Gaze:
    """Initialize a :py:class:`~pymovements.Gaze` from a BeGaze text export.

    Parameters
    ----------
    file: str | Path
        Path of BeGaze text export.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns (on BeGaze `MSG` lines).
    metadata_patterns: list[dict[str, Any] | str] | None
        List of patterns to match for extracting metadata from custom logged messages.
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns.
    experiment: Experiment | None
        The experiment definition. (default: None)
    trial_columns: str | list[str] | None
        The names of the columns (extracted by patterns) to use as trial columns.
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
    column_schema_overrides: dict[str, Any] | None
        Dictionary containing types for columns.
    encoding: str | None
        Text encoding of the file. Defaults to ASCII, which is the common BeGaze export encoding.
    prefer_eye: str
        Preferred eye to parse when both eyes are present ("L" or "R"). Defaults to "L".

    Returns
    -------
    Gaze
        The initialized gaze object read from the BeGaze text file.
    """
    # Read data via BeGaze parser.
    samples, event_data, metadata = parse_begaze(
        file,
        patterns=patterns,
        schema=schema,
        metadata_patterns=metadata_patterns,
        encoding=encoding or 'ascii',
        prefer_eye=prefer_eye,
    )

    if add_columns is not None:
        samples = samples.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in samples.columns
        ])

    if column_schema_overrides is not None:
        samples = samples.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in column_schema_overrides.items()
        ])

    # Fill experiment with parsed metadata.
    experiment = _fill_experiment_from_parsing_begaze_metadata(experiment, metadata)

    # Detect pixel columns to pass to Gaze (monocular naming from BeGaze uses 'x_pix', 'y_pix').
    detected_pixel_columns: list[str] | None = [c for c in samples.columns if '_pix' in c]

    gaze = Gaze(
        samples=samples,
        experiment=experiment,
        events=Events(event_data),
        trial_columns=trial_columns,
        time_column='time',
        time_unit='ms',
        pixel_columns=detected_pixel_columns,
    )
    return gaze
