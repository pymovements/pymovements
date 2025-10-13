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
"""Provides a definition for the PotsdamBingeWearablePVT dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.screen import Screen


@dataclass
class PotsdamBingeWearablePVT(DatasetDefinition):
    """PotsdamBingeWearablePVT dataset :cite:p:`PotsdamBingePVT`.

    This dataset includes monocular eye tracking data from 57 participants in two sessions with an
    interval of at least one week between two sessions. Eye movements are recorded at a sampling
    frequency of ~200 Hz (upsampled to 1000 Hz and synchronised with the EyeLink 1000 Plus
    tracking the right eye) using Pupil Core eye-tracking glasses and are provided as
    pixel coordinates. Participants are instructed to perform a PVT trial.

    Check the respective `repository <https://osf.io/qf7e6/>`_ for details.

    Attributes
    ----------
    name: str
        The name of the dataset.

    long_name: str
        The entire name of the dataset.

    resources: ResourceDefinitions
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    experiment: Experiment
        The experiment definition.

    filename_format: dict[str, str] | None
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]] | None
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    trial_columns: list[str] | None
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately.

    time_column: str
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``.

    time_unit: str
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds.

    distance_column: str
        The name of the distance column in the input data frame. These column will be
        used to convert pixel coordinates into degrees of visual angle.

    pixel_columns: list[str]
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.


    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.PotsdamBingeWearablePVT` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("PotsdamBingeWearablePVT", path='data/PotsdamBingeWearablePVT')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'PotsdamBingeWearablePVT'

    long_name: str = 'Potsdam Binge Wearable PVT dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                    {
                        'content': 'samples',
                        'url': 'https://osf.io/download/9vbs8/',
                        'filename': 'a.zip',
                        'md5': '87c6c74a9a17cbd093b91f9415e8dd9d',
                        'filename_pattern': r'{subject_id:d}_{session_id:d}_{condition:s}_{trial_id:d}_{block_id:d}.csv',  # noqa: E501 # pylint: disable=line-too-long
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'trial_id': int,
                            'block_id': int,
                        },
                    },
                    {
                        'content': 'samples',
                        'url': 'https://osf.io/download/yqukn/',
                        'filename': 'b.zip',
                        'md5': '54038547b1a373253b38999a227dde63',
                        'filename_pattern': r'{subject_id:d}_{session_id:d}_{condition:s}_{trial_id:d}_{block_id:d}.csv',  # noqa: E501 # pylint: disable=line-too-long
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'trial_id': int,
                            'block_id': int,
                        },
                    },
                    {
                        'content': 'samples',
                        'url': 'https://osf.io/download/yf2xa/',
                        'filename': 'e.zip',
                        'md5': 'a0d0203cbb273f6908c1b52a42750551',
                        'filename_pattern': r'{subject_id:d}_{session_id:d}_{condition:s}_{trial_id:d}_{block_id:d}.csv',  # noqa: E501 # pylint: disable=line-too-long
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'trial_id': int,
                            'block_id': int,
                        },
                    },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen=Screen(
                width_px=1920,
                height_px=1080,
                width_cm=59.76,
                height_cm=33.615,
                origin='center',
            ),
            eyetracker=EyeTracker(
                sampling_rate=1000,
                left=True,
                right=False,
                model='Pupil Core eye-tracking glasses',
                vendor='Pupil Labs',
                mount='Wearable',
            ),
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    trial_columns: list[str] | None = None

    time_column: str = 'eyelink_timestamp'

    time_unit: str = 'ms'

    distance_column: str = 'target_distance'

    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'x_pix_pupilcore_interpolated',
            'y_pix_pupilcore_interpolated',
        ],
    )

    column_map: dict[str, str] = field(
        default_factory=lambda: {},
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'samples': {
                'schema_overrides': {
                    'trial_id': pl.Float32,
                    'block_id': pl.Float32,
                    'x_pix_eyelink': pl.Float32,
                    'y_pix_eyelink': pl.Float32,
                    'eyelink_timestamp': pl.Int64,
                    'x_pix_pupilcore_interpolated': pl.Float32,
                    'y_pix_pupilcore_interpolated': pl.Float32,
                    'pupil_size_eyelink': pl.Float32,
                    'target_distance': pl.Float32,
                    'pupil_size_pupilcore_interpolated': pl.Float32,
                    'pupil_confidence_interpolated': pl.Float32,
                    'time_to_prev_bac': pl.Float32,
                    'time_to_next_bac': pl.Float32,
                    'prev_bac': pl.Float32,
                    'next_bac': pl.Float32,
                },
                'separator': ',',
            },
        },
    )
