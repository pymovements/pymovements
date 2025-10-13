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
"""Provides a definition for the HBN dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class HBN(DatasetDefinition):
    """HBN dataset :cite:p:`HBN`.

    This dataset consists of recordings from children
    watching four different age-appropriate videos: (1) an
    educational video clip (Fun with Fractals), (2) a short animated
    film (The Present), (3) a short clip of an animated film (Despicable Me),
    and (4) a trailer for a feature-length movie (Diary of a Wimpy Kid).
    The eye gaze was recorded at a sampling rate of 120 Hz.

    Check the respective paper for details :cite:p:`HBN`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    long_name: str
        The entire name of the dataset.

    resources: ResourceDefinitions
        A list of dataset gaze_resources. Each list entry must be a dictionary with the following
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

    time_column: str
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``.

    time_unit: str
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds.

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
    :py:class:`~pymovements.datasets.HBN` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("HBN", path='data/HBN')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'HBN'

    long_name: str = 'Healthy Brain Network dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                {
                    'content': 'gaze',
                    'url': 'https://files.osf.io/v1/resources/qknuv/providers/osfstorage/651190031e76a453918a9971',  # noqa: E501 # pylint: disable=line-too-long
                    'filename': 'data.zip',
                    'md5': '2c523e911022ffc0eab700e34e9f7f30',
                    'filename_pattern': r'{subject_id:12}_{video_id}.csv',
                    'filename_pattern_schema_overrides': {
                        'subject_id': str,
                        'video_id': str,
                    },
                    'load_kwargs': {
                        'time_column': 'time',
                        'time_unit': 'step',
                        'pixel_columns': ['x_pix', 'y_pix'],
                    },
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=800,
            screen_height_px=600,
            screen_width_cm=33.8,
            screen_height_cm=27.0,
            distance_cm=63.5,
            origin='center',
            sampling_rate=120,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    time_column: str | None = None

    time_unit: str | None = None

    pixel_columns: list[str] | None = None

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {
                'separator': ',',
                'columns': ['time', 'x_pix', 'y_pix'],
                'schema_overrides': {
                    'time': pl.Int64,
                    'x_pix': pl.Float32,
                    'y_pix': pl.Float32,
                },
            },
        },
    )
