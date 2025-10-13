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
"""Provides a definition for the EMTeC dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class EMTeC(DatasetDefinition):
    """EMTeC dataset :cite:p:`EMTeC`.

    This dataset includes eye-tracking data from 107 native speakers of English reading
    machine generated texts.  Eye movements are recorded at a sampling frequency of 1,000 Hz
    using an EyeLink 1000 eye tracker and are provided as pixel coordinates.

    Check the respective paper for details :cite:p:`EMTeC`.

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

    trial_columns: list[str]
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

    pixel_columns: list[str]
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.EMTeC` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("EMTeC", path='data/EMTeC')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'EMTeC'

    long_name: str = 'Eye movements on Machine-generated Texts Corpus'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                {
                    'content': 'samples',
                    'url': 'https://osf.io/download/374sk/',
                    'filename': 'subject_level_data.zip',
                    'md5': 'dca99e47ef43f3696acec4fd70967750',
                    'filename_pattern': r'ET_{subject_id:d}.csv',
                    'filename_pattern_schema_overrides': {'subject_id': int},
                },
                {
                    'content': 'precomputed_events',
                    'url': 'https://osf.io/download/2hs8p/',
                    'filename': 'fixations.csv',
                    'md5': '5e05a364a1d8a044d8b36506aa91437e',
                    'filename_pattern': r'fixations.csv',
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://osf.io/download/s4ny8/',
                    'filename': 'reading_measures.csv',
                    'md5': '56880f50af20682558065ac2d26be827',
                    'filename_pattern': r'reading_measures.csv',
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=38.2,
            screen_height_cm=30.2,
            distance_cm=60,
            origin='center',
            sampling_rate=2000,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    trial_columns: list[str] = field(default_factory=lambda: ['item_id'])

    time_column: str = 'time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(default_factory=lambda: ['x', 'y'])

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'samples': {
                'separator': '\t',
                'columns': [
                    'item_id',
                    'TRIAL_ID',
                    'Trial_Index_',
                    'model',
                    'decoding_strategy',
                    'time',
                    'x',
                    'y',
                    'pupil_right',
                ],
                'schema_overrides': {
                    'item_id': pl.Utf8,
                    'TRIAL_ID': pl.Int64,
                    'Trial_Index_': pl.Int64,
                    'model': pl.Utf8,
                    'decoding_strategy': pl.Utf8,
                    'time': pl.Int64,
                    'x': pl.Float32,
                    'y': pl.Float32,
                    'pupil_right': pl.Float32,
                },
            },
            'precomputed_events': {'separator': '\t'},
            'precomputed_reading_measures': {'separator': '\t'},
        },
    )
