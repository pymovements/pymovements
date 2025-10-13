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
"""Provides a definition for the ETDD70 dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class ETDD70(DatasetDefinition):
    """Eye-Tracking Dyslexia Dataset (ETDD70) :cite:p:`ETDD70`.

    This dataset includes binocular eye tracking data from 70 Czech children age 9-10.
    Eye movements are recorded at a sampling frequency of 250 Hz eye tracker and
    precomputed events are reported.

    Each participant is instructed to read three texts:
        - Task called Syllables contains 90 syllables arranged in a 9 x 10 matrix
        - Task called MeaningfulText consists of a passage about
          a young boy who watches a squirrel from his window.
        - Task called PseudoText comprises fictional, meaningless words.

    Check the respective paper for details :cite:p:`ETDD70`.

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

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.ETDD70` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("ETDD70", path='data/ETDD70')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'ETDD70'

    long_name: str = 'Eye-Tracking Dyslexia Dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                {
                    'content': 'gaze',
                    'url': 'https://zenodo.org/api/records/13332134/files-archive',
                    'filename': 'edd_raw.zip',
                    'md5': None,  # type: ignore
                    'filename_pattern': r'Subject_{subject_id:d}_{task:s}_raw.csv',
                    'load_kwargs': {
                        'time_column': 'time',
                        'time_unit': 'ms',
                        'pixel_columns': [
                            'gaze_x_left', 'gaze_y_left', 'gaze_x_right', 'gaze_y_right',
                        ],
                    },
                },
                {
                    'content': 'precomputed_events',
                    'url': 'https://zenodo.org/api/records/13332134/files-archive',
                    'filename': 'edd_fix.zip',
                    'md5': None,  # type: ignore
                    'filename_pattern': r'Subject_{subject_id:d}_{task:s}_fixations.csv',
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1680,
            screen_height_px=1050,
            screen_width_cm=None,
            screen_height_cm=None,
            distance_cm=65,  # in the paper it is written (60-70cm)
            origin='center',
            sampling_rate=250,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    time_column: str | None = None

    time_unit: str | None = None

    pixel_columns: list[str] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda:
            {
                'gaze': {},
                'precomputed_events': {},
            },
    )
