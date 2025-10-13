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
"""Provides a definition for the Gaze4Hate dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class Gaze4Hate(DatasetDefinition):
    """Gaze4Hate dataset :cite:p:`Gaze4Hate`.

    This dataset includes monocular eye tracking data from 43 participants annotating sentences for
    hate speech. Eye movements are recorded at a sampling frequency of 1,000 Hz using an EyeLink
    1000 Plus eye tracker and are provided as pixel coordinates.

    Check the respective paper for details :cite:p:`Gaze4Hate`.

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

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.Gaze4Hate` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("Gaze4Hate", path='data/Gaze4Hate')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'Gaze4Hate'

    long_name: str = 'Gaze4Hate dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                {
                    'content': 'precomputed_events',
                    'url': 'https://osf.io/download/dbshf/',
                    'filename': 'gaze4hate_sentence_reading_fix_report.csv',
                    'md5': 'c8cc645d1fad659f9442d61795da5481',
                    'filename_pattern': 'gaze4hate_sentence_reading_fix_report.csv',
                    'load_kwargs': {
                        'trial_columns': ['pno', 'sno'],
                    },
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://osf.io/download/fgdjw/',
                    'filename': 'gaze4hate_sentence_reading_IA_report.csv',
                    'md5': 'e09e791e7d31d6ac3c69cd862d139c57',
                    'filename_pattern': 'gaze4hate_sentence_reading_IA_report.csv',
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=2560,
            screen_height_px=1440,
            screen_width_cm=59.8,
            screen_height_cm=33.6,
            distance_cm=78.0,
            origin='center',
            sampling_rate=1000,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    trial_columns: list[str] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda:
        {
            'precomputed_events': {
                'separator': '\t',
                'null_values': '.',
            },
            'precomputed_reading_measures': {
                'separator': '\t',
            },
        },
    )
