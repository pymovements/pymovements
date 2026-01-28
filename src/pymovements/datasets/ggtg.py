# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Provides a definition for the CoLAGaze dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import KW_ONLY
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.experiment import EyeTracker


@dataclass
class GGTG(DatasetDefinition):
    """GGTG dataset :cite:p:`GGTG`.

    This dataset includes eye-tracking data from native (P13-24) and non-native (P01-P12) speakers
    of English reading multi-page texts generated via gaze-guided text generation. Eye movements
    are recorded at a sampling frequency of 1,000 Hz using an EyeLink 1000 Plus eye tracker and
    are provided as pixel coordinates.

    Check the respective paper :cite:p:`GGTG` and the `OSF repository <https://osf.io/rhgbk/>`__
    for details.

    Warning
    -------
    This dataset currently cannot be fully processed by ``pymovements`` due to an error during
    parsing of individual files.

    See issue `#1401 <https://github.com/pymovements/pymovements/issues/1401>`__ for reference.

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

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.GGTG` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("GGTG", path='data/GGTG')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'GGTG'

    _: KW_ONLY  # all fields below can only be passed as a positional argument.

    long_name: str = 'Corpus of Eye Movements for Linguistic Acceptability'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions(
            [
                {
                    'content': 'gaze',
                    'url': 'https://osf.io/download/wdeq7/',
                    'filename': 'samples.zip',
                    'md5': 'e26d81a46b7ffcdc3d0e8a0f069020c3',
                    'filename_pattern': '{subject_id}.csv',
                },
                {
                    'content': 'precomputed_events',
                    'url': 'https://osf.io/download/dvkwg/',
                    'filename': 'fixations.zip',
                    'md5': '9d8c7f84442671876c2190958c8bd0dc',
                    'filename_pattern': 'fixations_report_{subject_id}.csv',
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://osf.io/download/udmrk/',
                    'filename': 'measures.zip',
                    'md5': '63d33923d04f56c95a3d52e5b89a4ebd',
                    'filename_pattern': '{subject_id}.csv',
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1100,
            screen_height_px=900,
            screen_width_cm=31.2,
            screen_height_cm=25.2,
            distance_cm=66,
            origin='top left',
            eyetracker=EyeTracker(
                vendor='EyeLink',
                model='EyeLink 1000 Plus',
                mount='desktop',
                sampling_rate=1000,
            ),
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {},
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )
