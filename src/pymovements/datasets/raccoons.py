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
"""Provides a definition for the RaCCooNS dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.screen import Screen


@dataclass
class RaCCooNS(DatasetDefinition):
    """RaCCooNS dataset :cite:p:`RaCCooNS`.

    The Radboud Coregistration Corpus of Narrative Sentences (RaCCooNS) dataset consists
    simultaneously recorded eye-tracking and EEG data from Dutch sentence reading,
    aimed at studying human sentence comprehension and evaluating computational
    language models. The dataset includes 37 participants reading 200 narrative sentences,
    with eye movements and brain activity recorded to analyze reading behavior
    and neural responses. The dataset provides both raw and preprocessed data,
    including fixation-related potentials, enabling comparisons between cognitive
    and neural processes.

    Check the respective paper :cite:p:`RaCCooNS` for details.

    Attributes
    ----------
    name: str
        The name of the dataset.

    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.

    mirrors: dict[str, list[str]]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources: dict[str, list[dict[str, str | None]]]
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    extract: dict[str, bool]
        Decide whether to extract the data.

    experiment: Experiment
        The experiment definition.

    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    trial_columns: list[str]
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, Any]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.Dataset` object with the
    :py:class:`~pymovements.datasets.RaCCooNS` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("RaCCooNS", path='data/RaCCooNS')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'RaCCooNS'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )
    mirrors: dict[str, list[str]] = field(
        default_factory=lambda: {
            'gaze': [
                'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/',
            ],
            'precomputed_events': [
                'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/',
            ],
            'precomputed_reading_measures': [
                'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/',
            ],
        },
    )
    resources: dict[str, list[dict[str, str | None]]] = field(
        default_factory=lambda: {
            'gaze': [
                {
                    'resource': 'download/eyetracking%2FET_raw_data.zip',
                    'filename': 'ET_raw_data.zip',
                    'md5': None,  # type: ignore
                },
            ],
            'precomputed_events': [
                {
                    'resource': 'download/eyetracking%2FET_fix_data.tsv',
                    'filename': 'ET_fix_data.tsv',
                    'md5': None,  # type: ignore
                },
            ],
            'precomputed_reading_measures': [
                {
                    'resource': 'download/eyetracking%2FET_word_data.tsv',
                    'filename': 'ET_word_data.tsv',
                    'md5': None,  # type: ignore
                },
            ],
        },
    )

    extract: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen=Screen(
                width_px=1920,
                # in the paper its written 1018, but assume typo
                height_px=1080,
                width_cm=56.8,
                height_cm=33.5,
                distance_cm=105.5,
                origin='center',
            ),
            eyetracker=EyeTracker(
                sampling_rate=1000,
                # double check once data is back up
                left=True,
                right=False,
                model='EyeLink 1000',
                vendor='EyeLink',
                mount='Desktop',
            ),
        ),
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze': r'{participant_id:s}.asc',
            'precomputed_events': r'ET_fix_data.tsv',
            'precomputed_reading_measures': r'ET_word_data.tsv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {'participant_id': str},
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: [])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'gaze': {
                'patterns': [
                    r'TRIALID (?P<trial_index0>\d+)',
                    {'pattern': r'TRIAL_RESULT', 'column': 'trial_index0', 'value': None},
                ],
                'encoding': 'latin-1',
            },
            'precomputed_events': {'separator': '\t', 'encoding': 'latin-1'},
            'precomputed_reading_measures': {'separator': '\t', 'encoding': 'latin-1'},
        },
    )
