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
"""Provides a definition for the RaCCooNS dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
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
        The full name of the dataset.

    resources: ResourceDefinitions
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    experiment: Experiment
        The experiment definition.

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

    long_name: str = 'Radboud Coregistration Corpus of Narrative Sentences'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions(
            [
                {
                    'content': 'gaze',
                    'url': 'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/download/'
                    'eyetracking/ET_raw_data.zip',
                    'filename': 'ET_raw_data.zip',
                    'md5': '8b30241040071cee7afea367ae1e013e',
                    'filename_pattern': r'{participant_id:s}.asc',
                    'filename_pattern_schema_overrides': {'participant_id': str},
                    'load_kwargs': {
                        'patterns': [
                            r'TRIALID (?P<trial_index0>\d+)',
                            {'pattern': r'TRIAL_RESULT', 'column': 'trial_index0', 'value': None},
                        ],
                        'encoding': 'latin-1',
                        'trial_columns': ['trial_index0'],
                    },
                },
                {
                    'content': 'precomputed_events',
                    'url': 'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/download/'
                    'eyetracking/ET_fix_data.tsv',
                    'filename': 'ET_fix_data.tsv',
                    'md5': '98dff690022d0c0555987a6d88de992b',
                    'filename_pattern': r'ET_fix_data.tsv',
                    'filename_pattern_schema_overrides': {},
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t', 'encoding': 'latin-1'},
                    },
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://data.ru.nl/api/collectionfiles/ru/id/ru_395469/files/download/'
                    'eyetracking/ET_word_data.tsv',
                    'filename': 'ET_word_data.tsv',
                    'md5': 'c40886c4515c43187aba8fbc32c8c935',
                    'filename_pattern': r'ET_word_data.tsv',
                    'filename_pattern_schema_overrides': {},
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t', 'encoding': 'latin-1'},
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
                width_cm=56.8,
                height_cm=33.5,
                distance_cm=105.5,
                origin='center',
            ),
            eyetracker=EyeTracker(
                sampling_rate=1000,
                model='EyeLink 1000',
                vendor='EyeLink',
                mount='Desktop',
            ),
        ),
    )
