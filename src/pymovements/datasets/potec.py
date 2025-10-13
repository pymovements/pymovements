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
"""Provides a definition for the PoTeC dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class PoTeC(DatasetDefinition):
    """PoTeC dataset :cite:p:`PoTeC`.

    The Potsdam Textbook Corpus (PoTeC) is a naturalistic eye-tracking-while-reading
    corpus containing data from 75 participants reading 12 scientific texts.
    PoTeC is the first naturalistic eye-tracking-while-reading corpus that contains
    eye-movements from domain-experts as well as novices in a within-participant
    manipulation: It is based on a 2×2×2 fully-crossed factorial design which includes
    the participants' level of study and the participants' discipline of study as
    between-subject factors and the text domain as a within-subject factor. The
    participants' reading comprehension was assessed by a series of text comprehension
    questions and their domain knowledge was tested by text-independent
    background questions for each of the texts. The materials are annotated for a
    variety of linguistic features at different levels. We envision PoTeC to be used
    for a wide range of studies including but not limited to analyses of expert and
    non-expert reading strategies.

    The corpus and all the accompanying data at all
    stages of the preprocessing pipeline and all code used to preprocess the data are
    made available via `GitHub. <https://github.com/DiLi-Lab/PoTeC>`_

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
    :py:class:`~pymovements.datasets.PoTeC` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("PoTeC", path='data/PoTeC')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'PoTeC'

    long_name: str = 'Potsdam Textbook Corpus'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                    {
                        'content': 'samples',
                        'url': 'https://osf.io/download/tgd9q/',
                        'filename': 'PoTeC.zip',
                        'md5': 'cffd45039757c3777e2fd130e5d8a2ad',
                        'filename_pattern': r'reader{subject_id:d}_{text_id}_raw_data.tsv',
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'text_id': str,
                        },
                    },
                    {
                        'content': 'precomputed_events',
                        'url': 'https://osf.io/download/d8pyg/',
                        'filename': 'fixation.zip',
                        'md5': 'ecd9a998d07158922bb9b8cdd52f5688',
                        'filename_pattern': r'reader{subject_id:d}_{text_id}_uncorrected_fixations.tsv',  # noqa: E501 # pylint: disable=line-too-long
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'text_id': str,
                        },
                    },
                    {
                        'content': 'precomputed_reading_measures',
                        'url': 'https://osf.io/download/3ywhz/',
                        'filename': 'reading_measures.zip',
                        'md5': 'efafec5ce074d8f492cc2409b6c4d9eb',
                        'filename_pattern': r'reader{subject_id:d}_{text_id}_merged.tsv',
                        'filename_pattern_schema_overrides': {
                            'subject_id': int,
                            'text_id': str,
                        },
                    },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1680,
            screen_height_px=1050,
            screen_width_cm=47.5,
            screen_height_cm=30,
            distance_cm=65,
            origin='upper left',
            sampling_rate=1000,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    time_column: str = 'time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'x', 'y',
        ],
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'samples': {
                'schema_overrides': {
                    'time': pl.Int64,
                    'x': pl.Float32,
                    'y': pl.Float32,
                    'pupil_diameter': pl.Float32,
                },
                'separator': '\t',
            },
            'precomputed_events': {
                'separator': '\t',
                'null_values': '.',
            },
            'precomputed_reading_measures': {
                'separator': '\t',
                'null_values': '.',
                'infer_schema_length': 10000,
            },
        },
    )
