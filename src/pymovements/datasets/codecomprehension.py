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
"""Provides a definition for the CodeComprehension dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import KW_ONLY
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions


@dataclass
class CodeComprehension(DatasetDefinition):
    """CodeComprehension dataset :cite:p:`CodeComprehension`.

    This dataset includes eye-tracking-while-code-reading data from a participants in a single
    session. Eye movements are recorded at a sampling frequency of 1,000 Hz using an
    EyeLink 1000 eye tracker and are provided as pixel coordinates.

    The participant is instructed to read the code snippet and answer a code comprehension question.

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

    filename_format: dict[str, str] | None
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.
    filename_format_schema_overrides: dict[str, dict[str, type]] | None
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    column_map: dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]] | None
        If specified, these keyword arguments will be passed to the file reading function.
        (default: None)

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.CodeComprehension` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("CodeComprehension", path='data/CodeComprehension')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'CodeComprehension'

    _: KW_ONLY  # all fields below can only be passed as a positional argument.

    long_name: str = 'Code Comprehension dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions(
            [
                {
                    'content': 'precomputed_events',
                    'url': 'https://zenodo.org/records/11123101/files/Predicting%20Code%20Comprehension%20Package.zip?download=1',  # noqa: E501 # pylint: disable=line-too-long
                    'filename': 'data.zip',
                    'md5': '3a3c6fb96550bc2c2ddcf5d458fb12a2',
                    'filename_pattern': 'fix_report_P{subject_id:s}.txt',
                    'filename_pattern_schema_overrides': {'subject_id': pl.String},
                    'load_kwargs': {'separator': '\t', 'null_values': '.', 'quote_char': '"'},
                },
            ],
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    column_map: dict[str, str] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] | None = None
