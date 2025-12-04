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
"""Provides a definition for the UCL dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import KW_ONLY
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions


@dataclass
class UCL(DatasetDefinition):
    """UCL dataset :cite:p:`UCL`.

    UCL is a dataset of word-by-word reading times collected through
    self-paced reading and eye-tracking experiments to evaluate
    computational psycholinguistic models of English sentence comprehension.
    361 sentences from narrative sources, ensuring they were understandable without context,
    and recorded reading times from participants using both methods.

    For more details check out the original paper :cite:p:`UCL`.

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
    :py:class:`~pymovements.datasets.UCL` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("UCL", path='data/UCL')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'UCL'

    _: KW_ONLY  # all fields below can only be passed as a positional argument.

    long_name: str = 'University College London corpus'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions(
            [
                {
                    'content': 'precomputed_events',
                    'url': 'https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip',  # noqa: E501 # pylint: disable=line-too-long
                    'filename': 'UCL_events.zip',
                    'md5': '77e3c0cacccb0a074a55d23aa8531ca5',
                    'filename_pattern': r'eyetracking.fix',
                    'load_kwargs': {'separator': '\t', 'null_values': ['NaN']},
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip',  # noqa: E501 # pylint: disable=line-too-long
                    'filename': 'UCL_measures.zip',
                    'md5': '77e3c0cacccb0a074a55d23aa8531ca5',
                    'filename_pattern': r'eyetracking.RT',
                    'load_kwargs': {'separator': '\t'},
                },
            ],
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    column_map: dict[str, str] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] | None = None
