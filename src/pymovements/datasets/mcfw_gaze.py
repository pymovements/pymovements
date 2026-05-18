# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Provides a definition for the mcfw_gaze dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import KW_ONLY
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.gaze.experiment import Experiment


@dataclass
class MCFWGaze(DatasetDefinition):
    """MCFW-Gaze dataset :cite:p:`MCFW-Gaze`.

    Multi-context eye-tracking dataset recorded with a Tobii Pro Fusion (120 Hz)
    from 15 participants across four task types: natural image free-viewing
    (100 images, 3 sessions, 5 seconds per image), gaze-pattern authentication (5 trials),
    password experiment (5 blocks), and free web browsing
    (news, shopping, and video, 600 seconds per task).

    Eye-tracking data is provided as binocular gaze point coordinates on the
    display area, along with pupil diameter and gaze origin in user coordinates.

    Check the respective zenodo page for details :cite:p:`MCFW-Gaze`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    long_name: str
        The entire name of the dataset.

    resources: ResourceDefinitions
        A list of dataset resources.

    experiment: Experiment
        The experiment definition.

    filename_format: dict[str, str] | None
        Regular expression matched before loading the file. Named groups appear in `fileinfo`.

    filename_format_schema_overrides: dict[str, dict[str, type]] | None
        Allows casting named groups from `filename_format` to specific datatypes.

    trial_columns: list[str] | None
        The name of the trial columns in the input data frame.

    time_column: Any
        The name of the timestamp column in the input data frame.

    time_unit: Any
        The unit of the timestamps in the timestamp column.

    pixel_columns: list[str] | None
        The name of the pixel position columns in the input data frame.

    column_map: dict[str, str] | None
        Mapping from column names to read to their renamed targets.

    custom_read_kwargs: dict[str, dict[str, Any]] | None
        Extra keyword arguments passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.MCFWGaze` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("MCFW-Gaze", path='data/mcfw_gaze')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'MCFW-Gaze'

    _: KW_ONLY  # all fields below can only be passed as keyword arguments.

    long_name: str = 'Multi-Context Free-Viewing and Web-Browsing Gaze Dataset'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions(
            [
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://zenodo.org/records/19463338/files/dataset.zip?download=1',
                        'filename': 'dataset.zip',
                        'md5': '8fdb6e04df4ca2dee59b14edc3ec3aed',
                    },
                    'filename_pattern': r'dataset/data/participant_{participant_id:d}/image_\d+_\d+\.tsv',  # noqa: E501 # pylint: disable=line-too-long
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t'},
                        'time_column': 'device_time_stamp',
                        'time_unit': 'us',
                        'pixel_columns': [
                            'left_gaze_point_on_display_area_x',
                            'left_gaze_point_on_display_area_y',
                            'right_gaze_point_on_display_area_x',
                            'right_gaze_point_on_display_area_y',
                        ],
                    },
                },
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://zenodo.org/records/19463338/files/dataset.zip?download=1',
                        'filename': 'dataset.zip',
                        'md5': '8fdb6e04df4ca2dee59b14edc3ec3aed',
                    },
                    'filename_pattern': r'dataset/data/participant_{participant_id:d}/gaze_pattern_auth_trial\d+\.tsv',  # noqa: E501 # pylint: disable=line-too-long
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t'},
                        'time_column': 'device_time_stamp',
                        'time_unit': 'us',
                        'pixel_columns': [
                            'left_gaze_point_on_display_area_x',
                            'left_gaze_point_on_display_area_y',
                            'right_gaze_point_on_display_area_x',
                            'right_gaze_point_on_display_area_y',
                        ],
                    },
                },
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://zenodo.org/records/19463338/files/dataset.zip?download=1',
                        'filename': 'dataset.zip',
                        'md5': '8fdb6e04df4ca2dee59b14edc3ec3aed',
                    },
                    'filename_pattern': r'dataset/data/participant_{participant_id:d}/password_experiment_block\d+\.tsv',  # noqa: E501 # pylint: disable=line-too-long
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t'},
                        'time_column': 'device_time_stamp',
                        'time_unit': 'us',
                        'pixel_columns': [
                            'left_gaze_point_on_display_area_x',
                            'left_gaze_point_on_display_area_y',
                            'right_gaze_point_on_display_area_x',
                            'right_gaze_point_on_display_area_y',
                        ],
                    },
                },
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://zenodo.org/records/19463338/files/dataset.zip?download=1',
                        'filename': 'dataset.zip',
                        'md5': '8fdb6e04df4ca2dee59b14edc3ec3aed',
                    },
                    'filename_pattern': r'dataset/data/participant_{participant_id:d}/(?:news|shopping|video)\.tsv',  # noqa: E501 # pylint: disable=line-too-long
                    'load_kwargs': {
                        'read_csv_kwargs': {'separator': '\t'},
                        'time_column': 'device_time_stamp',
                        'time_unit': 'us',
                        'pixel_columns': [
                            'left_gaze_point_on_display_area_x',
                            'left_gaze_point_on_display_area_y',
                            'right_gaze_point_on_display_area_x',
                            'right_gaze_point_on_display_area_y',
                        ],
                    },
                },
            ],
        ),
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1920,
            screen_height_px=1080,
            screen_width_cm=31.0,
            screen_height_cm=17.5,
            distance_cm=65.0,
            origin='upper left',
            sampling_rate=120,
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    trial_columns: list[str] | None = None

    time_column: Any = None

    time_unit: Any = None

    pixel_columns: list[str] | None = None

    column_map: dict[str, str] | None = None

    custom_read_kwargs: dict[str, dict[str, Any]] | None = None
