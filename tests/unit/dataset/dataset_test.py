# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test all functionality in pymovements.dataset.dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
import os
from pathlib import Path
import re
import shutil
from unittest.mock import Mock

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from pymovements import __version__
from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import DatasetPaths
from pymovements import Events
from pymovements import Experiment
from pymovements import Gaze
from pymovements import ResourceDefinition
from pymovements.dataset.dataset_files import DatasetFile
from pymovements.events import fill
from pymovements.events import idt
from pymovements.events import ivt
from pymovements.events import microsaccades
from pymovements.exceptions import UnknownMeasure
from pymovements.stimulus.text import TextStimulus
from pymovements.warnings import ExperimentalWarning

# pylint: disable=too-many-lines

EXPECTED_AOI_MULTIPLEYE_STIMULI_TOY_X_1_TEXT_1_1 = pl.DataFrame(
    {
        'char_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'char': ['W', 'h', 'a', 't', ' ', 'i', 's', ' ', 'p', 'y'],
        'top_left_x': [81.0, 94.0, 107.0, 120.0, 133.0, 146.0, 159.0, 172.0, 185.0, 198.0],
        'top_left_y': [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        'width': [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
        'height': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        'char_idx_in_line': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'line_idx': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'page': ['question_01111' for _ in range(10)],
        'word_idx': [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        'word_idx_in_line': [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        'word': [
            'What',
            'What',
            'What',
            'What',
            ' ',
            'is',
            'is',
            ' ',
            'pymovements?',
            'pymovements?',
        ],
        'question_image_version': ['question_images_version_1' for _ in range(10)],
    },
)


class _UNSET: ...


@pytest.fixture(name='make_dataset', scope='function')
def fixture_make_dataset(tmp_path: Path) -> Callable[[list[str], Path | None], Path]:
    """Make a dataset of empty files.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory where files are copied to.

    Returns
    -------
    Callable[[list[str], Path | None], Path]
        Function that takes a directory structure and returns the Path to the root directory.

    """

    def _make_dataset(files: list[str], root: Path | None = None) -> Path:
        if root is None:
            root = tmp_path

        for relative_filepath in files:
            filepath = root / relative_filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Create new empty file.
            with open(filepath, 'x', encoding='utf-8'):
                pass
        return root

    return _make_dataset


@pytest.mark.parametrize(
    'files',
    [
        pytest.param([], id='no_files'),
        pytest.param(['test.abc'], id='single_file_in_root_directory'),
        pytest.param(['my/test.abc'], id='single_file_in_child_directory'),
        pytest.param(['one.abc', 'two.abc'], id='two_files_in_root_directory'),
        pytest.param(['one/test.abc', 'two/test.abc'], id='two_files_in_separate_directories'),
    ],
)
def test_make_dataset_creates_correct_tree(files, make_dataset, tmp_path):
    dataset_dirpath = make_dataset(files, tmp_path)

    created_files = {path for path in Path(dataset_dirpath).rglob('*') if not path.is_dir()}

    expected_files = {tmp_path / relative_filepath for relative_filepath in files}

    assert created_files == expected_files


def create_raw_gaze_files_from_fileinfo(gazes, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze, fileinfo_row in zip(gazes, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        for key in fileinfo_row:
            if key in gaze.columns:
                gaze = gaze.drop(key)

        gaze.write_csv(rootpath / filepath)


def create_preprocessed_gaze_files_from_fileinfo(gazes, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze, fileinfo_row in zip(gazes, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']
        filepath = filepath.replace('csv', 'feather')

        gaze.samples.write_ipc(rootpath / filepath)


def create_event_files_from_fileinfo(events_list, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for events, fileinfo_row in zip(events_list, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']
        filepath = filepath.replace('csv', 'feather')

        events.write_ipc(rootpath / filepath)


def create_precomputed_files_from_fileinfo(precomputed_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for precomputed_df, fileinfo_row in zip(precomputed_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        precomputed_df.write_csv(rootpath / filepath)


def create_precomputed_rm_files_from_fileinfo(precomputed_rm_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for precomputed_rm_df, fileinfo_row in zip(precomputed_rm_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        precomputed_rm_df.write_csv(rootpath / filepath)


def mock_toy(
    rootpath,
    raw_fileformat,
    eyes,
    remote=False,
    stimulus=False,
    extract=_UNSET,
    filename_format_schema_overrides=_UNSET,
    testfiles_dirpath=None,
):
    if extract is _UNSET:
        extract = None

    if filename_format_schema_overrides is _UNSET:
        filename_format_schema_overrides = {
            'gaze': {'subject_id': pl.Int64},
            'precomputed_events': {'subject_id': pl.Int64},
            'precomputed_reading_measures': {'subject_id': pl.Int64},
        }

    if filename_format_schema_overrides['precomputed_events']:
        subject_ids = list(range(1, 21))
        fileinfo = pl.DataFrame(
            data={'subject_id': subject_ids},
            schema={'subject_id': pl.Int64},
        )
    else:
        subject_ids = [str(idx) for idx in range(1, 21)]
        fileinfo = pl.DataFrame(
            data={'subject_id': subject_ids},
            schema={'subject_id': pl.String},
        )

    fileinfo = fileinfo.with_columns(
        [
            pl.format('{}.' + raw_fileformat, 'subject_id').alias('filepath'),
        ]
    )

    fileinfo = fileinfo.sort(by='filepath')

    gazes = []
    for _ in range(fileinfo.height):
        if eyes == 'both':
            gaze = pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                    'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                    'trial': np.concatenate([np.zeros(500), np.ones(500)]),
                },
                schema={
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'task': pl.String,
                    'trial': pl.Int64,
                },
            )
            pixel_columns = ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix']

        elif eyes == 'both+avg':
            gaze = pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                    'x_avg_pix': np.zeros(1000),
                    'y_avg_pix': np.zeros(1000),
                    'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                    'trial': np.concatenate([np.zeros(500), np.ones(500)]),
                },
                schema={
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'x_avg_pix': pl.Float64,
                    'y_avg_pix': pl.Float64,
                    'task': pl.String,
                    'trial': pl.Int64,
                },
            )
            pixel_columns = [
                'x_left_pix',
                'y_left_pix',
                'x_right_pix',
                'y_right_pix',
                'x_avg_pix',
                'y_avg_pix',
            ]

        elif eyes == 'left':
            gaze = pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                    'trial': np.concatenate([np.zeros(500), np.ones(500)]),
                },
                schema={
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'task': pl.String,
                    'trial': pl.Int64,
                },
            )
            pixel_columns = ['x_left_pix', 'y_left_pix']
        elif eyes == 'right':
            gaze = pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                    'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                    'trial': np.concatenate([np.zeros(500), np.ones(500)]),
                },
                schema={
                    'time': pl.Int64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'task': pl.String,
                    'trial': pl.Int64,
                },
            )
            pixel_columns = ['x_right_pix', 'y_right_pix']
        elif eyes == 'none':
            gaze = pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_pix': np.zeros(1000),
                    'y_pix': np.zeros(1000),
                    'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                    'trial': np.concatenate([np.zeros(500), np.ones(500)]),
                },
                schema={
                    'time': pl.Int64,
                    'x_pix': pl.Float64,
                    'y_pix': pl.Float64,
                    'task': pl.String,
                    'trial': pl.Int64,
                },
            )
            pixel_columns = ['x_pix', 'y_pix']
        else:
            raise ValueError(f'invalid value for eyes: {eyes}')

        if remote:
            gaze = gaze.with_columns(
                [
                    pl.lit(680.0).alias('distance'),
                ]
            )

            distance_column = 'distance'
            distance_cm = None
        else:
            distance_column = None
            distance_cm = 68

        gazes.append(gaze)

    create_raw_gaze_files_from_fileinfo(gazes, fileinfo, rootpath / 'raw')

    gaze_sample_resource_definition = ResourceDefinition(
        content='gaze',
        filename_pattern=r'{subject_id:d}.' + raw_fileformat,
        filename_pattern_schema_overrides=filename_format_schema_overrides.get('gaze'),
        load_kwargs={
            'time_column': 'time',
            'time_unit': 'ms',
            'distance_column': distance_column,
            'pixel_columns': pixel_columns,
            'trial_columns': ['task', 'trial'],
        },
    )
    resource_definitions = [gaze_sample_resource_definition]

    files = [
        DatasetFile(
            path=rootpath / 'raw' / fileinfo_row['filepath'],  # absolute path
            definition=gaze_sample_resource_definition,
            metadata={key: value for key, value in fileinfo_row.items() if key != 'filepath'},
        )
        for fileinfo_row in fileinfo.to_dicts()
    ]

    # Create Gazes for passing as ground truth
    gazes = [Gaze(gaze, pixel_columns=pixel_columns) for gaze in gazes]

    preprocessed_gazes = []
    for _ in range(fileinfo.height):
        position_columns = [pixel_column.replace('pix', 'pos') for pixel_column in pixel_columns]
        velocity_columns = [pixel_column.replace('pix', 'vel') for pixel_column in pixel_columns]
        acceleration_columns = [
            pixel_column.replace('pix', 'acc') for pixel_column in pixel_columns
        ]

        gaze_data = {
            'time': np.arange(1000),
        }
        gaze_schema = {
            'time': pl.Int64,
        }

        for column in pixel_columns + position_columns + velocity_columns + acceleration_columns:
            gaze_data[column] = np.zeros(1000)
            gaze_schema[column] = pl.Float64

        # Create Gazes for passing as ground truth
        gaze = Gaze(
            pl.from_dict(gaze_data, schema=gaze_schema),
            pixel_columns=pixel_columns,
            position_columns=position_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
        )

        preprocessed_gazes.append(gaze)

    create_preprocessed_gaze_files_from_fileinfo(
        preprocessed_gazes,
        fileinfo,
        rootpath / 'preprocessed',
    )

    events_list = []
    for _ in range(fileinfo.height):
        events = pl.from_dict(
            {
                'name': ['saccade', 'fixation'] * 5,
                'onset': np.arange(0, 100, 10),
                'offset': np.arange(5, 105, 10),
                'duration': np.array([5] * 10),
            },
            schema={
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
            },
        )
        events_list.append(events)

    create_event_files_from_fileinfo(events_list, fileinfo, rootpath / 'events')

    precomputed_dfs = []
    for _ in range(fileinfo.height):
        precomputed_events = pl.from_dict(
            {
                'CURRENT_FIXATION_DURATION': np.arange(1000),
                'CURRENT_FIX_X': np.zeros(1000),
                'CURRENT_FIX_Y': np.zeros(1000),
                'task': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                'trial': np.concatenate([np.zeros(500), np.ones(500)]),
            },
            schema={
                'CURRENT_FIXATION_DURATION': pl.Float64,
                'CURRENT_FIX_X': pl.Float64,
                'CURRENT_FIX_Y': pl.Float64,
                'task': pl.String,
                'trial': pl.Int64,
            },
        )
        precomputed_dfs.append(precomputed_events)

    create_precomputed_files_from_fileinfo(
        precomputed_dfs,
        fileinfo,
        rootpath / 'precomputed_events',
    )

    precomputed_events_resource_definition = ResourceDefinition(
        content='precomputed_events',
        filename_pattern=r'{subject_id:d}.' + raw_fileformat,
        filename_pattern_schema_overrides=filename_format_schema_overrides.get(
            'precomputed_events'
        ),
    )
    resource_definitions.append(precomputed_events_resource_definition)

    precomputed_events_files = [
        DatasetFile(
            path=rootpath / 'precomputed_events' / fileinfo_row['filepath'],  # absolute path
            definition=precomputed_events_resource_definition,
            metadata={key: value for key, value in fileinfo_row.items() if key != 'filepath'},
        )
        for fileinfo_row in fileinfo.to_dicts()
    ]
    files.extend(precomputed_events_files)

    precomputed_rm_dfs = []
    for _ in range(fileinfo.height):
        precomputed_rm_df = pl.from_dict(
            {
                'number_fix': np.arange(1000),
                'mean_fix_dur': np.zeros(1000),
            },
            schema={
                'number_fix': pl.Float64,
                'mean_fix_dur': pl.Float64,
            },
        )
        precomputed_rm_dfs.append(precomputed_rm_df)

    create_precomputed_rm_files_from_fileinfo(
        precomputed_rm_dfs,
        fileinfo,
        rootpath / 'precomputed_reading_measures',
    )

    precomputed_rm_resource_definition = ResourceDefinition(
        content='precomputed_reading_measures',
        filename_pattern=r'{subject_id:d}.' + raw_fileformat,
        filename_pattern_schema_overrides=filename_format_schema_overrides.get(
            'precomputed_reading_measures'
        ),
    )
    resource_definitions.append(precomputed_rm_resource_definition)

    precomputed_rm_files = [
        DatasetFile(
            path=rootpath / 'precomputed_reading_measures' / fileinfo_row['filepath'],
            definition=precomputed_rm_resource_definition,
            metadata={key: value for key, value in fileinfo_row.items() if key != 'filepath'},
        )
        for fileinfo_row in fileinfo.to_dicts()
    ]
    files.extend(precomputed_rm_files)

    if stimulus:
        stimulus_definition = ResourceDefinition(
            content='TextStimulus',
            filename_pattern=r'toy_text_{text_id:d}_{page_id:d}_aoi.' + raw_fileformat,
            filename_pattern_schema_overrides={'text_id': pl.Int64, 'page_id': pl.Int64},
            load_kwargs={
                'aoi_column': 'char',
                'start_x_column': 'top_left_x',
                'start_y_column': 'top_left_y',
                'width_column': 'width',
                'height_column': 'height',
                'page_column': 'page',
            },
        )
        resource_definitions.append(stimulus_definition)

        stimulus_filenames = {
            'toy_text_1_1_aoi.csv': {'text_id': 1, 'page_id': 1},
            'toy_text_2_5_aoi.csv': {'text_id': 2, 'page_id': 5},
            'toy_text_3_8_aoi.csv': {'text_id': 3, 'page_id': 8},
        }

        source_dirpath = testfiles_dirpath / 'stimuli'
        target_dirpath = rootpath / 'stimuli'
        target_dirpath.mkdir(parents=True, exist_ok=True)

        stimulus_files = []
        for stimulus_filename, stimulus_metadata in stimulus_filenames.items():
            source_filepath = source_dirpath / stimulus_filename
            target_filepath = target_dirpath / stimulus_filename
            shutil.copy2(source_filepath, target_filepath)
            stimulus_file = DatasetFile(
                path=target_filepath,
                definition=stimulus_definition,
                metadata=stimulus_metadata,
            )
            stimulus_files.append(stimulus_file)
        files.extend(stimulus_files)

    dataset_definition = DatasetDefinition(
        experiment=Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=38,
            screen_height_cm=30.2,
            distance_cm=distance_cm,
            origin='upper left',
            sampling_rate=1000,
        ),
        resources=resource_definitions,
        extract=extract,
    )

    return {
        'init_kwargs': {
            'definition': dataset_definition,
            'path': DatasetPaths(root=rootpath, dataset='.'),
        },
        'fileinfo': {
            'gaze': fileinfo,
            'precomputed_events': fileinfo,
            'precomputed_reading_measures': fileinfo,
        },
        'files': files,
        'raw_gazes': gazes,
        'preprocessed_gazes': preprocessed_gazes,
        'events_list': events_list,
        'precomputed_rm_dfs': precomputed_rm_dfs,
        'eyes': eyes,
        'trial_columns': ['task', 'trial'],
    }


@pytest.fixture(
    name='gaze_dataset_configuration',
    params=[
        'ToyMono',
        'ToyBino',
        'ToyLeft',
        'ToyRight',
        'ToyBino+Avg',
        'ToyRemote',
        'ToyAOI',
    ],
)
def gaze_fixture_dataset(request, tmp_path, testfiles_dirpath):
    rootpath = tmp_path

    dataset_type = request.param
    if dataset_type == 'ToyBino':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both')
    elif dataset_type == 'ToyBino+Avg':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both+avg')
    elif dataset_type == 'ToyMono':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='none')
    elif dataset_type == 'ToyLeft':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='left')
    elif dataset_type == 'ToyRight':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='right')
    elif dataset_type == 'ToyMat':
        dataset_dict = mock_toy(rootpath, raw_fileformat='mat', eyes='both')
    elif dataset_type == 'ToyRemote':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both', remote=True)
    elif dataset_type == 'ToyAOI':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='both',
            stimulus=True,
            testfiles_dirpath=testfiles_dirpath,
        )
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_init_with_definition_class():
    @dataclass
    class CustomPublicDataset(DatasetDefinition):
        name: str = 'CustomPublicDataset'

    dataset = Dataset(CustomPublicDataset, path='.')

    assert dataset.definition == CustomPublicDataset()


def test_scan_correct_files_from_dataset_configuration(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.scan()

    expected_files = gaze_dataset_configuration['files']
    assert dataset._files == expected_files


@pytest.mark.parametrize(
    ('dataset_tree', 'resources', 'expected_files'),
    [
        pytest.param(
            [],
            [],
            [],
            id='no_files_no_resources',
        ),
        pytest.param(
            ['test.csv'],
            [],
            [],
            id='single_file_no_resources',
        ),
        pytest.param(
            ['raw/test.csv'],
            [{'content': 'gaze', 'filename_pattern': 'test.csv'}],
            [
                {
                    'path': 'raw/test.csv',
                    'definition': ResourceDefinition(content='gaze', filename_pattern='test.csv'),
                },
            ],
            id='single_file_matches_resource_defintion',
        ),
        pytest.param(
            ['raw/01/test.csv', 'raw/02/test.csv'],
            [{'content': 'gaze', 'filename_pattern': 'test.csv'}],
            [
                {
                    'path': 'raw/01/test.csv',
                    'definition': ResourceDefinition(content='gaze', filename_pattern='test.csv'),
                },
                {
                    'path': 'raw/02/test.csv',
                    'definition': ResourceDefinition(content='gaze', filename_pattern='test.csv'),
                },
            ],
            id='two_files_match_single_resource_defintion',
        ),
        pytest.param(
            ['raw/01.csv', 'raw/02.csv'],
            [{'content': 'gaze', 'filename_pattern': '{participant_id:d}.csv'}],
            [
                {
                    'path': 'raw/01.csv',
                    'definition': ResourceDefinition(
                        content='gaze',
                        filename_pattern='{participant_id:d}.csv',
                    ),
                    'metadata': {'participant_id': '01'},
                },
                {
                    'path': 'raw/02.csv',
                    'definition': ResourceDefinition(
                        content='gaze',
                        filename_pattern='{participant_id:d}.csv',
                    ),
                    'metadata': {'participant_id': '02'},
                },
            ],
            id='two_files_match_single_resource_defintion_with_metadata',
        ),
        pytest.param(
            ['precomputed_events/fixations.csv', 'precomputed_events/saccades.csv'],
            [
                {'content': 'precomputed_events', 'filename_pattern': 'fixations.csv'},
                {'content': 'precomputed_events', 'filename_pattern': 'saccades.csv'},
            ],
            [
                {
                    'path': 'precomputed_events/fixations.csv',
                    'definition': ResourceDefinition(
                        content='precomputed_events',
                        filename_pattern='fixations.csv',
                    ),
                },
                {
                    'path': 'precomputed_events/saccades.csv',
                    'definition': ResourceDefinition(
                        content='precomputed_events',
                        filename_pattern='saccades.csv',
                    ),
                },
            ],
            id='two_files_match_two_resource_defintions',
        ),
        pytest.param(
            ['raw/01.csv', 'precomputed_events/01.csv'],
            [
                {'content': 'gaze', 'filename_pattern': '{participant_id:d}.csv'},
                {'content': 'precomputed_events', 'filename_pattern': '{participant_id:d}.csv'},
            ],
            [
                {
                    'path': 'raw/01.csv',
                    'definition': ResourceDefinition(
                        content='gaze',
                        filename_pattern='{participant_id:d}.csv',
                    ),
                    'metadata': {'participant_id': '01'},
                },
                {
                    'path': 'precomputed_events/01.csv',
                    'definition': ResourceDefinition(
                        content='precomputed_events',
                        filename_pattern='{participant_id:d}.csv',
                    ),
                    'metadata': {'participant_id': '01'},
                },
            ],
            id='two_files_match_two_resource_defintions_with_metadata',
        ),
    ],
)
def test_dataset_scan_correct_files_from_dataset_tree(
    dataset_tree,
    resources,
    expected_files,
    make_dataset,
):
    dirpath = make_dataset(dataset_tree)

    dataset_paths = DatasetPaths(root=dirpath, dataset='.')
    definition = DatasetDefinition(name='test', resources=resources)
    dataset = Dataset(definition=definition, path=dataset_paths)

    dataset.scan()
    scanned_files = dataset._files

    # add dataset dirpath to relative paths in specified expected files
    expected_files = [
        DatasetFile(
            path=dirpath / expected_file['path'],
            definition=expected_file['definition'],
            metadata=expected_file.get('metadata', {}),
        )
        for expected_file in expected_files
    ]

    # sort lists by filepath
    scanned_files = sorted(scanned_files, key=lambda file: file.path)
    expected_files = sorted(expected_files, key=lambda file: file.path)

    assert scanned_files == expected_files


@pytest.mark.filterwarnings('ignore:Stimulus support:pymovements.ExperimentalWarning')
def test_load_correct_fileinfo(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load()

    expected_fileinfo = gaze_dataset_configuration['fileinfo']
    assert_frame_equal(dataset.fileinfo['gaze'], expected_fileinfo['gaze'])


def test_load_correct_raw_gazes(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    expected_gazes = gaze_dataset_configuration['raw_gazes']
    for result_gaze, expected_gaze in zip(dataset.gaze, expected_gazes):
        assert_frame_equal(
            result_gaze.samples,
            expected_gaze.samples,
            check_column_order=False,
        )


def test_load_gaze_has_correct_metadata(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    expected_fileinfo = gaze_dataset_configuration['fileinfo']['gaze'].drop('filepath')
    for gaze, gaze_fileinfo in zip(dataset.gaze, expected_fileinfo.iter_rows(named=True)):
        assert gaze.metadata == gaze_fileinfo


def test_stimuli_list_exists(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])

    assert isinstance(dataset.stimuli, list)


def test_stimuli_not_loaded(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    assert not dataset.stimuli


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyAOI'],
    indirect=['gaze_dataset_configuration'],
)
def test_text_stimuli_list_not_empty(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])

    message = f'.*Stimulus support is experimental.*pymovements version.*{re.escape(__version__)}'
    with pytest.warns(ExperimentalWarning, match=message):
        dataset.load()

    assert dataset.stimuli
    assert all(isinstance(stim, TextStimulus) for stim in dataset.stimuli)


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyAOI'],
    indirect=['gaze_dataset_configuration'],
)
@pytest.mark.parametrize(
    'expected',
    [
        EXPECTED_AOI_MULTIPLEYE_STIMULI_TOY_X_1_TEXT_1_1,
    ],
)
def test_loaded_text_stimuli_list_correct(gaze_dataset_configuration, expected):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.scan()

    message = f'.*Stimulus support is experimental.*pymovements version.*{re.escape(__version__)}'
    with pytest.warns(ExperimentalWarning, match=message):
        dataset.load_stimuli()

    aois_list = dataset.stimuli
    assert len(aois_list) == 3
    head = aois_list[0].aois.head(10)

    assert_frame_equal(
        head,
        expected,
    )
    assert len(aois_list[0].aois.columns) == len(expected.columns)


def test_loaded_gazes_do_not_share_experiment_with_definition(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    definition = gaze_dataset_configuration['init_kwargs']['definition']

    for gaze in dataset.gaze:
        assert gaze.experiment is not definition.experiment


def test_loaded_gazes_do_not_share_experiment_with_other(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    for gaze1 in dataset.gaze:
        for gaze2 in dataset.gaze:
            if gaze1 is gaze2:
                continue

            assert gaze1.experiment is not gaze2.experiment


def test_load_gaze_has_position_columns(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, stimuli=False)

    for result_gaze in dataset.gaze:
        assert 'position' in result_gaze.columns


def test_load_correct_preprocessed_gazes(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, stimuli=False)

    expected_gazes = gaze_dataset_configuration['preprocessed_gazes']
    for result_gaze, expected_gaze in zip(dataset.gaze, expected_gazes):
        assert_frame_equal(
            result_gaze.samples,
            expected_gaze.samples,
            check_column_order=False,
        )


def test_load_correct_trial_columns(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    expected_trial_columns = gaze_dataset_configuration['trial_columns']
    for result_gaze in dataset.gaze:
        assert result_gaze.trial_columns == expected_trial_columns


def test_load_correct_events_list(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(events=True, stimuli=False)

    expected_events_list = gaze_dataset_configuration['events_list']
    for result_events, expected_events in zip(dataset.events, expected_events_list):
        assert_frame_equal(result_events.frame, expected_events)


@pytest.mark.filterwarnings('ignore:Stimulus support:pymovements.ExperimentalWarning')
@pytest.mark.parametrize(
    ('subset', 'fileinfo_idx'),
    [
        pytest.param(
            {'subject_id': 1},
            [0],
            id='subset_int',
        ),
        pytest.param(
            {'subject_id': [1, 11, 12]},
            [0, 2, 3],
            id='subset_list',
        ),
        pytest.param(
            {'subject_id': range(3)},
            [0, 11],
            id='subset_range',
        ),
    ],
)
def test_load_subset(subset, fileinfo_idx, gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(subset=subset)

    expected_fileinfo = gaze_dataset_configuration['fileinfo']
    expected_fileinfo = expected_fileinfo['gaze'][fileinfo_idx]

    assert_frame_equal(dataset.fileinfo['gaze'], expected_fileinfo)


@pytest.mark.parametrize(
    ('init_kwargs', 'load_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'subset': 1},
            TypeError,
            id='subset_no_dict',
        ),
        pytest.param(
            {},
            {'subset': {1: 1}},
            TypeError,
            id='subset_no_str_key',
        ),
        pytest.param(
            {},
            {'subset': {'unknown': 1}},
            ValueError,
            id='subset_key_not_in_fileinfo',
        ),
        pytest.param(
            {},
            {'subset': {'subject_id': None}},
            TypeError,
            id='subset_value_invalid_type',
        ),
    ],
)
def test_load_exceptions(init_kwargs, load_kwargs, exception, gaze_dataset_configuration):
    init_kwargs = {**gaze_dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load(stimuli=False, **load_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'save_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_save_gaze',
        ),
    ],
)
def test_save_gaze_exceptions(init_kwargs, save_kwargs, exception, gaze_dataset_configuration):
    init_kwargs = {**gaze_dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load(stimuli=False)
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.pos2acc()
        dataset.save_preprocessed(**save_kwargs)


@pytest.mark.parametrize(
    ('load_kwargs', 'exception'),
    [
        pytest.param(
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_load_events',
        ),
    ],
)
def test_load_events_exceptions(
    load_kwargs,
    exception,
    gaze_dataset_configuration,
):
    init_kwargs = {**gaze_dataset_configuration['init_kwargs']}
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        dataset.load(stimuli=False)
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.detect_events(
            method=ivt,
            velocity_threshold=45,
            minimum_duration=55,
        )
        dataset.save_events()
        dataset.load_event_files(**load_kwargs)

    (msg,) = excinfo.value.args
    assert (
        msg
        == """\
unsupported file format "invalid". \
Supported formats are: [\'csv\', \'txt\', \'tsv\', \'feather\']"""
    )


@pytest.mark.parametrize(
    ('init_kwargs', 'save_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_events',
        ),
    ],
)
def test_save_events_exceptions(init_kwargs, save_kwargs, exception, gaze_dataset_configuration):
    init_kwargs = {**gaze_dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load(stimuli=False)
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.detect_events(
            method=ivt,
            velocity_threshold=45,
            minimum_duration=55,
        )
        dataset.save_events(**save_kwargs)


def test_load_no_files_raises_exception(gaze_dataset_configuration):
    init_kwargs = {**gaze_dataset_configuration['init_kwargs']}
    dataset = Dataset(**init_kwargs)

    shutil.rmtree(dataset.paths.raw, ignore_errors=True)
    dataset.paths.raw.mkdir()

    with pytest.raises(RuntimeError):
        dataset.scan()


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyMat'],
    indirect=['gaze_dataset_configuration'],
)
def test_load_mat_file_exception(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])

    with pytest.raises(ValueError):
        dataset.load()


def test_pix2deg(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    original_schema = dataset.gaze[0].schema

    dataset.pix2deg()

    expected_schema = {**original_schema, 'position': pl.List(pl.Float64)}
    for result_gaze in dataset.gaze:
        assert result_gaze.schema == expected_schema


def test_deg2pix(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    original_schema = dataset.gaze[0].schema

    dataset.pix2deg()
    dataset.deg2pix(pixel_column='new_pixel')

    expected_schema = {
        **original_schema,
        'position': pl.List(pl.Float64),
        'new_pixel': pl.List(pl.Float64),
    }
    for result_gaze in dataset.gaze:
        assert result_gaze.schema == expected_schema


def test_pos2acc(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.pos2acc()

    expected_schema = {**original_schema, 'acceleration': pl.List(pl.Float64)}
    for result_gaze in dataset.gaze:
        assert result_gaze.schema == expected_schema


def test_pos2vel(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.pos2vel()

    expected_schema = {**original_schema, 'velocity': pl.List(pl.Float64)}
    for result_gaze in dataset.gaze:
        assert result_gaze.schema == expected_schema


def test_clip(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.clip(-1000, 1000, input_column='pixel', output_column='pixel_clipped', n_components=2)

    expected_schema = {**original_schema, 'pixel_clipped': pl.List(pl.Float64)}
    for result_gaze in dataset.gaze:
        assert result_gaze.schema == expected_schema


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    'detect_event_kwargs',
    [
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            id='microsaccades_class',
        ),
        pytest.param(
            {
                'method': 'microsaccades',
                'threshold': 1,
                'eye': 'auto',
            },
            id='microsaccades_string',
        ),
        pytest.param(
            {
                'method': fill,
                'eye': 'auto',
            },
            id='fill_class',
        ),
        pytest.param(
            {
                'method': 'fill',
                'eye': 'auto',
            },
            id='fill_string',
        ),
        pytest.param(
            {
                'method': 'ivt',
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'eye': 'auto',
            },
            id='ivt_string',
        ),
        pytest.param(
            {
                'method': ivt,
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'eye': 'auto',
            },
            id='ivt_class',
        ),
        pytest.param(
            {
                'method': 'idt',
                'eye': 'auto',
            },
            id='idt_string',
        ),
        pytest.param(
            {
                'method': idt,
                'eye': 'auto',
            },
            id='idt_class',
        ),
    ],
)
def test_detect_events_auto_eye(detect_event_kwargs, gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    expected_schema = {
        'task': pl.String,
        'trial': pl.Int64,
        'name': pl.String,
        'onset': pl.Int64,
        'offset': pl.Int64,
        'duration': pl.Int64,
    }
    for result_events in dataset.events:
        assert result_events.schema == expected_schema


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    'detect_event_kwargs',
    [
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'left',
            },
            id='left',
        ),
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'right',
            },
            id='right',
        ),
    ],
)
def test_detect_events_explicit_eye(detect_event_kwargs, gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()

    dataset_eyes = gaze_dataset_configuration['eyes']

    exception = None
    if not dataset_eyes.startswith('both') and detect_event_kwargs['eye'] is not None:
        exception = AttributeError

    if exception is None:
        dataset.detect_events(**detect_event_kwargs)

        expected_schema = {
            'task': pl.String,
            'trial': pl.Int64,
            'name': pl.String,
            'onset': pl.Int64,
            'offset': pl.Int64,
            'duration': pl.Int64,
        }

        for result_events in dataset.events:
            assert result_events.schema == expected_schema

    else:
        with pytest.raises(exception):
            dataset.detect_events(**detect_event_kwargs)


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    ('detect_event_kwargs_1', 'detect_event_kwargs_2', 'expected_schema'),
    [
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'task': pl.String,
                'trial': pl.Int64,
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
            },
            id='two-saccade-runs',
        ),
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'method': ivt,
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            {
                'task': pl.String,
                'trial': pl.Int64,
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
            },
            id='one-saccade-one-fixation-run',
        ),
    ],
)
def test_detect_events_multiple_calls(
    detect_event_kwargs_1,
    detect_event_kwargs_2,
    expected_schema,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs_1)
    dataset.detect_events(**detect_event_kwargs_2)

    for result_events in dataset.events:
        assert result_events.schema == expected_schema


@pytest.mark.parametrize(
    'detect_kwargs',
    [
        pytest.param(
            {
                'method': 'microsaccades',
                'threshold': 1,
                'eye': 'auto',
                'clear': False,
                'verbose': True,
            },
            id='left',
        ),
    ],
)
def test_detect_events_alias(gaze_dataset_configuration, detect_kwargs, monkeypatch):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()

    mock = Mock()
    monkeypatch.setattr(dataset, 'detect', mock)

    dataset.detect(**detect_kwargs)
    mock.assert_called_with(**detect_kwargs)


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyMono'],
    indirect=['gaze_dataset_configuration'],
)
def test_detect_events_attribute_error(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    detect_event_kwargs = {
        'method': microsaccades,
        'threshold': 1,
        'eye': 'right',
    }

    with pytest.raises(AttributeError):
        dataset.detect_events(**detect_event_kwargs)


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyMono'],
    indirect=['gaze_dataset_configuration'],
)
@pytest.mark.parametrize(
    ('rename_arg', 'detect_event_kwargs', 'expected_message'),
    [
        pytest.param(
            {'position': 'custom_position'},
            {
                'method': idt,
                'threshold': 1,
            },
            (
                "Column 'position' not found. Available columns are: "
                "['time', 'task', 'trial', 'pixel', 'custom_position', 'velocity']"
            ),
            id='no_position',
        ),
        pytest.param(
            {'velocity': 'custom_velocity'},
            {
                'method': microsaccades,
                'threshold': 1,
            },
            (
                "Column 'velocity' not found. Available columns are: "
                "['time', 'task', 'trial', 'pixel', 'position', 'custom_velocity']"
            ),
            id='no_velocity',
        ),
    ],
)
def test_detect_events_raises_column_not_found_error(
    gaze_dataset_configuration,
    rename_arg,
    detect_event_kwargs,
    expected_message,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    for file_id, _ in enumerate(dataset.gaze):
        dataset.gaze[file_id].samples = dataset.gaze[file_id].samples.rename(rename_arg)

    with pytest.raises(pl.exceptions.ColumnNotFoundError) as excinfo:
        dataset.detect_events(**detect_event_kwargs)

    (msg,) = excinfo.value.args
    assert msg == expected_message


@pytest.mark.parametrize(
    ('events_init', 'events_expected'),
    [
        pytest.param(
            [],
            [],
            id='empty_list_stays_empty_list',
        ),
        pytest.param(
            [Events()],
            [Events()],
            id='empty_df_stays_empty_df',
        ),
        pytest.param(
            [Events(name='event', onsets=[0], offsets=[99])],
            [Events()],
            id='single_instance_filled_df_gets_cleared_to_empty_df',
        ),
        pytest.param(
            [
                Events(name='event', onsets=[0], offsets=[99]),
                Events(name='event', onsets=[0], offsets=[99]),
            ],
            [Events(), Events()],
            id='two_instance_filled_df_gets_cleared_to_two_empty_dfs',
        ),
    ],
)
def test_clear_events(events_init, events_expected, tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)

    num_gazes = len(events_init)

    # add dummy gazes so events and gazes stay in sync
    for _ in range(num_gazes):
        dummy_gaze = Gaze(
            pl.DataFrame(
                {
                    'time': [],
                    'pixel_x': [],
                    'pixel_y': [],
                }
            ),
            pixel_columns=['pixel_x', 'pixel_y'],
            time_column='time',
            time_unit='ms',
        )
        dataset.gaze.append(dummy_gaze)

    dataset.events = events_init
    dataset.clear_events()

    for events_df_result, events_df_expected in zip(dataset.events, events_expected):
        assert_frame_equal(events_df_result.frame, events_df_expected.frame)


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    ('detect_event_kwargs', 'events_dirname', 'expected_save_dirpath', 'save_kwargs'),
    [
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            'events_test',
            'events_test',
            {},
            id='explicit_dirname',
        ),
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {'extension': 'csv'},
            id='save_events_extension_csv',
        ),
    ],
)
def test_save_events(
    detect_event_kwargs,
    events_dirname,
    expected_save_dirpath,
    save_kwargs,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    if events_dirname is None:
        events_dirname = 'events'
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_events(events_dirname, **save_kwargs)

    assert (dataset.path / expected_save_dirpath).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_dirpath)}'
    )


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    ('detect_event_kwargs', 'events_dirname', 'expected_save_dirpath', 'load_save_kwargs'),
    [
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            'events_test',
            'events_test',
            {},
            id='explicit_dirname',
        ),
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {'extension': 'csv'},
            id='load_events_extension_csv',
        ),
    ],
)
def test_load_previously_saved_events_gaze(
    detect_event_kwargs,
    events_dirname,
    expected_save_dirpath,
    load_save_kwargs,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.pos2acc()
    dataset.detect_events(**detect_event_kwargs)

    # We must not overwrite the original variable as it's needed in the end.
    if events_dirname is None:
        events_dirname_ = 'events'
    else:
        events_dirname_ = events_dirname

    shutil.rmtree(dataset.path / Path(events_dirname_), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_events(events_dirname, **load_save_kwargs)
    dataset.save_preprocessed(**load_save_kwargs)

    dataset.clear_events()

    dataset.load(
        events=True,
        preprocessed=True,
        stimuli=False,
        events_dirname=events_dirname,
        **load_save_kwargs,
    )
    assert dataset.events


@pytest.mark.parametrize(
    ('preprocessed_dirname', 'expected_save_dirpath'),
    [
        pytest.param(
            None,
            'preprocessed',
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed_test',
            'preprocessed_test',
            id='explicit_dirname',
        ),
    ],
)
def test_save_preprocessed_directory_exists(
    preprocessed_dirname,
    expected_save_dirpath,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.pos2acc()

    if preprocessed_dirname is None:
        preprocessed_dirname = 'preprocessed'
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_preprocessed(preprocessed_dirname)

    assert (dataset.path / expected_save_dirpath).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_dirpath)}'
    )


@pytest.mark.parametrize(
    'drop_column',
    [
        'time',
        'pixel',
        'position',
        'velocity',
        'acceleration',
    ],
)
def test_save_preprocessed(gaze_dataset_configuration, drop_column):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.pos2acc()

    dataset.gaze[0].samples = dataset.gaze[0].samples.drop(drop_column)

    preprocessed_dirname = 'preprocessed-test'
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    dataset.save_preprocessed(preprocessed_dirname, extension='csv')
    dataset.load_gaze_files(True, preprocessed_dirname, extension='csv')

    assert (dataset.path / preprocessed_dirname).is_dir(), (
        f'data was not written to {dataset.path / Path(preprocessed_dirname)}'
    )


@pytest.mark.parametrize(
    'drop_column',
    [
        'time',
        'pixel',
        'position',
        'velocity',
        'acceleration',
    ],
)
def test_save_preprocessed_has_no_side_effect(gaze_dataset_configuration, drop_column):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.pos2acc()

    dataset.gaze[0].samples = dataset.gaze[0].samples.drop(drop_column)

    old_frame = dataset.gaze[0].samples.clone()

    preprocessed_dirname = 'preprocessed-test'
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    dataset.save_preprocessed(preprocessed_dirname, extension='csv')

    new_frame = dataset.gaze[0].samples.clone()

    assert_frame_equal(old_frame, new_frame)


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    ('expected_save_preprocessed_path', 'expected_save_events_path', 'save_kwargs'),
    [
        pytest.param(
            'preprocessed',
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'verbose': 2},
            id='verbose_equals_2',
        ),
        pytest.param(
            'preprocessed_test',
            'events',
            {'preprocessed_dirname': 'preprocessed_test'},
            id='explicit_prepocessed_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events_test',
            {'events_dirname': 'events_test'},
            id='explicit_events_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'csv'},
            id='extension_equals_csv',
        ),
    ],
)
def test_save_creates_correct_directory(
    expected_save_preprocessed_path,
    expected_save_events_path,
    save_kwargs,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()

    detect_events_kwargs = {'method': microsaccades, 'threshold': 1, 'eye': 'auto'}
    dataset.detect_events(**detect_events_kwargs)

    preprocessed_dirname = save_kwargs.get('preprocessed_dirname', 'preprocessed')
    events_dirname = save_kwargs.get('events_dirname', 'events')

    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_preprocessed_path), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_events_path), ignore_errors=True)
    dataset.save(**save_kwargs)

    assert (dataset.path / Path(expected_save_preprocessed_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_preprocessed_path)}'
    )
    assert (dataset.path / Path(expected_save_events_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_events_path)}'
    )


@pytest.mark.filterwarnings('ignore:.*No events were detected.*:UserWarning')
@pytest.mark.parametrize(
    ('expected_save_preprocessed_path', 'expected_save_events_path', 'save_kwargs'),
    [
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'feather'},
            id='extension_equals_feather',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'csv'},
            id='extension_equals_csv',
        ),
    ],
)
def test_save_files_have_correct_extension(
    expected_save_preprocessed_path,
    expected_save_events_path,
    save_kwargs,
    gaze_dataset_configuration,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.pos2acc()

    detect_events_kwargs = {'method': microsaccades, 'threshold': 1, 'eye': 'auto'}
    dataset.detect_events(**detect_events_kwargs)

    preprocessed_dirname = save_kwargs.get('preprocessed_dirname', 'preprocessed')
    events_dirname = save_kwargs.get('events_dirname', 'events')
    extension = save_kwargs.get('extension', 'feather')

    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_preprocessed_path), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_events_path), ignore_errors=True)
    dataset.save(**save_kwargs)

    preprocessed_dir = dataset.path / Path(expected_save_preprocessed_path)
    preprocessed_file_list = os.listdir(preprocessed_dir)
    extension_list = [a.endswith(extension) for a in preprocessed_file_list]
    extension_sum = sum(extension_list)
    assert extension_sum == len(preprocessed_file_list), (
        f'not all preprocessed files created have correct extension {extension}'
    )

    events_dir = dataset.path / Path(expected_save_events_path)
    events_file_list = os.listdir(events_dir)
    extension_list = [a.endswith(extension) for a in events_file_list]
    extension_sum = sum(extension_list)
    assert extension_sum == len(events_file_list), (
        f'not all events files created have correct extension {extension}'
    )


@pytest.mark.parametrize(
    ('init_path', 'expected_paths'),
    [
        pytest.param(
            '/data/set/path',
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='dataset', events='custom_events'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/custom_events'),
            },
        ),
        pytest.param(
            DatasetPaths(
                root='/data/set/path',
                dataset='dataset',
                preprocessed='custom_preprocessed',
            ),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/custom_preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='dataset', raw='custom_raw'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/custom_raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
    ],
)
def test_paths(init_path, expected_paths):
    dataset = Dataset('ToyDataset', path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.path == expected_paths['dataset']
    assert dataset.paths.raw == expected_paths['raw']
    assert dataset.paths.preprocessed == expected_paths['preprocessed']
    assert dataset.paths.events == expected_paths['events']


@pytest.mark.parametrize(
    ('new_fileinfo', 'exception'),
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_fileinfo(new_fileinfo, exception, tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)

    dataset.fileinfo = new_fileinfo

    with pytest.raises(exception):
        dataset._check_fileinfo()


@pytest.mark.parametrize(
    ('new_gaze', 'exception'),
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_gaze(new_gaze, exception, tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)

    dataset.gaze = new_gaze

    with pytest.raises(exception):
        dataset._check_gaze()


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyBino'],
    indirect=['gaze_dataset_configuration'],
)
def test_check_experiment(gaze_dataset_configuration):
    gaze_dataset_configuration['init_kwargs']['definition'].experiment = None
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load()

    with pytest.raises(AttributeError):
        dataset.gaze[0]._check_experiment()


@pytest.mark.parametrize(
    'gaze_dataset_configuration',
    ['ToyBino'],
    indirect=['gaze_dataset_configuration'],
)
def test_velocity_columns(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True)

    for gaze in dataset.gaze:
        assert 'velocity' in gaze.columns


def test_dataset_compute_event_properties_warns_existing(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    measure = ('null_ratio', {'column': 'pixel', 'column_dtype': pl.List})
    dataset.compute_event_properties(measure)

    message = 'The following columns already exist in event and will be overwritten.*null_ratio.*'
    with pytest.warns(UserWarning, match=message):
        dataset.compute_event_properties(measure)


@pytest.mark.parametrize(
    ('property_kwargs', 'exception', 'message'),
    [
        pytest.param(
            {'event_properties': 'foo'},
            UnknownMeasure,
            "Measure 'foo' is unknown. Known measures are",
            id='unknown_measure',
        ),
    ],
)
def test_event_dataframe_add_property_raises_exceptions(
    gaze_dataset_configuration,
    property_kwargs,
    exception,
    message,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    with pytest.raises(exception, match=message):
        dataset.compute_event_properties(**property_kwargs)


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param({'event_properties': 'peak_velocity'}, id='peak_velocity'),
    ],
)
def test_event_dataframe_add_property_has_expected_height(
    gaze_dataset_configuration,
    property_kwargs,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    expected_heights = [len(events) for events in dataset.events]

    dataset.compute_event_properties(**property_kwargs)

    for events_df, expected_height in zip(dataset.events, expected_heights):
        assert events_df.frame.height == expected_height


@pytest.mark.parametrize(
    ('property_kwargs', 'expected_schema'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            {
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
                'peak_velocity': pl.Float64,
            },
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            {
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
                'location': pl.List(pl.Float64),
            },
            id='single_event_position',
        ),
    ],
)
def test_event_dataframe_add_property_has_expected_schema(
    gaze_dataset_configuration,
    property_kwargs,
    expected_schema,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    dataset.compute_event_properties(**property_kwargs)

    for events_df in dataset.events:
        assert events_df.frame.schema == expected_schema


@pytest.mark.parametrize(
    ('property_kwargs', 'expected_property_columns'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            ['peak_velocity'],
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            ['location'],
            id='single_event_location',
        ),
        pytest.param(
            {'event_properties': 'location', 'name': 'fixation'},
            ['location'],
            id='single_event_location_name_fixation',
        ),
        pytest.param(
            {'event_properties': 'peak_velocity', 'name': 'sacc.*'},
            ['peak_velocity'],
            id='single_event_peak_velocity_regex_name_sacc',
        ),
    ],
)
def test_event_dataframe_add_property_effect_property_columns(
    gaze_dataset_configuration,
    property_kwargs,
    expected_property_columns,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    dataset.compute_event_properties(**property_kwargs)

    for events_df in dataset.events:
        assert events_df.event_property_columns == expected_property_columns


@pytest.mark.parametrize(
    ('property_kwargs', 'warning', 'message'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity', 'name': 'taccade'},
            UserWarning,
            "No events found with name 'taccade'.",
            marks=pytest.mark.filterwarnings(
                'ignore:No events available for processing.*:UserWarning',
            ),
            id='name_missing',
        ),
    ],
)
def test_dataset_compute_event_properties_warns(
    gaze_dataset_configuration,
    property_kwargs,
    warning,
    message,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    with pytest.warns(warning, match=message):
        dataset.compute_event_properties(**property_kwargs)


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            id='single_event_position',
        ),
        pytest.param(
            {'event_properties': 'location', 'name': 'fixation'},
            id='single_event_position_name_fixation',
        ),
    ],
)
def test_event_dataframe_add_property_does_not_change_length(
    gaze_dataset_configuration,
    property_kwargs,
):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    lengths_pre = [len(events_df.frame) for events_df in dataset.events]
    dataset.compute_event_properties(**property_kwargs)
    lengths_post = [len(events_df.frame) for events_df in dataset.events]

    assert lengths_pre == lengths_post


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param(
            {
                'event_properties': 'peak_velocity',
                'name': None,
                'verbose': True,
            },
            id='peak_velocity',
        ),
    ],
)
def test_compute_event_properties_alias(gaze_dataset_configuration, property_kwargs, monkeypatch):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True, stimuli=False)

    mock = Mock()
    monkeypatch.setattr(dataset, 'compute_event_properties', mock)

    dataset.compute_properties(**property_kwargs)
    mock.assert_called_with(**property_kwargs)


@pytest.fixture(
    name='precomputed_dataset_configuration',
    params=[
        'ToyRightPrecomputedEventAndGaze',
        'ToyPrecomputedEvent',
        'ToyPrecomputedEventNoExtract',
        'ToyPrecomputedEventLoadKwargs',
    ],
)
def precomputed_fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    dataset_type = request.param
    if dataset_type == 'ToyRightPrecomputedEventAndGaze':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
        )
    elif dataset_type == 'ToyPrecomputedEvent':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
        )
        del dataset_dict['init_kwargs']['definition'].resources[0]  # remove gaze resources
    elif dataset_type == 'ToyPrecomputedEventNoExtract':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
            filename_format_schema_overrides={'precomputed_events': {}},
        )
    elif dataset_type == 'ToyPrecomputedEventLoadKwargs':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
        )
        new_resource = replace(
            dataset_dict['init_kwargs']['definition'].resources[1],
            load_kwargs={'separator': ','},
        )
        dataset_dict['init_kwargs']['definition'].resources[1] = new_resource
        dataset_dict['init_kwargs']['definition'].custom_read_kwargs = None
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_load_correct_fileinfo_precomputed(precomputed_dataset_configuration):
    dataset = Dataset(**precomputed_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    expected_fileinfo = precomputed_dataset_configuration['fileinfo']['precomputed_events']
    assert_frame_equal(dataset.fileinfo['precomputed_events'], expected_fileinfo)


def test_load_no_files_precomputed_raises_exception(precomputed_dataset_configuration):
    init_kwargs = {**precomputed_dataset_configuration['init_kwargs']}
    dataset = Dataset(**init_kwargs)

    shutil.rmtree(dataset.paths.precomputed_events, ignore_errors=True)
    dataset.paths.precomputed_events.mkdir()

    with pytest.raises(RuntimeError):
        dataset.scan()


@pytest.fixture(
    name='precomputed_rm_dataset_configuration',
    params=[
        'ToyRightPrecomputedEventAndGazeAndRM',
        'ToyPrecomputedRM',
        'ToyPrecomputedRMNoExtract',
        'ToyPrecomputedRMLoadKwargs',
    ],
)
def precomputed_rm_fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    dataset_type = request.param
    if dataset_type == 'ToyRightPrecomputedEventAndGazeAndRM' or dataset_type == 'ToyPrecomputedRM':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
        )
    elif dataset_type == 'ToyPrecomputedRMNoExtract':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
            filename_format_schema_overrides={
                'precomputed_events': {},
                'precomputed_reading_measures': {},
            },
        )
    elif dataset_type == 'ToyPrecomputedRMLoadKwargs':
        dataset_dict = mock_toy(
            rootpath,
            raw_fileformat='csv',
            eyes='right',
        )
        new_resource = replace(
            dataset_dict['init_kwargs']['definition'].resources[2],
            load_kwargs={'separator': ','},
        )
        dataset_dict['init_kwargs']['definition'].resources[2] = new_resource
        dataset_dict['init_kwargs']['definition'].custom_read_kwargs = None
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_load_correct_fileinfo_precomputed_rm(precomputed_rm_dataset_configuration):
    dataset = Dataset(**precomputed_rm_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)

    all_fileinfo = precomputed_rm_dataset_configuration['fileinfo']
    expected_fileinfo = all_fileinfo['precomputed_reading_measures']
    assert_frame_equal(dataset.fileinfo['precomputed_reading_measures'], expected_fileinfo)


def test_load_no_files_precomputed_rm_raises_exception(precomputed_rm_dataset_configuration):
    init_kwargs = {**precomputed_rm_dataset_configuration['init_kwargs']}
    dataset = Dataset(**init_kwargs)

    shutil.rmtree(dataset.paths.precomputed_reading_measures, ignore_errors=True)
    dataset.paths.precomputed_reading_measures.mkdir()

    with pytest.raises(RuntimeError):
        dataset.scan()


@pytest.mark.parametrize(
    ('by', 'expected_len'),
    [
        pytest.param(
            'task',
            60,
            id='subset_int',
        ),
        pytest.param(
            'trial',
            40,
            id='subset_int',
        ),
        pytest.param(
            ['task', 'trial'],
            80,
            id='subset_int',
        ),
    ],
)
def test_load_split_precomputed_events(precomputed_dataset_configuration, by, expected_len):
    dataset = Dataset(**precomputed_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.split_precomputed_events(by)
    assert len(dataset.precomputed_events) == expected_len


def test_dataset_definition_from_yaml(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'

    dataset_def = DatasetLibrary.get('ToyDataset')
    dataset_def.to_yaml(tmp_file)

    dataset_from_yaml = Dataset(tmp_file, '.')
    assert dataset_from_yaml.definition == dataset_def


@pytest.mark.parametrize(
    ('by', 'expected_len'),
    [
        pytest.param(
            'task',
            60,
            id='subset_int',
        ),
        pytest.param(
            'trial',
            40,
            id='subset_int',
        ),
        pytest.param(
            ['task', 'trial'],
            80,
            id='subset_int',
        ),
    ],
)
def test_load_split_gaze(gaze_dataset_configuration, by, expected_len):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.split_gaze_data(by)
    assert len(dataset.gaze) == expected_len


def test_two_resources_same_content_different_filename_pattern(tmp_path):
    dirpath = tmp_path / 'precomputed_events'
    dirpath.mkdir()

    # create empty files
    with open(dirpath / 'foo.csv', 'a', encoding='ascii') as f:
        f.close()
    with open(dirpath / 'bar.csv', 'a', encoding='ascii') as f:
        f.close()

    definition = DatasetDefinition(
        name='example',
        resources=[
            {'content': 'precomputed_events', 'filename_pattern': 'foo.csv'},
            {'content': 'precomputed_events', 'filename_pattern': 'bar.csv'},
        ],
    )

    dataset = Dataset(definition=definition, path=tmp_path)

    dataset.scan()

    assert dataset.fileinfo['precomputed_events']['filepath'].to_list() == ['foo.csv', 'bar.csv']


def test_unsupported_content_type(tmp_path):
    definition = DatasetDefinition(
        name='example',
        resources=[{'content': 'foobar'}],
    )
    dataset = Dataset(definition=definition, path=tmp_path)

    expected_msg = 'content type foobar is not supported'
    with pytest.warns(UserWarning, match=expected_msg):
        dataset.scan()


def test_drop_event_property(gaze_dataset_configuration):
    dataset = Dataset(**gaze_dataset_configuration['init_kwargs'])
    dataset.load(stimuli=False)
    dataset.pix2deg()
    dataset.detect_events('idt', dispersion_threshold=2.7, name='fixation.idt')
    dataset.pos2vel()
    dataset.compute_event_properties('peak_velocity')

    with pytest.raises(ValueError) as exinfo:
        dataset.drop_event_properties('alamakota')
    assert str(exinfo.value).startswith("The column 'alamakota' does not exist")

    # Nothing should be changed
    with pytest.raises(ValueError) as exinfo:
        dataset.drop_event_properties(['peak_velocity', 'alamakota'])
    assert 'peak_velocity' in dataset.gaze[0].events.columns

    # peak_velocity should be changed
    dd = dataset.drop_event_properties('peak_velocity')
    assert 'peak_velocity' not in dataset.gaze[0].events.columns
    assert isinstance(dd, Dataset)

    # Now error should be raised because peak_velocity does not exist
    with pytest.raises(ValueError) as exinfo:
        dataset.drop_event_properties('peak_velocity')
    assert str(exinfo.value).startswith("The column 'peak_velocity' does not exist")

    # onset should not be removed
    with pytest.raises(ValueError) as exinfo:
        dataset.drop_event_properties('onset')
    assert str(exinfo.value).startswith("The column 'onset' cannot be removed")
    assert 'onset' in dataset.gaze[0].events.columns


def test_events_setter_raises_on_length_mismatch(tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)
    # Add one gaze
    dataset.gaze.append(
        Gaze(
            pl.DataFrame({'time': [], 'pixel_x': [], 'pixel_y': []}),
            pixel_columns=['pixel_x', 'pixel_y'],
            time_column='time',
            time_unit='ms',
        ),
    )
    # Try to assign two events for one gaze
    with pytest.raises(ValueError, match='Number of events'):
        dataset.events = [Events(), Events()]


@pytest.mark.parametrize('n_gazes', [1, 3])
def test_events_getter_reflects_gazes(tmp_path, n_gazes):
    dataset = Dataset('ToyDataset', path=tmp_path)

    for _ in range(n_gazes):
        dataset.gaze.append(
            Gaze(
                pl.DataFrame({'time': [], 'pixel_x': [], 'pixel_y': []}),
                pixel_columns=['pixel_x', 'pixel_y'],
                time_column='time',
                time_unit='ms',
            ),
        )

    assert len(dataset.events) == n_gazes
    for i, ev in enumerate(dataset.events):
        assert isinstance(ev, Events)
        assert ev.frame.is_empty()
        assert ev is dataset.gaze[i].events


def test_events_setter_updates_gaze_events(tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)

    for _ in range(3):
        dataset.gaze.append(
            Gaze(
                pl.DataFrame({'time': [], 'pixel_x': [], 'pixel_y': []}),
                pixel_columns=['pixel_x', 'pixel_y'],
                time_column='time',
                time_unit='ms',
            ),
        )

    # create 3 Events objects
    ev_list = [Events(), Events(), Events()]
    dataset.events = ev_list
    for i, ev in enumerate(dataset.events):
        assert ev is ev_list[i]
        assert ev is dataset.gaze[i].events


def test_events_setter_identity_preserved(tmp_path):
    dataset = Dataset('ToyDataset', path=tmp_path)
    for _ in range(3):
        dataset.gaze.append(
            Gaze(
                pl.DataFrame({'time': [], 'pixel_x': [], 'pixel_y': []}),
                pixel_columns=['pixel_x', 'pixel_y'],
                time_column='time',
                time_unit='ms',
            ),
        )

    ev = Events(pl.DataFrame({'onset': [2], 'offset': [3], 'name': ['saccade']}))
    dataset.gaze[2].events = ev
    assert dataset.gaze[2].events is ev
    assert dataset.events[2] is ev
    assert dataset.gaze[0].events is not ev
    assert dataset.events[0] is not ev
    assert dataset.gaze[1].events is not ev
    assert dataset.events[1] is not ev
