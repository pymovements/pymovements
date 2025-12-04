# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test basic preprocessing on various datasets."""
import shutil
from pathlib import Path

import polars as pl
import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import DatasetPaths
from pymovements import Experiment
from pymovements import ResourceDefinitions


@pytest.fixture(name='make_example_dataset_files', scope='function')
def fixture_make_example_dataset_files(tmp_path, testfiles_dirpath):
    def _make_example_dataset_files(
            example_filename: str,
            filename_pattern: str,
            fill_filename_df: pl.DataFrame,
    ) -> Path:
        # Make samples subdirectory
        samples_dirpath = tmp_path / 'raw'
        samples_dirpath.mkdir()

        for filename_pattern_values in fill_filename_df.to_dicts():
            # Create target filepath by filling filename pattern with passed data.
            target_filename = filename_pattern.format(**filename_pattern_values)
            target_filepath = samples_dirpath / target_filename
            # Get filepath of source example file and copy file to target filepath.
            source_filepath = testfiles_dirpath / example_filename
            shutil.copy2(source_filepath, target_filepath)
        return tmp_path
    return _make_example_dataset_files


@pytest.fixture(
    name='dataset',
    params=[
        'csv_monocular',
        'csv_binocular',
        'ipc_monocular',
        'ipc_binocular',
        'emtec',
        'didec',
        'hbn',
        'sbsat',
        'gaze_on_faces',
        'gazebase',
        'gazebase_vr',
        'gazegraph',
        'judo1000',
        'potec',
        'potsdam_binge_remote_pvt',
        'potsdam_binge_wearable_pvt',
    ],
)
def fixture_dataset_init_kwargs(request, make_example_dataset_files):
    if request.param == 'csv_monocular':
        definition = DatasetDefinition(
            resources=[{
                'content': 'gaze',
                'filename_pattern': '{subject_id:d}.csv',
                'load_kwargs': {
                    'time_column': 'time',
                    'time_unit': 'ms',
                    'pixel_columns': ['x_left_pix', 'y_left_pix'],
                },
            }],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
        example_filename = 'monocular_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    elif request.param == 'csv_binocular':
        definition = DatasetDefinition(
            resources=[{
                'content': 'gaze',
                'filename_pattern': '{subject_id:d}.csv',
                'load_kwargs': {
                    'time_column': 'time',
                    'time_unit': 'ms',
                    'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                    'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
                },
            }],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
        example_filename = 'binocular_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    elif request.param == 'ipc_monocular':
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': '{subject_id:d}.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
        example_filename = 'monocular_example.feather'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    elif request.param == 'ipc_binocular':
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': '{subject_id:d}.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
        example_filename = 'binocular_example.feather'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    elif request.param == 'didec':
        definition = DatasetLibrary.get('DIDEC')
        example_filename = 'didec_example.txt'
        filename_config = pl.from_dict({
            'experiment': [1, 2],
            'list': [1, 2],
            'version': [1, 2],
            'participant': [1, 2],
            'session': [1, 2],
            'trial': [1, 2],
        })
    elif request.param == 'emtec':
        definition = DatasetLibrary.get('EMTeC')
        example_filename = 'emtec_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    elif request.param == 'hbn':
        definition = DatasetLibrary.get('HBN')
        example_filename = 'hbn_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3], 'video_id': [1, 2, 3]})
    elif request.param == 'gaze_on_faces':
        definition = DatasetLibrary.get('GazeOnFaces')
        example_filename = 'gaze_on_faces_example.csv'
        filename_config = pl.from_dict({'sub_id': [1, 2, 3], 'trial_id': [1, 2, 3]})
    elif request.param == 'gazebase':
        definition = DatasetLibrary.get('GazeBase')
        example_filename = 'gazebase_example.csv'
        filename_config = pl.from_dict({
            'round_id': [1, 2],
            'subject_id': [1, 2],
            'session_id': [1, 2],
            'task_name': [1, 2],
        })
    elif request.param == 'gazebase_vr':
        definition = DatasetLibrary.get('GazeBaseVR')
        example_filename = 'gazebase_vr_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
        filename_config = pl.from_dict({
            'round_id': [1, 2],
            'subject_id': [1, 2],
            'session_id': [1, 2],
            'task_name': [1, 2],
        })
    elif request.param == 'gazegraph':
        definition = DatasetLibrary.get('GazeGraph')
        example_filename = 'gazegraph_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
        filename_config = pl.from_dict({'subject_id': [1, 2], 'task': [1, 2]})
    elif request.param == 'judo1000':
        definition = DatasetLibrary.get('JuDo1000')
        example_filename = 'judo1000_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2], 'session_id': [1, 2]})
    elif request.param == 'potec':
        definition = DatasetLibrary.get('PoTeC')
        example_filename = 'potec_example.tsv'
        filename_config = pl.from_dict({'subject_id': [1, 2], 'text_id': [1, 2]})
    elif request.param == 'potsdam_binge_remote_pvt':
        definition = DatasetLibrary.get('PotsdamBingeRemotePVT')
        definition.resources = ResourceDefinitions([definition.resources[0]])
        example_filename = 'potsdam_binge_pvt_example.csv'
        filename_config = pl.from_dict({
            'subject_id': [1, 2],
            'session_id': [1, 2],
            'condition': ['A', 'B'],
            'trial_id': [1, 2],
            'block_id': [1, 2],
        })
    elif request.param == 'potsdam_binge_wearable_pvt':
        definition = DatasetLibrary.get('PotsdamBingeWearablePVT')
        definition.resources = ResourceDefinitions([definition.resources[1]])
        example_filename = 'potsdam_binge_pvt_example.csv'
        filename_config = pl.from_dict({
            'subject_id': [1, 2],
            'session_id': [1, 2],
            'condition': ['A', 'B'],
            'trial_id': [1, 2],
            'block_id': [1, 2],
        })
    elif request.param == 'sbsat':
        definition = DatasetLibrary.get('SBSAT')
        example_filename = 'sbsat_example.csv'
        filename_config = pl.from_dict({'subject_id': [1, 2, 3, 4, 5]})
    else:
        raise ValueError('unknown request.param {request.param}')

    dataset_path = make_example_dataset_files(
        example_filename,
        definition.resources.filter('gaze')[0].filename_pattern,
        filename_config,
    )

    # Only use gaze sample data in this functional test.
    definition.resources = definition.resources.filter('gaze')

    yield Dataset(
        definition=definition,
        path=DatasetPaths(root=dataset_path, dataset='.'),
    )


def test_dataset_save_load_preprocessed(dataset):
    dataset.scan()
    dataset.load()

    if 'pixel' in dataset.gaze[0].samples.columns:
        dataset.pix2deg()

    dataset.pos2vel()
    dataset.resample(resampling_rate=2000)
    dataset.save()
    dataset.load(preprocessed=True)
