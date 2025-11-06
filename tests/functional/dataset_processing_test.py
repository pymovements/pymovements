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

import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetPaths
from pymovements import datasets
from pymovements import Experiment
from pymovements import ResourceDefinitions


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
def fixture_dataset_init_kwargs(request, make_example_file):
    if request.param == 'csv_monocular':
        filepath = make_example_file('monocular_example.csv')
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'monocular_example.csv'}],
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix'],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
    elif request.param == 'csv_binocular':
        filepath = make_example_file('binocular_example.csv')
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'binocular_example.csv'}],
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            position_columns=['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
    elif request.param == 'ipc_monocular':
        filepath = make_example_file('monocular_example.feather')
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'monocular_example.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
    elif request.param == 'ipc_binocular':
        filepath = make_example_file('binocular_example.feather')
        definition = DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'binocular_example.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        )
    elif request.param == 'didec':
        filepath = make_example_file('didec_example.txt')
        definition = datasets.DIDEC()
        definition.resources[0].filename_pattern = 'didec_example.txt'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'emtec':
        filepath = make_example_file('emtec_example.csv')
        definition = datasets.EMTeC(trial_columns=None)
        definition.resources[0].filename_pattern = 'emtec_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'hbn':
        filepath = make_example_file('hbn_example.csv')
        definition = datasets.HBN()
        definition.resources[0].filename_pattern = 'hbn_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'sbsat':
        filepath = make_example_file('sbsat_example.csv')
        definition = datasets.SBSAT()
        definition.resources[0].filename_pattern = 'sbsat_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'gaze_on_faces':
        filepath = make_example_file('gaze_on_faces_example.csv')
        definition = datasets.GazeOnFaces()
        definition.resources[0].filename_pattern = 'gaze_on_faces_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'gazebase':
        filepath = make_example_file('gazebase_example.csv')
        definition = datasets.GazeBase()
        definition.resources[0].filename_pattern = 'gazebase_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'gazebase_vr':
        filepath = make_example_file('gazebase_vr_example.csv')
        definition = datasets.GazeBaseVR()
        definition.resources[0].filename_pattern = 'gazebase_vr_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'gazegraph':
        filepath = make_example_file('gazegraph_example.csv')
        definition = datasets.GazeGraph()
        definition.resources[0].filename_pattern = 'gazegraph_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'judo1000':
        filepath = make_example_file('judo1000_example.csv')
        definition = datasets.JuDo1000()
        definition.resources[0].filename_pattern = 'judo1000_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'potec':
        filepath = make_example_file('potec_example.tsv')
        definition = datasets.PoTeC()
        definition.resources[0].filename_pattern = 'potec_example.tsv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'potsdam_binge_remote_pvt':
        filepath = make_example_file('potsdam_binge_pvt_example.csv')
        definition = datasets.PotsdamBingeRemotePVT()
        definition.resources[0].filename_pattern = 'potsdam_binge_pvt_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    elif request.param == 'potsdam_binge_wearable_pvt':
        filepath = make_example_file('potsdam_binge_pvt_example.csv')
        definition = datasets.PotsdamBingeWearablePVT()
        definition.resources[0].filename_pattern = 'potsdam_binge_pvt_example.csv'
        definition.resources[0].filename_pattern_schema_overrides = None
    else:
        raise NotImplementedError(request.param)

    # make directory for raw data and move example file there
    dataset_path = filepath.parent
    raw_dirpath = dataset_path / 'raw'
    raw_dirpath.mkdir()
    filepath = shutil.move(filepath, raw_dirpath)

    # currently this tests only for a single gaze resource.
    definition.resources = ResourceDefinitions([definition.resources.filter('gaze')[0]])

    yield Dataset(
        definition=definition,
        path=DatasetPaths(
            root=dataset_path,
            dataset='.',
        ),
    )


def test_dataset_save_load_preprocessed(dataset):
    dataset.load()

    if 'pixel' in dataset.gaze[0].samples.columns:
        dataset.pix2deg()

    dataset.pos2vel()
    dataset.resample(resampling_rate=2000)
    dataset.save()
    dataset.load(preprocessed=True)
