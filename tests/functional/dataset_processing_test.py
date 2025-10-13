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
def fixture_dataset_init_kwargs(request):
    init_param_dict = {
        'csv_monocular': DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'monocular_example.csv'}],
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix'],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            custom_read_kwargs={'gaze': {}},
        ),
        'csv_binocular': DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'binocular_example.csv'}],
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            position_columns=['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            custom_read_kwargs={'gaze': {}},
        ),
        'ipc_monocular': DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'monocular_example.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            custom_read_kwargs={'gaze': {}},
        ),
        'ipc_binocular': DatasetDefinition(
            resources=[{'content': 'gaze', 'filename_pattern': 'binocular_example.feather'}],
            experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            custom_read_kwargs={'gaze': {}},
        ),
        'didec': datasets.DIDEC(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'didec_example.txt',
            }]),
            trial_columns=None,
        ),
        'emtec': datasets.EMTeC(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'emtec_example.csv',
            }]),
            trial_columns=None,
        ),
        'hbn': datasets.HBN(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'hbn_example.csv',
            }]),
            trial_columns=None,
        ),
        'sbsat': datasets.SBSAT(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'sbsat_example.csv',
            }]),
            trial_columns=None,
        ),
        'gaze_on_faces': datasets.GazeOnFaces(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'gaze_on_faces_example.csv',
            }]),
            trial_columns=None,
        ),
        'gazebase': datasets.GazeBase(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'gazebase_example.csv',
            }]),
            trial_columns=None,
        ),
        'gazebase_vr': datasets.GazeBaseVR(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'gazebase_vr_example.csv',
            }]),
            trial_columns=None,
        ),
        'gazegraph': datasets.GazeGraph(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'gazegraph_example.csv',
            }]),
            trial_columns=None,
        ),
        'judo1000': datasets.JuDo1000(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'judo1000_example.csv',
            }]),
            trial_columns=['trial_id'],
        ),
        'potec': datasets.PoTeC(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'potec_example.tsv',
            }]),
            trial_columns=None,
        ),
        'potsdam_binge_remote_pvt': datasets.PotsdamBingeRemotePVT(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'potsdam_binge_pvt_example.csv',
            }]),
            trial_columns=None,
        ),
        'potsdam_binge_wearable_pvt': datasets.PotsdamBingeWearablePVT(
            resources=ResourceDefinitions.from_dicts([{
                'content': 'samples',
                'filename_pattern': 'potsdam_binge_pvt_example.csv',
            }]),
        ),
    }
    yield Dataset(
        definition=init_param_dict[request.param],
        path=DatasetPaths(
            root='tests',
            dataset='.',
            raw='files',
            precomputed_events='files',
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
