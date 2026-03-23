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
"""Test basic preprocessing on various gaze files."""
import pytest

from pymovements import DatasetLibrary
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import gaze as gaze_module
from pymovements import ResourceDefinition


@pytest.fixture(
    name='init_kwargs',
    params=[
        'csv_monocular',
        'csv_binocular',
        'ipc_monocular',
        'ipc_binocular',
        'eyelink_monocular',
        'eyelink_monocular_2khz',
        'eyelink_monocular_no_dummy',
        'didec',
        'emtec',
        'hbn',
        'sbsat',
        'gaze_on_faces',
        'gazebase',
        'gazebase_vr',
        'judo1000',
        'potec',
        'raccoons',
    ],
)
def fixture_init_kwargs(request, make_example_file):
    init_param_dict = {
        'csv_monocular': {
            'file': make_example_file('monocular_example.csv'),
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            'definition': ResourceDefinition(
                content='gaze',
                load_kwargs={
                    'time_column': 'time',
                    'time_unit': 'ms',
                    'pixel_columns': ['x_left_pix', 'y_left_pix'],
                },
            ),
        },
        'csv_binocular': {
            'file': make_example_file('binocular_example.csv'),
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            'definition': ResourceDefinition(
                content='gaze',
                load_kwargs={
                    'time_column': 'time',
                    'time_unit': 'ms',
                    'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                    'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
                },
            ),
        },
        'ipc_monocular': {
            'file': make_example_file('monocular_example.feather'),
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            'definition': ResourceDefinition(content='gaze'),
        },
        'ipc_binocular': {
            'file': make_example_file('binocular_example.feather'),
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            'definition': ResourceDefinition(content='gaze'),
        },
        'eyelink_monocular': {
            'file': make_example_file('eyelink_monocular_example.asc'),
            'experiment': DatasetLibrary.get('ToyDatasetEyeLink').experiment,
            'definition': DatasetLibrary.get('ToyDatasetEyeLink').resources[0],
        },
        'eyelink_monocular_2khz': {
            'file': make_example_file('eyelink_monocular_2khz_example.asc'),
            'experiment': Experiment(
                1280, 1024, 38, 30.2, 68, 'upper left',
                eyetracker=EyeTracker(
                    sampling_rate=2000.0, left=True, right=False,
                    model='EyeLink Portable Duo', vendor='EyeLink',
                ),
            ),
            'definition': ResourceDefinition(content='gaze'),
        },
        'eyelink_monocular_no_dummy': {
            'file': make_example_file('eyelink_monocular_no_dummy_example.asc'),
            'experiment': Experiment(
                1920, 1080, 38, 30.2, 68, 'upper left',
                eyetracker=EyeTracker(
                    sampling_rate=500.0, left=True, right=False,
                    model='EyeLink 1000 Plus', vendor='EyeLink',
                ),
            ),
            'definition': ResourceDefinition(content='gaze'),
        },
        'didec': {
            'file': make_example_file('didec_example.txt'),
            'experiment': DatasetLibrary.get('DIDEC').experiment,
            'definition': DatasetLibrary.get('DIDEC').resources[0],
        },
        'emtec': {
            'file': make_example_file('emtec_example.csv'),
            'experiment': DatasetLibrary.get('EMTeC').experiment,
            'definition': DatasetLibrary.get('EMTeC').resources[0],
        },
        'hbn': {
            'file': make_example_file('hbn_example.csv'),
            'experiment': DatasetLibrary.get('HBN').experiment,
            'definition': DatasetLibrary.get('HBN').resources[0],
        },
        'sbsat': {
            'file': make_example_file('sbsat_example.csv'),
            'experiment': DatasetLibrary.get('SBSAT').experiment,
            'definition': DatasetLibrary.get('SBSAT').resources[0],
        },
        'gaze_on_faces': {
            'file': make_example_file('gaze_on_faces_example.csv'),
            'experiment': DatasetLibrary.get('GazeOnFaces').experiment,
            'definition': DatasetLibrary.get('GazeOnFaces').resources[0],
        },
        'gazebase': {
            'file': make_example_file('gazebase_example.csv'),
            'experiment': DatasetLibrary.get('GazeBase').experiment,
            'definition': DatasetLibrary.get('GazeBase').resources[0],
        },
        'gazebase_vr': {
            'file': make_example_file('gazebase_vr_example.csv'),
            'experiment': DatasetLibrary.get('GazeBaseVR').experiment,
            'definition': DatasetLibrary.get('GazeBaseVR').resources[0],
        },
        'judo1000': {
            'file': make_example_file('judo1000_example.csv'),
            'experiment': DatasetLibrary.get('JuDo1000').experiment,
            'definition': DatasetLibrary.get('JuDo1000').resources[0],
        },
        'potec': {
            'file': make_example_file('potec_example.tsv'),
            'experiment': DatasetLibrary.get('PoTeC').experiment,
            'definition': DatasetLibrary.get('PoTeC').resources[0],
        },
        'raccoons': {
            'file': make_example_file('raccoons.asc'),
            'experiment': DatasetLibrary.get('RaCCooNS').experiment,
            'definition': DatasetLibrary.get('RaCCooNS').resources[0],
        },

    }
    yield init_param_dict[request.param]


def test_gaze_file_processing(init_kwargs):
    # Load in gaze file.
    file_extension = init_kwargs['file'].suffix
    gaze = None

    resource_definition = init_kwargs['definition']

    # Load in gaze file.
    if resource_definition.load_function is not None:
        load_function_name = resource_definition.load_function
    elif file_extension in {'.csv', '.tsv', '.txt'}:
        load_function_name = 'from_csv'
    elif file_extension in {'.feather', '.ipc'}:
        load_function_name = 'from_ipc'
    elif file_extension == '.asc':
        load_function_name = 'from_asc'
    else:
        load_function_name = 'from_csv'

    if load_function_name == 'from_csv':
        load_function = gaze_module.from_csv
    elif load_function_name == 'from_ipc':
        load_function = gaze_module.from_ipc
    elif load_function_name == 'from_asc':
        load_function = gaze_module.from_asc
    elif load_function_name == 'from_begaze':
        load_function = gaze_module.from_begaze
    else:
        load_function = gaze_module.from_csv

    if resource_definition.load_kwargs is None:
        load_kwargs = {}
    else:
        load_kwargs = resource_definition.load_kwargs

    gaze = load_function(
        file=init_kwargs['file'],
        experiment=init_kwargs['experiment'],
        **load_kwargs,
    )

    assert gaze is not None
    assert gaze.samples.height > 0

    # Do some basic transformations.
    if 'pixel' in gaze.columns:
        gaze.pix2deg()
    gaze.pos2vel()
    gaze.pos2acc()
    gaze.resample(resampling_rate=2000)

    assert 'position' in gaze.columns
    assert 'velocity' in gaze.columns
    assert 'acceleration' in gaze.columns
