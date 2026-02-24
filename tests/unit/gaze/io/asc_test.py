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
"""Test read from eyelink asc files."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import DatasetLibrary
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Screen
from pymovements.gaze import from_asc


@pytest.mark.parametrize(
    ('header', 'body', 'kwargs', 'expected_samples'),
    [
        pytest.param(
            '',
            '\n',
            {'patterns': 'eyelink'},
            pl.DataFrame(
                schema={'time': pl.Int64, 'pupil': pl.Float64, 'pixel': pl.List(pl.Float64)},
            ),
            marks=[
                pytest.mark.filterwarnings('ignore:.*No metadata.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No mount configuration.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No recording configuration.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No samples configuration.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No screen resolution.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No sampling rate found.*:UserWarning'),
                pytest.mark.filterwarnings(
                    'ignore:.*No tracked eye information found.*:UserWarning',
                ),
                pytest.mark.filterwarnings('ignore:.*No eye tracker vendor found.*:UserWarning'),
                pytest.mark.filterwarnings('ignore:.*No eye tracker model found.*:UserWarning'),
                pytest.mark.filterwarnings(
                    'ignore:.*No eye tracker software version found.*:UserWarning',
                ),
            ],
            id='empty_file',
        ),

    ],
)
def test_from_asc_has_expected_samples(
        header, body, kwargs, expected_samples, make_text_file,
):
    filepath = make_text_file('test_eyelink.asc', header=header, body=body)
    gaze = from_asc(filepath, **kwargs)

    assert_frame_equal(gaze.samples, expected_samples, check_column_order=False)


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_samples'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {'patterns': 'eyelink'},
            pl.from_dict(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_eyelink',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'experiment': DatasetLibrary.get('ToyDatasetEyeLink').experiment,
                **DatasetLibrary.get('ToyDatasetEyeLink').resources.filter('gaze')[0].load_kwargs,
            },
            pl.DataFrame(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                    'trial_id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, None],
                    'point_id': 3 * [None] + [0, 1, 2, 3] + [None, 0] + [0, 0, 1, 2] + [0, 1, None],
                    'screen_id': [None, 0, 1] + 13 * [None],
                    'task': [None] + 2 * ['reading'] + 12 * ['judo'] + [None],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'task': pl.Utf8,
                    'screen_id': pl.Int64,
                    'point_id': pl.Int64,
                    'trial_id': pl.Int64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_list',
        ),

        pytest.param(
            'eyelink_monocular_2khz_example.asc',
            {'patterns': 'eyelink'},
            pl.from_dict(
                data={
                    'time': [
                        2154556.5, 2154557.0, 2154560.5, 2154564.0, 2154596.0, 2154598.5, 2154599.0,
                        2154695.0, 2154696.0, 2339227.0, 2339245.0, 2339246.0, 2339271.5, 2339272.0,
                        2339290.0, 2339291.0,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Float64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_2khz_pattern_eyelink',
        ),
    ],
)
def test_from_asc_example_file_has_expected_samples(
        filename, kwargs, expected_samples, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)
    assert_frame_equal(gaze.samples, expected_samples, check_column_order=False)


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'shape', 'schema'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {'patterns': 'eyelink'},
            (16, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_pattern_eyelink',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            {'patterns': 'eyelink', 'add_columns': {'test': 'A'}},
            (16, 4),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
                'test': pl.String,
            },
            id='eyelink_asc_mono_pattern_eyelink_add_columns',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'patterns': 'eyelink',
                'add_columns': {'test': 1}, 'column_schema_overrides': {'test': pl.Float64},
            },
            (16, 4),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
                'test': pl.Float64,
            },
            id='eyelink_asc_mono_pattern_eyelink_add_columns_with_schema',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'experiment': DatasetLibrary.get('ToyDatasetEyeLink').experiment,
                **DatasetLibrary.get('ToyDatasetEyeLink').resources.filter('gaze')[0].load_kwargs,
            },
            (16, 7),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'task': pl.Utf8,
                'screen_id': pl.Int64,
                'point_id': pl.Int64,
                'trial_id': pl.Int64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_toydataset_eyelink',
        ),

        pytest.param(
            'eyelink_monocular_2khz_example.asc',
            {'patterns': 'eyelink'},
            (16, 3),
            {
                'time': pl.Float64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_2khz_pattern_eyelink',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {
                'patterns': 'eyelink',
                'encoding': 'latin1',
            },
            (297, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_no_dummy_pattern_eyelink',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {
                'patterns': 'eyelink',
                'encoding': 'latin1',
            },
            (297, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_no_dummy_pattern_eyelink_encoding_latin1',
        ),
        pytest.param(
            'eyelink_binocular_example.asc',
            {'patterns': 'eyelink'},
            (368, 3),
            {
                'time': pl.Int64,
                'pixel': pl.List(pl.Float64),
                'pupil': pl.List(pl.Float64),
            },
            id='eyelink_asc_bino_pattern_eyelink',
        ),
    ],
)
def test_from_asc_example_file_has_shape_and_schema(
        filename, kwargs, shape, schema, make_example_file,
):
    filepath = make_example_file(filename)

    gaze = from_asc(filepath, **kwargs)

    assert gaze.samples.shape == shape
    assert dict(gaze.samples.schema) == schema


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'exception', 'message_prefix'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {'patterns': 'foobar'},
            ValueError,
            "unknown pattern key 'foobar'. Supported keys are: eyelink",
            id='unknown_pattern',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'ascii',
            },
            UnicodeDecodeError,
            'ascii',
            id='eyelink_monocular_no_dummy_example_encoding_ascii',
        ),
    ],
)
def test_from_asc_example_file_raises_exception(
        filename, kwargs, exception, message_prefix, make_example_file,
):
    filepath = make_example_file(filename)
    with pytest.raises(exception) as excinfo:
        from_asc(filepath, **kwargs)

    msg = excinfo.value.args[0]
    assert msg.startswith(message_prefix)


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_experiment'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {},
            Experiment(
                screen=Screen(
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='monocular_1khz',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'experiment': Experiment(
                    screen_width_cm=40, screen_height_cm=30, sampling_rate=1000,
                ),
            },
            Experiment(
                screen=Screen(
                    width_cm=40,
                    height_cm=30,
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='monocular_1khz_experiment',
        ),

        pytest.param(
            'eyelink_monocular_2khz_example.asc',
            {},
            Experiment(
                screen=Screen(
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=2000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='monocular_2khz',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {'encoding': 'latin1'},
            Experiment(
                screen=Screen(
                    width_px=1920,
                    height_px=1080,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=500.0,
                    left=True,
                    right=False,
                    model='EyeLink 1000 Plus',
                    version='5.50',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='monocular_500hz_no_dummy',
        ),

        pytest.param(
            'eyelink_binocular_example.asc',
            {'encoding': 'latin1'},
            Experiment(
                screen=Screen(
                    width_px=1921,
                    height_px=1081,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=True,
                    model='EyeLink Portable Duo',
                    version='6.14',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='binocular_1kHz_nonstandard_resolution',
        ),

        pytest.param(
            'eyelink_binocular_example.asc',
            # asc file was not recorded by SR Research software but misses required header
            # for auto-inferring if screen resolution needs to be extended or not.
            # The following header line is needed in the source asc file for auto-inferring:
            # ** Recorded by: libeyelink.py
            {'encoding': 'latin1', 'extend_resolution': False},
            Experiment(
                screen=Screen(
                    width_px=1920,
                    height_px=1080,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=True,
                    model='EyeLink Portable Duo',
                    version='6.14',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='binocular_1kHz_force_extend',
        ),
    ],
)
def test_from_asc_example_file_has_expected_experiment(
        filename, kwargs, expected_experiment, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)
    assert gaze.experiment == expected_experiment


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_trial_columns'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'experiment': DatasetLibrary.get('ToyDatasetEyeLink').experiment,
                **DatasetLibrary.get('ToyDatasetEyeLink').resources.filter('gaze')[0].load_kwargs,
            },
            ['task', 'trial_id'],
            id='eyelink_asc_mono',
        ),
    ],
)
def test_from_asc_example_file_has_expected_trial_columns(
        filename, kwargs, expected_trial_columns, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)
    assert gaze.trial_columns == expected_trial_columns


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_n_components'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'experiment': DatasetLibrary.get('ToyDatasetEyeLink').experiment,
                **DatasetLibrary.get('ToyDatasetEyeLink').resources.filter('gaze')[0].load_kwargs,
            },
            2,
            id='eyelink_asc_mono',
        ),

        pytest.param(
            'eyelink_binocular_example.asc',
            {},
            4,
            id='eyelink_asc_bino',
        ),

    ],
)
def test_from_asc_example_file_has_expected_n_components(
        filename, kwargs, expected_n_components, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)
    assert gaze.n_components == expected_n_components


@pytest.mark.parametrize(
    ('experiment_kwargs', 'issues'),
    [
        pytest.param(
            {
                'screen_width_px': 1920,
                'screen_height_px': 1080,
                'sampling_rate': 1000,
            },
            ['Screen resolution: 1920x1080 != 1280x1024'],
            id='screen_resolution',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(sampling_rate=500),
            },
            ['Sampling rate: 500 != 1000.0'],
            id='eyetracker_sampling_rate',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    left=False,
                    right=True,
                    sampling_rate=1000,
                    mount='Desktop',
                ),
            },
            [
                'Left eye tracked: False != True',
                'Right eye tracked: True != False',
            ],
            id='eyetracker_tracked_eye',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    vendor='Tobii',
                    model='Tobii Pro Spectrum',
                    version='1.0',
                    sampling_rate=1000,
                    left=True,
                    right=False,
                ),
            },
            [
                'Eye tracker vendor: Tobii != EyeLink',
                'Eye tracker model: Tobii Pro Spectrum != EyeLink Portable Duo',
                'Eye tracker software version: 1.0 != 6.12',
            ],
            id='eyetracker_vendor_model_version',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    mount='Remote',
                    sampling_rate=1000,
                    vendor='EyeLink',
                    model='EyeLink Portable Duo',
                    version='6.12',
                ),
            },
            ['Mount configuration: Remote != Desktop'],
            id='eyetracker_mount',
        ),
    ],
)
def test_from_asc_detects_mismatches_in_experiment_metadata(
        experiment_kwargs, issues, make_example_file,
):
    filepath = make_example_file('eyelink_monocular_example.asc')
    expected_message = (
        'Experiment metadata does not match the metadata parsed from the ASC file:\n'
        + '\n'.join(f'- {issue}' for issue in issues)
    )

    with pytest.warns(UserWarning, match=expected_message):
        from_asc(filepath, experiment=Experiment(**experiment_kwargs))


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_metadata'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'metadata_patterns': [
                    {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<subject_id>-?\d+)'},
                ],
            },
            {
                'subject_id': '-1',
            },
            id='eyelink_asc_mono_subject_id_metadata_patterns',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'utf8',
            },
            {
                'foobar': 'ÄÖÜ',
            },
            id='eyelink_monocular_no_dummy_example_encoding_utf8',
        ),

        pytest.param(
            'eyelink_monocular_no_dummy_example.asc',
            {
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'latin1',
            },
            {
                'foobar': 'Ã\x84Ã\x96Ã\x9c',
            },
            id='eyelink_monocular_no_dummy_example_encoding_latin1',
        ),
    ],
)
def test_from_asc_example_file_has_expected_metadata(
        filename, kwargs, expected_metadata, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)

    for key, value in expected_metadata.items():
        assert key in gaze._metadata
        assert gaze._metadata[key] == value


@pytest.mark.parametrize(
    'filename', [
        pytest.param('eyelink_monocular_example.asc', id='mono'),
    ],
)
def test_from_asc_sets_public_cal_interfaces(filename, make_example_file):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath)

    # Calibrations DataFrame present with the expected schema
    assert isinstance(gaze.calibrations, pl.DataFrame)
    assert gaze.calibrations.schema == {
        'time': pl.Float64,
        'num_points': pl.Int64,
        'eye': pl.Utf8,
        'tracking_mode': pl.Utf8,
    }

    # Example file should contain at least one calibration
    assert gaze.calibrations.height >= 1


@pytest.mark.parametrize(
    'filename', [
        pytest.param('eyelink_monocular_example.asc', id='mono'),
    ],
)
def test_from_asc_sets_public_val_interfaces(filename, make_example_file):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath)

    # Validations DataFrame present with the expected schema
    assert isinstance(gaze.validations, pl.DataFrame)
    assert gaze.validations.schema == {
        'time': pl.Float64,
        'num_points': pl.Int64,
        'eye': pl.Utf8,
        'accuracy_avg': pl.Float64,
        'accuracy_max': pl.Float64,
    }

    # Example file should contain at least one validation
    assert gaze.validations.height >= 1


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_event_frame'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {'events': False},
            pl.from_dict(
                data={
                    'name': [],
                    'onset': [],
                    'offset': [],
                    'duration': [],
                },
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_without_events',
        ),
        pytest.param(
            'eyelink_monocular_example.asc',
            {'events': True},
            pl.from_dict(
                data={
                    'name': [
                        'fixation_eyelink',
                        'saccade_eyelink',
                        'fixation_eyelink',
                    ],
                    'eye': ['left', 'left', 'left'],
                    'onset': [2154563, 2339227, 2339246],
                    'offset': [2154695, 2339245, 2339290],
                    'duration': [132, 18, 44],
                },
                schema={
                    'name': pl.Utf8,
                    'eye': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_with_events',
        ),
        pytest.param(
            'eyelink_monocular_2khz_example.asc',
            {'events': True},
            pl.from_dict(
                data={
                    'name': [
                        'fixation_eyelink',
                        'saccade_eyelink',
                        'fixation_eyelink',
                    ],
                    'eye': ['left', 'left', 'left'],
                    'onset': [2154563, 2339227, 2339246],
                    'offset': [2154695, 2339245, 2339290],
                    'duration': [132, 18, 44],
                },
                schema={
                    'name': pl.Utf8,
                    'eye': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_2khz_with_events',
        ),
        pytest.param(
            'eyelink_binocular_example.asc',
            {'events': True},
            pl.from_dict(
                data={
                    'name': [
                        'fixation_eyelink',
                        'fixation_eyelink',
                        'blink_eyelink',
                        'blink_eyelink',
                        'saccade_eyelink',
                        'saccade_eyelink',
                        'fixation_eyelink',
                        'fixation_eyelink',
                    ],
                    'eye': ['left', 'right', 'right', 'left', 'left', 'right', 'left', 'right'],
                    'onset': [
                        1408667, 1408667, 1408793, 1408787, 1408774, 1408778, 1408897, 1408899,
                    ],
                    'offset': [
                        1408773, 1408777, 1408872, 1408883, 1408896, 1408898, 1409025, 1409027,
                    ],
                    'duration': [106, 110, 79, 96, 122, 120, 128, 128],
                },
                schema={
                    'name': pl.Utf8,
                    'eye': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_bino_with_events',
        ),
    ],
)
def test_from_asc_example_file_has_expected_events(
        filename, kwargs, expected_event_frame, make_example_file,
):
    filepath = make_example_file(filename)
    gaze = from_asc(filepath, **kwargs)

    assert_frame_equal(gaze.events.frame, expected_event_frame, check_column_order=False)


@pytest.mark.filterwarnings('ignore:.*No metadata.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No mount configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No recording configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No samples configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No screen resolution.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No sampling rate found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No tracked eye information found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker vendor found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker model found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker software version found.*:UserWarning')
@pytest.mark.parametrize(
    ('header', 'body', 'expected_warning', 'expected_message', 'from_asc_kwargs'),
    [
        pytest.param(
            '', 'END	1408901 	SAMPLES	EVENTS	RES	  47.75	  45.92',
            UserWarning, 'END recording message without associated START recording message',
            {},
            id='no_start_recording',
        ),
    ],
)
def test_from_asc_warns(
        header, body, expected_warning, expected_message,
        make_text_file, from_asc_kwargs,
):
    filepath = make_text_file(filename='test.asc', header=header, body=body)

    with pytest.warns(expected_warning, match=expected_message):
        from_asc(filepath, **from_asc_kwargs)


@pytest.mark.filterwarnings('ignore:.*No metadata.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No mount configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No recording configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No samples configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No screen resolution.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No sampling rate found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No tracked eye information found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker vendor found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker model found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker software version found.*:UserWarning')
@pytest.mark.parametrize(
    ('body', 'messages', 'expected_data'),
    [
        pytest.param(
            'MSG 123 message here\nMSG 152 TEST 1',
            True,
            [(123, 152), ('message here', 'TEST 1')],
            id='multiple_messages',
        ),
        pytest.param(
            'MSG 123 message here\nMSG 152 TEST 1',
            [r'^.*TEST.*$'],
            [(152,), ('TEST 1',)],
            id='filter_messages',
        ),
        pytest.param(
            'MSG 123 message here\nMSG 152 TEEST 1',
            [r'^.*TEST.*$'],
            [],
            id='no_match',
        ),
        pytest.param(
            'MSG 123 message here\nMSG 152 TEEST 1',
            False,
            None,
            id='no_parsing',
        ),
    ],
)
def test_from_asc_messages(make_text_file, body, messages, expected_data):
    filepath = make_text_file(filename='test.asc', header='', body=body)

    gaze = from_asc(filepath, messages=messages)

    if expected_data is None:
        assert gaze.messages is None
    else:
        assert_frame_equal(
            gaze.messages,
            pl.DataFrame(
                schema={'time': pl.Float64, 'content': pl.String},
                data=expected_data,
            ),
        )


def test_from_asc_keeps_remaining_metadata_private_and_pops_cal_val(make_example_file):
    filepath = make_example_file('eyelink_monocular_example.asc')
    gaze = from_asc(filepath)

    # Public frames exist
    assert isinstance(gaze.calibrations, pl.DataFrame)
    assert isinstance(gaze.validations, pl.DataFrame)

    # Private _metadata exists and does NOT contain cal/val anymore
    assert isinstance(gaze._metadata, dict)
    assert 'calibrations' not in gaze._metadata
    assert 'validations' not in gaze._metadata

    # Data loss ratios should be present for consumers until we migrate to explicit preprocessing
    # utilities.
    assert 'data_loss_ratio' in gaze._metadata
    assert 'data_loss_ratio_blinks' in gaze._metadata


@pytest.mark.filterwarnings('ignore:.*No metadata.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No mount configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No recording configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No samples configuration.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No screen resolution.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No sampling rate found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No tracked eye information found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker vendor found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker model found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker software version found.*:UserWarning')
def test_from_asc_orphaned_event_end_marker_with_custom_patterns_does_not_raise_keyerror(
        make_text_file,
):
    """Orphaned event end markers with custom patterns should not raise KeyError, but should warn.

    This test reproduces a scenario where an event end marker appears before the
    associated context dictionary has been populated with keys from custom patterns.
    """
    body = (
        'EFIX R 1000 1100 100 500.0 500.0 1000\n'
        'MSG 1200 START_TRIAL_1\n'
    )
    patterns = [r'START_TRIAL_(?P<trial_id>\d+)']
    filepath = make_text_file(filename='orphaned_event.asc', body=body)

    with pytest.warns(UserWarning, match='Missing start marker before end for event'):
        gaze = from_asc(filepath, patterns=patterns, events=True)

    expected_events = pl.from_dict(
        data={
            'name': ['fixation_eyelink'],
            'eye': ['right'],
            'onset': [1000],
            'offset': [1100],
            'duration': [100],
            'trial_id': [None],
        },
        schema={
            'name': pl.Utf8,
            'eye': pl.Utf8,
            'onset': pl.Int64,
            'offset': pl.Int64,
            'duration': pl.Int64,
            'trial_id': pl.Null,
        },
    )

    assert_frame_equal(gaze.events.frame, expected_events, check_column_order=False)
