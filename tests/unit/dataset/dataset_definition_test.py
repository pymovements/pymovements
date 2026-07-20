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
"""Test dataset definition."""
from dataclasses import dataclass

import pytest
import yaml

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import Experiment
from pymovements import ResourceDefinition
from pymovements import ResourceDefinitions
from pymovements import WebSource


@pytest.mark.parametrize(
    'init_kwargs',
    [
        pytest.param(
            {'name': 'A'},
            id='name_only',
        ),
        pytest.param(
            {'name': 'A', 'experiment': Experiment(sampling_rate=1000)},
            id='name_and_experiment',
        ),
    ],
)
def test_dataset_definition_is_equal(init_kwargs):
    definition1 = DatasetDefinition(**init_kwargs)
    definition2 = DatasetDefinition(**init_kwargs)

    assert definition1 == definition2


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_resources'),
    [
        pytest.param(
            {},
            ResourceDefinitions(),
            id='default',
        ),

        pytest.param(
            {'resources': None},
            ResourceDefinitions(),
            id='none',
        ),

        pytest.param(
            {'resources': {}},
            ResourceDefinitions(),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='empty_dict',
        ),

        pytest.param(
            {'resources': []},
            ResourceDefinitions(),
            id='empty_list',
        ),

        pytest.param(
            {'resources': ResourceDefinitions([ResourceDefinition(content='gaze')])},
            ResourceDefinitions([ResourceDefinition(content='gaze')]),
            id='resource_definitions',
        ),

        pytest.param(
            {'resources': [ResourceDefinition(content='gaze')]},
            ResourceDefinitions([ResourceDefinition(content='gaze')]),
            id='resource_definitions_list',
        ),

        pytest.param(
            {'resources': [{'content': 'gaze'}]},
            ResourceDefinitions([ResourceDefinition(content='gaze')]),
            id='single_gaze_resource',
        ),

        pytest.param(
            {'resources': {'gaze': [{'resource': 'www.example.com'}]}},
            ResourceDefinitions(
                [ResourceDefinition(content='gaze', source=WebSource(url='www.example.com'))],
            ),
            marks=[
                pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
                pytest.mark.filterwarnings(
                    'ignore:.*Please use ResourceDefinition[.]source instead.*:DeprecationWarning',
                ),
            ],
            id='single_gaze_resource_legacy',
        ),

        pytest.param(
            {
                'resources': [
                    {'content': 'gaze', 'filename_pattern': 'test.csv'},
                ],
            },
            ResourceDefinitions([ResourceDefinition(content='gaze', filename_pattern='test.csv')]),
            id='single_gaze_resource_filename_pattern',
        ),

        pytest.param(
            {
                'resources': [
                    {
                        'content': 'gaze', 'filename_pattern': 'test.csv',
                        'url': 'https://example.com', 'mirrors': ['https://mirror.com'],
                    },
                ],
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze',
                    source=WebSource(url='https://example.com', mirrors=['https://mirror.com']),
                    filename_pattern='test.csv',
                ),
            ]),
            marks=pytest.mark.filterwarnings(
                'ignore:.*Please use ResourceDefinition[.]source instead.*:DeprecationWarning',
            ),
            id='single_gaze_resource_with_url_and_mirror_deprecated',
        ),

        pytest.param(
            {
                'resources': [
                    {
                        'content': 'gaze', 'filename_pattern': 'test.csv',
                        'source': {'url': 'https://example.com', 'mirrors': ['https://mirror.com']},
                    },
                ],
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze',
                    source=WebSource(url='https://example.com', mirrors=['https://mirror.com']),
                    filename_pattern='test.csv',
                ),
            ]),
            id='single_gaze_resource_with_url_and_mirror',
        ),

        pytest.param(
            {'resources': [{'content': 'precomputed_events'}]},
            ResourceDefinitions([ResourceDefinition(content='precomputed_events')]),
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            {
                'resources': [
                    {'content': 'gaze'},
                    {'content': 'precomputed_events'},
                ],
            },
            ResourceDefinitions([
                ResourceDefinition(content='gaze'),
                ResourceDefinition(content='precomputed_events'),
            ]),
            id='two_resources',
        ),

        pytest.param(
            {
                'resources': {
                    'gaze': [{'resource': 'www.example1.com'}],
                    'precomputed_events': [{'resource': 'www.example2.com'}],
                },
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze', source=WebSource(url='www.example1.com'),
                ),
                ResourceDefinition(
                    content='precomputed_events', source=WebSource(url='www.example2.com'),
                ),
            ]),
            marks=[
                pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
                pytest.mark.filterwarnings(
                    'ignore:.*Please use ResourceDefinition[.]source instead.*:DeprecationWarning',
                ),
            ],
            id='two_resources_legacy',
        ),

        pytest.param(
            {
                'resources': {
                    'gaze': [{'source': {'url': 'www.example1.com'}}],
                    'precomputed_events': [{'source': {'url': 'www.example2.com'}}],
                },
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze', source=WebSource(url='www.example1.com'),
                ),
                ResourceDefinition(
                    content='precomputed_events', source=WebSource(url='www.example2.com'),
                ),
            ]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='two_resources_legacy_with_sources',
        ),
    ],
)
def test_dataset_definition_resources_init_expected(init_kwargs, expected_resources):
    definition = DatasetDefinition(**init_kwargs)
    assert definition.resources == expected_resources


@pytest.mark.parametrize(
    ('definition', 'expected_dict'),
    [
        pytest.param(
            DatasetDefinition(
                name='Example',
                long_name='Example',
            ),
            {
                'name': 'Example',
                'long_name': 'Example',
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': None,
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': [],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='no_experiment',
        ),
        pytest.param(
            DatasetDefinition(
                name='Example',
                long_name='Example',
                experiment=Experiment(
                    screen_width_px=1280,
                    screen_height_px=1024,
                    screen_width_cm=38.2,
                    screen_height_cm=30.2,
                    distance_cm=60,
                    origin='center',
                    sampling_rate=2000,
                ),
            ),
            {
                'name': 'Example',
                'long_name': 'Example',
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': 2000,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': 60,
                        'height_cm': 30.2,
                        'height_px': 1024,
                        'origin': 'center',
                        'width_cm': 38.2,
                        'width_px': 1280,
                    },
                },
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': [],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='experiment',
        ),
    ],
)
def test_dataset_definition_to_dict_expected(definition, expected_dict):
    assert definition.to_dict(exclude_none=False) == expected_dict


@pytest.mark.parametrize(
    ('exclude_private', 'expected_dict'),
    [
        pytest.param(
            True,
            {
                'name': 'MyDatasetDefinition',
                'long_name': None,
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': None,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': None,
                        'height_cm': None,
                        'height_px': None,
                        'origin': None,
                        'width_cm': None,
                        'width_px': None,
                    },
                },
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': [],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='True',
        ),

        pytest.param(
            False,
            {
                'name': 'MyDatasetDefinition',
                'long_name': None,
                '_foobar': 'test',
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': None,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': None,
                        'height_cm': None,
                        'height_px': None,
                        'origin': None,
                        'width_cm': None,
                        'width_px': None,
                    },
                },
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': [],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='False',
        ),
    ],
)
def test_dataset_definition_to_dict_exclude_private_expected(exclude_private, expected_dict):
    @dataclass
    class MyDatasetDefinition(DatasetDefinition):
        name: str = 'MyDatasetDefinition'
        _foobar: str = 'test'

    definition = MyDatasetDefinition()

    assert definition.to_dict(exclude_private=exclude_private, exclude_none=False) == expected_dict


@pytest.mark.parametrize(
    ('definition'),
    [
        pytest.param(
            DatasetDefinition(
                name='Example',
            ),
            id='no_exp',
        ),

        pytest.param(
            DatasetDefinition(
                name='Example',
                experiment=Experiment(
                    screen_width_px=1280,
                    screen_height_px=1024,
                    screen_width_cm=38.2,
                    screen_height_cm=30.2,
                    distance_cm=60,
                    origin='center',
                    sampling_rate=2000,
                ),
            ),
            id='no_exp',
        ),
    ],
)
def test_dataset_definition_to_yaml_equal_dicts(definition, tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'

    definition.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    assert definition.to_dict() == yaml_dict


@pytest.mark.filterwarnings('ignore:DatasetDefinition.mirrors is deprecated.*:DeprecationWarning')
def test_write_yaml_already_existing_dataset_definition_w_tuple_screen(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    definition = DatasetLibrary.get('ToyDatasetEyeLink')
    definition.to_yaml(tmp_file, exclude_none=False)

    with open(tmp_file, encoding='utf-8') as f:
        yaml.safe_load(f)

    assert DatasetDefinition.from_yaml(tmp_file) == definition


def test_check_equality_of_load_from_yaml_and_load_from_dictionary_dump(tmp_path):
    dictionary_tmp_file = tmp_path / 'dictionary.yaml'
    yaml_encoding = {
        'name': 'Example',
    }

    with open(dictionary_tmp_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_encoding, f)

    yaml_definition = DatasetDefinition.from_yaml(dictionary_tmp_file)

    expected_definition = DatasetDefinition(
        name='Example',
    )

    assert yaml_definition == expected_definition


@pytest.mark.parametrize(
    ('dataset_definition', 'exclude_none', 'expected_dict'),
    [
        pytest.param(
            DatasetDefinition(),
            True,
            {
                'name': '.',
            },
            id='true_default',
        ),

        pytest.param(
            DatasetDefinition(experiment=Experiment(origin=None)),
            True,
            {
                'name': '.',
            },
            id='true_experiment_origin_none',
        ),

        pytest.param(
            DatasetDefinition(),
            False,
            {
                'name': '.',
                'long_name': None,
                'description': None,
                'mirrors': {},
                'resources': [],
                'experiment': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'trial_columns': None,
                'time_column': None,
                'time_unit': None,
                'pixel_columns': None,
                'position_columns': None,
                'velocity_columns': None,
                'acceleration_columns': None,
                'distance_column': None,
            },
            id='false_default',
        ),

        pytest.param(
            DatasetDefinition(experiment=None),
            False,
            {
                'name': '.',
                'long_name': None,
                'description': None,
                'mirrors': {},
                'resources': [],
                'experiment': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'trial_columns': None,
                'time_column': None,
                'time_unit': None,
                'pixel_columns': None,
                'position_columns': None,
                'velocity_columns': None,
                'acceleration_columns': None,
                'distance_column': None,
            },
            id='false_experiment_none',
        ),

        pytest.param(
            DatasetDefinition(experiment=Experiment(origin=None)),
            False,
            {
                'name': '.',
                'long_name': None,
                'description': None,
                'mirrors': {},
                'resources': [],
                'experiment': {
                    'eyetracker': {
                        'sampling_rate': None,
                        'vendor': None,
                        'model': None,
                        'version': None,
                        'mount': None,
                        'left': None,
                        'right': None,
                    },
                    'screen': {
                        'height_cm': None,
                        'width_cm': None,
                        'height_px': None,
                        'width_px': None,
                        'distance_cm': None,
                        'origin': None,
                    },
                },
                'column_map': None,
                'custom_read_kwargs': None,
                'trial_columns': None,
                'time_column': None,
                'time_unit': None,
                'pixel_columns': None,
                'position_columns': None,
                'velocity_columns': None,
                'acceleration_columns': None,
                'distance_column': None,
            },
            id='false_experiment_origin_none',
        ),

        pytest.param(
            DatasetDefinition(
                resources=[
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                    },
                ],
            ),
            True,
            {
                'name': '.',
                'resources': [
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                    },
                ],
            },
            id='true_resources',
        ),

        pytest.param(
            DatasetDefinition(
                resources=[
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                    },
                ],
            ),
            False,
            {
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': None,
                'long_name': None,
                'mirrors': {},
                'name': '.',
                'pixel_columns': None,
                'position_columns': None,
                'resources': [
                    {
                        'content': 'gaze',
                        'filename_pattern': None,
                        'filename_pattern_schema_overrides': None,
                        'load_function': None,
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': [
                                'test',
                                'foo',
                                'bar',
                            ],
                        },
                        'source': None,
                    },
                ],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='false_resources',
        ),

        pytest.param(
            DatasetDefinition(
                resources=[
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                        'source': WebSource(url='http://my.example.here'),
                    },
                ],
            ),
            True,
            {
                'name': '.',
                'resources': [
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                        'source': {'url': 'http://my.example.here'},
                    },
                ],
            },
            id='true_resources_with_source',
        ),

        pytest.param(
            DatasetDefinition(
                resources=[
                    {
                        'content': 'gaze',
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': ['test', 'foo', 'bar'],
                        },
                        'source': WebSource(url='http://my.example.here'),
                    },
                ],
            ),
            False,
            {
                'acceleration_columns': None,
                'column_map': None,
                'custom_read_kwargs': None,
                'description': None,
                'distance_column': None,
                'experiment': None,
                'long_name': None,
                'mirrors': {},
                'name': '.',
                'pixel_columns': None,
                'position_columns': None,
                'resources': [
                    {
                        'content': 'gaze',
                        'filename_pattern': None,
                        'filename_pattern_schema_overrides': None,
                        'load_function': None,
                        'load_kwargs': {
                            'distance_column': 'test',
                            'position_columns': [
                                'test',
                                'foo',
                                'bar',
                            ],
                        },
                        'source': {
                            'url': 'http://my.example.here',
                            'filename': None,
                            'md5': None,
                            'mirrors': None,
                        },
                    },
                ],
                'time_column': None,
                'time_unit': None,
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='false_resources_with_source',
        ),
    ],
)
def test_dataset_to_dict_exclude_none(dataset_definition, exclude_none, expected_dict):
    assert dataset_definition.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    ('init_kwargs', 'scheduled_version'),
    [
        pytest.param(
            {'mirrors': {'gaze': ['https://mirror.com']}},
            '0.29.0',
            id='mirrors',
        ),
        pytest.param(
            {'trial_columns': ['trial']},
            '0.30.0',
            id='trial_columns',
        ),
        pytest.param(
            {'time_column': 't'},
            '0.30.0',
            id='time_column',
        ),
        pytest.param(
            {'time_unit': 'ms'},
            '0.30.0',
            id='time_unit',
        ),
        pytest.param(
            {'pixel_columns': ['x', 'y']},
            '0.30.0',
            id='pixel_columns',
        ),
        pytest.param(
            {'position_columns': ['x', 'y']},
            '0.30.0',
            id='position_columns',
        ),
        pytest.param(
            {'velocity_columns': ['x', 'y']},
            '0.30.0',
            id='velocity_columns',
        ),
        pytest.param(
            {'acceleration_columns': ['x', 'y']},
            '0.30.0',
            id='acceleration_columns',
        ),
        pytest.param(
            {'distance_column': 'd'},
            '0.30.0',
            id='distance_column',
        ),
        pytest.param(
            {'column_map': {'a': 'b'}},
            '0.30.0',
            id='column_map',
        ),
        pytest.param(
            {'custom_read_kwargs': {'gaze': {'asd': 'def'}}},
            '0.30.0',
            id='custom_read_kwargs',
        ),
    ],
)
def test_dataset_definition_init_parameter_is_deprecated_or_removed(
        init_kwargs, scheduled_version, assert_deprecation_is_removed,
):
    with pytest.raises(DeprecationWarning) as info:
        DatasetDefinition(**init_kwargs)

    assert_deprecation_is_removed(
        function_name=f'DatasetDefinition init keyword argument {list(init_kwargs.keys())[0]}',
        warning_message=info.value.args[0],
        scheduled_version=scheduled_version,

    )


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            {'resources': 1},
            TypeError,
            'resources is of type int but must be of type ResourceDefinitions, list, or dict.',
            id='resources_int',
        ),
    ],
)
def test_dataset_definition_init_raises_exception(init_kwargs, exception, exception_msg):
    with pytest.raises(exception) as excinfo:
        DatasetDefinition(**init_kwargs)

    msg, = excinfo.value.args
    assert msg == exception_msg
