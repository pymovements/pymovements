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
"""Test dataset definition."""
import re
from dataclasses import dataclass

import pytest
import yaml

from pymovements import __version__
from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import Experiment
from pymovements import ResourceDefinition
from pymovements import ResourceDefinitions


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
            {'resources': [{'content': 'gaze'}]},
            ResourceDefinitions([ResourceDefinition(content='gaze')]),
            id='single_gaze_resource',
        ),

        pytest.param(
            {'resources': {'gaze': [{'resource': 'www.example.com'}]}},
            ResourceDefinitions([ResourceDefinition(content='gaze', url='www.example.com')]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
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
                'resources': {'gaze': [{'content': 'gaze'}]},
                'filename_format': {'gaze': 'test.csv'},
            },
            ResourceDefinitions([ResourceDefinition(content='gaze', filename_pattern='test.csv')]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='single_gaze_resource_filename_format_legacy',
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
                    content='gaze', filename_pattern='test.csv',
                    url='https://example.com', mirrors=['https://mirror.com'],
                ),
            ]),
            id='single_gaze_resource_with_url_and_mirror',
        ),

        pytest.param(
            {
                'filename_format': {'gaze': 'test.csv'},
            },
            ResourceDefinitions([ResourceDefinition(content='gaze', filename_pattern='test.csv')]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='filename_format_without_resources_legacy',
        ),

        pytest.param(
            {
                'resources': {'gaze': [{'content': 'gaze'}]},
                'filename_format': {'gaze': '{subject_id:d}.csv'},
                'filename_format_schema_overrides': {
                    'gaze': {
                        'subject_id': int,
                    },
                },
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze',
                    filename_pattern='{subject_id:d}.csv',
                    filename_pattern_schema_overrides={'subject_id': int},
                ),
            ]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='single_gaze_resource_filename_format_schema_overrides_legacy',
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
                ResourceDefinition(content='gaze', url='www.example1.com'),
                ResourceDefinition(content='precomputed_events', url='www.example2.com'),
            ]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='two_resources_legacy',
        ),

        pytest.param(
            {
                'resources': {
                    'gaze': [{'resource': 'www.example1.com'}],
                    'precomputed_events': [{'resource': 'www.example2.com'}],
                },
                'filename_format': {
                    'gaze': 'test1.csv',
                    'precomputed_events': 'test2.csv',
                },
            },
            ResourceDefinitions([
                ResourceDefinition(
                    content='gaze',
                    url='www.example1.com',
                    filename_pattern='test1.csv',
                ),
                ResourceDefinition(
                    content='precomputed_events',
                    url='www.example2.com',
                    filename_pattern='test2.csv',
                ),
            ]),
            marks=pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning'),
            id='two_resources_filename_format_legacy',
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
                'column_map': {},
                'custom_read_kwargs': {},
                'distance_column': None,
                'experiment': None,
                'extract': None,
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
                'column_map': {},
                'custom_read_kwargs': {},
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
                'extract': None,
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
                'column_map': {},
                'custom_read_kwargs': {},
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
                'extract': None,
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
                'column_map': {},
                'custom_read_kwargs': {},
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
                'extract': None,
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


@pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:.*has_resources.*:DeprecationWarning')
@pytest.mark.parametrize(
    ('resources', 'expected_has_resources'),
    [
        pytest.param(
            None,
            False,
            id='none',
        ),

        pytest.param(
            {},
            False,
            id='empty_resources_dict',
        ),

        pytest.param(
            {'gaze': None},
            False,
            id='none_value_as_resources',
        ),

        pytest.param(
            {'gaze': []},
            False,
            id='empty_list_as_resources',
        ),

        pytest.param(
            {'gaze': [{'resource': 'foo'}]},
            True,
            id='gaze_resources',
        ),

        pytest.param(
            {'precomputed_events': [{'resource': 'foo'}]},
            True,
            id='precomputed_event_resources',
        ),

        pytest.param(
            {'precomputed_reading_measures': [{'resource': 'foo'}]},
            True,
            id='precomputed_reading_measures_resources',
        ),

        pytest.param(
            {
                'gaze': [{'resource': 'foo'}],
                'precomputed_events': [{'resource': 'foo'}],
                'precomputed_reading_measures': [{'resource': 'foo'}],
            },
            True,
            id='all_resources',
        ),

        pytest.param(
            {
                'foo': [{'resource': 'bar'}],
            },
            True,
            id='custom_resources',
        ),
    ],
)
def test_dataset_definition_has_resources_boolean(resources, expected_has_resources):
    definition = DatasetDefinition(resources=resources)

    # there are multiple contexts of using booleans.
    assert bool(definition.has_resources) == expected_has_resources
    assert str(definition.has_resources) == str(expected_has_resources)
    assert definition.has_resources == expected_has_resources
    assert not (definition.has_resources and not expected_has_resources)
    assert not (not definition.has_resources and expected_has_resources)


@pytest.mark.filterwarnings('ignore:.*from_dict.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:.*has_resources.*:DeprecationWarning')
@pytest.mark.parametrize(
    ('resources', 'expected_resources'),
    [
        pytest.param(
            {},
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='empty_resources_dict',
        ),

        pytest.param(
            {'gaze': None},
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='none_value_as_resources',
        ),

        pytest.param(
            {'gaze': []},
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='empty_list_as_resources',
        ),

        pytest.param(
            {'gaze': [{'resource': 'foo'}]},
            {
                'gaze': True,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='gaze_resources',
        ),

        pytest.param(
            {'precomputed_events': [{'resource': 'foo'}]},
            {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
            id='precomputed_event_resources',
        ),

        pytest.param(
            {'precomputed_reading_measures': [{'resource': 'foo'}]},
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
            id='precomputed_reading_measures_resources',
        ),

        pytest.param(
            {
                'gaze': [{'resource': 'foo'}],
                'precomputed_events': [{'resource': 'foo'}],
                'precomputed_reading_measures': [{'resource': 'foo'}],
            },
            {'gaze': True, 'precomputed_events': True, 'precomputed_reading_measures': True},
            id='all_resources',
        ),

        pytest.param(
            {
                'foo': [{'resource': 'bar'}],
            },
            {
                'foo': True,
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='custom_resources',
        ),
    ],
)
def test_dataset_definition_has_resources_indexable(resources, expected_resources):
    definition = DatasetDefinition(resources=resources)

    for key, value in expected_resources.items():
        assert definition.has_resources[key] == value


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_dataset_definition_has_resources_not_equal():
    definition1 = DatasetDefinition(resources=[{'content': 'gaze', 'url': 'http://www.xample.com'}])
    definition2 = DatasetDefinition(resources=[])

    assert definition1.has_resources != definition2.has_resources


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
            DatasetDefinition(
                distance_column='test',
                position_columns=['test', 'foo', 'bar'],
            ),
            True,
            {
                'name': '.',
                'position_columns': ['test', 'foo', 'bar'],
                'distance_column': 'test',
            },
            id='true_str_dict_list',
        ),

        pytest.param(
            DatasetDefinition(
                experiment=Experiment(origin=None),
                distance_column='test',
                position_columns=['test', 'foo', 'bar'],
            ),
            True,
            {
                'name': '.',
                'position_columns': ['test', 'foo', 'bar'],
                'distance_column': 'test',
            },
            id='true_str_dict_list_experiment_origin_none',
        ),

        pytest.param(
            DatasetDefinition(),
            False,
            {
                'name': '.',
                'long_name': None,
                'mirrors': {},
                'resources': [],
                'experiment': None,
                'extract': None,
                'custom_read_kwargs': {},
                'column_map': {},
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
                'mirrors': {},
                'resources': [],
                'experiment': None,
                'extract': None,
                'custom_read_kwargs': {},
                'column_map': {},
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
                'extract': None,
                'custom_read_kwargs': {},
                'column_map': {},
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
    ],
)
def test_dataset_to_dict_exclude_none(dataset_definition, exclude_none, expected_dict):
    assert dataset_definition.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    'attribute_kwarg',
    [
        pytest.param(
            {'extract': True},
            id='extract_true',
        ),
        pytest.param(
            {'extract': False},
            id='extract_false',
        ),
        pytest.param(
            {'has_files': {'gaze': True}},
            id='has_files',
        ),
        pytest.param(
            {'mirrors': {'gaze': ['https://mirror.com']}},
            id='mirrors',
        ),
    ],
)
def test_dataset_definition_attribute_is_deprecated_or_removed(attribute_kwarg):
    with pytest.raises(DeprecationWarning) as info:
        DatasetDefinition(**attribute_kwarg)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    attribute_name = list(attribute_kwarg.keys())[0]
    assert current_version < remove_version, (
        f'{attribute_name} is scheduled to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
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


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    ('definition', 'expected'),
    [
        pytest.param(
            DatasetDefinition(
                resources=[{'content': 'gaze', 'filename_pattern': 'abc'}],
            ),
            {'gaze': 'abc'},
            id='single_gaze',
        ),
    ],
)
def test_dataset_definition_get_filename_format_expected(definition, expected):
    assert definition.filename_format == expected


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    ('definition', 'new_value', 'expected'),
    [
        pytest.param(
            DatasetDefinition(
                resources=[{'content': 'gaze', 'filename_pattern': 'abc'}],
            ),
            {'gaze': 'def'},
            ResourceDefinitions([ResourceDefinition(content='gaze', filename_pattern='def')]),
            id='gaze_resource',
        ),
        pytest.param(
            DatasetDefinition(
                resources=[
                    {'content': 'gaze', 'filename_pattern': 'abc'},
                    {'content': 'precomputed_events', 'filename_pattern': 'cba'},
                ],
            ),
            {'gaze': 'def'},
            ResourceDefinitions([
                ResourceDefinition(content='gaze', filename_pattern='def'),
                ResourceDefinition(content='precomputed_events', filename_pattern='cba'),
            ]),
            id='two_resources',
        ),
    ],
)
def test_dataset_definition_set_filename_format_expected(definition, new_value, expected):
    definition.filename_format = new_value
    assert definition.resources == expected


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    ('definition', 'expected'),
    [
        pytest.param(
            DatasetDefinition(
                resources=[{'content': 'gaze', 'filename_pattern_schema_overrides': {'a': int}}],
            ),
            {'gaze': {'a': int}},
            id='single_gaze',
        ),
    ],
)
def test_dataset_definition_filename_get_format_schema_expected(definition, expected):
    assert definition.filename_format_schema_overrides == expected


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    ('definition', 'new_value', 'expected'),
    [
        pytest.param(
            DatasetDefinition(
                resources=[{'content': 'gaze', 'filename_pattern_schema_overrides': {'a': int}}],
            ),
            {'gaze': {'b': str}},
            ResourceDefinitions(
                [ResourceDefinition(content='gaze', filename_pattern_schema_overrides={'b': str})],
            ),
            id='gaze_resource',
        ),
        pytest.param(
            DatasetDefinition(
                resources=[
                    {'content': 'gaze', 'filename_pattern_schema_overrides': {'a': int}},
                    {'content': 'precomputed_events'},
                ],
            ),
            {'gaze': {'b': str}},
            ResourceDefinitions(
                [
                    ResourceDefinition(
                        content='gaze', filename_pattern_schema_overrides={'b': str},
                    ),
                    ResourceDefinition(content='precomputed_events'),
                ],
            ),
            id='two_resources',
        ),
    ],
)
def test_dataset_definition_set_filename_format_schema_expected(definition, new_value, expected):
    definition.filename_format_schema_overrides = new_value
    assert definition.resources == expected


@pytest.mark.parametrize(
    ('definition', 'attribute'),
    [
        pytest.param(
            DatasetDefinition(),
            'filename_format',
            id='filename_format',
        ),
    ],
)
def test_dataset_definition_get_attribute_is_deprecated(definition, attribute):
    with pytest.warns(DeprecationWarning):
        getattr(definition, attribute)


@pytest.mark.parametrize(
    ('definition', 'attribute', 'value'),
    [
        pytest.param(
            DatasetDefinition(),
            'filename_format',
            {'gaze': 'test.csv'},
            id='filename_format',
        ),
    ],
)
def test_dataset_definition_set_attribute_is_deprecated(definition, attribute, value):
    with pytest.warns(DeprecationWarning):
        setattr(definition, attribute, value)


@pytest.mark.parametrize(
    'attribute',
    [
        'filename_format',
        'filename_format_schema_overrides',
    ],
)
def test_dataset_definition_get_attribute_is_removed(attribute):
    definition = DatasetDefinition()
    with pytest.raises(DeprecationWarning) as info:
        getattr(definition, attribute)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'DatasetDefinition.{attribute} was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )
