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
"""Test dataset resources."""
import re

import pytest

from pymovements import __version__
from pymovements import ResourceDefinition
from pymovements import ResourceDefinitions


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
            },
            id='gaze_content_filename',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
            },
            id='gaze_content_filename_url',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'mirrors': ['https://mirror.com'],
            },
            id='gaze_content_filename_mirror',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'md5': 'abcdefgh',
            },
            id='gaze_content_filename_url_md5',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': '{subject_id}.csv',
            },
            id='gaze_content_filename_pattern',
        ),
    ],
)
def test_resource_is_equal(kwargs):
    assert ResourceDefinition(**kwargs) == ResourceDefinition(**kwargs)


@pytest.mark.parametrize(
    ('resource1', 'resource2'),
    [
        pytest.param(
            ResourceDefinition(content='gaze', filename='test1.csv'),
            ResourceDefinition(content='gaze', filename='test2.csv'),
            id='different_filename',
        ),

        pytest.param(
            ResourceDefinition(content='gaze', filename='test.csv'),
            ResourceDefinition(content='precomputed_events', filename='test.csv'),
            id='different_content',
        ),

        pytest.param(
            ResourceDefinition(content='gaze', filename='test.csv', url='https://example.com'),
            ResourceDefinition(content='gaze', filename='test.csv', url='https://examples.com'),
            id='different_url',
        ),

        pytest.param(
            ResourceDefinition(
                content='gaze', filename='test.csv', url='https://example.com',
                mirrors=['https://this.mirror.com'],
            ),
            ResourceDefinition(
                content='gaze', filename='test.csv', url='https://examples.com',
                mirrors=['https://that.mirror.com'],
            ),
            id='different_mirror',
        ),

        pytest.param(
            ResourceDefinition(
                content='gaze', filename='test.csv', url='https://example.com',
                md5='abcdefgh',
            ),
            ResourceDefinition(
                content='gaze', filename='test.csv', url='https://example.com',
                md5='ijklmnop',
            ),
            id='different_md5',
        ),

        pytest.param(
            ResourceDefinition(
                content='gaze',
                filename_pattern='{subject_id}.csv',
            ),
            ResourceDefinition(
                content='gaze',
                filename_pattern='{task_id}.csv',
            ),
            id='different_filename_pattern',
        ),

        pytest.param(
            ResourceDefinition(
                content='gaze',
                load_function='from_csv',
            ),
            ResourceDefinition(
                content='gaze',
                load_function='from_asc',
            ),
            id='different_load_function',
        ),

        pytest.param(
            ResourceDefinition(
                content='gaze',
                load_function='from_csv',
                load_kwargs={'pixel_columns': ['x', 'y']},
            ),
            ResourceDefinition(
                content='gaze',
                load_function='from_csv',
                load_kwargs={'pixel_columns': ['xr', 'yr', 'xl', 'yl']},
            ),
            id='different_load_kwargs',
        ),
    ],
)
def test_resource_is_not_equal(resource1, resource2):
    assert resource1 != resource2


@pytest.mark.parametrize(
    ('resource_dict', 'expected_resource'),
    [
        pytest.param(
            {
                'content': 'gaze',
            },
            ResourceDefinition(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
            ),
            id='content',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
            },
            ResourceDefinition(
                content='gaze',
                filename='test.csv',
                url=None,
                md5=None,
            ),
            id='content_filename',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
            },
            ResourceDefinition(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5=None,
            ),
            id='content_filename_url',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'mirrors': ['https://mirror.com'],
            },
            ResourceDefinition(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                mirrors=['https://mirror.com'],
                md5=None,
            ),
            id='content_filename_url_mirror',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'md5': 'abcdefgh',
            },
            ResourceDefinition(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5='abcdefgh',
            ),
            id='content_filename_url_md5',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'resource': 'https://example.com',
            },
            ResourceDefinition(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5=None,
            ),
            marks=pytest.mark.filterwarnings('ignore::DeprecationWarning'),
            id='deprecated_resource_key',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': 'test.csv',
            },
            ResourceDefinition(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
                filename_pattern='test.csv',
            ),
            id='filename_pattern',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': '{subject_id:d}.csv',
                'filename_pattern_schema_overrides': {'subject_id': int},
            },
            ResourceDefinition(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
                filename_pattern='{subject_id:d}.csv',
                filename_pattern_schema_overrides={'subject_id': int},
            ),
            id='filename_pattern_schema_overrides',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': '{subject_id:d}.csv',
                'load_function': 'from_csv',
                'load_kwargs': {'pixel_columns': ['x', 'y']},
            },
            ResourceDefinition(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
                filename_pattern='{subject_id:d}.csv',
                load_function='from_csv',
                load_kwargs={'pixel_columns': ['x', 'y']},
            ),
            id='load_function_and_kwargs',
        ),
    ],
)
def test_resource_from_dict_expected(resource_dict, expected_resource):
    assert ResourceDefinition.from_dict(resource_dict) == expected_resource


@pytest.mark.filterwarnings('ignore:.*from_dicts.*:DeprecationWarning')
@pytest.mark.parametrize(
    ('init_resources', 'expected_resources'),
    [
        pytest.param(
            None,
            ResourceDefinitions(),
            id='none',
        ),

        pytest.param(
            {
                'gaze': None,
            },
            ResourceDefinitions(),
            id='none_gaze_list',
        ),

        pytest.param(
            {
                'gaze': [],
            },
            ResourceDefinitions(),
            id='empty_gaze_list',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile.txt'}],
            },
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myfile.txt', content='gaze'),
                ],
            ),
            id='single_gaze_resource',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile1.zip'}, {'filename': 'myfile2.zip'}],
            },
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myfile1.zip', content='gaze'),
                    ResourceDefinition(filename='myfile2.zip', content='gaze'),
                ],
            ),
            id='two_gaze_resources',
        ),

        pytest.param(
            {
                'precomputed_events': [{'filename': 'myevents.csv'}],
            },
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myevents.csv', content='precomputed_events'),
                ],
            ),
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            {
                'precomputed_reading_measures': [{'filename': 'reading_measures.csv'}],
            },
            ResourceDefinitions(
                [
                    ResourceDefinition(
                        filename='reading_measures.csv',
                        content='precomputed_reading_measures',
                    ),
                ],
            ),
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_from_dict_expected(init_resources, expected_resources):
    assert ResourceDefinitions.from_dict(init_resources) == expected_resources


def test_resource_definitions_from_dict_deprecated():
    resources_dict = {'gaze': [{'filename': 'myfile.txt'}]}

    with pytest.raises(DeprecationWarning) as info:
        ResourceDefinitions.from_dict(resources_dict)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'ResourceDefinitions.from_dict() was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )


@pytest.mark.parametrize(
    ('resources', 'expected_dicts'),
    [
        pytest.param(
            ResourceDefinitions(),
            [],
            id='default',
        ),

        pytest.param(
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myfile.txt', content='gaze'),
                ],
            ),
            [
                {'filename': 'myfile.txt', 'content': 'gaze'},
            ],
            id='single_gaze_resource',
        ),

        pytest.param(
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myfile1.zip', content='gaze'),
                    ResourceDefinition(filename='myfile2.zip', content='gaze'),
                ],
            ),
            [
                {'filename': 'myfile1.zip', 'content': 'gaze'},
                {'filename': 'myfile2.zip', 'content': 'gaze'},
            ],
            id='two_gaze_resources',
        ),

        pytest.param(
            ResourceDefinitions(
                [
                    ResourceDefinition(filename='myevents.csv', content='precomputed_events'),
                ],
            ),
            [
                {'filename': 'myevents.csv', 'content': 'precomputed_events'},
            ],
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            ResourceDefinitions(
                [
                    ResourceDefinition(
                        filename='reading_measures.csv',
                        content='precomputed_reading_measures',
                    ),
                ],
            ),
            [
                {
                    'filename': 'reading_measures.csv',
                    'content': 'precomputed_reading_measures',
                },
            ],
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_to_dicts_expected(resources, expected_dicts):
    assert resources.to_dicts() == expected_dicts


@pytest.mark.parametrize(
    ('resources', 'content_type', 'expected_resources'),
    [
        pytest.param(
            ResourceDefinitions(),
            None,
            [],
            id='default_filter_none',
        ),

        pytest.param(
            ResourceDefinitions(),
            'gaze',
            [],
            id='default_filter_gaze',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'gaze',
            [
                ResourceDefinition(filename='myfile.txt', content='gaze'),
            ],
            id='single_gaze_filter_gaze',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'precomputed_events',
            [],
            id='single_gaze_filter_precomputed_events',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts(
                [{'filename': 'events.csv', 'content': 'precomputed_events'}],
            ),
            'precomputed_events',
            [
                ResourceDefinition(filename='events.csv', content='precomputed_events'),
            ],
            id='single_precomputed_events_filter_precomputed_events',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts(
                [{'filename': 'events.csv', 'content': 'precomputed_events'}],
            ),
            'gaze',
            [],
            id='single_precomputed_events_filter_gaze',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            None,
            [
                ResourceDefinition(filename='myfile.txt', content='gaze'),
                ResourceDefinition(filename='events.csv', content='precomputed_events'),
            ],
            id='gaze_and_precomputed_events_filter_none',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            'gaze',
            [
                ResourceDefinition(filename='myfile.txt', content='gaze'),
            ],
            id='gaze_and_precomputed_events_filter_gaze',
        ),

        pytest.param(
            ResourceDefinitions.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            'precomputed_events',
            [
                ResourceDefinition(filename='events.csv', content='precomputed_events'),
            ],
            id='gaze_and_precomputed_events_filter_precomputed_events',
        ),

    ],
)
def test_resources_filter_expected(resources, content_type, expected_resources):
    assert resources.filter(content_type) == expected_resources


@pytest.mark.parametrize(
    ('resources', 'expected_has_content'),
    [
        pytest.param(
            ResourceDefinitions(),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='default',
        ),

        pytest.param(
            ResourceDefinitions([]),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='empty_list',
        ),

        pytest.param(
            ResourceDefinitions([ResourceDefinition(content='gaze')]),
            {
                'gaze': True,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='gaze_resources',
        ),

        pytest.param(
            ResourceDefinitions([ResourceDefinition(content='precomputed_events')]),
            {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
            id='precomputed_event_resources',
        ),

        pytest.param(
            ResourceDefinitions([ResourceDefinition(content='precomputed_reading_measures')]),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
            id='precomputed_reading_measures_resources',
        ),

        pytest.param(
            ResourceDefinitions(
                [
                    ResourceDefinition(content='gaze'),
                    ResourceDefinition(content='precomputed_events'),
                    ResourceDefinition(content='precomputed_reading_measures'),
                ],
            ),
            {
                'gaze': True,
                'precomputed_events': True,
                'precomputed_reading_measures': True,
            },
            id='all_resources',
        ),

        pytest.param(
            ResourceDefinitions([ResourceDefinition(content='foo')]),
            {
                'foo': True,
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='custom_content',
        ),
    ],
)
def test_resources_has_content_expected(resources, expected_has_content):
    for key, value in expected_has_content.items():
        assert resources.has_content(key) == value, (resources, value)


@pytest.mark.parametrize(
    ('dicts', 'expected_resources'),
    [
        pytest.param(
            None,
            [],
            id='none',
        ),

        pytest.param(
            [],
            [],
            id='empty_list',
        ),

        pytest.param(
            [{'filename': 'myfile.txt', 'content': 'gaze'}],
            [
                ResourceDefinition(filename='myfile.txt', content='gaze'),
            ],
            id='single_resource',
        ),

        pytest.param(
            [
                {'filename': 'myfile.txt', 'content': 'gaze'},
                {'filename': 'events.csv', 'content': 'precomputed_events'},
            ],
            [
                ResourceDefinition(filename='myfile.txt', content='gaze'),
                ResourceDefinition(filename='events.csv', content='precomputed_events'),
            ],
            id='two_resources',
        ),
    ],
)
def test_resources_from_dicts_expected(dicts, expected_resources):
    assert ResourceDefinitions.from_dicts(dicts) == expected_resources


def test_resource_definition_from_dict_resource_key_deprecated():
    resource_dict = {'content': 'samples', 'resource': 'http://www.example.com'}
    with pytest.raises(DeprecationWarning) as info:
        ResourceDefinition.from_dict(resource_dict)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'from_dict() key "resources" was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )
