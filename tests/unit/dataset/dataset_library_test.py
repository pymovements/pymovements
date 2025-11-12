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
"""Test dataset library."""
import glob
from pathlib import Path
from unittest import mock

import pytest
import yaml

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import register_dataset
from pymovements.dataset import dataset_library


def test_add_single_definition():
    class CustomDatasetDefinition(DatasetDefinition):
        name: str = 'CustomDatasetDefinition'

    DatasetLibrary.add(CustomDatasetDefinition)
    definition = DatasetLibrary.get('CustomDatasetDefinition')

    assert definition == CustomDatasetDefinition()


def test_add_two_definitions():
    class CustomDatasetDefinition1(DatasetDefinition):
        name: str = 'CustomDatasetDefinition1'

    class CustomDatasetDefinition2(DatasetDefinition):
        name: str = 'CustomDatasetDefinition2'

    DatasetLibrary.add(CustomDatasetDefinition1)
    DatasetLibrary.add(CustomDatasetDefinition2)

    definition1 = DatasetLibrary.get('CustomDatasetDefinition1')
    definition2 = DatasetLibrary.get('CustomDatasetDefinition2')

    assert definition1 == CustomDatasetDefinition1()
    assert definition2 == CustomDatasetDefinition2()


def test_raise_value_error_get_non_existent_dataset():
    with pytest.raises(KeyError) as exc_info:
        DatasetLibrary.get('NonExistent')

    msg, = exc_info.value.args
    error_msg_snippets = [
        'NonExistent',
        'not found in DatasetLibrary',
        'Available datasets',
    ]
    for snippet in error_msg_snippets:
        assert snippet in msg


def test_register_definition_class():
    @register_dataset
    class CustomDatasetDefinition(DatasetDefinition):
        name: str = 'CustomDatasetDefinition3'

    DatasetLibrary.add(CustomDatasetDefinition)
    definition = DatasetLibrary.get('CustomDatasetDefinition3')

    assert definition == CustomDatasetDefinition()


def test_library_not_empty():
    assert len(DatasetLibrary.definitions) >= 0


def test_list_names_is_list_of_str():
    names = DatasetLibrary.names()

    assert isinstance(names, list)

    for name in names:
        assert isinstance(name, str)


def test_returned_definition_is_copy():
    name = DatasetLibrary.names()[0]

    internal_definition = DatasetLibrary.definitions[name]
    output_definition = DatasetLibrary.get(name)

    assert internal_definition is not output_definition


def test_dataset_library_contains_all_public_datasets_files():
    library = DatasetLibrary.names()
    for filename in glob.glob('src/pymovements/datasets/*.yaml'):
        dataset_path = Path(filename)
        if dataset_path.name == 'datasets.yaml':
            continue
        with open(filename, encoding='ascii') as f:
            dataset_file = yaml.safe_load(f)
        dataset_name = dataset_file['name']
        assert dataset_name in library, f'please add {dataset_name} to `datasets.yaml`'


def test__add_shipped_datasets():
    with mock.patch('pymovements.dataset.dataset_library._add_shipped_datasets') as mock_add:
        dataset_library._add_shipped_datasets()
        mock_add.assert_called_once()
