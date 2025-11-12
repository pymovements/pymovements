# Copyright (c) 2025 The pymovements Project Authors
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
"""Test write_datasets_yaml script."""
from pathlib import Path

import pytest
import yaml

from pymovements._scripts import write_datasets_yaml


@pytest.fixture(name='make_datasets_directory')
def make_datasets_directory_fixture(tmp_path):
    def _make_datasets_directory(param: str, make_yaml: bool) -> Path:
        datasets = []
        if param in {'single', 'two'}:
            filepath = tmp_path / 'first.yaml'
            filepath.touch()
            datasets.append('first')
        if param == 'two':
            filepath = tmp_path / 'second.yaml'
            filepath.touch()
            datasets.append('second')
        if make_yaml:
            with open(tmp_path / 'datasets.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(datasets, f)
        return tmp_path

    yield _make_datasets_directory


@pytest.mark.parametrize(
    ('fixture_args', 'expected_return', 'expected_yaml'),
    [
        pytest.param(
            ['empty', False],
            1,
            [],
            id='empty_new',
        ),

        pytest.param(
            ['empty', True],
            0,
            [],
            id='empty_exist',
        ),

        pytest.param(
            ['single', False],
            1,
            ['first'],
            id='single_new',
        ),

        pytest.param(
            ['single', True],
            0,
            ['first'],
            id='single_exist',
        ),

        pytest.param(
            ['two', False],
            1,
            ['first', 'second'],
            id='two_new',
        ),

        pytest.param(
            ['two', True],
            0,
            ['first', 'second'],
            id='two_exist',
        ),
    ],
)
def test_write_datasets_yaml(
        fixture_args, expected_return, expected_yaml, make_datasets_directory,
):
    datasets_dirpath = make_datasets_directory(*fixture_args)
    datasets_yaml_filename = 'datasets.yaml'

    return_value = write_datasets_yaml.main(
        datasets_dirpath=datasets_dirpath,
        datasets_yaml_filename=datasets_yaml_filename,
    )

    with open(datasets_dirpath / datasets_yaml_filename, encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)

    assert return_value == expected_return
    assert yaml_content == expected_yaml
