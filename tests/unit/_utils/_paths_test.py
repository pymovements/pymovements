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
"""Test pymovements paths."""
import pathlib
import re
import unittest
from pathlib import Path

import pytest

from pymovements._utils._paths import get_filepaths
from pymovements._utils._paths import match_filepaths


def test_get_filepaths_mut_excl_extension_and_regex_error():
    """Test mutually exclusive extension and regex in get_filepaths."""
    with pytest.raises(ValueError) as excinfo:
        get_filepaths(path='tmp', extension='extension', regex=re.compile('regex'))
    msg, = excinfo.value.args
    assert msg == 'extension and regex are mutually exclusive'


@pytest.mark.parametrize(
    ('extension', 'regex', 'expected_paths'),
    [
        pytest.param(
            None,
            re.compile('regex'),
            [],
            id='regex_empty_dir',
        ),
        pytest.param(
            'extension',
            None,
            [],
            id='extension_empty_dir',
        ),
    ],
)
def test_get_filepaths_empty_directory(
        extension,
        regex,
        expected_paths,
):
    assert get_filepaths(path='tmp', extension=extension, regex=regex) == expected_paths


def create_directory(tmp_path, sub_dirs, files):
    for sub_dir in sub_dirs:
        dir_path = tmp_path / sub_dir
        dir_path.mkdir()
        for file_name in files:
            file_path = dir_path / file_name
            file_path.write_text('test')


@pytest.mark.parametrize(
    ('extension', 'regex', 'sub_dirs', 'files', 'expected_paths'),
    [
        pytest.param(
            '.txt',
            None,
            ['tmp_dir'],
            ['foo.txt', 'bar.py'],
            ['tmp_dir/foo.txt'],
            id='extension_regex_get_filepaths_list',
        ),
        pytest.param(
            None,
            re.compile('foo'),
            ['tmp_dir'],
            ['foo.txt', 'foo.py', 'bar.py'],
            ['tmp_dir/foo.txt', 'tmp_dir/foo.py'],
            id='regex_get_filepaths_list',
        ),
    ],
)
def test_get_filepaths_expected_output(
        extension,
        regex,
        tmp_path,
        sub_dirs,
        files,
        expected_paths,
):
    create_directory(tmp_path, sub_dirs, files)
    ret = get_filepaths(path=tmp_path, extension=extension, regex=regex)
    expected_list = [tmp_path / pathlib.Path(expected_path) for expected_path in expected_paths]
    assert sorted(ret) == sorted(expected_list)


@pytest.mark.parametrize(
    ('regex', 'sub_dirs', 'files', 'expected_dicts'),
    [
        pytest.param(
            re.compile('.*'),
            ['tmp_dir'],
            [],
            [],
            id='empty_directory_empty_list',
        ),
        pytest.param(
            re.compile('.*'),
            ['tmp_dir'],
            ['foo.txt'],
            [{'filepath': str(Path('tmp_dir/foo.txt'))}],
            id='match_all_no_groups_single_file',
        ),
        pytest.param(
            re.compile('.*'),
            ['tmp_dir'],
            ['foo.txt', 'bar.py'],
            [{'filepath': str(Path('tmp_dir/foo.txt'))}, {'filepath': str(Path('tmp_dir/bar.py'))}],
            id='match_all_no_groups_two_files',
        ),
        pytest.param(
            re.compile('foo'),
            ['tmp_dir'],
            ['foo.txt', 'bar.py'],
            [{'filepath': str(Path('tmp_dir/foo.txt'))}],
            id='match_substring_single_file_out_of_two_no_groups',
        ),
        pytest.param(
            re.compile(r'(?P<foo>\d+)_(?P<bar>\d+).ext'),
            ['a'],
            ['123_456.ext', '456_789.ext'],
            [
                {'foo': '123', 'bar': '456', 'filepath': str(Path('a/123_456.ext'))},
                {'foo': '456', 'bar': '789', 'filepath': str(Path('a/456_789.ext'))},
            ],
            id='match_groups_two_files',
        ),
    ],
)
def test_match_filepaths(
        regex,
        sub_dirs,
        files,
        expected_dicts,
        tmp_path,
):
    create_directory(tmp_path, sub_dirs, files)
    result_dicts = match_filepaths(path=tmp_path, regex=regex)

    case = unittest.TestCase()
    case.assertCountEqual(result_dicts, expected_dicts)


def test_match_filepaths_not_relative(tmp_path):
    create_directory(tmp_path, ['tmp_dir'], ['foo.txt'])
    result_dicts = match_filepaths(path=tmp_path, regex=re.compile('.*'), relative=False)

    expected_dicts = [{'filepath': str(tmp_path / 'tmp_dir/foo.txt')}]
    case = unittest.TestCase()
    case.assertCountEqual(result_dicts, expected_dicts)


def test_match_filepaths_not_exists_raises_value_error(tmp_path):
    filepath = tmp_path / 'not_existing'

    with pytest.raises(ValueError) as excinfo:
        match_filepaths(path=filepath, regex=re.compile('.*'))
    msg, = excinfo.value.args

    assert 'not exist' in msg
    assert str(filepath) in msg


def test_match_filepaths_no_directory_raises_value_error(tmp_path):
    create_directory(tmp_path, ['tmp_dir'], ['foo.txt'])
    filepath = tmp_path / 'tmp_dir' / 'foo.txt'

    with pytest.raises(ValueError) as excinfo:
        match_filepaths(path=filepath, regex=re.compile('.*'))
    msg, = excinfo.value.args

    assert 'must point to a directory' in msg
    assert str(filepath) in msg
