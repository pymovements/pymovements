# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Test file fixtures."""
import filecmp
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def test_testfiles_dirpath_has_files(testfiles_dirpath):
    assert len(list(testfiles_dirpath.iterdir())) > 0


@pytest.mark.parametrize(
    'filename',
    [
        'eyelink_binocular_example.asc',
        'monocular_example.feather',
        'judo1000_example.csv',
        'potec_word_aoi_b0.tsv',
        'rda_test_file.rda',
    ],
)
def test_make_example_file_returns_copy(filename, make_example_file, testfiles_dirpath):
    fixture_filepath = make_example_file(filename)
    testfiles_filepath = testfiles_dirpath / filename

    assert fixture_filepath != testfiles_filepath  # different filepath
    assert filecmp.cmp(fixture_filepath, testfiles_filepath)  # same content


def test_make_text_file_accepts_relative_path_object(make_text_file):
    p = make_text_file(Path('nested') / 'custom.txt', header='H', body='B')
    assert p.name == 'custom.txt'
    assert p.parent.name == 'nested'
    print(f"dir in parent: {p.parent}")
    assert p.exists()
    assert p.read_text(encoding='utf-8') == 'HB'


def test_make_text_file_rejects_non_pathlike(make_text_file):
    with pytest.raises(TypeError, match='filename must be a str or Path, got int'):
        make_text_file(123, header='h', body='b')


def test_make_text_file_writes_concatenated_content(make_text_file):
    header = 'HEADER LINE\nSECOND\n'
    body = 'BODY LINE\nEND'
    p = make_text_file('custom.txt', header=header, body=body)
    assert p.name == 'custom.txt'
    assert p.exists()
    assert p.read_text(encoding='utf-8') == header + body


def test_make_text_file_defaults(make_text_file):
    p = make_text_file('default.txt')
    assert p.exists()
    # default header is '' and default body is '\n'
    assert p.read_text(encoding='utf-8') == '\n'


def test_make_text_file_non_utf8_encoding(make_text_file):
    header = 'über header\n(seit 2017 gibt es ein großgeschriebenes ß: NĀMLICH ẞ 🤷\n'
    body = 'Le type naïf de la fête de l\'été'
    p = make_text_file('utf16.txt', header=header, body=body, encoding='utf-16')
    assert p.read_text(encoding='utf-16') == header + body


def test_make_text_file_overwrites_existing(make_text_file):
    p1 = make_text_file('same.txt', header='A', body='B')
    p2 = make_text_file('same.txt', header='C', body='D')
    assert p2 == p1
    assert p1.read_text(encoding='utf-8') == 'CD'


def test_make_text_file_rejects_absolute_paths(make_text_file):
    with pytest.raises(ValueError, match='relative path'):
        make_text_file(Path('/absolute.txt'), header='h', body='b')
    with pytest.raises(ValueError, match='relative path'):
        make_text_file('/absolute.txt', header='h', body='b')


def test_make_text_file_rejects_tilde_home(make_text_file):
    with pytest.raises(ValueError, match=r"~\' \(home\) is not allowed|relative path"):
        make_text_file('~/.secret.txt', header='h', body='b')


@pytest.mark.parametrize(
    ('filename', 'data', 'header', 'kwargs', 'read_kwargs'),
    [
        pytest.param(
            'test.csv', pl.DataFrame({'a': [1, 2, 3]}), None, None, None,
            id='no_header_single_column_no_kwargs',
        ),

        pytest.param(
            'test.csv', pl.DataFrame({'a': [1, 2, 3], 'b': ['A', 'B', 'C']}), None, None, None,
            id='no_header_two_columns_no_kwargs',
        ),

        pytest.param(
            'test.csv', pl.DataFrame({'a': [1, 2, 3]}), '# test header', None, {'skip_lines': 1},
            id='single_line_header_single_column_no_kwargs',
        ),

        pytest.param(
            'test.csv', pl.DataFrame({'a': [1, 2, 3]}), None,
            {'separator': '\t'}, {'separator': '\t'},
            id='no_header_single_column_tab_separated',
        ),
    ],
)
def test_make_csv_file(filename, data, header, kwargs, read_kwargs, make_csv_file):
    if kwargs is not None:
        filepath = make_csv_file(filename, data, header=header, **kwargs)
    else:
        filepath = make_csv_file(filename, data, header=header)

    if header is not None:
        with open(filepath, encoding='utf-8') as opened_file:
            header_lines = len(header.split('\n'))
            written_header = ''.join([next(opened_file) for _ in range(header_lines)])
    else:
        written_header = None

    # read dataframe contents csv file
    if read_kwargs is not None:
        written_data = pl.read_csv(filepath, **read_kwargs)
    else:
        written_data = pl.read_csv(filepath)

    assert filepath.name == filename
    if header is None:
        assert written_header is None
    else:
        assert written_header == header + '\n'
    assert_frame_equal(written_data, data)
