# Copyright (c) 2026 The pymovements Project Authors
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
"""Test pymovements sources utilities."""
from pathlib import Path

import pytest

from pymovements._utils._sources import add_source
from pymovements._utils._sources import merge_sources
from pymovements._utils._sources import relativize_sources


@pytest.mark.parametrize('metadata', [None, {}])
def test_add_source_empty_metadata(metadata, tmp_path):
    filepath = tmp_path / 'sub_1.csv'
    assert add_source(metadata, filepath) == {'sources': [filepath.resolve().as_posix()]}


@pytest.mark.parametrize('file', [str, Path])
def test_add_source_str_and_path(file, tmp_path):
    filepath = file(tmp_path / 'sub_1.csv')
    assert add_source({}, filepath) == {'sources': [Path(filepath).resolve().as_posix()]}


def test_add_source_does_not_mutate_passed_metadata(tmp_path):
    metadata = {'subject_id': 1}
    result = add_source(metadata, tmp_path / 'sub_1.csv')

    assert metadata == {'subject_id': 1}
    assert result == {
        'subject_id': 1,
        'sources': [(tmp_path / 'sub_1.csv').resolve().as_posix()],
    }


def test_add_source_respects_existing_sources(tmp_path):
    metadata = {'sources': ['my/custom/source.csv']}
    result = add_source(metadata, tmp_path / 'sub_1.csv')
    assert result['sources'] == ['my/custom/source.csv']


def test_add_source_ignores_non_path_file():
    assert add_source({'key': 'value'}, object()) == {'key': 'value'}


def test_relativize_sources_below_root(tmp_path):
    metadata = {'sources': [(tmp_path / 'raw' / 'sub_1.csv').resolve().as_posix()]}
    relativize_sources(metadata, tmp_path)
    assert metadata['sources'] == ['raw/sub_1.csv']


def test_relativize_sources_outside_root_kept(tmp_path):
    source = (tmp_path / 'elsewhere' / 'sub_1.csv').resolve().as_posix()
    metadata = {'sources': [source]}
    relativize_sources(metadata, tmp_path / 'dataset')
    assert metadata['sources'] == [source]


def test_relativize_sources_normalizes_path_entries(tmp_path):
    metadata = {
        'sources': [
            (tmp_path / 'raw' / 'sub_1.csv').resolve(),
            (tmp_path / 'elsewhere' / 'sub_1.csv').resolve(),
        ],
    }
    relativize_sources(metadata, tmp_path / 'raw')
    assert metadata['sources'] == [
        'sub_1.csv',
        (tmp_path / 'elsewhere' / 'sub_1.csv').resolve().as_posix(),
    ]


def test_relativize_sources_raises_on_non_list_sources(tmp_path):
    metadata = {'sources': 'raw/sub_1.csv'}
    with pytest.raises(TypeError) as excinfo:
        relativize_sources(metadata, tmp_path)
    assert str(excinfo.value) == (
        "metadata['sources'] must be a list of path strings "
        "but is of type str: 'raw/sub_1.csv'"
    )


def test_relativize_sources_raises_on_non_path_entry(tmp_path):
    metadata = {'sources': ['raw/sub_1.csv', 123]}
    with pytest.raises(TypeError) as excinfo:
        relativize_sources(metadata, tmp_path)
    assert str(excinfo.value) == (
        "metadata['sources'] entries must be path strings "
        'but found entry of type int: 123'
    )


@pytest.mark.parametrize('metadata', [None, {}, {'sources': []}, {'key': 'value'}])
def test_relativize_sources_no_sources_noop(metadata, tmp_path):
    metadata_pre = dict(metadata) if metadata else metadata
    relativize_sources(metadata, tmp_path)
    assert metadata == metadata_pre


def test_merge_sources_appends_and_deduplicates():
    metadata = {'sources': ['a.csv', 'b.csv']}
    merge_sources(metadata, {'sources': ['b.csv', 'c.csv']})
    assert metadata['sources'] == ['a.csv', 'b.csv', 'c.csv']


def test_merge_sources_into_metadata_without_sources():
    metadata = {'key': 'value'}
    merge_sources(metadata, {'sources': ['a.csv']})
    assert metadata == {'key': 'value', 'sources': ['a.csv']}


@pytest.mark.parametrize('other', [None, {}, {'sources': []}])
def test_merge_sources_without_other_sources_noop(other):
    metadata = {'key': 'value'}
    merge_sources(metadata, other)
    assert metadata == {'key': 'value'}
