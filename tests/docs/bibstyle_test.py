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
"""Tests for AuthorYearLabelStyle in docs/source/conf.py."""
from __future__ import annotations

import string

import pytest
from conf import AuthorYearLabelStyle  # pylint: disable=import-error
from pybtex.database import Entry
from pybtex.database import Person


def _make_entry(last_name: str, year: str) -> Entry:
    """Create a pybtex Entry with given author last name and year."""
    return Entry('article', fields={'year': year}, persons={'author': [Person(last_name)]})


@pytest.mark.parametrize(
    ('entries_data', 'expected_labels'),
    [
        pytest.param(
            [],
            [],
            id='empty_entries',
        ),
        pytest.param(
            [('Smith', '2020')],
            ['Smith et al., 2020'],
            id='single_entry',
        ),
        pytest.param(
            [('Smith', '2020'), ('Jones', '2021'), ('Brown', '2019')],
            [
                'Smith et al., 2020',
                'Jones et al., 2021',
                'Brown et al., 2019',
            ],
            id='multiple_distinct_entries',
        ),
        pytest.param(
            [('Smith', '2020'), ('Smith', '2020')],
            ['Smith et al., 2020', 'Smith et al., 2020a'],
            id='two_duplicates',
        ),
        pytest.param(
            [('Smith', '2020'), ('Smith', '2020'), ('Smith', '2020')],
            [
                'Smith et al., 2020',
                'Smith et al., 2020a',
                'Smith et al., 2020b',
            ],
            id='three_duplicates',
        ),
        pytest.param(
            [('Smith', '2020'), ('Jones', '2021'), ('Smith', '2020')],
            [
                'Smith et al., 2020',
                'Jones et al., 2021',
                'Smith et al., 2020a',
            ],
            id='duplicates_mixed_with_unique',
        ),
        pytest.param(
            [('Smith', '2020'), ('Smith', '2021')],
            ['Smith et al., 2020', 'Smith et al., 2021'],
            id='same_author_different_years',
        ),
        pytest.param(
            [('Smith', '2020'), ('Jones', '2020')],
            ['Smith et al., 2020', 'Jones et al., 2020'],
            id='different_authors_same_year',
        ),
        pytest.param(
            [
                ('Smith', '2020'), ('Smith', '2020'), ('Jones', '2021'), ('Jones', '2021'),
            ],
            [
                'Smith et al., 2020',
                'Smith et al., 2020a',
                'Jones et al., 2021',
                'Jones et al., 2021a',
            ],
            id='multiple_duplicate_groups',
        ),
    ],
)
def test_format_labels(entries_data, expected_labels):
    style = AuthorYearLabelStyle()
    entries = [_make_entry(name, year) for name, year in entries_data]

    labels = list(style.format_labels(entries))

    assert labels == expected_labels


def test_format_labels_all_26_suffixes_exhausted():
    style = AuthorYearLabelStyle()
    entries = [_make_entry('Smith', '2020') for _ in range(27)]

    labels = list(style.format_labels(entries))

    assert labels[0] == 'Smith et al., 2020'
    for i, char in enumerate(string.ascii_lowercase):
        assert labels[i + 1] == f'Smith et al., 2020{char}'


def test_format_labels_28_duplicates_raises():
    """Test that the 28th duplicate gets no suffix when all letters are used."""
    style = AuthorYearLabelStyle()
    entries = [_make_entry('Smith', '2020') for _ in range(28)]

    with pytest.raises(Exception, match="character suffixes exhausted for 'Smith et al., 2020'"):
        list(style.format_labels(entries))
