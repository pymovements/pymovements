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
"""Test pymovements string utilities."""
from __future__ import annotations

import re

import pytest

from pymovements._utils._strings import curly_to_regex


@pytest.mark.parametrize(
    ('pattern', 'expected_regex'),
    [
        pytest.param(
            r'{subject_id:d}_{session_name}.csv',
            re.compile(r'(?P<subject_id>[0-9]+)_(?P<session_name>.+).csv'),
            id='test_basic_pattern',
        ),
        pytest.param(
            r'',
            re.compile(r''),
            id='test_empty_string',
        ),
        pytest.param(
            r'test',
            re.compile(r'test'),
            id='test_no_curly_braces',
        ),
        pytest.param(
            r'{test}',
            re.compile(r'(?P<test>.+)'),
            id='test_one_curly_brace',
        ),
        pytest.param(
            r'{t3ST}',
            re.compile(r'(?P<t3ST>.+)'),
            id='test_various_characters_as_name',
        ),
        pytest.param(
            r'{test1}_{test2}',
            re.compile(r'(?P<test1>.+)_(?P<test2>.+)'),
            id='test_two_curly_braces',
        ),
        pytest.param(
            r'{{{test1}}}',
            re.compile(r'{(?P<test1>.+)}'),
            id='test_nested_curly_braces',
        ),
        pytest.param(
            r'{{test1}}',
            re.compile(r'{test1}'),
            id='test_escaped_curly_braces',
        ),
        pytest.param(
            r'{num:s}',
            re.compile(r'(?P<num>.+)'),
            id='test_explicit_string',
        ),
        pytest.param(
            r'{num:d}',
            re.compile(r'(?P<num>[0-9]+)'),
            id='test_numbers',
        ),
        pytest.param(
            r'{num:42}',
            re.compile(r'(?P<num>.{42})'),
            id='test_quantities',
        ),
        pytest.param(
            r'{num:5d}',
            re.compile(r'(?P<num>[0-9]{5})'),
            id='test_quantities_2',
        ),
    ],
)
def test_curly_to_regex(pattern, expected_regex):
    assert curly_to_regex(pattern) == expected_regex
