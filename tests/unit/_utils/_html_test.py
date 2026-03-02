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
"""Test pymovements HTML representations."""

from pathlib import Path
import re

import polars as pl
import pytest

from pymovements._utils import _html
from pymovements.dataset.resources import _HasResourcesIndexer
from pymovements.dataset.resources import ResourceDefinitions

DATAFRAME = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})


class Foo:
    """Test class for HTML representation."""

    def __init__(self, a: int, b: str) -> None:
        self.a = a
        self.b = b
        self._private = 'private'  # Should be excluded from the HTML representation

    @property
    def working_property(self) -> str:
        """Properties should be included in the HTML representation."""
        return f'{self.a} {self.b}'

    @property
    def failing_property(self) -> None:
        """Properties that raise an error should be excluded from the HTML representation."""
        raise RuntimeError()

    def method(self) -> None:
        """All methods should be excluded from the HTML representation."""


class ShortRepr:
    """Object with a short repr (< 50 chars)."""

    def __repr__(self) -> str:  # noqa: D105 - missing docstring
        return 'ShortRepr(1)'


class LongRepr:
    """Object with a long repr (>= 50 chars)."""

    def __repr__(self) -> str:  # noqa: D105 - missing docstring
        return 'X' * 60


@pytest.mark.parametrize(
    ('cls', 'attrs', 'init_args', 'init_kwargs', 'expected_html'),
    [
        pytest.param(
            Foo,
            None,
            (123, 'test'),
            {},
            r'<span class="pymovements-section-title">Foo</span>\s*'
            r'<ul class="pymovements-section-list">\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">a:</span>\s*'
            r'<div class="pymovements-section-inline-details">123</div>\s*'
            r'</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">b:</span>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;test&#x27;</div>\s*'
            r'</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">working_property:</span>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;123 test&#x27;</div>\s*'
            r'</li>\s*'
            r'</ul>\s*',
            id='all_attrs',
        ),
        pytest.param(
            Foo,
            ['a'],
            (123, 'test'),
            {},
            r'<span class="pymovements-section-title">Foo</span>\s*'
            r'<ul class="pymovements-section-list">\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">a:</span>\s*'
            r'<div class="pymovements-section-inline-details">123</div>\s*'
            r'</li>\s*'
            r'</ul>\s*',
            id='one_attr',
        ),
    ],
)
def test_html_repr(cls, attrs, init_args, init_kwargs, expected_html):
    # Apply decorator
    cls = _html.repr_html(attrs)(cls)
    # Create instance of the class
    obj = cls(*init_args, **init_kwargs)
    # Get HTML representation
    html = obj._repr_html_()
    assert re.search(expected_html, html, re.MULTILINE)


@pytest.mark.parametrize(
    ('obj', 'expected_inline', 'expected_expandable'),
    [
        pytest.param(
            'abc\ndef',
            '&#x27;abc\ndef&#x27;',
            False,
            id='string_short',
        ),
        pytest.param(
            'x' * 100,
            '&#x27;' + 'x' * 50 + '...&#x27;',
            True,
            id='long_short',
        ),
        pytest.param(
            [1, 2, 3],
            'list (3 items)',
            True,
            id='list',
        ),
        pytest.param(
            (1, 2, 3),
            'tuple (3 items)',
            True,
            id='tuple',
        ),
        pytest.param(
            {'a': 1, 'b': 2},
            'dict (2 items)',
            True,
            id='dict',
        ),
        pytest.param(
            DATAFRAME,
            'DataFrame (2 columns, 2 rows)',
            True,
            id='dataframe',
        ),
        # branch: elif len(repr(obj)) < 50
        pytest.param(
            ShortRepr(),
            'ShortRepr(1)',
            True,
            id='short_repr_branch',
        ),
        # branch: else -> type(obj).__name__
        pytest.param(
            LongRepr(),
            'LongRepr',
            True,
            id='long_repr_else_branch',
        ),
        # branch: elif repr(obj) in ['True', 'False']
        pytest.param(
            _HasResourcesIndexer(ResourceDefinitions([{'content': 'test', 'filename': 'f'}])),
            'True',
            False,
            id='boolean_like_true',
        ),
        pytest.param(
            _HasResourcesIndexer(ResourceDefinitions([])),
            'False',
            False,
            id='boolean_like_false',
        ),
    ],
)
def test_attr_inline_details_html(obj, expected_inline, expected_expandable):
    inline, expandable = _html._attr_inline_details_html(obj)
    assert inline == expected_inline
    assert expandable == expected_expandable


@pytest.mark.parametrize(
    ('obj', 'expected_html', 'regex'),
    [
        pytest.param(
            'abc\ndef',
            '&#x27;abc\\ndef&#x27;',
            False,
            id='string',
        ),
        pytest.param(
            [1, 2],
            '<ul><li>1</li><li>2</li></ul>',
            False,
            id='list_short',
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            '<ul><li>1</li><li>2</li><li>(3 more)</li></ul>',
            False,
            id='list_long',
        ),
        pytest.param(
            [1, [2, [3, [4, [5]]]]],
            '<ul>'
            '<li>1</li>'
            '<li><ul><li>2</li>'
            '<li><ul><li>3</li>'
            '<li>[4, [5]]</li>'
            '</ul></li></ul></li></ul>',
            False,
            id='list_deep',
        ),
        pytest.param(
            {'a': 1, 'b': 2},
            r'^<ul class="pymovements-section-list">\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">a:</span>\s*'
            r'<div class="pymovements-section-inline-details">1</div>\s*</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">b:</span>\s*'
            r'<div class="pymovements-section-inline-details">2</div>\s*'
            r'</li>\s*'
            r'</ul>$',
            True,
            id='dict_short',
        ),
        pytest.param(
            {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
            r'^<ul class="pymovements-section-list">\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">a:</span>\s*'
            r'<div class="pymovements-section-inline-details">1</div>\s*</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">b:</span>\s*'
            r'<div class="pymovements-section-inline-details">2</div>\s*'
            r'</li>\s*'
            r'<li>\(3 more\)</li>\s*'
            r'</ul>$',
            True,
            id='dict_long',
        ),
        pytest.param(
            DATAFRAME,
            DATAFRAME._repr_html_(),
            False,
            id='dataframe',
        ),
    ],
)
def test_attr_details_html(obj, expected_html, regex):
    html = _html._attr_details_html(obj)
    if regex:
        assert re.search(expected_html, html, re.MULTILINE)
    else:
        assert html == expected_html


@pytest.mark.parametrize(
    ('name', 'obj', 'expected_html'),
    [
        pytest.param(
            'none_attr',
            None,
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">none_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">None</div>\s*'
            r'</li>',
            id='none',
        ),
        pytest.param(
            'int_attr',
            123,
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">int_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">123</div>\s*'
            r'</li>',
            id='int',
        ),
        pytest.param(
            'float_attr',
            1.23,
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">float_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">1.23</div>\s*'
            r'</li>',
            id='float',
        ),
        pytest.param(
            'bool_attr',
            True,
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">bool_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">True</div>\s*'
            r'</li>',
            id='bool',
        ),
        pytest.param(
            'path_attr',
            Path('data/ToyDataset'),
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">path_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">'
            r'(PosixPath|WindowsPath)\(&#x27;data/ToyDataset&#x27;\)</div>\s*'
            r'</li>',
            id='path',
        ),
        pytest.param(
            'type_attr',
            int,
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">type_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">&lt;class &#x27;int&#x27;&gt;'
            r'</div>\s*'
            r'</li>',
            id='type',
        ),
        pytest.param(
            'empty_list_attr',
            [],
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">empty_list_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">list \(0 items\)</div>\s*'
            r'</li>',
            id='empty_list',
        ),
        pytest.param(
            'empty_tuple_attr',
            (),
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">empty_tuple_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">tuple \(0 items\)</div>\s*'
            r'</li>',
            id='empty_tuple',
        ),
        pytest.param(
            'empty_dict_attr',
            {},
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">empty_dict_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">dict \(0 items\)</div>\s*'
            r'</li>',
            id='empty_dict',
        ),
        pytest.param(
            'short_str_attr',
            'abc',
            r'<li class="pymovements-section">\s*'
            r'<span class="pymovements-section-label-empty">short_str_attr:</span>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;abc&#x27;</div>\s*'
            r'</li>',
            id='short_str',
        ),
        pytest.param(
            'long_str_attr',
            'x' * 100,
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-.*" class="pymovements-section-toggle" type="checkbox">\s*'
            r'<label for="pymovements-.*" class="pymovements-section-label">long_str_attr:'
            r'</label>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;'
            + 'x'
            * 50
            + r'\.\.\.&#x27;</div>\s*'
            r'<div class="pymovements-section-details">&#x27;' + 'x' * 100 + r'&#x27;</div>\s*'
            r'</li>',
            id='long_str',
        ),
        pytest.param(
            'list_attr',
            [1, 2, 3],
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-.*" class="pymovements-section-toggle" type="checkbox">\s*'
            r'<label for="pymovements-.*" class="pymovements-section-label">list_attr:</label>\s*'
            r'<div class="pymovements-section-inline-details">list \(3 items\)</div>\s*'
            r'<div class="pymovements-section-details">.*</div>\s*'
            r'</li>',
            id='list',
        ),
    ],
)
def test_attr_html(name, obj, expected_html):
    html = _html._attr_html(name, obj)
    assert re.search(expected_html, html, re.MULTILINE)
