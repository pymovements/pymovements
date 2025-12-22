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
"""Test pymovements HTML representations."""
import re

import polars as pl
import pytest

from pymovements._utils import _html

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
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\1" class="pymovements-section-label">a:</label>\s*'
            r'<div class="pymovements-section-inline-details">123</div>\s*'
            r'<div class="pymovements-section-details">123</div>\s*'
            r'</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\2" class="pymovements-section-label">b:</label>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;test&#x27;</div>\s*'
            r'<div class="pymovements-section-details">&#x27;test&#x27;</div>\s*'
            r'</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\3" class="pymovements-section-label">working_property:'
            r'</label>\s*'
            r'<div class="pymovements-section-inline-details">&#x27;123 test&#x27;</div>\s*'
            r'<div class="pymovements-section-details">&#x27;123 test&#x27;</div>\s*'
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
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\1" class="pymovements-section-label">a:</label>\s*'
            r'<div class="pymovements-section-inline-details">123</div>\s*'
            r'<div class="pymovements-section-details">123</div>\s*'
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
    ('obj', 'expected_html'),
    [
        pytest.param(
            'abc\ndef',
            '&#x27;abc\\ndef&#x27;',
            id='string_short',
        ),
        pytest.param(
            'x' * 100,
            'str',
            id='long_short',
        ),
        pytest.param(
            [1, 2, 3],
            'list (3 items)',
            id='list',
        ),
        pytest.param(
            {'a': 1, 'b': 2},
            'dict (2 items)',
            id='dict',
        ),
        pytest.param(
            DATAFRAME,
            'DataFrame (2 columns, 2 rows)',
            id='dataframe',
        ),
    ],
)
def test_attr_inline_details_html(obj, expected_html):
    html = _html._attr_inline_details_html(obj)
    assert html == expected_html


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
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\1" class="pymovements-section-label">a:</label>\s*'
            r'<div class="pymovements-section-inline-details">1</div>\s*'
            r'<div class="pymovements-section-details">1</div>\s*</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\2" class="pymovements-section-label">b:</label>\s*'
            r'<div class="pymovements-section-inline-details">2</div>\s*'
            r'<div class="pymovements-section-details">2</div>\s*'
            r'</li>\s*'
            r'</ul>$',
            True,
            id='dict_short',
        ),
        pytest.param(
            {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
            r'^<ul class="pymovements-section-list">\s*'
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\1" class="pymovements-section-label">a:</label>\s*'
            r'<div class="pymovements-section-inline-details">1</div>\s*'
            r'<div class="pymovements-section-details">1</div>\s*</li>\s*'
            r'<li class="pymovements-section">\s*'
            r'<input id="pymovements-([0-9a-f-]+)" class="pymovements-section-toggle" '
            r'type="checkbox">\s*'
            r'<label for="pymovements-\2" class="pymovements-section-label">b:</label>\s*'
            r'<div class="pymovements-section-inline-details">2</div>\s*'
            r'<div class="pymovements-section-details">2</div>\s*'
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
