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
"""Provides functions for generating HTML representations of objects for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from itertools import islice
import pathlib
from typing import TypeVar
from uuid import uuid4

import polars as pl

STYLE = """
<style>
    .pymovements-section-list {
        margin: 0;
        padding: 0;
        font-family: sans-serif;
    }
    .pymovements-section {
        list-style: none;
        padding-bottom: 0.5em;
    }
    .pymovements-section-title {
        font-size: 120%;
        font-weight: bold;
    }
    .pymovements-section-toggle {
        display: none;
    }
    .pymovements-section-label {
        cursor: pointer;
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        transition: background-color 0.1s, border-color 0.1s;
    }
    .pymovements-section-label:hover {
        background-color: rgba(0, 120, 215, 0.1);
        border-color: #0078d7;
        text-decoration: none;
    }
    .pymovements-section-label-empty {
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 0px 6px;
        border-radius: 4px;
        display: inline-block;
    }
    .pymovements-section-label:before {
        display: inline-block;
        content: "►";
        margin-right: 6px;
        color: green;
        font-size: 0.8em;
    }
    .pymovements-section-toggle:checked + .pymovements-section-label:before {
        content: "▼";
    }
    .pymovements-section-inline-details {
        display: inline-block;
    }
    .pymovements-section-details {
        display: none;
        padding-left: 1em;
    }
    .pymovements-section-toggle:checked ~ .pymovements-section-details {
        display: block;
    }
    .pymovements-section-details > ul {
        padding-left: 1em;
    }
</style>
"""


T = TypeVar('T')


def repr_html(attrs: list[str] | None = None) -> Callable[[T], T]:
    """Add an HTML representation to the class for Jupyter notebooks.

    Parameters
    ----------
    attrs : list[str] | None
        List of attributes to include in the HTML representation.
        If None, all public attributes are included.

    Returns
    -------
    Callable[[T], T]
        Decorator function that adds the HTML representation to the decorated class.
    """

    def decorator(cls: T) -> T:
        cls._repr_html_ = lambda self: _obj_html(self, attrs)  # type: ignore[attr-defined]
        return cls

    return decorator


def _obj_html(obj: object, attrs: list[str] | None = None) -> str:
    """Generate HTML representation for an object."""
    if attrs is None:
        attrs = []
        for attr in dir(obj):
            # Skip private attributes
            if attr.startswith('_'):
                continue
            # Skip properties that raise errors
            try:
                value = getattr(obj, attr)
            except BaseException:
                continue
            # Skip methods
            if not callable(value):
                attrs.append(attr)

    title = escape(type(obj).__name__)

    sections = []
    for attr in attrs:
        attr_obj = getattr(obj, attr)
        sections.append(_attr_html(attr, attr_obj))

    return f"""
    {STYLE}
    <span class="pymovements-section-title">{title}</span>
    <ul class="pymovements-section-list">
        {''.join(sections)}
    </ul>
    """


def _attr_html(name: str, obj: object, depth: int = 0) -> str:
    """Generate HTML representation for an attribute."""
    section_id = uuid4()
    name = escape(name)

    inline_details, is_expandable = _attr_inline_details_html(obj)

    if is_expandable:
        details = _attr_details_html(obj, depth=depth)
        return f"""
        <li class="pymovements-section">
            <input id="pymovements-{section_id}" class="pymovements-section-toggle" type="checkbox">
            <label for="pymovements-{section_id}" class="pymovements-section-label">{name}:</label>
            <div class="pymovements-section-inline-details">{inline_details}</div>
            <div class="pymovements-section-details">{details}</div>
        </li>
        """

    return f"""
    <li class="pymovements-section">
        <span class="pymovements-section-label-empty">{name}:</span>
        <div class="pymovements-section-inline-details">{inline_details}</div>
    </li>
    """


def _attr_inline_details_html(obj: object) -> tuple[str, bool]:
    """Generate inline (collapsed) details for HTML representation."""
    is_expandable = True

    if obj is None:
        inline_details = 'None'
        is_expandable = False

    elif isinstance(obj, pl.DataFrame):
        inline_details = f'DataFrame ({len(obj.columns)} columns, {len(obj)} rows)'

    elif isinstance(obj, list):
        inline_details = f'list ({len(obj)} items)'
        if len(obj) == 0:
            is_expandable = False

    elif isinstance(obj, tuple):
        inline_details = f'tuple ({len(obj)} items)'
        if len(obj) == 0:
            is_expandable = False

    elif isinstance(obj, dict):
        inline_details = f'dict ({len(obj)} items)'
        if len(obj) == 0:
            is_expandable = False

    elif isinstance(obj, str):
        if len(obj) < 50:
            inline_details = f"'{obj}'"
            is_expandable = False
        else:
            inline_details = f"'{obj[:50]}...'"

    elif isinstance(obj, (int, float, bool, pathlib.Path, type)):
        inline_details = repr(obj).replace('\n', ' ')
        is_expandable = False

    # boolean-like objects like `_HasResourcesIndexer`
    elif repr(obj) in {'True', 'False'}:
        inline_details = repr(obj)
        is_expandable = False

    elif len(repr(obj)) < 50:
        inline_details = repr(obj).replace('\n', ' ')

    else:
        inline_details = type(obj).__name__

    return escape(str(inline_details)), is_expandable


def _attr_details_html(obj: object, depth: int = 0, max_depth: int = 3) -> str:
    """Generate expanded details for HTML representation."""
    max_items = 2

    if isinstance(obj, (list, tuple)) and depth < max_depth:
        details = '<ul>'
        for item in obj[:max_items]:
            details += f'<li>{_attr_details_html(item, depth + 1)}</li>'
        if len(obj) > max_items:
            details += f'<li>({len(obj) - 2} more)</li>'
        details += '</ul>'

    elif isinstance(obj, dict) and depth < max_depth:
        details = '<ul class="pymovements-section-list">'
        for key in islice(obj.keys(), max_items):
            value = obj[key]
            details += _attr_html(key, value, depth + 1)
        if len(obj) > max_items:
            details += f'<li>({len(obj) - 2} more)</li>'
        details += '</ul>'

    elif hasattr(obj, '_repr_html_'):
        details = obj._repr_html_()  # pylint: disable=protected-access

    else:
        details = escape(repr(obj))

    return details
