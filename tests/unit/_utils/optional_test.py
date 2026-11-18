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
"""Test optional dependency imports.

The missing and outdated cases are covered through the public loaders in the dataset tests.
Only the ``ValueError`` for a name outside the extra needs the helper directly.
"""
import importlib.metadata
import re

import pytest

from pymovements._utils.optional import import_optional_dependency


@pytest.mark.parametrize(
    ('name', 'extra'),
    [
        pytest.param('numpy', 'rdata', id='core_dependency'),
        pytest.param('pyreadr', 'docs', id='other_extra'),
        pytest.param('nonexistent', 'rdata', id='unknown_name'),
    ],
)
def test_import_optional_dependency_not_in_extra_raises_value_error(name, extra):
    expected_message = f"'{name}' is not a requirement of the pymovements extra '{extra}'"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        import_optional_dependency(name, extra=extra)


def test_import_optional_dependency_conditional_core_requirement_raises_value_error(monkeypatch):
    requirements = [
        'pyreadr>=0.5.2; python_version >= "3.9"',
        'fastexcel>=0.13.0; extra == "excel"',
    ]
    monkeypatch.setattr(importlib.metadata, 'requires', lambda name: requirements)

    expected_message = "'pyreadr' is not a requirement of the pymovements extra 'rdata'"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        import_optional_dependency('pyreadr', extra='rdata')
