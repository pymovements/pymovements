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
"""Test optional dependency imports."""
import importlib.metadata
import re
import sys

import pytest

from pymovements._utils._optional import import_optional_dependency


def test_import_optional_dependency_returns_module():
    assert import_optional_dependency('pyreadr', extra='datasets') is sys.modules['pyreadr']


def test_import_optional_dependency_missing_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyreadr', None)

    expected_message = (
        "Missing optional dependency 'pyreadr'. "
        "Install it with: pip install 'pymovements[datasets]'"
    )
    with pytest.raises(ImportError, match=re.escape(expected_message)):
        import_optional_dependency('pyreadr', extra='datasets')


@pytest.mark.parametrize(
    'installed_version',
    [
        pytest.param('0.4.9', id='older_minor'),
        pytest.param('0.5.1', id='older_patch'),
        pytest.param('0.5.1.post1', id='older_patch_with_suffix'),
    ],
)
def test_import_optional_dependency_too_old_raises_upgrade_hint(monkeypatch, installed_version):
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: installed_version)

    expected_message = (
        "pymovements requires version '0.5.2' or newer of 'pyreadr' "
        f"(version '{installed_version}' is installed). "
        "Upgrade it with: pip install --upgrade 'pymovements[datasets]'"
    )
    with pytest.raises(ImportError, match=re.escape(expected_message)):
        import_optional_dependency('pyreadr', extra='datasets')


@pytest.mark.parametrize(
    'installed_version',
    [
        pytest.param('0.5.2', id='equal'),
        pytest.param('0.5.10', id='newer_patch_two_digits'),
        pytest.param('1.0.0rc1', id='newer_major_prerelease'),
    ],
)
def test_import_optional_dependency_new_enough_returns_module(monkeypatch, installed_version):
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: installed_version)

    assert import_optional_dependency('pyreadr', extra='datasets') is sys.modules['pyreadr']
