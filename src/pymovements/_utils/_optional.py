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
"""Provides the import of optional dependencies."""
from __future__ import annotations

import importlib.metadata
import re
from types import ModuleType

# Keep in sync with the extras in pyproject.toml.
MINIMUM_VERSIONS = {
    'fastexcel': '0.13.0',
    'pyreadr': '0.5.2',
}


def import_optional_dependency(name: str, extra: str) -> ModuleType:
    """Import an optional dependency or raise an ``ImportError`` with an install hint.

    Parameters
    ----------
    name: str
        Name of the module to import.
    extra: str
        Name of the pymovements extra that installs the module.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed or its version is older than the minimum version.
    """
    try:
        module = importlib.import_module(name)
    except ImportError as exception:
        raise ImportError(
            f"Missing optional dependency '{name}'. "
            f"Install it with: pip install 'pymovements[{extra}]'",
        ) from exception

    minimum_version = MINIMUM_VERSIONS.get(name)
    if minimum_version is not None:
        installed_version = importlib.metadata.version(name)
        if _release(installed_version) < _release(minimum_version):
            raise ImportError(
                f"pymovements requires version '{minimum_version}' or newer of '{name}' "
                f"(version '{installed_version}' is installed). "
                f"Upgrade it with: pip install --upgrade 'pymovements[{extra}]'",
            )

    return module


def _release(version_string: str) -> tuple[int, ...]:
    """Return the numeric release segment of a version string, e.g. ``(1, 2, 0)``."""
    match = re.match(r'\d+(\.\d+)*', version_string)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group().split('.'))
