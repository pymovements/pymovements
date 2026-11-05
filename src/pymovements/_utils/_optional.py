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
from types import ModuleType

from packaging.requirements import Requirement


def import_optional_dependency(name: str, extra: str) -> ModuleType:
    """Import an optional dependency or raise an ``ImportError`` with an install hint.

    The minimum version is read from the requirement that the pymovements extra declares in
    ``pyproject.toml``.

    Parameters
    ----------
    name: str
        Name of the module to import. Must equal the distribution name in the extra.
    extra: str
        Name of the pymovements extra that installs the module.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed or its version does not satisfy the extra's requirement.
    ValueError
        If the extra does not declare a requirement for ``name``.
    """
    requirement = _requirement(name, extra)

    try:
        module = importlib.import_module(name)
    except ImportError as exception:
        raise ImportError(
            f"Missing optional dependency '{name}'. "
            f"Install it with: pip install 'pymovements[{extra}]'",
        ) from exception

    installed_version = importlib.metadata.version(name)
    if not requirement.specifier.contains(installed_version, prereleases=True):
        raise ImportError(
            f"pymovements requires '{requirement.name}{requirement.specifier}' "
            f"(version '{installed_version}' is installed). "
            f"Upgrade it with: pip install --upgrade 'pymovements[{extra}]'",
        )

    return module


def _requirement(name: str, extra: str) -> Requirement:
    """Return the requirement for ``name`` that the pymovements ``extra`` declares."""
    for requirement_string in importlib.metadata.requires('pymovements') or []:
        requirement = Requirement(requirement_string)
        if requirement.name != name or requirement.marker is None:
            continue
        if requirement.marker.evaluate({'extra': extra}):
            return requirement
    raise ValueError(f"'{name}' is not a requirement of the pymovements extra '{extra}'")
