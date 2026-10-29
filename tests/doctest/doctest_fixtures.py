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
"""Fixtures for doctest support.

Modules in this directory are pytest plugins for doctest support. They are
registered as plugins so their fixtures reach the doctests in ``src/``. A
doctest requests a fixture from within a hidden ``.. testsetup::`` block via
``getfixture(...)``.
"""
from __future__ import annotations

import warnings
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(name='doctest_tmp_cwd')
def fixture_doctest_tmp_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Change the working directory of the requesting doctest to pytest's tmp path.

    Doctests that write relative paths request this fixture in a hidden
    ``.. testsetup::`` block at the start of their Examples section::

        .. testsetup::

            >>> getfixture('doctest_tmp_cwd')

    This changes the working directory of the requesting doctest to pytest's
    temporary directory, so artifacts never land in the repository. Doctests
    that read repository files via relative paths must not request it.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(name='doctest_synthetic')
def fixture_doctest_synthetic() -> Generator[None]:
    """Let the requesting doctest run synthetic generators like a user would.

    Doctests of the synthetic generation framework request this fixture in a hidden
    ``.. testsetup::`` block::

        .. testsetup::

            >>> getfixture('doctest_synthetic')

    It silences the ``ExperimentalWarning`` that every generation run emits, which the test
    configuration otherwise turns into an error, and restores the generator libraries afterwards,
    so generators registered in an example do not leak into other tests.
    """
    # Imported here: this plugin loads before pytest-cov starts, so a module-level import of
    # pymovements would hide the package's import-time lines from the coverage report.
    # pylint: disable=import-outside-toplevel
    from pymovements.synthetic import EventGeneratorLibrary
    from pymovements.synthetic import SampleGeneratorLibrary
    from pymovements.warnings import ExperimentalWarning

    event_generators = EventGeneratorLibrary.generators.copy()
    sample_generators = SampleGeneratorLibrary.generators.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ExperimentalWarning)
        yield
    EventGeneratorLibrary.generators.clear()
    EventGeneratorLibrary.generators.update(event_generators)
    SampleGeneratorLibrary.generators.clear()
    SampleGeneratorLibrary.generators.update(sample_generators)
