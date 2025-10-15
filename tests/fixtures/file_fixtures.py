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
"""Provide fixtures to securely make files from examples in ``tests/files``."""
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(name='testfiles_dirpath', scope='session')
def fixture_testfiles_dirpath(request):
    """Return the path to tests/files."""
    return request.config.rootpath / 'tests' / 'files'


@pytest.fixture(name='make_example_file', scope='function')
def fixture_make_example_file(testfiles_dirpath: Path, tmp_path: Path) -> Callable[[str], Path]:
    """Make a temporary copy of a file from one of the example files in tests/files.

    This way each file can be used in tests without the risk of changing contents.

    Parameters
    ----------
    testfiles_dirpath : Path
        Path to the tests/files directory.
    tmp_path : Path
        Temporary directory where files are copied to.

    Returns
    -------
    Callable[[str], Path]
        Function that takes a filename and returns the Path to the copied file.

    """
    def _make_example_file(filename: str) -> Path:
        source_filepath = testfiles_dirpath / filename
        target_filepath = tmp_path / filename
        shutil.copy2(source_filepath, target_filepath)
        return target_filepath
    return _make_example_file
