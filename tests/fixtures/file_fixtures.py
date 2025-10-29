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
from __future__ import annotations

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


@pytest.fixture(name='make_text_file', scope='function')
def fixture_make_text_file(tmp_path: Path) -> Callable[[str | Path, str, str, str], Path]:
    """Make a custom file with self-written header and body.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory where files are copied to. Existing fixture.

    Returns
    -------
    Callable[[str | Path, str, str, str], Path]
        Function that takes a filename, the file header and body and the encoding of the file.
        Returns the path to the created file.
        Header and body are simply concatenated into a single string and written to the file.
        The file is saved into a temporary directory.

    """
    def _ensure_rel_filename(filename: str | Path) -> Path:
        """Ensure filename is a relative path.

        Accepts str or Path. Rejects other types with TypeError.
        Rejects absolute paths, drive letters (Windows),
        and user home shortcuts '~' with ValueError.
        Returns a Path relative to current directory (to be joined with tmp_path).
        """
        if not isinstance(filename, (str, Path)):
            raise TypeError(f"filename must be a str or Path, got {type(filename).__name__}")

        if isinstance(filename, str) and filename.startswith('~'):
            raise ValueError("filename must be a relative path; '~' (home) is not allowed")

        p = Path(filename)
        # On Windows, p.drive captures drive letters; p.anchor is non-empty for absolute paths
        if p.is_absolute() or p.anchor or getattr(p, 'drive', ''):
            raise ValueError('filename must be a relative path without drive or root')
        return p

    def _make_text_file(
            filename: str | Path, header: str = '', body: str = '\n', encoding: str = 'utf-8',
    ) -> Path:
        rel = _ensure_rel_filename(filename)
        content = header + body
        filepath = tmp_path / rel
        # assure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # write contents to a file
        filepath.write_text(content, encoding=encoding)
        return filepath
    return _make_text_file
