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
"""Tests deprecated utils.archives."""
import gzip

import pytest

from pymovements import __version__
from pymovements.utils.archives import extract_archive


@pytest.fixture(name='compressed_file')
def fixture_compressed_file(tmp_path):
    source_filepath = tmp_path / 'test.file'
    source_filepath.write_bytes(b'test')

    # declare archive path
    compressed_filepath = tmp_path / 'test.gz'

    with gzip.open(compressed_filepath, 'wb') as fp:
        fp.write(source_filepath.read_bytes())

    # now remove original file again
    source_filepath.unlink()

    yield compressed_filepath


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_archive_extract(compressed_file):
    extract_archive(compressed_file)


def test_archive_extract_removed(compressed_file, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        extract_archive(compressed_file)
    assert_deprecation_is_removed('utils/archives.py', info.value.args[0], __version__)
