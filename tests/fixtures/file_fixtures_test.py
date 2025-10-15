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
"""Test file fixtures."""
import filecmp

import pytest


def test_testfiles_dirpath_has_files(testfiles_dirpath):
    assert len(list(testfiles_dirpath.iterdir())) > 0


@pytest.mark.parametrize(
    'filename',
    [
        'eyelink_binocular_example.asc',
        'monocular_example.feather',
        'judo1000_example.csv',
        'potec_word_aoi_b0.tsv',
        'rda_test_file.rda',
    ],
)
def test_make_example_file_returns_copy(filename, make_example_file, testfiles_dirpath):
    fixture_filepath = make_example_file(filename)
    testfiles_filepath = testfiles_dirpath / filename

    assert fixture_filepath != testfiles_filepath  # different filepath
    assert filecmp.cmp(fixture_filepath, testfiles_filepath)  # same content
