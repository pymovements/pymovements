# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test the resource definitions of the EMTeC dataset."""
from pathlib import Path

import pytest

from pymovements import Dataset


@pytest.fixture(name='emtec_path')
def fixture_emtec_path(tmp_path: Path) -> Path:
    """Return a dataset directory with empty files laid out like the extracted EMTeC archives.

    Parameters
    ----------
    tmp_path: Path
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        The dataset directory.

    """
    relative_paths = [
        'raw/ET_1.csv',
        'precomputed_events/fixations.csv',
        'precomputed_events/fixations_corrected.csv',
        'precomputed_events/__MACOSX/._fixations_corrected.csv',
        'precomputed_reading_measures/reading_measures.csv',
    ]
    for relative_path in relative_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return tmp_path


def test_emtec_scan_matches_each_fixation_file_once(emtec_path):
    fileinfo = Dataset('EMTeC', path=emtec_path).scan().fileinfo['precomputed_events']

    assert fileinfo.to_dicts() == [
        {'filepath': 'fixations.csv'},
        {'filepath': 'fixations_corrected.csv'},
    ]
