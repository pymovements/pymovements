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
import pytest

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements._utils._strings import curly_to_regex


@pytest.fixture(name='emtec')
def fixture_emtec() -> DatasetDefinition:
    """Return the EMTeC dataset definition.

    Returns
    -------
    DatasetDefinition
        The definition registered in the dataset library.

    """
    return DatasetLibrary.get('EMTeC')


def test_emtec_has_corrected_and_uncorrected_fixations(emtec):
    patterns = sorted(
        candidate.filename_pattern for candidate in emtec.resources
        if candidate.content == 'precomputed_events'
    )

    assert patterns == ['fixations.csv', 'fixations_corrected.csv']


@pytest.mark.parametrize(
    ('pattern', 'filename', 'matches'),
    [
        pytest.param('fixations.csv', 'fixations.csv', True, id='uncorrected_own'),
        pytest.param('fixations.csv', 'fixations_corrected.csv', False, id='uncorrected_other'),
        pytest.param(
            'fixations_corrected.csv', 'fixations_corrected.csv', True, id='corrected_own',
        ),
        pytest.param('fixations_corrected.csv', 'fixations.csv', False, id='corrected_other'),
    ],
)
def test_emtec_fixation_patterns_do_not_overlap(emtec, pattern, filename, matches):
    """Both fixation resources land in the same directory, so their patterns must be disjoint."""
    resource = next(
        candidate for candidate in emtec.resources
        if candidate.content == 'precomputed_events' and candidate.filename_pattern == pattern
    )

    assert bool(curly_to_regex(resource.filename_pattern).fullmatch(filename)) is matches
