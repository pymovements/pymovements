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
"""Tests deprecated utils.parsing."""
import re

import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements.stimulus import text
from pymovements.utils.aois import get_aoi


@pytest.fixture(name='text_stimulus')
def fixture_text_stimulus(make_example_file):
    filepath = make_example_file('toy_text_1_1_aoi.csv')
    yield text.from_file(
        filepath,
        aoi_column='word',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        end_x_column='bottom_left_x',
        end_y_column='bottom_left_y',
        page_column='page',
    )


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    'row',
    [
        {'x': 400, 'y': 125},
        {'x': 500, 'y': 300},
    ],
)
def test_get_aoi_equal_to_new(text_stimulus, row):
    aoi_old = get_aoi(text_stimulus, row, 'x', 'y')
    aoi_new = text_stimulus.get_aoi(row=row, x_eye='x', y_eye='y')

    assert_frame_equal(aoi_old, aoi_new)


def test_get_aoi_deprecated(text_stimulus):
    with pytest.raises(DeprecationWarning):
        get_aoi(text_stimulus, {'x': 400, 'y': 125}, 'x', 'y')


def test_get_aoi_removed(text_stimulus, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        get_aoi(text_stimulus, {'x': 400, 'y': 125}, 'x', 'y')

    re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')
    assert_deprecation_is_removed('utils/parsing.py', info.value.args[0], __version__)
