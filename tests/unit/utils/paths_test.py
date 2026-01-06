# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Tests deprecated utils.paths."""
import re

import pytest

from pymovements.utils.paths import get_filepaths
from pymovements.utils.paths import match_filepaths


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize('path_function', [get_filepaths, match_filepaths])
def test_downloads_function(path_function, tmp_path):
    path_function(path=tmp_path, regex=re.compile('foo'))


@pytest.mark.parametrize('path_function', [get_filepaths, match_filepaths])
def test_parse_eyelink_removed(path_function, tmp_path, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        path_function(path=tmp_path, regex=re.compile('foo'))

    assert_deprecation_is_removed(
        function_name='utils/paths.py',
        warning_message=info.value.args[0],
        scheduled_version='0.26.0',

    )
