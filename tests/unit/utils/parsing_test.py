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
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_parse_eyelink_equal_gaze(make_example_file):
    filepath = make_example_file('eyelink_monocular_example.asc')

    gaze, _, _, _ = pm.gaze._utils._parsing_eyelink.parse_eyelink(filepath)
    gaze_depr, _ = pm.utils.parsing.parse_eyelink(filepath)

    assert_frame_equal(gaze, gaze_depr)


def test_parse_eyelink_is_deprecated(make_example_file, assert_deprecation_is_removed):
    filepath = make_example_file('eyelink_monocular_example.asc')

    with pytest.raises(DeprecationWarning) as info:
        pm.utils.parsing.parse_eyelink(filepath)

    assert_deprecation_is_removed(
        function_name='utils/parsing.py',
        warning_message=info.value.args[0],
        scheduled_version='0.26.0',

    )
