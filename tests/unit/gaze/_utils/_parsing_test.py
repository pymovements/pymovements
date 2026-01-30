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
"""Tests pymovements asc to csv processing - shared functionality."""
import pytest

from pymovements.gaze._utils import _parsing


@pytest.mark.parametrize(
    (
        'num_expected_samples',
        'num_actual_samples',
        'num_blink_samples',
        'expected_total',
        'expected_blink',
    ),
    [
        pytest.param(0, 10, 5, 0.0, 0.0, id='zero_expected_samples'),
        pytest.param(0, 0, 0, 0.0, 0.0, id='all_zero'),
        pytest.param(100, 90, 5, 0.1, 0.05, id='normal_case'),
    ],
)
def test_calculate_data_loss_ratio(
        num_expected_samples, num_actual_samples, num_blink_samples,
        expected_total, expected_blink,
):
    """Test data loss ratio calculation."""
    total, blink = _parsing._calculate_data_loss_ratio(
        num_expected_samples, num_actual_samples, num_blink_samples,
    )
    assert total == expected_total
    assert blink == expected_blink
