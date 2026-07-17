# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Reading measure tests."""
import polars as pl
import pytest

from pymovements.measure.reading.processing import compute_reading_measures


@pytest.mark.parametrize(
    'fixations_df, aoi_df',
    [
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 2, 3],
                    'duration': [100, 100, 100, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'character': ['a', 'b', 'c'],
                },
            ),
            id='standard',
        ),
    ],
)
def test_compute_reading_measures(fixations_df, aoi_df):
    result = compute_reading_measures(fixations_df, aoi_df)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(aoi_df)
    assert 'FFD' in result.columns
