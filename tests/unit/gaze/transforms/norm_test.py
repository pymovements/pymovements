# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Test pymovements.gaze.transforms.norm."""
import polars as pl
import pytest
from polars.testing import assert_series_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('columns', 'df', 'expected_series'),
    [
        pytest.param(
            ('x', 'y'),
            pl.DataFrame([
                pl.Series('x', [], pl.Float64),
                pl.Series('y', [], pl.Float64),
            ]),
            pl.Series(None, [], pl.Float64),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            ('x', 'y'),
            pl.DataFrame([
                pl.Series('x', [1], pl.Float64),
                pl.Series('y', [1], pl.Float64),
            ]),
            pl.Series(None, [1.41421356], pl.Float64),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            ('x', 'y'),
            pl.DataFrame([
                pl.Series('x', [1, 1], pl.Float64),
                pl.Series('y', [1, 1], pl.Float64),
            ]),
            pl.Series(None, [1.4142, 1.4142], pl.Float64),
            id='empty_series_returns_empty_series',
        ),
    ],
)
def test_norm_returns(columns, df, expected_series):
    result_df = df.select(
        pm.gaze.transforms.norm(columns=columns).alias('norm'),
    )
    assert_series_equal(result_df['norm'], expected_series, check_names=False)
