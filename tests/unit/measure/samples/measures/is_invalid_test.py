# Copyright (c) 2026 The pymovements Project Authors
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
"""Test is_invalid sample measure helper."""

from math import inf
from math import nan

import polars as pl
from polars.testing import assert_series_equal
import pytest

from pymovements.measure.samples.measures import _is_invalid
from pymovements.measure.samples.measures import _is_invalid_value


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param(None, True, id='scalar_none'),
        pytest.param(nan, True, id='scalar_nan'),
        pytest.param(inf, True, id='scalar_inf'),
        pytest.param(-inf, True, id='scalar_neginf'),
        pytest.param(1.0, False, id='scalar_float_valid'),
        pytest.param(1, False, id='scalar_int_valid'),
        pytest.param('a', False, id='scalar_str_valid'),
        pytest.param([1.0, 1.0], False, id='list_valid'),
        pytest.param([1.0, nan], True, id='list_with_nan'),
        pytest.param([1.0, None], True, id='list_with_none'),
        pytest.param((1.0, inf), True, id='tuple_with_inf'),
        pytest.param(iter([1.0, 1.0]), False, id='iter_valid'),
        pytest.param(iter([nan]), True, id='iter_invalid'),
    ],
)
def test_is_invalid_value_parametrized(value, expected):
    """Test _is_invalid_value helper function."""
    assert _is_invalid_value(value) == expected


@pytest.mark.parametrize(
    ('data', 'dtype', 'expected'),
    [
        pytest.param(
            [1.0, None, nan, inf, -inf, 2.0],
            pl.Float64,
            [False, True, True, True, True, False],
            id='float64',
        ),
        pytest.param(
            [1, None, 2],
            pl.Int64,
            [False, True, False],
            id='int64',
        ),
        pytest.param(
            ['a', None, 'b'],
            pl.Utf8,
            [False, True, False],
            id='utf8',
        ),
        pytest.param(
            [[1.0, 1.0], [1.0, nan], [1.0, inf], [1.0, None], None],
            pl.List(pl.Float64),
            [False, True, True, True, True],
            id='list_float',
        ),
        pytest.param(
            [1.0, None, nan, inf, 2.0],
            None,
            [False, True, True, True, False],
            id='no_dtype',
        ),
    ],
)
def test_is_invalid_parametrized(data, dtype, expected):
    """Test is_invalid expression generator."""
    df = pl.DataFrame({'A': data}, schema={'A': dtype} if dtype else None)
    # Test both string and expression input
    expr_str = _is_invalid('A', dtype=dtype)
    expr_col = _is_invalid(pl.col('A'), dtype=dtype)

    result_str = df.select(expr_str)['A']
    result_col = df.select(expr_col)['A']

    expected_series = pl.Series('A', expected, dtype=pl.Boolean)
    assert_series_equal(result_str, expected_series)
    assert_series_equal(result_col, expected_series)
