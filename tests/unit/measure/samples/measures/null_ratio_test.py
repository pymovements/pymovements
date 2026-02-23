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
"""Test null ratio measure."""

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from pymovements.measure.samples import null_ratio


@pytest.mark.parametrize(
    ('df', 'kwargs', 'expected'),
    [
        pytest.param(
            pl.from_dict(data={'A': [0.1, 0.2, 0.3, 0.4]}, schema={'A': pl.Float64}),
            {'column': 'A', 'column_dtype': pl.Float64},
            pl.from_dict(data={'null_ratio': [0.0]}),
            id='float_dtype_4_rows_no_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [0.1, None, 0.3, None]}, schema={'A': pl.Float64}),
            {'column': 'A', 'column_dtype': pl.Float64},
            pl.from_dict(data={'null_ratio': [0.5]}),
            id='float_dtype_4_rows_half_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Float64}),
            {'column': 'A', 'column_dtype': pl.Float64},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='float_dtype_4_rows_all_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [1, 2, 3, 4]}, schema={'A': pl.Int64}),
            {'column': 'A', 'column_dtype': pl.Int64},
            pl.from_dict(data={'null_ratio': [0.0]}),
            id='int_dtype_4_rows_no_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [1, None, 3, None]}, schema={'A': pl.Int64}),
            {'column': 'A', 'column_dtype': pl.Int64},
            pl.from_dict(data={'null_ratio': [0.5]}),
            id='int_dtype_4_rows_half_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Int64}),
            {'column': 'A', 'column_dtype': pl.Int64},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='int_dtype_4_rows_all_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Utf8}),
            {'column': 'A', 'column_dtype': pl.Utf8},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='str_dtype_4_rows_all_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': ['1', '2', '3', '4']}, schema={'A': pl.Utf8}),
            {'column': 'A', 'column_dtype': pl.Utf8},
            pl.from_dict(data={'null_ratio': [0.0]}),
            id='str_dtype_4_rows_no_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': ['1', None, '3', None]}, schema={'A': pl.Utf8}),
            {'column': 'A', 'column_dtype': pl.Utf8},
            pl.from_dict(data={'null_ratio': [0.5]}),
            id='str_dtype_4_rows_half_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [[0.1, 0.2], [0.3, 0.4]]}),
            {'column': 'A', 'column_dtype': pl.List(pl.Float64)},
            pl.from_dict(data={'null_ratio': [0.0]}),
            id='list_dtype_2_elem_2_rows_no_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [[None, None], [None, None]]}),
            {'column': 'A', 'column_dtype': pl.List(pl.Float64)},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='list_dtype_2_elem_2_rows_all_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [[0.1, 0.2], [None, None]]}),
            {'column': 'A', 'column_dtype': pl.List(pl.Float64)},
            pl.from_dict(data={'null_ratio': [0.5]}),
            id='list_dtype_2_elem_2_rows_half_nulls',
        ),
        pytest.param(
            pl.from_dict(data={'A': [[0.1, None], [0.2, None]]}),
            {'column': 'A', 'column_dtype': pl.List(pl.Float64)},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='list_dtype_2_elem_2_rows_half_nulls_each_row',
        ),
    ],
)
def test_null_ratio_expected(df, kwargs, expected):
    result = df.select(null_ratio(**kwargs))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('df', 'kwargs', 'exception', 'message'),
    [
        pytest.param(
            pl.DataFrame({'A': [1, 2], 'B': [True, False]}).select(pl.struct(pl.all()).alias('C')),
            {'column': 'C', 'column_dtype': pl.Struct},
            TypeError,
            'column_dtype must be of type {Float64, Int64, Utf8, List} but is of type Struct',
            id='struct_column',
        ),
    ],
)
def test_null_ratio_raises(df, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        df.select(null_ratio(**kwargs))
