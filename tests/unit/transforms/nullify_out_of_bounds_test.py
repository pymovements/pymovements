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
"""Test pymovements.transforms.nullify_out_of_bounds."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.transforms import nullify_out_of_bounds


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[1., 2.], [3., 4.], [5., 6.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[1., 2.], [3., 4.], [5., 6.]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='all_in_range_unchanged',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[-1., 2.], [3., -4.], [5., 6.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [None, None, [5., 6.]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='below_lower_bound_nulled',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[11., 2.], [3., 4.], [5., 60.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [None, [3., 4.], None]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='above_upper_bound_nulled',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[0., 10.], [10., 0.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[0., 10.], [10., 0.]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='bounds_inclusive',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[1., 20.], [-1., 2.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [None, None]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='one_component_out_nulls_whole_sample',
        ),
        pytest.param(
            {'lower_bound': None, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[-100., 2.], [3., 40.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[-100., 2.], None]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='no_lower_bound',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': None, 'n_components': 2},
            pl.Series('velocity', [[-100., 2.], [3., 40.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [None, [3., 40.]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='no_upper_bound',
        ),
        pytest.param(
            {'lower_bound': None, 'upper_bound': None, 'n_components': 2},
            pl.Series('velocity', [[-100., 2.], [3., 40.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[-100., 2.], [3., 40.]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='no_bounds_unchanged',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 2},
            pl.Series('velocity', [[None, 2.], [None, 20.], None], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[None, 2.], None, None]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='existing_nulls_kept',
        ),
        pytest.param(
            {'lower_bound': 0, 'upper_bound': 10, 'n_components': 3},
            pl.Series('velocity', [[1., 2., 3.], [1., 2., 30.]], pl.List(pl.Float64)),
            pl.DataFrame(
                {'velocity': [[1., 2., 3.], None]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='three_components',
        ),
    ],
)
def test_nullify_out_of_bounds_returns(kwargs, series, expected_df):
    df = pl.DataFrame(series)
    result_df = df.select(
        nullify_out_of_bounds(
            input_column='velocity', output_column='velocity', **kwargs,
        ),
    )
    assert_frame_equal(result_df, expected_df)


def test_nullify_out_of_bounds_writes_output_column():
    df = pl.DataFrame(
        {'velocity': [[1., 2.], [30., 4.]]},
        schema={'velocity': pl.List(pl.Float64)},
    )
    result_df = df.with_columns(
        nullify_out_of_bounds(
            lower_bound=0, upper_bound=10, n_components=2,
            input_column='velocity', output_column='velocity_nulled',
        ),
    )
    expected_df = pl.DataFrame(
        {
            'velocity': [[1., 2.], [30., 4.]],
            'velocity_nulled': [[1., 2.], None],
        },
        schema={'velocity': pl.List(pl.Float64), 'velocity_nulled': pl.List(pl.Float64)},
    )
    assert_frame_equal(result_df, expected_df)
