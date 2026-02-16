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
"""Test module pymovements.events.event_properties."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.samples import dispersion


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'exception', 'message'),
    [
        pytest.param(
            {'degree_column': 'degree'},
            pl.DataFrame(schema={'_degree': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            'degree',
            id='dispersion_missing_degree_column',
        ),
    ],
)
def test_dispersion_exceptions(init_kwargs, input_df, exception, message):
    expression = dispersion(**init_kwargs)
    with pytest.raises(exception, match=message):
        input_df.select([expression])


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            {},
            pl.DataFrame(
                {'degree': [[2, 3]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [0]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_one_sample',
        ),

        pytest.param(
            {},
            pl.DataFrame(
                {'degree': [[0, 0], [2, 0]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [2]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_x_move',
        ),

        pytest.param(
            {},
            pl.DataFrame(
                {'degree': [[0, 0], [0, 3]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [3]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_y_move',
        ),

        pytest.param(
            {},
            pl.DataFrame(
                {'degree': [[0, 0], [2, 3]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [5]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_xy_move',
        ),
    ],
)
def test_dispersion_has_expected_result(init_kwargs, input_df, expected_df):
    expression = dispersion(**init_kwargs)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
