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
"""Test module for std_rms sample measure."""

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from pymovements.measure.samples import std_rms


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'exception', 'message'),
    [
        pytest.param(
            {'column': 'position'},
            pl.DataFrame(schema={'_position': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            'position',
            id='std_rms_missing_position_column',
        ),
    ],
)
def test_std_rms_exceptions(init_kwargs, input_df, exception, message):
    expression = std_rms(**init_kwargs)
    with pytest.raises(exception, match=message):
        input_df.select([expression])


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[5, 5]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [None]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_one_sample_constancy',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [2, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [1.4142135623730951]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_two_samples_x_move',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [0, 4]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [2.8284271247461903]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_two_samples_y_move',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [3, 4]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [3.5355339059327378]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_two_samples_xy_move',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[-1, -1], [1, -1], [-1, 1], [1, 1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [1.632993161855452]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_four_samples_corners',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1, 1], [2, 2]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [1.4142135623730951]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_three_samples_diagonal',
        ),
        pytest.param(
            {'column': 'gaze_pos'},
            pl.DataFrame(
                {'gaze_pos': [[10, 20], [10, 20], [10, 20]]},
                schema={'gaze_pos': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [0.0]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_custom_column_constant_positions',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[-5, -5], [5, 5]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [10.0]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_negative_positions',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0.5, 0.5], [1.5, 1.5]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'std_rms': [1.0]},
                schema={'std_rms': pl.Float64},
            ),
            id='std_rms_decimals',
        ),
    ],
)
def test_std_rms_has_expected_result(init_kwargs, input_df, expected_df):
    expression = std_rms(**init_kwargs)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
