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
"""Test module for bcea sample measure."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.samples import bcea


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'exception', 'message'),
    [
        pytest.param(
            {'column': 'position'},
            pl.DataFrame(schema={'_position': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            'position',
            id='bcea_missing_position_column',
        ),
    ],
)
def test_bcea_exceptions(init_kwargs, input_df, exception, message):
    expression = bcea(**init_kwargs)
    with pytest.raises(exception, match=message):
        input_df.select([expression])


@pytest.mark.parametrize(
    ('confidence', 'message'),
    [
        pytest.param(
            -1.0,
            r'confidence must be between 0 and 100',
            id='bcea_confidence_below_zero',
        ),
        pytest.param(
            100.0,
            r'confidence must be between 0 and 100',
            id='bcea_confidence_at_upper_bound',
        ),
        pytest.param(
            100.0001,
            r'confidence must be between 0 and 100',
            id='bcea_confidence_above_upper_bound',
        ),
    ],
)
def test_bcea_confidence_bounds(confidence, message):
    with pytest.raises(ValueError, match=message):
        bcea(confidence=confidence)


@pytest.mark.parametrize(
    'confidence',
    [
        pytest.param(0.0, id='bcea_confidence_lower_bound'),
        pytest.param(99.999, id='bcea_confidence_below_upper_bound'),
    ],
)
def test_bcea_confidence_allows_valid_bounds(confidence):
    assert isinstance(bcea(confidence=confidence), pl.Expr)


@pytest.mark.parametrize(
    'confidence',
    [
        pytest.param(None, id='bcea_confidence_none'),
        pytest.param('100', id='bcea_confidence_string'),
    ],
)
def test_bcea_confidence_invalid_type_raises_type_error(confidence):
    with pytest.raises(TypeError):
        bcea(confidence=confidence)


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
                {'bcea': [None]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_one_sample_returns_none',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [2, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [None]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_two_samples_x_only',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [0, 2]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [None]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_two_samples_y_only',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[-1, 0], [1, 0], [0, 1], [0, -1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [4.808344028750191]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_four_samples_cross_pattern',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[-1, -1], [1, 1], [-1, 1], [1, -1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [9.61668805750038]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_four_samples_corners',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1, 1], [2, 2], [3, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [0.0]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_four_samples_perfect_correlation',
        ),
        pytest.param(
            {'column': 'gaze_pos'},
            pl.DataFrame(
                {'gaze_pos': [[10, 20], [10, 20]]},
                schema={'gaze_pos': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [None]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_custom_column_constant_positions',
        ),
        pytest.param(
            {},
            pl.DataFrame(
                {'position': [[-5, -5], [5, -5], [-5, 5], [5, 5]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'bcea': [240.41720143750956]},
                schema={'bcea': pl.Float64},
            ),
            id='bcea_negative_positions',
        ),
    ],
)
def test_bcea_has_expected_result(init_kwargs, input_df, expected_df):
    expression = bcea(**init_kwargs)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
