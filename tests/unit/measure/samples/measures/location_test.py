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

from pymovements.measure.samples import location


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'message'),
    [
        pytest.param(
            {'method': 'foo'},
            ValueError,
            "Method 'foo' not supported",
            id='degree_unsupported_method_raises_value_error',
        ),
    ],
)
def test_location_exceptions(init_kwargs, exception, message):
    with pytest.raises(exception, match=message):
        location(**init_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            {'method': 'mean'},
            pl.DataFrame(
                {'degree': [[0, 0], [1, 0]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[0.5, 0]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='degree_two_samples_mean',
        ),

        pytest.param(
            {'method': 'mean'},
            pl.DataFrame(
                {'degree': [[0, 0], [0, 1], [0, 3]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[0, 1.3333333333333333]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='degree_three_samples_mean',
        ),

        pytest.param(
            {'method': 'median'},
            pl.DataFrame(
                {'degree': [[0, 0], [2, 1], [3, 3]]},
                schema={'degree': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[2, 1]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='degree_three_samples_median',
        ),
    ],
)
def test_location_has_expected_result(init_kwargs, input_df, expected_df):
    expression = location(**init_kwargs)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
