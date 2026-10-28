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

from pymovements.measure.samples import peak_velocity


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'message'),
    [
        pytest.param(
            {'n_components': 3},
            ValueError,
            'data must have exactly two components',
            id='peak_velocity_not_2_components_raise_value_error',
        ),
    ],
)
def test_peak_velocity_init_exceptions(init_kwargs, exception, message):
    with pytest.raises(exception, match=message):
        peak_velocity(**init_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'exception', 'message'),
    [
        pytest.param(
            {'velocity_column': 'velocity'},
            pl.DataFrame(schema={'_velocity': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            'velocity',
            id='peak_velocity_missing_velocity_column',
        ),
    ],
)
def test_peak_velocity_exceptions(init_kwargs, input_df, exception, message):
    expression = peak_velocity(**init_kwargs)
    with pytest.raises(exception, match=message):
        input_df.select([expression])


@pytest.mark.parametrize(
    ('init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            {},
            pl.DataFrame(
                {'velocity': [[0, 0], [0, 1]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'peak_velocity': [1]},
                schema={'peak_velocity': pl.Float64},
            ),
            id='single_event_peak_velocity',
        ),
    ],
)
def test_peak_velocity_has_expected_result(init_kwargs, input_df, expected_df):
    expression = peak_velocity(**init_kwargs).alias('peak_velocity')
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
