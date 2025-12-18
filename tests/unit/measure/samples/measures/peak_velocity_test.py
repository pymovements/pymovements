# Copyright (c) 2023-2025 The pymovements Project Authors
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
    ('event_property', 'init_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            peak_velocity,
            {'n_components': 3},
            ValueError,
            ('data must have exactly two components',),
            id='peak_velocity_not_2_components_raise_value_error',
        ),
    ],
)
def test_property_init_exceptions(event_property, init_kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        event_property(**init_kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('event_property', 'init_kwargs', 'input_df', 'exception', 'msg_substrings'),
    [
        pytest.param(
            peak_velocity,
            {'velocity_column': 'velocity'},
            pl.DataFrame(schema={'_velocity': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('velocity',),
            id='peak_velocity_missing_velocity_column',
        ),
    ],
)
def test_property_exceptions(event_property, init_kwargs, input_df, exception, msg_substrings):
    property_expression = event_property(**init_kwargs)
    with pytest.raises(exception) as excinfo:
        input_df.select([property_expression])

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('event_property', 'init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            peak_velocity,
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
def test_property_has_expected_result(event_property, init_kwargs, input_df, expected_df):
    expression = event_property(**init_kwargs).alias(event_property.__name__)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
