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
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.samples import location


@pytest.mark.parametrize(
    ('event_property', 'init_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            location,
            {'method': 'foo'},
            ValueError,
            ('method', 'foo', 'not', 'supported', 'mean', 'median'),
            id='position_unsupported_method_raises_value_error',
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
    ('event_property', 'init_kwargs', 'input_df', 'expected_df'),
    [
        pytest.param(
            location,
            {'method': 'mean'},
            pl.DataFrame(
                {'position': [[0, 0], [1, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[0.5, 0]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='position_two_samples_mean',
        ),
        pytest.param(
            location,
            {'method': 'mean'},
            pl.DataFrame(
                {'position': [[0, 0], [0, 1], [0, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[0, 1.3333333333333333]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='position_three_samples_mean',
        ),
        pytest.param(
            location,
            {'method': 'median'},
            pl.DataFrame(
                {'position': [[0, 0], [2, 1], [3, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'location': [[2, 1]]},
                schema={'location': pl.List(pl.Float64)},
            ),
            id='position_three_samples_median',
        ),
    ],
)
def test_property_has_expected_result(event_property, init_kwargs, input_df, expected_df):
    expression = event_property(**init_kwargs).alias(event_property.__name__)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)
