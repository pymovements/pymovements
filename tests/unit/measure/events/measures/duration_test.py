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



@pytest.mark.parametrize(
    ('event_property', 'init_kwargs', 'input_df', 'exception', 'msg_substrings'),
    [
        pytest.param(
            pm.events.duration,
            {},
            pl.DataFrame(schema={'onset': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('offset',),
            id='duration_missing_offset_column',
        ),
        pytest.param(
            pm.events.duration,
            {},
            pl.DataFrame(schema={'offset': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('onset',),
            id='duration_missing_onset_column',
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
            pm.events.duration,
            {},
            pl.DataFrame(schema={'onset': pl.Int64, 'offset': pl.Int64}),
            pl.DataFrame(schema={'duration': pl.Int64}),
            id='empty_dataframe_results_in_empty_dataframe_with_correct_schema',
        ),

        pytest.param(
            pm.events.duration,
            {},
            pl.DataFrame({'onset': 0, 'offset': 1}, schema={'onset': pl.Int64, 'offset': pl.Int64}),
            pl.DataFrame({'duration': 1}, schema={'duration': pl.Int64}),
            id='single_event_duration',
        ),

        pytest.param(
            pm.events.duration,
            {},
            pl.DataFrame(
                {'onset': [0, 10], 'offset': [9, 23]},
                schema={'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.DataFrame(
                {'duration': [9, 13]},
                schema={'duration': pl.Int64},
            ),
            id='two_events_different_durations',
        ),
    ],
)
def test_property_has_expected_result(event_property, init_kwargs, input_df, expected_df):
    expression = event_property(**init_kwargs).alias(event_property.__name__)
    result_df = input_df.select([expression])

    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    ('property_function', 'property_function_name'),
    [
        pytest.param(pm.events.duration, 'duration', id='duration'),
    ],
)
def test_property_registered(property_function, property_function_name):
    property_dict = pm.events.EVENT_PROPERTIES

    assert property_function_name in property_dict
    assert property_dict[property_function_name] == property_function
    assert property_dict[property_function_name].__name__ == property_function_name


@pytest.mark.parametrize('property_function', pm.events.EVENT_PROPERTIES.values())
def test_property_returns_polars_expression(property_function):
    assert isinstance(property_function(), pl.Expr)
