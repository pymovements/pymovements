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
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('event_property', 'init_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            pm.events.peak_velocity,
            {'n_components': 3},
            ValueError,
            ('data must have exactly two components',),
            id='peak_velocity_not_2_components_raise_value_error',
        ),
        pytest.param(
            pm.events.location,
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
        pytest.param(
            pm.events.peak_velocity,
            {'velocity_column': 'velocity'},
            pl.DataFrame(schema={'_velocity': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('velocity',),
            id='peak_velocity_missing_velocity_column',
        ),
        pytest.param(
            pm.events.dispersion,
            {'position_column': 'position'},
            pl.DataFrame(schema={'_position': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('position',),
            id='dispersion_missing_position_column',
        ),
        pytest.param(
            pm.events.amplitude,
            {'position_column': 'position'},
            pl.DataFrame(schema={'_position': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('position',),
            id='amplitude_missing_position_column',
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

        pytest.param(
            pm.events.peak_velocity,
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

        pytest.param(
            pm.events.dispersion,
            {},
            pl.DataFrame(
                {'position': [[2, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [0]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_one_sample',
        ),

        pytest.param(
            pm.events.dispersion,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [2, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [2]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_x_move',
        ),

        pytest.param(
            pm.events.dispersion,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [0, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [3]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_y_move',
        ),

        pytest.param(
            pm.events.dispersion,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [2, 3]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'dispersion': [5]},
                schema={'dispersion': pl.Float64},
            ),
            id='dispersion_two_samples_xy_move',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[2, 1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [0]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_single_sample',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [1]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_two_samples_x_move',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [0, 1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [1]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_two_samples_y_move',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1, 1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [np.sqrt(2)]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_two_samples_xy_move',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1.1, 0], [1, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [1]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_three_samples_overshoot',
        ),

        pytest.param(
            pm.events.disposition,
            {},
            pl.DataFrame(
                {'position': [[-1, 0], [0, 0], [1, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'disposition': [2]},
                schema={'disposition': pl.Float64},
            ),
            id='disposition_three_samples_negative',
        ),

        pytest.param(
            pm.events.amplitude,
            {},
            pl.DataFrame(
                {'position': [[4, 5]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'amplitude': [0]},
                schema={'amplitude': pl.Float64},
            ),
            id='amplitude_one_sample',
        ),

        pytest.param(
            pm.events.amplitude,
            {},
            pl.DataFrame(
                {'position': [[2, 0], [0, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'amplitude': [2]},
                schema={'amplitude': pl.Float64},
            ),
            id='amplitude_two_samples_x_move',
        ),

        pytest.param(
            pm.events.amplitude,
            {},
            pl.DataFrame(
                {'position': [[0, 3], [0, 0]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'amplitude': [3]},
                schema={'amplitude': pl.Float64},
            ),
            id='amplitude_two_samples_y_move',
        ),
        pytest.param(
            pm.events.amplitude,
            {},
            pl.DataFrame(
                {'position': [[0, 0], [1, 1]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            pl.DataFrame(
                {'amplitude': [np.sqrt(2)]},
                schema={'amplitude': pl.Float64},
            ),
            id='amplitude_two_samples_xy_move',
        ),
        pytest.param(
            pm.events.location,
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
            pm.events.location,
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
            pm.events.location,
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


@pytest.mark.parametrize(
    ('property_function', 'property_function_name'),
    [
        pytest.param(pm.events.duration, 'duration', id='duration'),
        pytest.param(pm.events.peak_velocity, 'peak_velocity', id='peak_velocity'),
        pytest.param(pm.events.dispersion, 'dispersion', id='dispersion'),
        pytest.param(pm.events.amplitude, 'amplitude', id='amplitude'),
        pytest.param(pm.events.disposition, 'disposition', id='disposition'),
        pytest.param(pm.events.location, 'location', id='location'),
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
