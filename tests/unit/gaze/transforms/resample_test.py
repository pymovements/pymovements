# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Test the resample transformation function."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'df', 'expected_df'),
    [
        # Without filling nulls after resampling
        pytest.param(
            {
                'resampling_rate': 1000,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5],
                    'pixel': [1, 2, 3, 4, 5, 6],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5],
                    'pixel': [1, 2, 3, 4, 5, 6],
                },
            ),
            id='resample_same_sampling_rate',
        ),
        pytest.param(
            {
                'resampling_rate': 2000,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5],
                    'pixel': [1, 2, 3, 4, 5, 6],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    'pixel': [1, None, 2, None, 3, None, 4, None, 5, None, 6],
                },
                schema={'time': pl.Float64, 'pixel': pl.Int64},
            ),
            id='upsample_1000_to_2000_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 500,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5],
                    'pixel': [1, 2, 3, 4, 5, 6],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 3, 5],
                },
            ),
            id='downsample_1000_to_500_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 500,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 4, 6, 8, 10],
                    'pixel': [1, 2, 4, 5, 6, 7],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 2, 4, 6, 8, 10],
                    'pixel': [1, None, 4, 5, 6, 7],
                },
            ),
            id='resample_inconsistent_sampling_rate_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 4, 6, 8, 10],
                    'pixel': [1, 2, 4, 5, 6, 7],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'pixel': [1, 2, None, None, 4, None, 5, None, 6, None, 7],
                },
            ),
            id='upsample_inconsistent_sampling_rate_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 4000,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 0.5, 1, 1.5, 2, 2.5],
                    'pixel': [1, 2, 3, 4, 5, 6],
                },
                schema={'time': pl.Float64, 'pixel': pl.Int64},
            ),
            pl.DataFrame(
                {
                    'time': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
                    'pixel': [1, None, 2, None, 3, None, 4, None, 5, None, 6],
                },
                schema={'time': pl.Float64, 'pixel': pl.Int64},
            ),
            id='upsample_2000_to_4000_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 333,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'pixel': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 3.003, 6.006],
                    'pixel': [1, None, None],
                },
                schema={'time': pl.Float64, 'pixel': pl.Int64},
            ),
            id='downsample_1000_to_333_no_interpolation',
        ),
        pytest.param(
            {
                'resampling_rate': 333,
                'columns': None,
            },
            pl.DataFrame(
                {
                    'time': [0, 10, 20],
                    'pixel': [1, 2, 3],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 3.003, 6.006, 9.009, 12.012, 15.015, 18.018],
                    'pixel': [1, None, None, None, None, None, None],
                },
                schema={'time': pl.Float64, 'pixel': pl.Int64},
            ),
            id='upsample_100_to_333_no_interpolation',
        ),
        # With filling nulls, single component
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [1, 1.5, 2, 3, 4],
                },
                schema={'time': pl.Int64, 'pixel': pl.Float64},
            ),
            id='upsample_500_to_1000_interpolate_linear_one_component',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_nearest',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [1., 2., 2., 4., 4.],
                },
                schema={'time': pl.Int64, 'pixel': pl.Float64},
            ),
            id='upsample_500_to_1000_interpolate_nearest_one_component',
        ),
        pytest.param(
            {
                'resampling_rate': 2000,
                'fill_null_strategy': 'interpolate_nearest',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                    'pixel': [1., 1., 2., 2., 2., 2., 4., 4., 4.],
                },
                schema={'time': pl.Float64, 'pixel': pl.Float64},
            ),
            id='upsample_500_to_2000_interpolate_nearest_one_component',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'forward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [1., 1., 2., 2., 4.],
                },
                schema={'time': pl.Int64, 'pixel': pl.Float64},
            ),
            id='upsample_500_to_1000_fill_forward_one_component',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'backward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [1., 2., 2., 4., 4.],
                },
                schema={'time': pl.Int64, 'pixel': pl.Float64},
            ),
            id='upsample_500_to_1000_fill_backward_one_component',
        ),
        # With fill nulls, multiple components
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [1.5, 2.5], [2., 3.], [2.5, 3.5], [3., 4.]],
                },
            ),
            id='upsample_500_to_1000_interpolate_linear_two_components',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_nearest',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [2., 3.], [2., 3.], [3., 4.], [3., 4.]],
                },
            ),
            id='upsample_500_to_1000_interpolate_nearest_two_components',
        ),
        pytest.param(
            {
                'resampling_rate': 2000,
                'fill_null_strategy': 'interpolate_nearest',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                    'pixel': [
                        [1., 2.], [1., 2.], [2., 3.], [2., 3.], [2., 3.],
                        [2., 3.], [3., 4.], [3., 4.], [3., 4.],
                    ],
                },
                schema={'time': pl.Float64, 'pixel': pl.List(pl.Float64)},
            ),
            id='upsample_500_to_2000_interpolate_nearest_two_components',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'forward',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [1., 2.], [2., 3.], [2., 3.], [3., 4.]],
                },
            ),
            id='upsample_500_to_1000_fill_forward_two_components',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'backward',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [2., 3.], [2., 3.], [3., 4.], [3., 4.]],
                },
            ),
            id='upsample_500_to_1000_fill_backward_two_components',
        ),
        # With filling null on specific columns
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
                'columns': ['pixel', 'distance'],
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                    'distance': [1, 2, 4],
                    'other': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [1.5, 2.5], [2., 3.], [2.5, 3.5], [3., 4.]],
                    'distance': [1, 1.5, 2, 3, 4],
                    'other': [1, None, 2, None, 4],
                },
                schema={
                    'time': pl.Int64,
                    'pixel': pl.List(
                        pl.Float64,
                    ),
                    'distance': pl.Float64,
                    'other': pl.Int64,
                },
            ),
            id='upsample_500_to_1000_interpolate_linear_one_component_specific_columns_list',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
                'columns': 'pixel',
                'n_components': 2,
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [[1, 2], [2, 3], [3, 4]],
                    'distance': [1, 2, 4],
                    'other': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [[1., 2.], [1.5, 2.5], [2., 3.], [2.5, 3.5], [3., 4.]],
                    'distance': [1, None, 2, None, 4],
                    'other': [1, None, 2, None, 4],
                },
            ),
            id='upsample_500_to_1000_interpolate_linear_one_component_specific_columnn_string',
        ),
        # Test for handling pre-existing null values
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, None, 2., 3., 4.],
                },
            ),
            id='upsample_500_to_1000_interpolate_linear_one_component_with_none_values',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_nearest',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, None, 2., 4., 4.],
                },
            ),
            id='upsample_500_to_1000_interpolate_nearest_one_component_with_none_values',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'forward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, None, 2., 2., 4.],
                },
            ),
            id='upsample_500_to_1000_fill_forward_one_component_with_none_values',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'backward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, 2., 2., 4., 4.],
                },
            ),
            id='upsample_500_to_1000_fill_backward_one_component_with_none_values',
        ),
        # Test for handling large timestamps (e.g. millis since unix epoch)
        pytest.param(
            {
                'resampling_rate': 2000,
                'fill_null_strategy': None,
            },
            pl.DataFrame(
                {
                    'time': [1713398400010, 1713398400011, 1713398400012],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [
                        1713398400010.0,
                        1713398400010.5,
                        1713398400011.0,
                        1713398400011.5,
                        1713398400012.0,
                    ],
                    'pixel': [1.0, None, 2.0, None, 4.0],
                },
            ),
            id='upsample_unix_timestamps_int',
        ),
        pytest.param(
            {
                'resampling_rate': 2000,
                'fill_null_strategy': None,
            },
            pl.DataFrame(
                {
                    'time': [1713398400010., 1713398400011., 1713398400012.],
                    'pixel': [1, 2, 4],
                },
            ),
            pl.DataFrame(
                {
                    'time': [
                        1713398400010.0,
                        1713398400010.5,
                        1713398400011.0,
                        1713398400011.5,
                        1713398400012.0,
                    ],
                    'pixel': [1.0, None, 2.0, None, 4.0],
                },
            ),
            id='upsample_unix_timestamps_float',
        ),
        # Handling empty DataFrame
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'interpolate_linear',
            },
            pl.DataFrame(schema={'time': pl.Int32, 'pixel': pl.Float32}),
            pl.DataFrame(schema={'time': pl.Int32, 'pixel': pl.Float32}),
            id='resample_empty_df',
        ),
        # Handle non-numeric columns
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'backward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                    'description': ['text', 'more_text', None],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, 2., 2., 4., 4.],
                    'description': ['text', 'more_text', 'more_text', None, None],
                },
            ),
            id='resample_non_numeric_columns_backfilled',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'forward',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                    'description': ['text', 'more_text', None],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, None, 2., 2., 4.],
                    'description': ['text', 'text', 'more_text', 'more_text', 'more_text'],
                },
            ),
            id='resample_non_numeric_columns_forwardfilled',
        ),
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'forward',
                'columns': 'description',
            },
            pl.DataFrame(
                {
                    'time': [0, 2, 4],
                    'pixel': [None, 2, 4],
                    'description': ['text', 'more_text', None],
                },
            ),
            pl.DataFrame(
                {
                    'time': [0, 1, 2, 3, 4],
                    'pixel': [None, None, 2, None, 4],
                    'description': ['text', 'text', 'more_text', 'more_text', 'more_text'],
                },
            ),
            id='resample_non_numeric_columns_forwardfilled_no_numerical',
        ),
    ],
)
def test_resample_returns(kwargs, df, expected_df):
    """Test if resample returns expected DataFrame."""
    result_df = pm.gaze.transforms.resample(df, **kwargs)

    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        # Invalid fill_null_strategy
        pytest.param(
            {
                'resampling_rate': 1000,
                'fill_null_strategy': 'unknown',
                'n_components': 2,
            },
            ValueError,
            ['Unknown fill_null_strategy'],
            id='invalid_fill_null_strategy',
        ),
        # Invalid n_components
        pytest.param(
            {
                'resampling_rate': 1000,
            },
            ValueError,
            ['n_components must be specified when processing nested column'],
            id='missing_n_components',
        ),
        # Invalid resampling_rate
        pytest.param(
            {
                'resampling_rate': 0,
            },
            ValueError,
            ['resampling_rate', '0'],
            id='zero_resampling_rate',
        ),
        # Invalid resampling_rate
        pytest.param(
            {
                'resampling_rate': -1,
            },
            ValueError,
            ['resampling_rate', '-1'],
            id='negative_resampling_rate',
        ),
        # Unsuporrted sample rate
        pytest.param(
            {
                'resampling_rate': 7,
                'n_components': 2,
            },
            ValueError,
            ['unsupported resampling rate'],
            id='unsupported_resampling_rate',
        ),
    ],
)
def test_resample_raises_error(kwargs, exception, msg_substrings):
    """Test if resample raises expected error."""
    df = pl.DataFrame({
        'time': [0, 1, 2, 3, 4, 5],
        'pixel': [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
    })

    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.resample(df, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()
