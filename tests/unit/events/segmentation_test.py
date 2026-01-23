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
"""Test segmentation utilities."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.events import Events
from pymovements.events.segmentation import events2segmentation
from pymovements.events.segmentation import segmentation2events


@pytest.mark.parametrize(
    'events, num_samples, expected, onset_col, offset_col',
    [
        pytest.param(
            pl.DataFrame({'onset': [2, 7], 'offset': [5, 9]}),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32),
            None,
            None,
            id='basic_df',
        ),
        pytest.param(
            Events(pl.DataFrame({'onset': [2, 7], 'offset': [5, 9]})),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32),
            None,
            None,
            id='basic_events',
        ),
        pytest.param(
            pl.DataFrame(
                {'onset': [], 'offset': []},
                schema={
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                },
            ),
            5,
            np.array([0, 0, 0, 0, 0], dtype=np.int32),
            None,
            None,
            id='empty',
        ),
        pytest.param(
            pl.DataFrame({'onset': [0], 'offset': [5]}),
            5,
            np.array([1, 1, 1, 1, 1], dtype=np.int32),
            None,
            None,
            id='full',
        ),
        pytest.param(
            pl.DataFrame({'onset': [0, 5], 'offset': [2, 10]}),
            10,
            np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            None,
            None,
            id='boundaries',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2, 5], 'offset': [5, 8]}),
            10,
            np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int32),
            None,
            None,
            id='adjacent',
        ),
        pytest.param(
            pl.DataFrame({'onset': [], 'offset': []}),
            1,
            np.array([0], dtype=np.int32),
            None,
            None,
            id='edge_case_single_sample_no_event',
        ),
        pytest.param(
            pl.DataFrame({'start': [2], 'end': [5]}),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            'start',
            'end',
            id='custom_column_names_df',
        ),
        pytest.param(
            Events(pl.DataFrame({'foo': [2, 7], 'bar': [5, 9]})),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32),
            'foo',
            'bar',
            id='custom_column_names_events',
        ),
        pytest.param(
            pl.DataFrame(
                {'onset': [2], 'offset': [5]},
                schema={'onset': pl.Int8, 'offset': pl.Int8},
            ),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            None,
            None,
            id='int8_dtype',
        ),
        pytest.param(
            pl.DataFrame(
                {'onset': [2], 'offset': [5]},
                schema={'onset': pl.Int16, 'offset': pl.Int16},
            ),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            None,
            None,
            id='int16_dtype',
        ),
        pytest.param(
            pl.DataFrame(
                {'onset': [2], 'offset': [5]},
                schema={'onset': pl.Int64, 'offset': pl.Int64},
            ),
            10,
            np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            None,
            None,
            id='int64_dtype',
        ),
    ],
)
def test_events2segmentation(events, num_samples, expected, onset_col, offset_col):
    if onset_col is None and offset_col is None:
        result = events2segmentation(events, num_samples)
    else:
        result = events2segmentation(
            events, num_samples, onset_column=onset_col, offset_column=offset_col,
        )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int32


@pytest.mark.parametrize(
    'segmentation, expected_df_dict',
    [
        pytest.param(
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32),
            {'onset': [2, 7], 'offset': [5, 9]},
            id='basic',
        ),
        pytest.param(
            np.array([0, 0, 0], dtype=np.int32),
            {'onset': [], 'offset': []},
            id='empty',
        ),
        pytest.param(
            np.array([1, 1, 1], dtype=np.int32),
            {'onset': [0], 'offset': [3]},
            id='full',
        ),
        pytest.param(
            np.array([0, 1, 1], dtype=np.int32),
            {'onset': [1], 'offset': [3]},
            id='ends_with_one',
        ),
        pytest.param(
            np.array([1, 1, 0], dtype=np.int32),
            {'onset': [0], 'offset': [2]},
            id='starts_with_one',
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0], dtype=np.int32),
            {'onset': [1, 3], 'offset': [2, 4]},
            id='multiple_short',
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1], dtype=np.int32),
            {'onset': [0, 2, 4], 'offset': [1, 3, 5]},
            id='alternating',
        ),
        pytest.param(
            np.array([0, 1, 0], dtype=np.int8),
            {'onset': [1], 'offset': [2]},
            id='int8_dtype',
        ),
        pytest.param(
            np.array([0, 1, 0], dtype=np.int16),
            {'onset': [1], 'offset': [2]},
            id='int16_dtype',
        ),
        pytest.param(
            np.array([0, 1, 0], dtype=np.int64),
            {'onset': [1], 'offset': [2]},
            id='int64_dtype',
        ),
        pytest.param(
            np.array([0, 1, 0], dtype=np.uint8),
            {'onset': [1], 'offset': [2]},
            id='uint8_dtype',
        ),
    ],
)
def test_segmentation2events(segmentation, expected_df_dict):
    result = segmentation2events(segmentation)
    assert isinstance(result, Events)
    result_df = result.frame.select(['onset', 'offset'])
    expected_df = pl.DataFrame(expected_df_dict).with_columns(
        pl.col('onset').cast(pl.Int64),
        pl.col('offset').cast(pl.Int64),
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    'segmentation',
    [
        pytest.param(
            np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.int32), id='mixed',
        ),
        pytest.param(np.array([0, 0, 0, 0, 0], dtype=np.int32), id='all_zeros'),
        pytest.param(np.array([1, 1, 1, 1, 1], dtype=np.int32), id='all_ones'),
        pytest.param(np.array([1, 0, 1, 0, 1, 0], dtype=np.int32), id='alternating'),
    ],
)
def test_roundtrip(segmentation):
    events = segmentation2events(segmentation)
    result = events2segmentation(events, len(segmentation))
    np.testing.assert_array_equal(result, segmentation)

    # Mathematical invariance: total event duration should be preserved
    expected_total_duration = np.sum(segmentation)
    actual_total_duration = np.sum(
        events.frame.select(['offset', 'onset']).to_numpy()[:, 0]
        - events.frame.select(['offset', 'onset']).to_numpy()[:, 1],
    )
    assert actual_total_duration == expected_total_duration


@pytest.mark.parametrize(
    'events, num_samples, expected_exception, expected_match',
    [
        pytest.param(
            pl.DataFrame({'onset': [2, 7], 'offset': [5, 9]}),
            -1,
            ValueError,
            'num_samples must be non-negative',
            id='negative_num_samples',
        ),
        pytest.param(
            pl.DataFrame({'foo': [2], 'offset': [5]}),
            10,
            ValueError,
            'not found in events',
            id='missing_onset_column',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2], 'bar': [5]}),
            10,
            ValueError,
            'not found in events',
            id='missing_offset_column',
        ),
        pytest.param(
            pl.DataFrame({'onset': [-1], 'offset': [5]}),
            10,
            ValueError,
            'must be non-negative',
            id='negative_onset',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2], 'offset': [-1]}),
            10,
            ValueError,
            'must be non-negative',
            id='negative_offset',
        ),
        pytest.param(
            pl.DataFrame({'onset': [5], 'offset': [5]}),
            10,
            ValueError,
            'Onset must be less than offset',
            id='onset_equal_offset',
        ),
        pytest.param(
            pl.DataFrame({'onset': [6], 'offset': [5]}),
            10,
            ValueError,
            'Onset must be less than offset',
            id='onset_greater_offset',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2], 'offset': [11]}),
            10,
            ValueError,
            'exceeds num_samples',
            id='offset_out_of_bounds',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2, 4], 'offset': [5, 7]}),
            10,
            ValueError,
            'Overlapping events detected',
            id='overlapping_events',
        ),
    ],
)
def test_events2segmentation_errors(
    events,
    num_samples,
    expected_exception,
    expected_match,
):
    with pytest.raises(expected_exception, match=expected_match):
        events2segmentation(events, num_samples)


@pytest.mark.parametrize(
    'segmentation, expected_exception, expected_match',
    [
        pytest.param(
            [0, 1, 0], TypeError, 'segmentation must be a numpy.ndarray', id='not_numpy_array',
        ),
        pytest.param(
            np.array([[0, 1], [1, 0]]),
            ValueError,
            'segmentation must be a 1D array',
            id='not_1d_array',
        ),
        pytest.param(
            np.array([0, 1, 2]),
            ValueError,
            'segmentation must only contain binary values',
            id='not_binary_array',
        ),
        pytest.param(
            np.array([0.0, 1.0, 0.5]),
            ValueError,
            'segmentation must only contain binary values',
            id='not_binary_float_array',
        ),
    ],
)
def test_segmentation2events_errors(segmentation, expected_exception, expected_match):
    with pytest.raises(expected_exception, match=expected_match):
        segmentation2events(segmentation)
