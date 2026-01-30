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
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.segmentation import _has_overlap
from pymovements.events.segmentation import events2segmentation
from pymovements.events.segmentation import segmentation2events


@pytest.fixture(name='events_df')
def fixture_events_df():
    return pl.DataFrame({
        'name': ['blink', 'blink'],
        'onset': pl.Series([2, 7], dtype=pl.Int64),
        'offset': pl.Series([5, 9], dtype=pl.Int64),
    })


@pytest.mark.parametrize(
    'name, time_column, expected',
    [
        pytest.param(
            'blink',
            'time',
            [False, False, True, True, True, True, False, True, True, True],
            id='basic',
        ),
        pytest.param(
            'saccade',
            'time',
            [False] * 10,
            id='no_matching_events',
        ),
    ],
)
def test_events2segmentation_basic(events_df, name, time_column, expected):
    gaze_df = pl.DataFrame({'time': np.arange(10, dtype=np.int64)})
    result_expr = events2segmentation(events_df, name=name, time_column=time_column)
    result_df = gaze_df.select(result_expr)

    assert result_df.columns == [name]
    assert result_df[name].to_list() == expected


@pytest.mark.parametrize(
    'events_df, gaze_df, kwargs, expected',
    [
        pytest.param(
            pl.DataFrame({'name': ['blink'], 'start': [2], 'end': [5]}),
            pl.DataFrame({'timestamp': np.arange(10, dtype=np.int64)}),
            {
                'name': 'blink',
                'time_column': 'timestamp',
                'onset_column': 'start',
                'offset_column': 'end',
            },
            [False, False, True, True, True, True, False, False, False, False],
            id='custom_columns',
        ),
        pytest.param(
            pl.DataFrame({
                'name': ['blink', 'blink'],
                'onset': [2, 1],
                'offset': [4, 5],
                'trial': [1, 2],
            }),
            pl.DataFrame({
                'time': pl.Series([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=pl.Int64),
                'trial': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            }),
            {'name': 'blink', 'trial_columns': ['trial']},
            [
                False, False, True, True, True, False,  # Trial 1
                False, True, True, True, True, True,  # Trial 2
            ],
            id='trialized',
        ),
        pytest.param(
            pl.DataFrame({'name': ['blink'], 'onset': [-2], 'offset': [1]}),
            pl.DataFrame({'time': np.arange(-5, 5, dtype=np.int64)}),
            {'name': 'blink'},
            [False, False, False, True, True, True, True, False, False, False],
            id='negative_time',
        ),
        pytest.param(
            pl.DataFrame(
                {'name': [], 'onset': [], 'offset': []},
                schema={'name': pl.String, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.DataFrame({'time': np.arange(5, dtype=np.int64)}),
            {'name': 'blink'},
            [False] * 5,
            id='empty',
        ),
        pytest.param(
            pl.DataFrame({'name': ['saccade'], 'onset': [2], 'offset': [5]}),
            pl.DataFrame({'time': np.arange(10, dtype=np.int64)}),
            {'name': 'blink'},
            [False] * 10,
            id='mismatched_name',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2], 'offset': [5]}),
            pl.DataFrame({'time': np.arange(10, dtype=np.int64)}),
            {'name': 'blink'},
            [False, False, True, True, True, True, False, False, False, False],
            id='no_name_column',
        ),
    ],
)
def test_events2segmentation_advanced(events_df, gaze_df, kwargs, expected):
    result_expr = events2segmentation(events_df, **kwargs)
    result_df = gaze_df.select(result_expr)
    name = kwargs.get('name', 'blink')

    assert result_df[name].to_list() == expected


def test_events2segmentation_overlap_warning():
    events_df = pl.DataFrame({'name': ['blink', 'blink'], 'onset': [2, 4], 'offset': [5, 7]})
    gaze_df = pl.DataFrame({'time': np.arange(10, dtype=np.int64)})

    with pytest.warns(UserWarning, match='Overlapping events detected'):
        result_expr = events2segmentation(events_df, name='blink')

    result_df = gaze_df.select(result_expr)
    # 2, 3, 4, 5, 6, 7 are blink
    expected = [False, False, True, True, True, True, True, True, False, False]
    assert result_df['blink'].to_list() == expected


def test_events2segmentation_overlap_warning_trial_hint():
    events_df = pl.DataFrame({
        'name': ['blink', 'blink'],
        'onset': [2, 1],
        'offset': [4, 5],
        'trial': [1, 2],
    })
    with pytest.warns(UserWarning, match='Consider providing trial_columns'):
        events2segmentation(events_df, name='blink')


@pytest.mark.parametrize(
    'segmentation, name, expected_dict',
    [
        pytest.param(
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32),
            'blink',
            {'name': ['blink', 'blink'], 'onset': [2, 7], 'offset': [4, 8]},
            id='int32',
        ),
        pytest.param(
            np.array([False, False, True, True, True, False, False, True, True, False]),
            'blink',
            {'name': ['blink', 'blink'], 'onset': [2, 7], 'offset': [4, 8]},
            id='bool',
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64),
            'blink',
            {'name': ['blink', 'blink'], 'onset': [2, 7], 'offset': [4, 8]},
            id='int64',
        ),
        pytest.param(
            np.array([0, 0, 0], dtype=np.int32),
            'blink',
            {'name': [], 'onset': [], 'offset': []},
            id='empty',
        ),
        pytest.param(
            np.array([1, 1, 1], dtype=np.int32),
            'fixation',
            {'name': ['fixation'], 'onset': [0], 'offset': [2]},
            id='full',
        ),
    ],
)
def test_segmentation2events(segmentation, name, expected_dict):
    result_df = segmentation2events(segmentation, name=name)
    expected_df = pl.DataFrame(
        expected_dict, schema={
            'name': pl.String,
            'onset': pl.Int64,
            'offset': pl.Int64,
        },
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    'segmentation',
    [
        pytest.param(
            np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.int32),
            id='int32',
        ),
        pytest.param(
            np.array([False, True, True, False, True, False, False, True, True, True]),
            id='bool',
        ),
    ],
)
def test_roundtrip_indices(segmentation):
    name = 'event'
    events_df = segmentation2events(segmentation, name=name)

    gaze_df = pl.DataFrame({'time': np.arange(len(segmentation), dtype=np.int64)})
    result_expr = events2segmentation(events_df, name=name)
    result_df = gaze_df.select(result_expr)

    np.testing.assert_array_equal(result_df[name].to_numpy(), segmentation.astype(bool))


@pytest.mark.parametrize(
    'segmentation, time_column, expected_match',
    [
        pytest.param(
            np.array([0, 1, 0]), np.array([1, 2]),
            'length .* must match', id='mismatched_length',
        ),
    ],
)
def test_segmentation2events_invalid_parameters(segmentation, time_column, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        segmentation2events(segmentation, name='blink', time_column=time_column)


@pytest.mark.parametrize(
    'time_column, segmentation, expected_onset, expected_offset',
    [
        pytest.param(
            pl.Series([1, 2, 3], dtype=pl.Int64),
            np.array([0, 1, 0]),
            2,
            2,
            id='series',
        ),
        pytest.param(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1, 0]),
            2,
            2,
            id='numpy',
        ),
        pytest.param(
            np.array([100]),
            np.array([1]),
            100,
            100,
            id='single_sample_event',
        ),
    ],
)
def test_segmentation2events_time_column_types(
    time_column, segmentation, expected_onset, expected_offset,
):
    result_df = segmentation2events(segmentation, name='blink', time_column=time_column)
    assert result_df.get_column('onset')[0] == expected_onset
    assert result_df.get_column('offset')[0] == expected_offset


@pytest.mark.parametrize(
    'segmentation, time',
    [
        pytest.param(
            np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.int32),
            np.arange(100, 110, dtype=np.int64),
            id='int_time',
        ),
        pytest.param(
            np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.int32),
            np.linspace(0, 1, 10),
            id='float_time',
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1.1, 2.2, 3.3, 4.4]),
            id='float_time_with_end_event',
        ),
    ],
)
def test_roundtrip_time(segmentation, time):
    name = 'event'
    events_df = segmentation2events(segmentation, name=name, time_column=time)

    gaze_df = pl.DataFrame({'time': time})
    result_expr = events2segmentation(events_df, name=name, time_column='time')
    result_df = gaze_df.select(result_expr)

    np.testing.assert_array_equal(result_df[name].to_numpy(), segmentation.astype(bool))


@pytest.mark.parametrize(
    'events, name, expected_exception, expected_match',
    [
        pytest.param(
            pl.DataFrame({'foo': [2], 'offset': [5]}),
            'blink',
            ValueError,
            'not found in events',
            id='missing_onset_column',
        ),
        pytest.param(
            pl.DataFrame({'onset': [2], 'bar': [5]}),
            'blink',
            ValueError,
            'not found in events',
            id='missing_offset_column',
        ),
        pytest.param(
            pl.DataFrame({'name': ['blink'], 'onset': [6], 'offset': [5]}),
            'blink',
            ValueError,
            'Onset must be less than or equal to offset',
            id='onset_greater_offset',
        ),
    ],
)
def test_events2segmentation_errors(
    events,
    name,
    expected_exception,
    expected_match,
):
    with pytest.raises(expected_exception, match=expected_match):
        events2segmentation(events, name=name)


def test_events2segmentation_trialized_overlap_warning():
    events_df = pl.DataFrame({
        'name': ['blink', 'blink'],
        'onset': [2, 4],
        'offset': [5, 7],
        'trial': [1, 1],
    })
    gaze_df = pl.DataFrame({
        'time': np.arange(10, dtype=np.int64),
        'trial': [1] * 10,
    })

    with pytest.warns(UserWarning, match='Overlapping events detected for trial'):
        result_expr = events2segmentation(events_df, name='blink', trial_columns=['trial'])

    result_df = gaze_df.select(result_expr)
    # 2, 3, 4, 5, 6, 7 are blink
    expected = [False, False, True, True, True, True, True, True, False, False]
    assert result_df['blink'].to_list() == expected


@pytest.mark.parametrize(
    'faulty_segmentation, expected_exception, expected_match',
    [
        pytest.param(np.array([0, 1, 2]), ValueError, 'binary values', id='int_values_not_binary'),
        pytest.param(
            np.array([6.0, 7.0]), ValueError,
            'binary values', id='float_values_not_binary',
        ),
        pytest.param(np.array([1.1, 2.2]), ValueError, 'binary values', id='float_non_binary'),
        pytest.param(
            np.array([0.0, 1.0, 0.5]), ValueError,
            'binary values', id='not_binary_float_array',
        ),
        pytest.param(
            [0, 1, 0], TypeError,
            'must be a polars.Series or numpy.ndarray', id='list_input',
        ),
        pytest.param(
            np.array([[0, 1], [1, 0]]), ValueError, 'must be a 1D array', id='2d_array',
        ),
        pytest.param(
            pl.Series([0, 1, 0]), ValueError, 'trial_columns length .* must match',
            id='invalid_trial_length',
        ),
    ],
)
def test_segmentation2events_invalid_values(
    faulty_segmentation, expected_exception, expected_match,
):
    kwargs = {}
    if expected_match == 'trial_columns length .* must match':
        kwargs['trial_columns'] = pl.DataFrame({'trial': [1, 1]})

    with pytest.raises(expected_exception, match=expected_match):
        segmentation2events(faulty_segmentation, name='blink', **kwargs)


@pytest.mark.parametrize(
    'onsets, offsets, expected',
    [
        pytest.param(np.array([]), np.array([]), False, id='empty'),
        pytest.param(np.array([1]), np.array([2]), False, id='single_event'),
        pytest.param(np.array([1, 3]), np.array([2, 4]), False, id='no_overlap'),
        pytest.param(np.array([1, 2]), np.array([2, 3]), True, id='boundary_touch'),
        pytest.param(np.array([1, 4]), np.array([3, 5]), False, id='gap'),
        pytest.param(np.array([1, 2]), np.array([4, 3]), True, id='overlap_basic'),
        pytest.param(np.array([2, 1]), np.array([3, 2]), True, id='unsorted_no_overlap'),
        pytest.param(np.array([2, 1]), np.array([4, 3]), True, id='unsorted_overlap'),
        pytest.param(np.array([1, 2, 5]), np.array([3, 6, 7]), True, id='multiple_overlap'),
        pytest.param(np.array([1, 4, 7]), np.array([3, 6, 9]), False, id='multiple_no_overlap'),
    ],
)
def test_has_overlap(onsets, offsets, expected):
    assert _has_overlap(onsets, offsets) == expected


@pytest.mark.parametrize(
    'segmentation, name, trial_columns, expected_dict',
    [
        pytest.param(
            pl.Series([0, 1, 1, 0, 1, 1]),
            'blink',
            pl.DataFrame({'trial': [1, 1, 1, 2, 2, 2]}),
            {
                'name': ['blink', 'blink'],
                'onset': [1, 4],
                'offset': [2, 5],
                'trial': [1, 2],
            },
            id='basic',
        ),
        pytest.param(
            pl.Series([0, 0, 0]),
            'blink',
            pl.DataFrame({'trial': [1, 1, 1]}),
            {
                'name': [],
                'onset': [],
                'offset': [],
                'trial': [],
            },
            id='empty',
        ),
    ],
)
def test_segmentation2events_trialized(segmentation, name, trial_columns, expected_dict):
    result_df = segmentation2events(segmentation, name=name, trial_columns=trial_columns)

    expected_df = pl.DataFrame(
        expected_dict,
        schema={
            'name': pl.String,
            'onset': pl.Int64,
            'offset': pl.Int64,
            **trial_columns.schema,
        },
    )

    assert_frame_equal(result_df, expected_df)
