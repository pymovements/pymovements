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
"""Test the duration event measure."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.events import duration
from pymovements.measure.events import offset


def _events(onsets, offsets):
    return pl.DataFrame(
        {'onset': onsets, 'offset': offsets},
        schema={'onset': pl.Duration('us'), 'offset': pl.Duration('us')},
    )


@pytest.mark.parametrize(
    ('kwargs', 'input_df', 'exception', 'message'),
    [
        pytest.param(
            {'offsets_inclusive': False},
            pl.DataFrame(schema={'onset': pl.Duration('us')}),
            pl.exceptions.ColumnNotFoundError,
            'offset',
            id='missing_offset_column',
        ),
        pytest.param(
            {'offsets_inclusive': False},
            pl.DataFrame(schema={'offset': pl.Duration('us')}),
            pl.exceptions.ColumnNotFoundError,
            'onset',
            id='missing_onset_column',
        ),
    ],
)
def test_duration_exceptions(kwargs, input_df, exception, message):
    expression = duration(**kwargs)
    with pytest.raises(exception, match=message):
        input_df.select([expression])


def test_duration_inclusive_without_sampling_rate_raises():
    with pytest.raises(ValueError, match='offsets_inclusive=True requires a sampling_rate'):
        duration(offsets_inclusive=True)


def test_duration_without_offsets_inclusive_raises():
    with pytest.raises(
            TypeError,
            match="missing 1 required keyword-only argument: 'offsets_inclusive'",
    ):
        duration()  # pylint: disable=missing-kwoa


@pytest.mark.parametrize(
    ('kwargs', 'input_df', 'expected'),
    [
        pytest.param(
            {'offsets_inclusive': False},
            _events([], []),
            [],
            id='empty_dataframe_results_in_empty_dataframe_with_correct_schema',
        ),
        pytest.param(
            {'offsets_inclusive': False},
            _events([0, 10_000, 20_000], [9000, 23_000, None]),
            [9000, 13_000, None],
            id='exclusive_offsets',
        ),
        pytest.param(
            {'offsets_inclusive': True, 'sampling_rate': 1000},
            _events([0, 10_000, 20_000], [9000, 23_000, None]),
            [10_000, 14_000, None],
            id='inclusive_offsets_add_the_interval',
        ),
        pytest.param(
            {'offsets_inclusive': True, 'sampling_rate': 600},
            _events([0], [9000]),
            [10_667],
            id='inclusive_offsets_round_the_interval_to_microseconds',
        ),
        pytest.param(
            {'offsets_inclusive': False, 'sampling_rate': 1000},
            _events([0], [9000]),
            [9000],
            id='exclusive_ignores_the_rate',
        ),
    ],
)
def test_duration_has_expected_result(kwargs, input_df, expected):
    result = input_df.select(duration(**kwargs))

    expected_frame = pl.DataFrame(
        {'duration': expected}, schema={'duration': pl.Duration('us')},
    )
    assert_frame_equal(result, expected_frame)


@pytest.mark.parametrize('inclusive', [True, False], ids=['inclusive', 'exclusive'])
def test_offset_then_duration_is_the_identity_from_one_interval(inclusive):
    events = pl.DataFrame(
        {'onset': [0, 10_000, 20_000], 'duration': [1000, 1500, 0]},
        schema={'onset': pl.Duration('us'), 'duration': pl.Duration('us')},
    )

    round_trip = (
        events
        .with_columns(offset(inclusive=inclusive, sampling_rate=1000))
        .select(duration(offsets_inclusive=inclusive, sampling_rate=1000))
    )

    # An impulse has no inclusive offset, so its duration does not survive the round trip.
    expected = [1000, 1500, None if inclusive else 0]
    assert round_trip['duration'].dt.total_microseconds().to_list() == expected
