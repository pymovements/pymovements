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
"""Test the offset event measure."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.events import offset


def _events(onsets, durations):
    return pl.DataFrame(
        {'onset': onsets, 'duration': durations},
        schema={'onset': pl.Duration('us'), 'duration': pl.Duration('us')},
    )


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {'inclusive': True, 'sampling_rate': 1000},
            [4000, 7000, None, None],
            id='inclusive_last_sample_null_for_impulse_and_null',
        ),
        pytest.param(
            {'inclusive': False},
            [5000, 8000, 10_000, None],
            id='exclusive_one_past_the_end_onset_for_impulse',
        ),
        pytest.param(
            {'inclusive': False, 'sampling_rate': 1000},
            [5000, 8000, 10_000, None],
            id='exclusive_ignores_the_rate',
        ),
        pytest.param(
            {'inclusive': True, 'sampling_rate': 600},
            [3333, 6333, None, None],
            id='inclusive_rounds_the_interval_to_microseconds',
        ),
    ],
)
def test_offset_has_expected_result(kwargs, expected):
    events = _events([0, 5000, 10_000, 15_000], [5000, 3000, 0, None])

    result = events.select(offset(**kwargs))

    expected_frame = pl.DataFrame({'offset': expected}, schema={'offset': pl.Duration('us')})
    assert_frame_equal(result, expected_frame)


def test_offset_inclusive_without_sampling_rate_raises():
    with pytest.raises(ValueError, match='inclusive=True requires a sampling_rate'):
        offset(inclusive=True)


def test_offset_without_inclusive_raises():
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'inclusive'"):
        offset()  # pylint: disable=missing-kwoa


def test_offset_missing_duration_column_raises():
    events = pl.DataFrame({'onset': [0]}, schema={'onset': pl.Duration('us')})
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='duration'):
        events.select(offset(inclusive=False))
