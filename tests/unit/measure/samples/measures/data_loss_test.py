# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Test data_loss sample measure."""
from __future__ import annotations

import re
from math import inf
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('unit', 'expected_col'),
    [
        pytest.param('count', 'data_loss_count'),
        pytest.param('time', 'data_loss_time'),
        pytest.param('ratio', 'data_loss_ratio'),
    ],
)
class TestDataLoss:
    """Test data_loss sample measure."""

    @pytest.mark.parametrize(
        ('times', 'values', 'sampling_rate', 'expected_vals'),
        [
            # 1 Hz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 4.0] (4 samples) -> 1 missing by time
            # One NaN in values -> 1 missing by validity
            # Total missing = 2
            pytest.param(
                [0.0, 1.0, 2.0, 4.0], [1.0, 1.0, nan, 1.0], 1.0,
                {'count': 2, 'time': 2.0, 'ratio': 2 / 5},
                id='gap_plus_nan_invalid',
            ),

            # 1 Hz: expected [0, 1, 2, 3, 4, 5, 6, 7] (8 samples)
            # Observed: [0, 1, 2, 6, 7] (5 samples) -> 3 missing by time
            # One None in values -> 1 missing by validity
            # Total missing = 4
            pytest.param(
                [0.0, 1.0, 2.0, 6.0, 7.0], [1.0, None, 1.0, 1.0, 1.0], 1.0,
                {'count': 4, 'time': 4.0, 'ratio': 4 / 8},
                id='large_gap_with_invalid',
            ),

            # 2 Hz: expected [0.0, 0.5, 1.0, 1.5] (4 samples)
            # Observed: 4 samples, all valid -> 0 missing
            pytest.param(
                [0.0, 0.5, 1.0, 1.5], [1.0, 1.0, 1.0, 1.0], 2.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid',
            ),
        ],
    )
    def test_internal_gaps_and_invalids(
            self, unit, expected_col, times, values, sampling_rate,
            expected_vals,
    ):
        """Test data loss with internal time gaps and invalid values."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = pm.measure.data_loss('time', 'value', sampling_rate=sampling_rate, unit=unit)
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ('times', 'values', 'sampling_rate', 'start', 'end', 'expected_vals'),
        [
            # 1 Hz, start=9, end=13: expected [9, 10, 11, 12, 13] (5 samples)
            # Observed: [10, 11, 12] (3 samples) -> 2 missing by time
            # One inf in values -> 1 missing by validity
            # Total missing = 3
            pytest.param(
                [10.0, 11.0, 12.0], [1.0, inf, 1.0], 1.0, 9.0, 13.0,
                {'count': 3, 'time': 3.0, 'ratio': 3 / 5},
                id='explicit_bounds_with_inf',
            ),

            # 1/3 Hz, start=0, end=12: expected [0, 3, 6, 9, 12] (5 samples)
            # Observed: [0, 2, 6] (3 samples) -> 2 missing by time
            # One None in values -> 1 missing by validity
            # Total missing = 3
            pytest.param(
                [0.0, 2.0, 6.0], [1.0, None, 1.0], 1 / 3, 0.0, 12.0,
                {'count': 3, 'time': 9.0, 'ratio': 3 / 5},
                id='irregular_with_none',
            ),
        ],
    )
    def test_bounds_missing(
            self, unit, expected_col, times, values, sampling_rate, start, end,
            expected_vals,
    ):
        """Test data loss with explicit start and end bounds."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = pm.measure.data_loss(
            'time', 'value', sampling_rate=sampling_rate, start_time=start, end_time=end, unit=unit,
        )
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ('value', 'expected_vals'),
        [
            pytest.param(inf, {'count': 1, 'time': 1.0, 'ratio': 1.0}, id='pos_inf'),
            pytest.param(-inf, {'count': 1, 'time': 1.0, 'ratio': 1.0}, id='neg_inf'),
        ],
    )
    def test_inf_missing(self, unit, expected_col, value, expected_vals):
        """Test data loss when input contains infinite values."""
        df = pl.DataFrame({'time': [0.0], 'value': [value]})
        expr = pm.measure.data_loss('time', 'value', sampling_rate=1.0, unit=unit)
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ('times', 'values', 'expected_vals'),
        [
            # min=0, max=4, 1 Hz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: 4 samples -> 1 missing by time
            # One -inf in values -> 1 missing by validity
            # Total missing = 2
            pytest.param(
                [3.0, 1.0, 0.0, 4.0], [1.0, 1.0, -inf, 1.0],
                {'count': 2, 'time': 2.0, 'ratio': 2 / 5},
                id='unsorted_with_neginf',
            ),
        ],
    )
    def test_unsorted_input(self, unit, expected_col, times, values, expected_vals):
        """Test data loss with unsorted input timestamps."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = pm.measure.data_loss('time', 'value', sampling_rate=1.0, unit=unit)
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_single_sample_invalid(self, unit, expected_col):
        """Test data loss with a single sample that is invalid."""
        df = pl.DataFrame({'time': [5.0], 'value': [None]})
        expr = pm.measure.data_loss('time', 'value', sampling_rate=1.0, unit=unit)
        result = df.select(expr)
        expected_vals = {'count': 1, 'time': 1.0, 'ratio': 1.0}
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_degenerate_interval(self, unit, expected_col):
        """Test data loss with a degenerate interval (start == end)."""
        # end == start -> expected = (0*rate).floor() + 1 = 1 sample
        # Observed: 2 samples (1.0, 2.0) but we requested start=5, end=5
        # span = 0, expected = 1. observed = 2
        # time_missing = max(1 - 2, 0) = 0
        df = pl.DataFrame({'time': [1.0, 2.0], 'value': [1.0, 1.0]})
        expr = pm.measure.data_loss(
            'time', 'value', sampling_rate=1.0, start_time=5.0,
            end_time=5.0, unit=unit,
        )
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [0.0 if unit != 'count' else 0]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_clip_negative(self, unit, expected_col):
        """Test that data loss count is clipped to zero if observed > expected."""
        # Observed (3) > expected (floor(1.5*1)+1 = 2)
        # time_missing = max(2 - 3, 0) = 0
        df = pl.DataFrame({'time': [0.0, 1.0, 2.0], 'value': [1.0, 1.0, 1.0]})
        expr = pm.measure.data_loss(
            'time', 'value', sampling_rate=1.0, start_time=0.0,
            end_time=1.5, unit=unit,
        )
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [0.0 if unit != 'count' else 0]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_list_input(self, unit, expected_col):
        """Test data loss with list-valued input columns."""
        # 1 Hz, [1, 9] -> expected [1..9] (9 samples)
        # Observed: [1, 2, 3, 4, 5, 9] (6 samples) -> 3 missing by time
        # Invalid rows:
        # index 2: None
        # index 3: None
        # index 5: [1, None] -> contains None
        # Total invalid = 3
        # Total missing = 3 + 3 = 6
        df = pl.DataFrame({
            'time': [1, 2, 3, 4, 5, 9],
            'pixel': [[1, 1], [1, 1], None, None, [1, 1], [1, None]],
        })
        expr = pm.measure.data_loss('time', 'pixel', sampling_rate=1.0, unit=unit)
        result = df.select(expr)
        expected_vals = {'count': 6, 'time': 6.0, 'ratio': 6 / 9}
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize('bad_unit', ['invalid', '', None, 'COUNT'])
def test_data_loss_invalid_unit_raises(bad_unit):
    """Test that providing an invalid unit raises a ValueError."""
    message = re.escape(
        f"unit must be one of ('count', 'time', 'ratio') but got: {repr(bad_unit)}",
    )
    with pytest.raises(ValueError, match=message):
        pm.measure.data_loss('time', 'value', sampling_rate=1.0, unit=bad_unit)


@pytest.mark.parametrize('bad_time_column', [123, None, {'col': 'time'}])
def test_data_loss_invalid_time_column_raises(bad_time_column):
    """Test that providing an invalid time column type raises a TypeError."""
    message = (
        f"invalid type for 'time_column'. Expected 'str' , got "
        f"'{type(bad_time_column).__name__}'"
    )
    with pytest.raises(TypeError, match=message):
        pm.measure.data_loss(bad_time_column, 'value', sampling_rate=1.0)


@pytest.mark.parametrize('bad_sampling_rate', [0, -1, 0.0, -10.5, '1Hz'])
def test_data_loss_invalid_sampling_rate_raises(bad_sampling_rate):
    """Test that providing an invalid sampling rate raises a ValueError."""
    message = (
        f"sampling_rate must be a positive number, but got: {repr(bad_sampling_rate)}"
    )
    with pytest.raises(ValueError, match=message):
        pm.measure.data_loss('time', 'value', sampling_rate=bad_sampling_rate)


def test_data_loss_invalid_range_raises_python():
    """Test that providing an invalid time range raises a ValueError."""
    message = r'end_time \(0.0\) must be greater than or equal to start_time \(1.0\)'
    with pytest.raises(ValueError, match=message):
        pm.measure.data_loss('time', 'value', sampling_rate=1.0, start_time=1.0, end_time=0.0)
