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

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure import data_loss


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
            # 1 Hz: expected [0] (1 sample)
            # Observed: [0.0] (1 sample) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0.0], [1.0], 1.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='single_sample_1hz_float',
            ),

            # 1 Hz: expected [0.000, 0.001, 0.002, 0.003, 0.004] (5 samples)
            # Observed: [0.000, 0.001, 0.002, 0.003, 0.004] (5 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0.0, 0.001, 0.002, 0.003, 0.004], [1.0, 1.0, 1.0, 1.0, 1.0], 1.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid_1hz_float',
            ),

            # 1 Hz: expected [0.000, 0.001, 0.002, 0.003, 0.004] (5 samples)
            # Observed: [0, 0.001, 0.002, 0.004] (4 samples) -> 1 missing by time
            # Total missing = 1
            pytest.param(
                [0, 1000, 2000, 4000], [1.0, 1.0, 1.0, 1.0], 1.0,
                {'count': 1, 'time': 1.0, 'ratio': 1 / 5},
                id='one_missing_sample_1hz_float',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0, 1.0], 1000.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid_1khz_float',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                1000.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid_1khz_list',
            ),

            # 1 kHz: expected [100, 101, 102, 103, 104] (5 samples)
            # Observed: [100.0, 101.0, 102.0, 103.0, 104.0] (5 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [100.0, 101.0, 102.0, 103.0, 104.0], [1.0, 1.0, 1.0, 1.0, 1.0], 1000.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid_1khz_start_time_100',
            ),

            # 2 Hz: expected [0.0, 0.5, 1.0, 1.5] (4 samples)
            # Observed: 4 samples, all valid -> 0 missing
            pytest.param(
                [0.0, 0.5, 1.0, 1.5], [1.0, 1.0, 1.0, 1.0], 2000.0,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_missing_regular_all_valid_2khz',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 4.0] (4 samples) -> 1 missing by time
            # Total missing = 2
            pytest.param(
                [0.0, 1.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0], 1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_missing_sample_float_1khz',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 4.0] (4 samples) -> 1 missing
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 4.0],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_missing_sample_list',
            ),

            # 1 kHz: expected [100, 101, 102, 103, 104] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 4.0] (4 samples) -> 1 missing by time
            # Total missing = 1
            pytest.param(
                [100.0, 101.0, 102.0, 104.0], [1.0, 1.0, 1.0, 1.0], 1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_missing_sample_start_time_100',
            ),

            # 1 kHz: expected [0, 1, 2, 3] (4 samples)
            # Observed: [0.0, 3.0] (2 samples) -> 2 missing by time
            # Total missing = 2
            pytest.param(
                [0.0, 3.0], [1.0, 1.0], 1000.0,
                {'count': 2, 'time': 0.002, 'ratio': 2 / 4},
                id='half_missing_samples',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One null in values -> 1 missing by validity
            # Total missing = 1
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.0, None, 1.0, 1.0], 1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_null_sample_1khz_float',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One null in values -> 1 missing by validity
            # Total missing = 1
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.0, np.nan, 1.0, 1.0], 1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_nan_sample_1khz_float',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One inf in values -> 1 missing by validity
            # Total missing = 1
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.0, np.inf, 1.0, 1.0], 1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_inf_sample_1khz_float',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One null in values -> 1 missing by validity
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [[1.0, 1.0], None, [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_null_sample_1khz_list',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One null component in values -> 1 missing by validity
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [[1.0, 1.0], [None, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_null_x_component_1khz_list',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 3.0, 4.0] (5 samples) -> 0 missing
            # One null component in values -> 1 missing by validity
            # Total missing = 0
            pytest.param(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [[1.0, 1.0], [1.0, None], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                1000.0,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 5},
                id='one_null_y_component_1khz_list',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4] (5 samples)
            # Observed: [0.0, 1.0, 2.0, 4.0] (4 samples) -> 1 missing by time
            # One NaN in values -> 1 missing by validity
            # Total missing = 2
            pytest.param(
                [0.0, 1.0, 2.0, 4.0], [1.0, 1.0, nan, 1.0], 1000.0,
                {'count': 2, 'time': 0.002, 'ratio': 2 / 5},
                id='gap_plus_nan_invalid',
            ),

            # 1 kHz: expected [0, 1, 2, 3, 4, 5, 6, 7] (8 samples)
            # Observed: [0, 1, 2, 6, 7] (5 samples) -> 3 missing by time
            # One None in values -> 1 missing by validity
            # Total missing = 4
            pytest.param(
                [0.0, 1.0, 2.0, 6.0, 7.0], [1.0, None, 1.0, 1.0, 1.0], 1000.0,
                {'count': 4, 'time': 0.004, 'ratio': 4 / 8},
                id='large_gap_with_invalid',
            ),
        ],
    )
    def test_internal_gaps_and_invalids(
            self, unit, expected_col, times, values, sampling_rate, expected_vals,
    ):
        """Test data loss with internal time gaps and invalid values."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = data_loss('value', sampling_rate=sampling_rate, unit=unit)
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ('times', 'values', 'sampling_rate', 'start', 'end', 'expected_vals'),
        [
            # 1 Hz, start=0, end=5: expected [0, 1000, 2000, 3000, 4000, 5000] (6 samples)
            # Observed: [0, 1000, 2000, 3000, 4000, 5000] (6 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0, 1000, 2000, 3000, 4000, 5000], [1, 2, 3, 4, 5, 6], 1, 0, 5,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_samples_missing_1hz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (6 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], 1000, 0, 5,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_samples_missing_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (6 samples) -> 0 missing
            # Total missing = 0
            pytest.param(
                [100, 101, 102, 103, 104, 105], [1, 2, 3, 4, 5, 6], 1000, 100, 105,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='no_samples_missing_100_start_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 5] (5 samples) -> 1 missing
            # Total missing = 1
            pytest.param(
                [0, 1, 2, 3, 5], [1, 2, 3, 4, 6], 1000, 0, 5,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 6},
                id='one_sample_missing_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 3, 5] (4 samples) -> 2 missing
            # Total missing = 2
            pytest.param(
                [0, 1, 3, 5], [1, 3, 4, 6], 1000, 0, 5,
                {'count': 2, 'time': 0.002, 'ratio': 2 / 6},
                id='two_samples_missing_1khz',
            ),

            # 1000 Hz, start=0, end=2000: expected [0, 500, 1000, 1500, 2000] (5 samples)
            # Observed: [0, 500, 1000, 1500] (4 samples) -> 1 missing
            # Total missing = 1
            pytest.param(
                [0, 500, 1000, 1500], [1, 2, 4, 6], 2, 0, 2000,
                {'count': 1, 'time': 0.5, 'ratio': 1 / 5},
                id='one_sample_missing_2hz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (5 samples) -> 0 missing
            # One nan in values -> 1 missing by validity
            # Total missing = 1
            pytest.param(
                [0, 1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, nan, 6.0], 1000, 0, 5,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 6},
                id='one_sample_nan_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (5 samples) -> 0 missing
            # Two nan in values -> 2 missing by validity
            # Total missing = 2
            pytest.param(
                [0, 1, 2, 3, 4, 5], [1.0, np.inf, 3.0, 4.0, np.inf, 6.0], 1000, 0, 5,
                {'count': 2, 'time': 0.002, 'ratio': 2 / 6},
                id='two_samples_nan_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (5 samples) -> 0 missing
            # One inf in values -> 1 missing by validity
            # Total missing = 1
            pytest.param(
                [0, 1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, np.inf, 6.0], 1000, 0, 5,
                {'count': 1, 'time': 0.001, 'ratio': 1 / 6},
                id='one_sample_inf_1khz',
            ),

            # 1000 Hz, start=0, end=5: expected [0, 1, 2, 3, 4, 5] (6 samples)
            # Observed: [0, 1, 2, 3, 4, 5] (5 samples) -> 0 missing
            # Two inf in values -> 2 missing by validity
            # Total missing = 2
            pytest.param(
                [0, 1, 2, 3, 4, 5], [1.0, 2.0, np.inf, 4.0, np.inf, 6.0], 1000, 0, 5,
                {'count': 2, 'time': 0.002, 'ratio': 2 / 6},
                id='two_samples_inf_1khz',
            ),

            # 1 Hz, start=9, end=13: expected [10, 1010, 2010, 3010, 4010] (5 samples)
            # Observed: [10, 1010, 2010] (3 samples) -> 2 missing by time
            # One inf in values -> 1 missing by validity
            # Total missing = 3
            pytest.param(
                [10.0, 1010.0, 2010.0], [1.0, inf, 1.0], 1.0, 10.0, 4010.0,
                {'count': 3, 'time': 3.0, 'ratio': 3 / 5},
                id='one_sample_inf_1hz_start_10',
            ),

            # 1 Hz, start=9, end=13: expected [9, 1009, 2009, 3009, 4009] (5 samples)
            # Observed: [10, 1010, 2010] (3 samples) -> 2 missing by time
            # One inf in values -> 1 missing by validity
            # Total missing = 3
            pytest.param(
                [10.0, 1010.0, 2010.0], [1.0, inf, 1.0], 1.0, 9.0, 4009,
                {'count': 3, 'time': 3.0, 'ratio': 3 / 5},
                id='one_sample_inf_1hz_start_9_shifted',
            ),

            # 10/3 Hz, start=0, end=1000: expected [0, 333.3, 666.6, 1000] (4 samples)
            # Observed: [0, 234.0, 678.0, 1000.0] (3 samples) -> 0 missing by time
            # Total missing = 1
            pytest.param(
                [0.0, 234.0, 678.0, 1000.0], [1.0, 1.0, 1.0, 1.0], 10 / 3, 0.0, 1000,
                {'count': 0, 'time': 0.0, 'ratio': 0.0},
                id='irregular_complete_333hz',
            ),

            # 10/3 Hz, start=0, end=1000: expected [0, 333.3, 666.6, 1000] (4 samples)
            # Observed: [0, 234.0, 678.0] (3 samples) -> 1 missing by time
            # Total missing = 1
            pytest.param(
                [0.0, 234.0, 678.0], [1.0, 1.0, 1.0], 10 / 3, 0.0, 1000,
                {'count': 1, 'time': 0.3, 'ratio': 1 / 4},
                id='irregular_one_missing_3.33hz',
            ),

            # 10/3 Hz, start=0, end=1000: expected [0, 333.3, 666.6, 1000] (4 samples)
            # Observed: [0, 234.0, 678.0, 1000.0] (3 samples) -> 0 missing by time
            # Two None in values -> 2 missing by validity
            # Total missing = 2
            pytest.param(
                [0.0, 234.0, 678.0, 1000.0], [1.0, None, None, 1.0], 10 / 3, 0.0, 1000,
                {'count': 2, 'time': 0.6, 'ratio': 2 / 4},
                id='irregular_with_none_333hz',
            ),
        ],
    )
    def test_explicit_bounds(
            self, unit, expected_col, times, values, sampling_rate, start, end,
            expected_vals,
    ):
        """Test data loss with explicit start and end bounds."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = data_loss(
            'value', sampling_rate=sampling_rate, start_time=start, end_time=end, unit=unit,
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
        expr = data_loss('value', sampling_rate=1.0, unit=unit)
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
                {'count': 2, 'time': 0.002, 'ratio': 2 / 5},
                id='unsorted_with_neginf',
            ),
        ],
    )
    def test_unsorted_input(self, unit, expected_col, times, values, expected_vals):
        """Test data loss with unsorted input timestamps."""
        df = pl.DataFrame({'time': times, 'value': values})
        expr = data_loss('value', sampling_rate=1000.0, unit=unit)
        result = df.select(expr)
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_single_sample_invalid(self, unit, expected_col):
        """Test data loss with a single sample that is invalid."""
        df = pl.DataFrame({'time': [5.0], 'value': [None]})
        expr = data_loss('value', sampling_rate=1.0, unit=unit)
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
        expr = data_loss(
            'value', sampling_rate=1.0, start_time=5.0,
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
        expr = data_loss(
            'value', sampling_rate=1.0, start_time=0.0,
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
        expr = data_loss('pixel', sampling_rate=1000.0, unit=unit)
        result = df.select(expr)
        expected_vals = {'count': 6, 'time': 0.006, 'ratio': 6 / 9}
        expected = pl.DataFrame({expected_col: [expected_vals[unit]]})
        assert_frame_equal(result, expected, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize('bad_unit', ['invalid', '', None, 'COUNT'])
def test_data_loss_invalid_unit_raises(bad_unit):
    """Test that providing an invalid unit raises a ValueError."""
    message = re.escape(
        f"unit must be one of ('count', 'time', 'ratio') but got: {repr(bad_unit)}",
    )
    with pytest.raises(ValueError, match=message):
        data_loss('value', sampling_rate=1.0, unit=bad_unit)


@pytest.mark.parametrize('bad_time_column', [123, None, {'col': 'time'}])
def test_data_loss_invalid_time_column_raises(bad_time_column):
    """Test that providing an invalid time column type raises a TypeError."""
    message = (
        f"invalid type for 'time_column'. Expected 'str' , got "
        f"'{type(bad_time_column).__name__}'"
    )
    with pytest.raises(TypeError, match=message):
        data_loss('value', time_column=bad_time_column, sampling_rate=1.0)


@pytest.mark.parametrize('bad_sampling_rate', [0, -1, 0.0, -10.5, '1Hz'])
def test_data_loss_invalid_sampling_rate_raises(bad_sampling_rate):
    """Test that providing an invalid sampling rate raises a ValueError."""
    message = (
        f"sampling_rate must be a positive number, but got: {repr(bad_sampling_rate)}"
    )
    with pytest.raises(ValueError, match=message):
        data_loss('value', sampling_rate=bad_sampling_rate)


def test_data_loss_invalid_range_raises_python():
    """Test that providing an invalid time range raises a ValueError."""
    message = r'end_time \(0.0\) must be greater than or equal to start_time \(1.0\)'
    with pytest.raises(ValueError, match=message):
        data_loss('value', sampling_rate=1.0, start_time=1.0, end_time=0.0)


def test_data_loss_time_column_not_found():
    """Test that exception is raised if time column is not found."""
    df = pl.DataFrame({'time': [1, 2, 3], 'value': [0, 1, 2]})
    expr = data_loss('value', time_column='t', sampling_rate=1000)
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='t'):
        df.select(expr)
