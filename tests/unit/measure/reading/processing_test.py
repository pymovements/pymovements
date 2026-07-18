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
"""Reading measure tests."""
import polars as pl
import pytest

from pymovements.measure.reading.processing import compute_reading_measures


@pytest.mark.parametrize(
    'fixations_df, aoi_df, expected_results',
    [
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 2, 3],
                    'duration': [100, 110, 120, 130],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'character': ['a', 'b', 'c'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100, 'FPRT': 100, 'TFC': 1},
                1: {'FFD': 110, 'TFT': 230, 'FPRT': 230, 'TFC': 2},
                2: {'FFD': 130, 'TFT': 130, 'FPRT': 130, 'TFC': 1},
            },
            id='forward',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 1, 3],
                    'duration': [100, 110, 120, 130],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'character': ['a', 'b', 'c'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 220, 'FPRT': 100, 'RRT': 120, 'TFC': 2},
                1: {'FFD': 110, 'TFT': 110, 'FPRT': 110, 'TFC': 1},
                2: {'FFD': 130, 'TFT': 130, 'FPRT': 130, 'TFC': 1},
            },
            id='regression',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 3],
                    'duration': [100, 130],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'character': ['a', 'b', 'c'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100, 'TFC': 1},
                1: {'FFD': 0, 'TFT': 0, 'TFC': 0},
                2: {'FFD': 130, 'TFT': 130, 'TFC': 1},
            },
            id='skipping',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 0, 2],
                    'duration': [100, 100, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2],
                    'character': ['a', 'b'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100, 'TFC': 1},
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1},
            },
            id='out_of_bounds_aois',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1],
                    'duration': [100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1],
                    'character': ['a'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'SFD': 100},
            },
            id='single_word_sfd',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'duration': [100, 100, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 4],
                    'character': ['a', 'd'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100},
                3: {'FFD': 0, 'TFT': 0},
            },
            id='missing_aois_in_middle',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': pl.Series(['not_an_int'], dtype=pl.Utf8),
                    'duration': [100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1],
                    'character': ['a'],
                },
            ),
            {
                0: {'FFD': 0, 'TFT': 0},
            },
            id='invalid_type_aoi',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1],
                    'duration': [None],
                },
                schema={'aoi': pl.Int64, 'duration': pl.Int64},
            ),
            pl.DataFrame(
                {
                    'aoi': [1],
                    'character': ['a'],
                },
            ),
            {
                0: {'FFD': 0, 'TFT': 0},
            },
            id='null_duration',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 1],
                    'duration': [100, 100, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2],
                    'character': ['a', 'b'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 200, 'TFC': 2, 'TRC_out': 1},
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'TRC_out': 1},
            },
            id='trc_out',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 1, 2],
                    'duration': [100, 100, 100, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2],
                    'character': ['a', 'b'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 200, 'TRC_out': 0, 'FPRT': 100},
                1: {'FFD': 100, 'TFT': 200, 'TRC_out': 2, 'FPRT': 100},
            },
            id='trc_out_multiple_passes',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'aoi': [1, 2, 2, 3],
                    'duration': [100, 100, 0, 100],
                },
            ),
            pl.DataFrame(
                {
                    'aoi': [1, 2, 3],
                    'character': ['a', 'b', 'c'],
                },
            ),
            {
                0: {'FFD': 100, 'TFT': 100},
                1: {'FFD': 100, 'TFT': 100},
                2: {'FFD': 100, 'TFT': 100},
            },
            id='zero_duration_fixation',
        ),
    ],
)
def test_compute_reading_measures(fixations_df, aoi_df, expected_results):
    result = compute_reading_measures(fixations_df, aoi_df)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(aoi_df)

    for word_idx, expected in expected_results.items():
        row = result.filter(pl.col('word_index') == word_idx)
        assert not row.is_empty()
        for col, val in expected.items():
            assert row[col][0] == val
