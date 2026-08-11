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
    ('fixations', 'aois', 'expected_results'),
    [
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 2, 3], 'duration': [100, 110, 120, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'FPRT': 100, 'TFC': 1, 'SL_in': 0, 'SL_out': 1},
                2: {'FFD': 110, 'TFT': 230, 'FPRT': 230, 'TFC': 2, 'SFD': 0},
                # last fixated word: no spurious regression out, no negative saccade out
                3: {'FFD': 130, 'TFT': 130, 'TFC': 1, 'TRC_out': 0, 'SL_out': 0},
            },
            id='forward',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1, 3], 'duration': [100, 110, 120, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 220, 'FPRT': 100, 'RRT': 120, 'TFC': 2, 'TRC_in': 1, 'SL_in': 0},
                2: {'FFD': 110, 'TFT': 110, 'TRC_out': 1, 'SL_out': -1, 'RPD_exc': 120, 'FPReg': 1},
                3: {'FFD': 130, 'TFT': 130, 'TRC_out': 0, 'SL_out': 0, 'SL_in': 2},
            },
            id='regression',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 3], 'duration': [100, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'SL_out': 2},
                2: {'FFD': 0, 'TFT': 0, 'TFC': 0, 'Fix': 0},
                3: {'FFD': 130, 'TFT': 130, 'TFC': 1, 'SL_in': 2, 'SL_out': 0},
            },
            id='skipping',
        ),
        pytest.param(
            # Out-of-bounds fixations carry a null word index (as produced by map_to_aois) and are
            # ignored; the null between the two words must not create a spurious regression.
            pl.DataFrame(
                {'word_idx': [1, None, 2], 'duration': [100, 100, 100]},
                schema={'word_idx': pl.Int64, 'duration': pl.Int64},
            ),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'TRC_out': 0, 'SL_out': 1},
                2: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'SL_in': 1},
            },
            id='out_of_bounds_null',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'SFD': 100, 'SL_in': 0, 'SL_out': 0},
            },
            id='single_word_sfd',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 3], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 4], 'word': ['a', 'd']}),
            {
                1: {'FFD': 100, 'TFT': 100},
                4: {'FFD': 0, 'TFT': 0, 'TFC': 0},
            },
            id='missing_aois_in_middle',
        ),
        pytest.param(
            # Non-integer word indices become null and are dropped, leaving the word unfixated.
            pl.DataFrame(
                {'word_idx': pl.Series(['not_an_int'], dtype=pl.Utf8), 'duration': [100]},
            ),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            {
                1: {'FFD': 0, 'TFT': 0, 'TFC': 0},
            },
            id='invalid_type_aoi',
        ),
        pytest.param(
            pl.DataFrame(
                {'word_idx': [1], 'duration': [None]},
                schema={'word_idx': pl.Int64, 'duration': pl.Int64},
            ),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            {
                1: {'FFD': 0, 'TFT': 0},
            },
            id='null_duration',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                # regression out belongs to word 2 (2 -> 1), not to the last fixated word 1
                1: {'TFT': 200, 'TFC': 2, 'TRC_in': 1, 'TRC_out': 0, 'RR': 1},
                2: {'TFT': 100, 'TFC': 1, 'TRC_out': 1, 'SL_out': -1},
            },
            id='trc_out',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1, 2], 'duration': [100, 100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'TFT': 200, 'FPRT': 100, 'TRC_out': 0, 'TRC_in': 1},
                2: {'TFT': 200, 'FPRT': 100, 'TRC_out': 1, 'SL_out': -1},
            },
            id='trc_out_multiple_passes',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 2, 3], 'duration': [100, 100, 0, 100]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100},
                # two first-pass fixations (one zero-duration) -> not a single fixation
                2: {'FFD': 100, 'TFT': 100, 'SFD': 0, 'TFC': 2},
                3: {'FFD': 100, 'TFT': 100, 'TRC_out': 0},
            },
            id='zero_duration_fixation',
        ),
    ],
)


def test_compute_reading_measures(fixations, aois, expected_results):
    result = compute_reading_measures(fixations, aois)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == aois['word_idx'].n_unique()

    for word_idx, expected in expected_results.items():
        row = result.filter(pl.col('word_index') == word_idx)
        assert not row.is_empty()
        for col, val in expected.items():
            assert row[col][0] == val
