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
                1: {
                    'FFD': 100, 'TFT': 100, 'FPRT': 100, 'FPFC': 1, 'TFC': 1,
                    'SL_in': 0, 'SL_out': 1,
                },
                2: {'FFD': 110, 'TFT': 230, 'FPRT': 230, 'FPFC': 2, 'TFC': 2, 'SFD': 0},
                # last fixated word: no spurious regression out, no negative saccade out
                3: {'FFD': 130, 'TFT': 130, 'FPFC': 1, 'TFC': 1, 'TRC_out': 0, 'SL_out': 0},
            },
            id='forward',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1, 3], 'duration': [100, 110, 120, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {
                    'FFD': 100, 'TFT': 220, 'FPRT': 100, 'FPFC': 1, 'RRT': 120, 'TFC': 2,
                    'TRC_in': 1, 'SL_in': 0,
                },
                2: {
                    'FFD': 110, 'TFT': 110, 'FPFC': 1, 'TRC_out': 1, 'SL_out': -1,
                    'RPD_exc': 120, 'FPReg': 1,
                },
                3: {'FFD': 130, 'TFT': 130, 'FPFC': 1, 'TRC_out': 0, 'SL_out': 0, 'SL_in': 2},
            },
            id='regression',
        ),
        pytest.param(
            # Pins the intent of main's dropped old-engine test
            # test_regression_path_duration_no_first_pass: word 1 is entered only from the
            # right, so it never has a first pass and never opens a regression-path window.
            pl.DataFrame({'word_idx': [2, 1, 2], 'duration': [100, 150, 120]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {
                    'RPD_inc': 0, 'RPD_exc': 0, 'RBRT': 0, 'FFD': 0, 'FPRT': 0, 'RRT': 150,
                    'TFT': 150, 'FPFC': 0, 'TFC': 1, 'TRC_in': 1, 'SL_in': -1,
                },
                2: {
                    'RPD_inc': 370, 'RPD_exc': 150, 'RBRT': 220, 'FFD': 100, 'FPRT': 100,
                    'RRT': 120, 'TFT': 220, 'FPFC': 1, 'TRC_out': 1, 'SL_out': -1, 'FPReg': 1,
                },
            },
            id='rpd_no_first_pass',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 3], 'duration': [100, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TFC': 1, 'SL_out': 2, 'skipped': 0},
                2: {'FFD': 0, 'TFT': 0, 'FPFC': 0, 'TFC': 0, 'Fix': 0, 'skipped': 1},
                3: {
                    'FFD': 130, 'TFT': 130, 'FPFC': 1, 'TFC': 1, 'SL_in': 2,
                    'SL_out': 0, 'skipped': 0,
                },
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
                1: {'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TFC': 1, 'TRC_out': 0, 'SL_out': 1},
                2: {'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TFC': 1, 'SL_in': 1},
            },
            id='out_of_bounds_null',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            {
                1: {
                    'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TFC': 1, 'SFD': 100,
                    'SL_in': 0, 'SL_out': 0,
                },
            },
            id='single_word_sfd',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 3], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 4], 'word': ['a', 'd']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'FPFC': 1},
                4: {'FFD': 0, 'TFT': 0, 'FPFC': 0, 'TFC': 0},
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
                1: {'FFD': 0, 'TFT': 0, 'FPFC': 0, 'TFC': 0},
            },
            id='invalid_type_aoi',
            marks=pytest.mark.filterwarnings(
                'ignore:no fixations left to annotate',
            ),
        ),
        pytest.param(
            # Fractional word indices are no valid word indices: they become null and are
            # dropped instead of being silently truncated to the next lower word.
            pl.DataFrame({'word_idx': [1.0, 2.7], 'duration': [100, 110]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TFC': 1, 'SL_out': 0},
                2: {'FFD': 0, 'TFT': 0, 'FPFC': 0, 'TFC': 0, 'skipped': 1},
            },
            id='fractional_word_idx',
        ),
        pytest.param(
            pl.DataFrame(
                {'word_idx': [1], 'duration': [None]},
                schema={'word_idx': pl.Int64, 'duration': pl.Int64},
            ),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            {
                # the null-duration fixation contributes no time but still counts as a
                # first-pass fixation
                1: {'FFD': 0, 'TFT': 0, 'FPFC': 1},
            },
            id='null_duration',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                # regression out belongs to word 2 (2 -> 1), not to the last fixated word 1
                1: {'TFT': 200, 'FPFC': 1, 'TFC': 2, 'TRC_in': 1, 'TRC_out': 0, 'RR': 1},
                2: {'TFT': 100, 'FPFC': 1, 'TFC': 1, 'TRC_out': 1, 'SL_out': -1},
            },
            id='trc_out',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 1, 2], 'duration': [100, 100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'TFT': 200, 'FPRT': 100, 'FPFC': 1, 'TRC_out': 0, 'TRC_in': 1},
                2: {'TFT': 200, 'FPRT': 100, 'FPFC': 1, 'TRC_out': 1, 'SL_out': -1},
            },
            id='trc_out_multiple_passes',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 2, 2, 3], 'duration': [100, 100, 0, 100]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'FPFC': 1},
                # two first-pass fixations (one zero-duration) -> not a single fixation
                2: {'FFD': 100, 'TFT': 100, 'SFD': 0, 'FPFC': 2, 'TFC': 2},
                3: {'FFD': 100, 'TFT': 100, 'FPFC': 1, 'TRC_out': 0},
                # PoTeC reference (zero-duration folding, see the divergence tests below and
                # tests/functional/reading_measures_potec_test.py): the zero-duration fixation
                # is folded into the current word, so word 2 keeps single-fixation status:
                # 2: {'FFD': 100, 'TFT': 100, 'SFD': 100, 'TFC': 2},
                # 3: {'FFD': 0, 'TFT': 0, 'TRC_out': 0},  # last fixation never processed
            },
            id='zero_duration_fixation',
        ),
        pytest.param(
            # Sentinel indices without an AOI entry (here -1) are excluded from the sequence:
            # no spurious regression out of word 1 and no inflated saccade into word 2.
            pl.DataFrame({'word_idx': [1, -1, 2], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'TFT': 100, 'FPFC': 1, 'TRC_out': 0, 'SL_out': 1},
                2: {'TFT': 100, 'FPFC': 1, 'TRC_in': 0, 'SL_in': 1},
            },
            id='sentinel_out_of_aoi',
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


@pytest.mark.parametrize(
    ('fixations', 'aois', 'expected_results'),
    [
        pytest.param(
            # Divergence: the reference loop never processes the last fixation of a sequence,
            # pymovements processes all fixations.
            pl.DataFrame({'word_idx': [1, 2, 3], 'duration': [100, 110, 120]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                2: {'TFT': 110, 'TFC': 1, 'SL_out': 1},
                3: {'FFD': 120, 'TFT': 120, 'TFC': 1, 'Fix': 1, 'skipped': 0},
                # PoTeC reference (word 3 carries the last fixation, which is never processed):
                # 3: {'FFD': 0, 'TFT': 0, 'TFC': 0, 'Fix': 0, 'skipped': 1},
            },
            id='unprocessed_last_fixation',
        ),
        pytest.param(
            # Divergence: the reference loop starts from a -1 word sentinel, so SL_in of the
            # first fixated word equals its one-based word position; pymovements has no
            # previous fixation there and reports 0.
            pl.DataFrame({'word_idx': [2, 3], 'duration': [100, 110]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                2: {'SL_in': 0, 'TFT': 100, 'TFC': 1},
                # PoTeC reference (saccade from the -1 sentinel into word 2):
                # 2: {'SL_in': 2, 'TFT': 100, 'TFC': 1},
            },
            id='sl_in_sentinel_at_first_word',
        ),
        pytest.param(
            # Divergence: end-of-sequence handling. The reference loop still uses the final
            # fixation as lookahead for word 3 (same FRT/SL_out/TRC_out as pymovements) but
            # never processes it, leaving the last fixated word 2 without any measures. The
            # functional test masks FRT/SL_out/TRC_out at the last fixated words because it
            # feeds pymovements the sequence without the final fixation, which also removes
            # this lookahead transition.
            pl.DataFrame({'word_idx': [1, 3, 2], 'duration': [100, 110, 120]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                3: {'FRT': 110, 'SL_out': -1, 'TRC_out': 1},
                2: {'FRT': 120, 'TFT': 120, 'TFC': 1, 'TRC_in': 1, 'SL_in': -1, 'Fix': 1},
                # PoTeC reference (word 2 carries the last fixation, which is never processed):
                # 3: {'FRT': 110, 'SL_out': -1, 'TRC_out': 1},
                # 2: {'FRT': 0, 'TFT': 0, 'TFC': 0, 'TRC_in': 0, 'SL_in': 0, 'Fix': 0},
            },
            id='end_of_sequence_lookahead',
        ),
        pytest.param(
            # Divergence: FRT of a run that is still open at the end of the sequence. The
            # reference loop sets FRT only when the word is left, so a first run lasting until
            # the sequence end keeps FRT at 0; pymovements closes the run at the sequence end.
            pl.DataFrame({'word_idx': [1, 2, 2], 'duration': [100, 110, 120]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                2: {'FRT': 230, 'FPRT': 230, 'TFT': 230, 'TFC': 2, 'SFD': 0},
                # PoTeC reference (run never left before the sequence ended, last fixation
                # never processed):
                # 2: {'FRT': 0, 'FPRT': 110, 'TFT': 110, 'TFC': 1, 'SFD': 110},
            },
            id='frt_of_run_open_at_sequence_end',
        ),
        pytest.param(
            # Divergence: the reference loop folds a zero-duration fixation into the current
            # word (next_fix_word_idx = cur_fix_word_idx), pymovements treats it as an
            # ordinary fixation on its own word.
            pl.DataFrame({'word_idx': [1, 2, 3], 'duration': [100, 0, 110]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'TFT': 100, 'TFC': 1, 'SL_out': 1},
                2: {'TFT': 0, 'TFC': 1, 'FFD': 0, 'Fix': 0, 'skipped': 0},
                3: {'TFT': 110, 'TFC': 1},
                # PoTeC reference (the zero-duration fixation on word 2 is folded into word 1,
                # word 3 carries the last fixation, which is never processed):
                # 1: {'TFT': 100, 'TFC': 2, 'SL_out': 2},
                # 2: {'TFT': 0, 'TFC': 0, 'FFD': 0, 'Fix': 0, 'skipped': 1},
                # 3: {'TFT': 0, 'TFC': 0},
            },
            id='zero_duration_fixation_folded_into_current_word',
        ),
    ],
)
def test_compute_reading_measures_potec_reference_divergences(fixations, aois, expected_results):
    # Pins current pymovements behavior where it is known to diverge from the PoTeC reference
    # implementation. The commented-out lines next to each expectation give the values the
    # reference loop would produce for the same input; the functional comparison in
    # tests/functional/reading_measures_potec_test.py documents and masks these differences.
    result = compute_reading_measures(fixations, aois)

    for word_idx, expected in expected_results.items():
        row = result.filter(pl.col('word_index') == word_idx)
        assert not row.is_empty()
        for col, val in expected.items():
            assert row[col][0] == val, (word_idx, col)


def test_compute_reading_measures_broadcasts_aois_across_trials():
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 1],
        'duration': [100, 110, 120],
        'trial': ['t1', 't1', 't2'],
    })
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    with pytest.warns(UserWarning, match='broadcast'):
        result = compute_reading_measures(fixations, aois)

    assert result.columns[:3] == ['trial', 'word_index', 'word']
    assert result['trial'].to_list() == ['t1', 't1', 't2', 't2']
    assert result['word_index'].to_list() == [1, 2, 1, 2]
    assert result.filter(pl.col('trial') == 't2')['TFT'].to_list() == [120, 0]


def test_compute_reading_measures_aois_dict():
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 1],
        'duration': [100, 110, 120],
        'trial': ['t1', 't1', 't2'],
    })
    aois = {
        't1': pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
        't2': pl.DataFrame({'word_idx': [1], 'word': ['c']}),
    }

    result = compute_reading_measures(fixations, aois)

    assert result['trial'].to_list() == ['t1', 't1', 't2']
    assert result['word'].to_list() == ['a', 'b', 'c']
    assert result['TFT'].to_list() == [100, 110, 120]


def test_compute_reading_measures_aois_dict_warns_on_unused_keys():
    fixations = pl.DataFrame({
        'word_idx': [1, 2],
        'duration': [100, 110],
        'trial': ['t1', 't1'],
    })
    aois = {
        't1': pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
        't2': pl.DataFrame({'word_idx': [1], 'word': ['c']}),
    }

    with pytest.warns(
        UserWarning, match=r"keys without any matching fixations: \['t2'\]",
    ):
        result = compute_reading_measures(fixations, aois)

    # the unused entry stays in the output as a fully skipped word
    assert result['trial'].to_list() == ['t1', 't1', 't2']
    unused = result.filter(pl.col('trial') == 't2')
    assert unused['word'].to_list() == ['c']
    assert unused['TFT'].to_list() == [0]
    assert unused['TFC'].to_list() == [0]
    assert unused['Fix'].to_list() == [0]
    assert unused['skipped'].to_list() == [1]


def test_compute_reading_measures_aois_dict_warns_on_key_with_only_non_fixation_events():
    # t2 exists in the fixation frame but only via a saccade row, which never produces
    # measures, so its dict entry counts as unused and gets the warning.
    fixations = pl.DataFrame({
        'name': ['fixation', 'saccade'],
        'word_idx': [1, 1],
        'duration': [100, 50],
        'trial': ['t1', 't2'],
    })
    aois = {
        't1': pl.DataFrame({'word_idx': [1], 'word': ['a']}),
        't2': pl.DataFrame({'word_idx': [1], 'word': ['b']}),
    }

    with pytest.warns(
        UserWarning, match=r"keys without any matching fixations: \['t2'\]",
    ):
        result = compute_reading_measures(fixations, aois)

    assert result.filter(pl.col('trial') == 't2')['skipped'].to_list() == [1]


def test_compute_reading_measures_aois_dict_ignores_trials_with_only_non_fixation_events():
    # t2 exists in the fixation frame but only via a saccade row, so it does not demand a
    # dict entry.
    fixations = pl.DataFrame({
        'name': ['fixation', 'saccade'],
        'word_idx': [1, 1],
        'duration': [100, 50],
        'trial': ['t1', 't2'],
    })
    aois = {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})}

    result = compute_reading_measures(fixations, aois)

    assert result['trial'].to_list() == ['t1']
    assert result['TFT'].to_list() == [100]


def test_compute_reading_measures_aois_dict_mixed_char_idx_dtypes():
    # char_idx dtypes may differ between dict entries; they are cast to a common dtype and
    # LP is computed per entry.
    fixations = pl.DataFrame({
        'word_idx': [1, 1],
        'char_idx': [1, 2],
        'duration': [100, 110],
        'trial': ['t1', 't2'],
    })
    aois = {
        't1': pl.DataFrame({
            'word_idx': [1, 1],
            'word': ['ab', 'ab'],
            'char_idx': pl.Series([0, 1], dtype=pl.UInt32),
        }),
        't2': pl.DataFrame({
            'word_idx': [1, 1],
            'word': ['cd', 'cd'],
            'char_idx': pl.Series([2, 3], dtype=pl.Int64),
        }),
    }

    result = compute_reading_measures(fixations, aois)

    # t1: fixated char 1, word starts at char 0, one-based position 2; t2: fixated char 2,
    # word starts at char 2, position 1
    assert result['trial'].to_list() == ['t1', 't2']
    assert result['LP'].to_list() == [2, 1]


def test_compute_reading_measures_aois_dict_entries_with_page_column():
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 1],
        'duration': [100, 110, 120],
        'trial': ['t1', 't1', 't2'],
        'page': ['p1', 'p2', 'p1'],
    })
    aois = {
        't1': pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b'], 'page': ['p1', 'p2']}),
        't2': pl.DataFrame({'word_idx': [1], 'word': ['c'], 'page': ['p1']}),
    }

    result = compute_reading_measures(fixations, aois)

    assert result.columns[:4] == ['trial', 'page', 'word_index', 'word']
    assert result['trial'].to_list() == ['t1', 't1', 't2']
    assert result['page'].to_list() == ['p1', 'p2', 'p1']
    assert result['word'].to_list() == ['a', 'b', 'c']
    assert result['TFT'].to_list() == [100, 110, 120]


def test_compute_reading_measures_shared_trial_and_page_columns_preserved():
    fixations = pl.DataFrame({
        'word_idx': [1, 1],
        'duration': [100, 110],
        'trial': ['t1', 't1'],
        'page': ['p2', 'p1'],
    })
    aois = pl.DataFrame({
        'word_idx': [1, 1],
        'word': ['a', 'b'],
        'trial': ['t1', 't1'],
        'page': ['p1', 'p2'],
    })

    result = compute_reading_measures(fixations, aois)

    assert result.columns[:4] == ['trial', 'page', 'word_index', 'word']
    assert result['page'].to_list() == ['p1', 'p2']
    assert result['TFT'].to_list() == [110, 100]


@pytest.mark.parametrize(
    ('fixations', 'aois', 'match'),
    [
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1], 'word': ['a'], 'trial': ['t1']}),
            "aois has a 'trial' column",
            id='trial_only_in_aois',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1], 'word': ['a'], 'page': ['p1']}),
            "aois has a 'page' column",
            id='page_only_in_aois',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'page': ['p1']}),
            pl.DataFrame({'word_idx': [1], 'word': ['a']}),
            "fixations has a 'page' column",
            id='page_only_in_fixations',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': [7]}),
            pl.DataFrame({'word_idx': [1], 'word': ['a'], 'trial': ['t1']}),
            'dtype mismatch',
            id='trial_dtype_mismatch',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})},
            "no 'trial' column",
            id='dict_without_fixation_trial',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': ['t1']}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a'], 'trial': ['t1']})},
            "must not contain a 'trial' column",
            id='dict_entry_with_trial_column',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 1], 'duration': [100, 100], 'trial': ['t1', 't2']}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})},
            'without an entry in the aois dict',
            id='dict_missing_fixation_trial',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': ['t1']}),
            {},
            'aois dict is empty',
            id='empty_dict',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': [1]}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})},
            'aois dict keys are not compatible with the fixation trial dtype Int64',
            id='dict_key_dtype_mismatch',
        ),
        pytest.param(
            pl.DataFrame({
                'word_idx': [1, 1], 'duration': [100, 100],
                'trial': ['t1', 't2'], 'page': ['p1', 'p1'],
            }),
            {
                't1': pl.DataFrame({'word_idx': [1], 'word': ['a'], 'page': ['p1']}),
                't2': pl.DataFrame({'word_idx': [1], 'word': ['b']}),
            },
            "either all or no aois dict entries must have a 'page' column",
            id='dict_mixed_page_entries',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': ['t1']}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a'], 'page': ['p1']})},
            "aois has a 'page' column but fixations does not",
            id='dict_page_only_in_aois',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': ['t1'], 'page': ['p1']}),
            {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})},
            "fixations has a 'page' column but aois does not",
            id='dict_page_only_in_fixations',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 1], 'duration': [100, 100], 'trial': ['t1', 't2']}),
            {
                't1': pl.DataFrame({'word_idx': [1], 'word': ['a'], 'char_idx': [0]}),
                't2': pl.DataFrame({'word_idx': [1], 'word': ['b']}),
            },
            "either all or no aois dict entries must have a 'char_idx' column",
            id='dict_mixed_char_level_entries',
        ),
    ],
)
def test_compute_reading_measures_raises_value_error(fixations, aois, match):
    with pytest.raises(ValueError, match=match):
        compute_reading_measures(fixations, aois)


def test_compute_reading_measures_deduplicates_inconsistent_word_labels():
    fixations = pl.DataFrame({'word_idx': [1, 2], 'duration': [100, 200]})
    aois = pl.DataFrame({
        'word_idx': [1, 1, 2, None],
        'word': ['a', 'a ', 'b', 'junk'],
    })

    result = compute_reading_measures(fixations, aois)

    assert result['word_index'].to_list() == [1, 2]
    assert result['word'].to_list() == ['a', 'b']


def test_compute_reading_measures_count_dtypes_are_uint64():
    fixations = pl.DataFrame({'word_idx': [1, 2, 1], 'duration': [100, 110, 120]})
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    result = compute_reading_measures(fixations, aois)

    for column in ('TFC', 'TRC_in', 'TRC_out'):
        assert result.schema[column] == pl.UInt64


def test_compute_reading_measures_landing_position_within_word():
    # Char-level AOI table: word 1 spans chars 0-2, word 2 spans chars 3-7, word 3 chars 8-9.
    aois = pl.DataFrame({
        'word_idx': [1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
        'word': ['The'] * 3 + ['quick'] * 5 + ['ox'] * 2,
        'char_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    })
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 2],
        'char_idx': [1, 5, 3],
        'duration': [100, 110, 120],
    })

    result = compute_reading_measures(fixations, aois)

    # Word 1: first fixation on char 1, one-based position 2. Word 2: first fixation on char 5,
    # word starts at char 3, position 3. Word 3 was never fixated and gets 0.
    assert result['LP'].to_list() == [2, 3, 0]


def test_compute_reading_measures_landing_position_null_for_null_char_idx_fixation():
    # Char-level AOI table: word 1 spans chars 0-1, word 2 spans chars 3-4.
    aois = pl.DataFrame({
        'word_idx': [1, 1, 2, 2],
        'word': ['ab'] * 2 + ['cd'] * 2,
        'char_idx': [0, 1, 3, 4],
    })
    # Word 1 is fixated, but its first fixation carries no char_idx value.
    fixations = pl.DataFrame({
        'word_idx': [1, 2],
        'char_idx': [None, 4],
        'duration': [100, 100],
    })

    result = compute_reading_measures(fixations, aois)

    # Word 1: fixated (TFC 1, not skipped) but the landing position is not determinable, so LP
    # stays null instead of colliding with the 0 = never fixated fill. Word 2: first fixation on
    # char 4, word starts at char 3, one-based position 2.
    assert result['LP'].to_list() == [None, 2]
    assert result['TFC'].to_list() == [1, 1]
    assert result['skipped'].to_list() == [0, 0]


def test_compute_reading_measures_custom_group_columns():
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 1],
        'duration': [100, 110, 120],
        'subject_id': [1, 1, 2],
    })
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    with pytest.warns(UserWarning, match='broadcast'):
        result = compute_reading_measures(fixations, aois, group_columns=['subject_id'])

    assert result.columns[:3] == ['subject_id', 'word_index', 'word']
    assert result.filter(pl.col('subject_id') == 2)['TFT'].to_list() == [120, 0]


@pytest.mark.parametrize(
    ('group_columns', 'match'),
    [
        pytest.param(['word_idx'], 'reserved', id='reserved_word_idx'),
        pytest.param(['trial', 'word'], 'reserved', id='reserved_word'),
        # annotation output columns would be overwritten by the annotation step
        pytest.param(['run_id'], r"reserved columns \['run_id'\]", id='reserved_run_id'),
        pytest.param(['trial', 'prev_word_idx'], 'reserved', id='reserved_prev_word_idx'),
        # internal working columns of the pipeline
        pytest.param(['_group'], r"reserved columns \['_group'\]", id='reserved_group'),
        pytest.param(['word_start_char'], 'reserved', id='reserved_word_start_char'),
        # measure output columns
        pytest.param(['TFT'], 'reserved', id='reserved_measure_column'),
    ],
)
def test_compute_reading_measures_invalid_group_columns(group_columns, match):
    fixations = pl.DataFrame({'word_idx': [1], 'duration': [100]})
    aois = pl.DataFrame({'word_idx': [1], 'word': ['a']})

    with pytest.raises(ValueError, match=match):
        compute_reading_measures(fixations, aois, group_columns=group_columns)


def test_compute_reading_measures_empty_group_columns_disable_grouping():
    # The trial column is ignored: no broadcast, no per-trial rows, one single sequence.
    fixations = pl.DataFrame({
        'word_idx': [1, 2],
        'duration': [100, 110],
        'trial': ['t1', 't2'],
    })
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    result = compute_reading_measures(fixations, aois, group_columns=[])

    assert result.columns[:2] == ['word_index', 'word']
    assert result['TFT'].to_list() == [100, 110]
    # word 2 is entered from word 1 in the single combined sequence
    assert result['SL_in'].to_list()[1] == 1


def test_compute_reading_measures_aois_dict_requires_group_columns():
    fixations = pl.DataFrame({'word_idx': [1], 'duration': [100], 'trial': ['t1']})
    aois = {'t1': pl.DataFrame({'word_idx': [1], 'word': ['a']})}

    with pytest.raises(ValueError, match='at least one group column'):
        compute_reading_measures(fixations, aois, group_columns=[])


def test_compute_reading_measures_custom_word_index_column():
    fixations = pl.DataFrame({'aoi': [1, 2, 1], 'duration': [100, 110, 120]})
    aois = pl.DataFrame({'aoi': [1, 2], 'word': ['a', 'b']})

    result = compute_reading_measures(fixations, aois, word_index_column='aoi')

    assert result['word_index'].to_list() == [1, 2]
    assert result['TFT'].to_list() == [220, 110]
    assert result['TRC_in'].to_list() == [1, 0]


def test_compute_reading_measures_custom_word_index_column_is_reserved():
    fixations = pl.DataFrame({'aoi': [1], 'duration': [100]})
    aois = pl.DataFrame({'aoi': [1], 'word': ['a']})

    with pytest.raises(ValueError, match='reserved'):
        compute_reading_measures(
            fixations, aois, word_index_column='aoi', group_columns=['aoi'],
        )


def test_compute_reading_measures_custom_event_name():
    fixations = pl.DataFrame({
        'word_idx': [1, 2],
        'duration': [100, 110],
        'name': ['fixation.custom', 'saccade'],
    })
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    result = compute_reading_measures(fixations, aois, event_name='fixation.custom')

    assert result['TFT'].to_list() == [100, 0]


@pytest.mark.parametrize(
    ('fixations', 'aois'),
    [
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            id='no_char_idx_at_all',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'char_idx': [0], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            id='char_idx_only_in_fixations',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'duration': [100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b'], 'char_idx': [0, 1]}),
            id='char_idx_only_in_aois',
        ),
    ],
)
def test_compute_reading_measures_landing_position_null_without_char_idx(fixations, aois):
    result = compute_reading_measures(fixations, aois)

    assert result['LP'].to_list() == [None, None]
