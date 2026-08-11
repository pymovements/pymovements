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
                1: {
                    'FFD': 100, 'TFT': 220, 'FPRT': 100, 'RRT': 120, 'TFC': 2,
                    'TRC_in': 1, 'SL_in': 0,
                },
                2: {'FFD': 110, 'TFT': 110, 'TRC_out': 1, 'SL_out': -1, 'RPD_exc': 120, 'FPReg': 1},
                3: {'FFD': 130, 'TFT': 130, 'TRC_out': 0, 'SL_out': 0, 'SL_in': 2},
            },
            id='regression',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1, 3], 'duration': [100, 130]}),
            pl.DataFrame({'word_idx': [1, 2, 3], 'word': ['a', 'b', 'c']}),
            {
                1: {'FFD': 100, 'TFT': 100, 'TFC': 1, 'SL_out': 2, 'skipped': 0},
                2: {'FFD': 0, 'TFT': 0, 'TFC': 0, 'Fix': 0, 'skipped': 1},
                3: {'FFD': 130, 'TFT': 130, 'TFC': 1, 'SL_in': 2, 'SL_out': 0, 'skipped': 0},
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
            marks=pytest.mark.filterwarnings(
                'ignore:no fixations left to annotate',
            ),
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
        pytest.param(
            # Sentinel indices without an AOI entry (here -1) are excluded from the sequence:
            # no spurious regression out of word 1 and no inflated saccade into word 2.
            pl.DataFrame({'word_idx': [1, -1, 2], 'duration': [100, 100, 100]}),
            pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']}),
            {
                1: {'TFT': 100, 'TRC_out': 0, 'SL_out': 1},
                2: {'TFT': 100, 'TRC_in': 0, 'SL_in': 1},
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
    # Char-level AOI table: word 1 spans chars 0-2, word 2 spans chars 3-7.
    aois = pl.DataFrame({
        'word_idx': [1, 1, 1, 2, 2, 2, 2, 2],
        'word': ['The'] * 3 + ['quick'] * 5,
        'char_idx': [0, 1, 2, 3, 4, 5, 6, 7],
    })
    fixations = pl.DataFrame({
        'word_idx': [1, 2, 2],
        'char_idx': [1, 5, 3],
        'duration': [100, 110, 120],
    })

    result = compute_reading_measures(fixations, aois)

    assert result['LP'].to_list() == [1, 2]


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


def test_compute_reading_measures_landing_position_null_without_char_idx():
    fixations = pl.DataFrame({'word_idx': [1], 'duration': [100]})
    aois = pl.DataFrame({'word_idx': [1, 2], 'word': ['a', 'b']})

    result = compute_reading_measures(fixations, aois)

    assert result['LP'].to_list() == [None, None]
