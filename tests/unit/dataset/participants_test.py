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
"""Unit tests of Participants class functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Participants


def test_participants_init_default():
    participants = Participants()
    assert_frame_equal(participants.data, pl.DataFrame(schema={'participant_id': pl.String}))


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['sub-01']}),
    ],
)
def test_participants_init_data(data):
    participants = Participants(data)
    assert_frame_equal(participants.data, data)


@pytest.mark.parametrize(
    ('data', 'expected_metadata'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01']}),
            {'participant_id': {'Format': 'string'}},
            id='participant_id_string',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'age': [21]}),
            {'participant_id': {'Format': 'string'}, 'age': {'Format': 'integer'}},
            id='age_integer',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'age': [21.0]}),
            {'participant_id': {'Format': 'string'}, 'age': {'Format': 'number'}},
            id='age_number',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'test': [True]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'bool'}},
            id='bool',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'test': [42]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'integer'}},
            id='integer',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'test': [42]},
                schema={'participant_id': pl.String, 'test': pl.UInt64},
            ),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'index'}},
            id='index',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'test': [6.7]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'number'}},
            id='number',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'test': ['a']}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'string'}},
            id='string',
        ),
    ],
)
def test_participants_init_infers_correct_format(data, expected_metadata):
    participants = Participants(data)
    assert participants.metadata == expected_metadata


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['sub-01']}),
        pl.DataFrame({'participant_id': ['sub-01'], 'age': [21]}),
        pl.DataFrame({'participant_id': ['sub-01'], 'test': ['a']}),
    ],
)
def test_participants_init_no_metadata_infer(data):
    participants = Participants(data, infer_metadata=False)
    assert isinstance(participants.metadata, dict)
    assert not participants.metadata


@pytest.mark.parametrize(
    ('data', 'expected_exception', 'expected_message'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': [1], 'test': [(1, 2)]}),
            TypeError,
            r'polars datatype List\(Int64\) has no mapping to bids format descriptor',
            id='list_format_not_supported',
        ),
    ],
)
def test_participants_init_data_raises(data, expected_exception, expected_message):
    with pytest.raises(expected_exception, match=expected_message):
        Participants(data)


@pytest.mark.parametrize(
    ('data', 'metadata', 'expected_exception', 'expected_message'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'test': 1}),
            {'test': {'Format': 'test'}},
            TypeError,
            r'unknown bids format descriptor "test"',
            id='unknown_bids_format',
        ),
    ],
)
def test_participants_init_metadata_raises(
    data,
    metadata,
    expected_exception,
    expected_message,
):
    with pytest.raises(expected_exception, match=expected_message):
        Participants(data, metadata)


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['sub-1']}),
        pl.DataFrame({'participant_id': ['sub-1'], 'age': [21.0]}),
    ],
)
def test_participants_load_data_from_tsv(data, make_csv_file):
    filename = 'participants.tsv'
    path = make_csv_file(filename, data, separator='\t')

    participants = Participants.load(path)

    assert_frame_equal(participants.data, data)


def test_participants_load_data_from_directory(make_csv_file):
    """Test that participants.tsv is used for loading if path is directory."""
    filename = 'participants.tsv'
    data = pl.DataFrame({'participant_id': ['sub-1']})
    path = make_csv_file(filename, data, separator='\t')

    participants = Participants.load(path.parent)

    assert_frame_equal(participants.data, data)


@pytest.mark.parametrize(
    ('source_data', 'rename', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'subject_id': ['sub-1']}),
            {'subject_id': 'participant_id'},
            pl.DataFrame({'participant_id': ['sub-1']}),
            id='subject_id_to_participant_id',
        ),
    ],
)
def test_participants_load_and_rename_data_from_file(
    source_data,
    rename,
    expected_data,
    make_csv_file,
):
    filename = 'participants.tsv'
    path = make_csv_file(filename, source_data, separator='\t')

    participants = Participants.load(path, rename=rename)

    assert_frame_equal(participants.data, expected_data)


@pytest.mark.parametrize(
    ('data', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': [1]}),
            pl.DataFrame(
                {'participant_id': ['1']},
                schema={'participant_id': pl.String},
            ),
            id='autocast_participant_id_to_string',
        ),
    ],
)
def test_participants_init_autocasts(data, expected_data):
    participants = Participants(data)
    assert_frame_equal(participants.data, expected_data)


@pytest.mark.parametrize(
    ('before', 'update_data', 'after'),
    [
        pytest.param(
            pl.DataFrame(schema={'participant_id': pl.String}),
            pl.DataFrame(schema={'participant_id': pl.String}),
            pl.DataFrame(schema={'participant_id': pl.String}),
            id='empty_update_empty',
        ),
        pytest.param(
            pl.DataFrame(schema={'participant_id': pl.String}),
            pl.DataFrame({'participant_id': ['sub-1']}),
            pl.DataFrame({'participant_id': ['sub-1']}),
            id='empty_update_single_participant',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-1']}),
            pl.DataFrame({'participant_id': ['sub-2']}),
            pl.DataFrame({'participant_id': ['sub-1', 'sub-2']}),
            id='one_participant_update_new_participant',
        ),
    ],
)
def test_participants_update_data(before, update_data, after):
    participants = Participants(before)
    participants.update(update_data)
    assert_frame_equal(participants.data, after)


@pytest.mark.parametrize(
    ('before', 'update_metadata', 'after'),
    [
        pytest.param(
            {'participant_id': {'Format': 'string'}},
            {'age': {'Format': 'integer'}},
            {'participant_id': {'Format': 'string'}, 'age': {'Format': 'integer'}},
            id='update_metadata',
        ),
    ],
)
def test_participants_update_metadata(before, update_metadata, after):
    participants = Participants(metadata=before)
    participants.update(data=participants.data, metadata=update_metadata)
    assert participants.metadata == after


@pytest.mark.parametrize(
    ('data', 'metadata', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-1'], 'age': ['21']}),
            {'age': {'Format': 'integer'}},
            pl.DataFrame(
                {'participant_id': ['sub-1'], 'age': [21]},
                schema={'participant_id': pl.String, 'age': pl.Int64},
            ),
            id='cast_age_column_to_int',
        ),
    ],
)
def test_participants_init_casts_from_metadata(data, metadata, expected_data):
    participants = Participants(data, metadata)
    assert_frame_equal(participants.data, expected_data)


def test_participants_save_data_to_filepath(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-1'], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'participants.tsv'
    participants.save(save_path, verify_bids=False)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


def test_participants_save_data_to_dirpath(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-1'], 'age': [21.0]})
    participants = Participants(data)

    participants.save(tmp_path, verify_bids=False)

    save_path = tmp_path / 'participants.tsv'

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


def test_verify_bids_valid():
    data = pl.DataFrame({'participant_id': ['sub-01', 'sub-02']})
    participants = Participants(data, verify_bids=False)
    assert participants.verify_bids('REQUIRED') == []


def test_verify_bids_invalid_id_format():
    data = pl.DataFrame({'participant_id': ['01']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert any("match 'sub-<label>' pattern" in w for w in warnings_list)


def test_verify_bids_id_not_first():
    data = pl.DataFrame({'age': [20], 'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert 'participant_id column must be the first column' in warnings_list


def test_verify_bids_duplicate_id():
    data = pl.DataFrame({'participant_id': ['sub-01', 'sub-01']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert 'participant_id values must be unique' in warnings_list


def test_verify_bids_id_null():
    data = pl.DataFrame({'participant_id': [None]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert 'participant_id column contains null values' in warnings_list


def test_verify_bids_id_not_string():
    data = pl.DataFrame({'participant_id': ['sub-1']})
    participants = Participants(data, verify_bids=False)
    assert participants.data['participant_id'].dtype == pl.String
    assert participants.verify_bids('REQUIRED') == []


def test_verify_bids_age_over_89():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'age': [100]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert any('age should be capped at 89' in w for w in warnings_list)


def test_verify_bids_recommended_missing():
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert "Recommended column 'age' is missing" in warnings_list
    assert "Recommended column 'sex' is missing" in warnings_list
    assert "Recommended column 'handedness' is missing" in warnings_list


def test_verify_bids_sex_invalid():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'sex': ['invalid']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert any('sex must be one of' in w for w in warnings_list)


def test_verify_bids_handedness_invalid():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'handedness': ['invalid']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert any('handedness must be one of' in w for w in warnings_list)


def test_verify_bids_na_conformity():
    # 'NaN' instead of 'n/a'
    data = pl.DataFrame({'participant_id': ['sub-01'], 'age': [float('nan')]})
    # Cast to float to ensure numeric dtype if it's not already
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert any("Column 'age' contains invalid null values" in w for w in warnings_list)
    assert any("BIDS requires missing values to be coded as 'n/a'" in w for w in warnings_list)


def test_verify_bids_metadata_description_missing():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'custom': [1]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert "Column 'custom' is missing a 'Description' field in metadata (json)." in warnings_list


def test_verify_bids_column_name_not_snake_case():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'AgeInYears': [20]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert "Column name 'AgeInYears' should be written in snake_case." in warnings_list


def test_verify_bids_init_raises():
    data = pl.DataFrame({'participant_id': ['01']})
    with pytest.raises(ValueError, match='BIDS non-conformities found'):
        Participants(data, verify_bids=True)


def test_verify_bids_init_warns():
    data = pl.DataFrame({'participant_id': ['01']})
    with pytest.warns(UserWarning, match="match 'sub-<label>' pattern"):
        Participants(data, verify_bids='REQUIRED')


def test_verify_bids_load_raises(make_csv_file):
    data = pl.DataFrame({'participant_id': ['01']})
    path = make_csv_file('participants.tsv', data, separator='\t')
    with pytest.raises(ValueError, match='BIDS non-conformities found'):
        Participants.load(path, verify_bids=True)


def test_verify_bids_load_warns(make_csv_file):
    data = pl.DataFrame({'participant_id': ['01']})
    path = make_csv_file('participants.tsv', data, separator='\t')
    with pytest.warns(UserWarning, match="match 'sub-<label>' pattern"):
        Participants.load(path, verify_bids='REQUIRED')


def test_verify_bids_save_raises(tmp_path):
    data = pl.DataFrame({'participant_id': ['01']})
    participants = Participants(data, verify_bids=False)
    with pytest.raises(ValueError, match='BIDS non-conformities found'):
        participants.save(tmp_path / 'participants.tsv', verify_bids=True)


def test_verify_bids_save_warns(tmp_path):
    data = pl.DataFrame({'participant_id': ['01']})
    participants = Participants(data, verify_bids=False)
    with pytest.warns(UserWarning, match="match 'sub-<label>' pattern"):
        participants.save(tmp_path / 'participants.tsv', verify_bids='REQUIRED')


def test_verify_bids_load_format_warnings(make_csv_file):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    # Wrong filename
    path = make_csv_file('wrong_name.tsv', data, separator='\t')
    with pytest.warns(
            UserWarning, match="requires participant file to be named 'participants.tsv'",
    ):
        Participants.load(path, verify_bids='REQUIRED')

    # Wrong separator
    path = make_csv_file('participants.tsv', data, separator=',')
    with pytest.warns(UserWarning, match='requires tab-separated files'):
        Participants.load(path, verify_bids='REQUIRED', separator=',')


def test_verify_bids_save_format_warnings(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)

    # Wrong filename
    with pytest.warns(
            UserWarning, match="requires participant file to be named 'participants.tsv'",
    ):
        participants.save(tmp_path / 'wrong_name.tsv', verify_bids='REQUIRED')

    # Wrong separator
    with pytest.warns(UserWarning, match='requires tab-separated files'):
        participants.save(tmp_path / 'participants.tsv', verify_bids='REQUIRED', separator=',')


def test_verify_bids_regex_plus_allowed():
    data = pl.DataFrame({'participant_id': ['sub-01+02']})
    participants = Participants(data, verify_bids=False)
    assert participants.verify_bids('REQUIRED') == []


def test_save_encodes_null_as_na(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-01'], 'age': [None]})
    participants = Participants(data, verify_bids=False)
    save_path = tmp_path / 'participants.tsv'
    participants.save(save_path, verify_bids=False)

    with open(save_path) as f:
        content = f.read()
        assert 'n/a' in content
        assert 'null' not in content.lower()
