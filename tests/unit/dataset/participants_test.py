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
import warnings
from math import nan

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Participants
from pymovements.dataset.participants import _validate_age
from pymovements.dataset.participants import _validate_sex


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
            r"unknown bids format descriptor 'test'",
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


def test_participants_load_missing_participant_id_raises(make_csv_file):
    """Test that ValueError is raised if loaded participant data has no participant_id column."""
    path = make_csv_file(
        'participants.tsv',
        pl.DataFrame({'a': [1, 2]}),
        separator='\t',
    )

    with pytest.raises(ValueError, match='participant_id'):
        Participants.load(path, verify_bids=True)


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

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(save_path, verify_bids=False)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


@pytest.mark.parametrize(
    'separator',
    ['\t', ','],
)
def test_participants_save_data_to_filepath_custom_separator(separator, tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-1'], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(save_path, separator=separator, verify_bids=False)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator=separator)
    assert_frame_equal(saved_data, data)


def test_participants_save_data_write_csv_kwargs_precedence_over_separator(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-1'], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(
        save_path,
        separator=',',
        write_csv_kwargs={
            'separator': '\t',
        },
        verify_bids=False,
    )

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
    assert not participants.verify_bids('REQUIRED')


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
    assert not participants.verify_bids('REQUIRED')


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
    data = pl.DataFrame({'participant_id': ['sub-01'], 'age': [nan]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert any("Column 'age' contains invalid null values" in w for w in warnings_list)
    assert any(
        "BIDS requires missing values to be coded as 'n/a'" in w for w in warnings_list
    )


def test_verify_bids_metadata_description_missing():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'custom': [1]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert (
        "Column 'custom' is missing a 'Description' field in metadata (json)."
        in warnings_list
    )


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
    assert not participants.verify_bids('REQUIRED')


@pytest.mark.parametrize(
    ('before', 'update_data', 'expected_exception', 'expected_match'),
    [
        pytest.param(
            pl.DataFrame(schema={'participant_id': pl.String}),
            pl.DataFrame({'not_id': ['sub-1']}),
            ValueError,
            "data must have column named 'participant_id'",
            id='update_missing_participant_id',
        ),
    ],
)
def test_participants_update_raises(
    before,
    update_data,
    expected_exception,
    expected_match,
):
    participants = Participants(before)
    with pytest.raises(expected_exception, match=expected_match):
        participants.update(update_data)


@pytest.mark.parametrize(
    ('before', 'update_data', 'after'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-1']}),
            pl.DataFrame({'participant_id': ['sub-2'], 'age': [21]}),
            pl.DataFrame(
                {'participant_id': ['sub-1', 'sub-2'], 'age': [None, 21]},
                schema={'participant_id': pl.String, 'age': pl.Int64},
            ),
            id='update_adds_new_column',
        ),
    ],
)
def test_participants_update_new_columns(before, update_data, after):
    participants = Participants(before)
    participants.update(update_data)
    assert_frame_equal(participants.data, after)


@pytest.mark.parametrize(
    ('data', 'load_kwargs', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-1']}),
            {'read_csv_kwargs': {'separator': ','}},
            pl.DataFrame({'participant_id': ['sub-1']}),
            id='load_with_custom_read_csv_kwargs',
        ),
    ],
)
def test_participants_load_custom_kwargs(
    data,
    load_kwargs,
    expected_data,
    make_csv_file,
):
    filename = 'participants.tsv'
    path = make_csv_file(filename, data, separator=',')
    participants = Participants.load(path, **load_kwargs)
    assert_frame_equal(participants.data, expected_data)


def test_participants_load_auto_detect_metadata(make_csv_file, make_json_file):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    metadata = {'participant_id': {'Description': 'The participant ID'}}

    path = make_csv_file('participants.tsv', data, separator='\t')
    make_json_file('participants.json', metadata)

    participants = Participants.load(path.parent)
    assert (
        participants.metadata.get('participant_id', {}).get('Description')
        == 'The participant ID'
    )


@pytest.mark.parametrize(
    ('metadata_input', 'expected_description'),
    [
        pytest.param(
            {'participant_id': {'Description': 'ID from dict'}},
            'ID from dict',
            id='metadata_dict',
        ),
    ],
)
def test_participants_load_with_metadata_dict(
    metadata_input,
    expected_description,
    make_csv_file,
):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    path = make_csv_file('participants.tsv', data, separator='\t')
    participants = Participants.load(path, metadata=metadata_input)
    assert (
        participants.metadata.get('participant_id', {}).get(
            'Description',
        )
        == expected_description
    )


def test_participants_load_with_metadata_path(make_csv_file, make_json_file):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    metadata = {'participant_id': {'Description': 'ID from path'}}

    path = make_csv_file('participants.tsv', data, separator='\t')
    json_path = make_json_file('participants.json', metadata)

    participants = Participants.load(path, metadata=json_path)
    assert (
        participants.metadata.get('participant_id', {}).get(
            'Description',
        )
        == 'ID from path'
    )


def test_participants_load_with_metadata_path_wrong_name(make_csv_file, make_json_file):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    metadata = {'participant_id': {'Description': 'ID'}}

    path = make_csv_file('participants.tsv', data, separator='\t')
    custom_metadata_path = make_json_file('custom.json', metadata)

    with pytest.warns(
        UserWarning,
        match="requires metadata file to be named 'participants.json'",
    ):
        Participants.load(
            path,
            metadata=custom_metadata_path,
            verify_bids='REQUIRED',
        )


@pytest.mark.parametrize(
    'write_csv_kwargs',
    [
        pytest.param({'separator': ','}, id='save_custom_write_kwargs'),
    ],
)
def test_participants_save_custom_write_kwargs(tmp_path, write_csv_kwargs):
    data = pl.DataFrame({'participant_id': ['sub-1']})
    participants = Participants(data)
    participants.save(
        tmp_path / 'participants.tsv',
        verify_bids=False,
        write_csv_kwargs=write_csv_kwargs,
    )
    saved = pl.read_csv(tmp_path / 'participants.tsv', separator=',')
    assert_frame_equal(saved, data)


def test_participants_save_metadata_warning(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    with pytest.warns(
        UserWarning,
        match="requires metadata file to be named 'participants.json'",
    ):
        participants.save(
            tmp_path / 'participants.tsv',
            verify_bids='REQUIRED',
            metadata_path=tmp_path / 'custom.json',
        )


def test_verify_bids_invalid_level():
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    assert not participants.verify_bids('INVALID')  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('data', 'metadata', 'level', 'expected_warnings', 'unexpected_warnings'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'age': [25]}),
            None,
            'RECOMMENDED',
            [],
            ["Column 'age' must be of numeric type"],
            id='age_valid_up_to_89',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01', 'sub-01'], 'age': ['n/a', 'abc']},
            ),
            None,
            'RECOMMENDED',
            ["Column 'age' must be of numeric type"],
            [],
            id='age_na_and_exception',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'sex': [1]}),
            None,
            'RECOMMENDED',
            ["Column 'sex' must be of string (Utf8) type"],
            [],
            id='sex_non_string',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'sex': ['F']}),
            None,
            'RECOMMENDED',
            [],
            ['sex must be one of'],
            id='sex_valid_no_invalid',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'handedness': [1]}),
            None,
            'RECOMMENDED',
            ["Column 'handedness' must be of string (Utf8) type"],
            [],
            id='handedness_non_string',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'handedness': ['right']}),
            None,
            'RECOMMENDED',
            [],
            ['handedness must be one of'],
            id='handedness_valid_no_invalid',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'species': [True]}),
            None,
            'RECOMMENDED',
            ["Column 'species' must be of string or numeric type"],
            [],
            id='species_non_standard_dtype',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'strain': ['some_strain']}),
            None,
            'RECOMMENDED',
            [],
            [],
            id='strain_with_values',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'strain_rrid': ['invalid']}),
            None,
            'RECOMMENDED',
            ["strain_rrid must match 'RRID:<identifier>' pattern"],
            [],
            id='strain_rrid_invalid_pattern',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'null': [1]}),
            None,
            'RECOMMENDED',
            ['Column names must not be blank or null'],
            [],
            id='column_name_null',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'custom_col': [1]}),
            {'custom_col': {'Description': 'A custom column'}},
            'RECOMMENDED',
            [],
            ["Column 'custom_col' is missing a 'Description' field"],
            id='custom_column_with_description',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01'], 'species': ['human']}),
            None,
            'RECOMMENDED',
            [],
            ["Column 'species' must be of string or numeric type"],
            id='species_valid_string',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'strain': [None]},
                schema={'participant_id': pl.String, 'strain': pl.String},
            ),
            None,
            'RECOMMENDED',
            [],
            [],
            id='strain_null_values',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'strain_rrid': [None]},
                schema={'participant_id': pl.String, 'strain_rrid': pl.String},
            ),
            None,
            'RECOMMENDED',
            [],
            [],
            id='strain_rrid_null_values',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'strain_rrid': ['RRID:valid_123']},
            ),
            None,
            'RECOMMENDED',
            [],
            ["strain_rrid must match 'RRID:<identifier>' pattern"],
            id='strain_rrid_valid_pattern',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'age': [None]},
                schema={'participant_id': pl.String, 'age': pl.Float64},
            ),
            None,
            'REQUIRED',
            ["Column 'age' contains invalid null values"],
            [],
            id='age_null_value',
        ),
    ],
)
def test_verify_bids_edge_cases(
    data,
    metadata,
    level,
    expected_warnings,
    unexpected_warnings,
):
    participants = Participants(data, metadata=metadata, verify_bids=False)
    warnings_list = participants.verify_bids(level)
    for expected in expected_warnings:
        assert any(expected in w for w in warnings_list), (
            f"Expected '{expected}' in warnings: {warnings_list}"
        )
    for unexpected in unexpected_warnings:
        assert not any(unexpected in w for w in warnings_list), (
            f"Did not expect '{unexpected}' in warnings: {warnings_list}"
        )


def test_verify_bids_missing_participant_id_column():
    data = pl.DataFrame({'not_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert 'participant_id column is missing' in warnings_list


def test_verify_bids_non_string_participant_id_dtype():
    data = pl.DataFrame({'participant_id': [1]})
    participants = Participants(data, verify_bids=False, infer_metadata=False)
    warnings_list = participants.verify_bids('REQUIRED')
    assert any(
        'participant_id column must have string (Utf8) data type' in w
        for w in warnings_list
    )


def test_verify_bids_age_over_89():
    data = pl.DataFrame({'participant_id': ['sub-01'], 'age': [100]})
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('RECOMMENDED')
    assert any('age should be capped at 89' in w for w in warnings_list)


@pytest.mark.parametrize(
    ('data', 'expected_warnings'),
    [
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'age': [nan]},
                schema={'participant_id': pl.String, 'age': pl.Float64},
            ),
            ["Column 'age' contains invalid null values"],
            id='age_nan',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['sub-01'], 'sex': ['N/A']},
            ),
            ["Column 'sex' contains invalid null values"],
            id='sex_na_string',
        ),
    ],
)
def test_verify_bids_na_conformity_detailed(data, expected_warnings):
    participants = Participants(data, verify_bids=False)
    warnings_list = participants.verify_bids('REQUIRED')
    for expected in expected_warnings:
        assert any(expected in w for w in warnings_list), (
            f"Expected '{expected}' in warnings: {warnings_list}"
        )


def test_participants_load_relative_metadata_path_with_verify(
    make_csv_file,
    make_json_file,
):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    metadata = {'participant_id': {'Description': 'The participant ID'}}

    path = make_csv_file('participants.tsv', data, separator='\t')
    make_json_file('participants.json', metadata)

    participants = Participants.load(
        path.parent,
        metadata='participants.json',
        verify_bids='REQUIRED',
    )
    assert (
        participants.metadata.get('participant_id', {}).get('Description')
        == 'The participant ID'
    )


def test_participants_init_null_dtype():
    data = pl.DataFrame(
        {'participant_id': ['sub-01'], 'col': [None]},
        schema={'participant_id': pl.String, 'col': pl.Null},
    )
    participants = Participants(data)
    assert participants.metadata.get('col', {}).get('Format') == 'string'


class TestVerifyBidsInit:
    @pytest.mark.parametrize(
        ('data', 'verify_bids', 'expected_exception', 'expected_message'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                'REQUIRED',
                None,
                None,
                id='verify_required_no_exception',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['01']}),
                'REQUIRED',
                None,
                None,
                id='verify_required_warning',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['01']}),
                True,
                ValueError,
                'BIDS non-conformities found',
                id='verify_true_raises',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                False,
                None,
                None,
                id='verify_false_no_check',
            ),
        ],
    )
    def test_verify_bids_init(
        self,
        data,
        verify_bids,
        expected_exception,
        expected_message,
    ):
        if expected_exception:
            with pytest.raises(expected_exception, match=expected_message):
                Participants(data, verify_bids=verify_bids)
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                _ = Participants(data, verify_bids=verify_bids)
                if verify_bids not in (False, None):
                    if expected_message:
                        warning_messages = [str(warning.message) for warning in w]
                        assert any(expected_message in msg for msg in warning_messages)
                else:
                    assert not w


class TestVerifyBidsLoad:
    def test_verify_bids_load_with_warning(self, make_csv_file):
        data = pl.DataFrame({'participant_id': ['01']})
        path = make_csv_file('participants.tsv', data, separator='\t')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = Participants.load(path, verify_bids='REQUIRED')
            assert len(w) > 0

    def test_verify_bids_load_true_raises(self, make_csv_file):
        data = pl.DataFrame({'participant_id': ['01']})
        path = make_csv_file('participants.tsv', data, separator='\t')

        with pytest.raises(ValueError, match='BIDS non-conformities found'):
            Participants.load(path, verify_bids=True)

    def test_verify_bids_load_false_no_check(self, make_csv_file):
        data = pl.DataFrame({'participant_id': ['01']})
        path = make_csv_file('participants.tsv', data, separator='\t')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = Participants.load(path, verify_bids=False)
            assert not w

    def test_verify_bids_load_with_recommended_level(self, make_csv_file):
        data = pl.DataFrame({'participant_id': ['01'], 'age': [100]})
        path = make_csv_file('participants.tsv', data, separator='\t')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = Participants.load(path, verify_bids='RECOMMENDED')
            warning_messages = [str(warning.message) for warning in w]
            assert any('participant_id' in msg for msg in warning_messages)
            assert any('age' in msg for msg in warning_messages)

    def test_verify_bids_load_required_level_no_age_check(self, make_csv_file):
        data = pl.DataFrame({'participant_id': ['01'], 'age': [100]})
        path = make_csv_file('participants.tsv', data, separator='\t')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = Participants.load(path, verify_bids='REQUIRED')
            warning_messages = [str(warning.message) for warning in w]
            assert any('participant_id' in msg for msg in warning_messages)
            assert not any('age' in msg for msg in warning_messages)


class TestVerifyBidsSave:
    def test_verify_bids_save_with_warning(self, tmp_path):
        data = pl.DataFrame({'participant_id': ['01']})
        participants = Participants(data, verify_bids=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            participants.save(tmp_path / 'participants.tsv', verify_bids='REQUIRED')
            assert len(w) > 0

    def test_verify_bids_save_true_raises(self, tmp_path):
        data = pl.DataFrame({'participant_id': ['01']})
        participants = Participants(data, verify_bids=False)

        with pytest.raises(ValueError, match='BIDS non-conformities found'):
            participants.save(tmp_path / 'participants.tsv', verify_bids=True)

    def test_verify_bids_save_false_no_check(self, tmp_path):
        data = pl.DataFrame({'participant_id': ['01']})
        participants = Participants(data, verify_bids=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            participants.save(tmp_path / 'participants.tsv', verify_bids=False)
            assert not w

    def test_verify_bids_save_default_required(self, tmp_path):
        data = pl.DataFrame({'participant_id': ['01']})
        participants = Participants(data, verify_bids=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            participants.save(tmp_path / 'participants.tsv')
            assert len(w) > 0


class TestValidateAge:
    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [34]}),
                [],
                id='valid_age',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [100]}),
                ['age should be capped at 89, found 100.0'],
                id='age_over_89',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                [],
                id='no_age_column',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': ['not_a_number']}),
                ["Column 'age' must be of numeric type (integer or float)"],
                id='non_numeric_age',
            ),
        ],
    )
    def test_validate_age(self, data, expected):
        warnings_list = _validate_age(data)
        assert warnings_list == expected


class TestValidateSex:
    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'sex': ['M']}),
                [],
                id='valid_male',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'sex': ['invalid']}),
                [
                    "sex must be one of ['F', 'FEMALE', 'Female', 'M', 'MALE', "
                    "'Male', 'O', 'OTHER', 'Other', 'f', 'female', 'm', 'male', "
                    "'o', 'other'], found: ['invalid']",
                ],
                id='invalid_sex',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                [],
                id='no_sex_column',
            ),
        ],
    )
    def test_validate_sex(self, data, expected):
        warnings_list = _validate_sex(data)
        assert warnings_list == expected


def test_verify_bids_with_sex_na(make_csv_file):
    data = pl.DataFrame(
        {
            'participant_id': ['sub-01', 'sub-02'],
            'sex': ['M', 'n/a'],
        },
    )
    path = make_csv_file('participants.tsv', data, separator='\t')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = Participants.load(path, verify_bids='RECOMMENDED')
        warning_messages = [str(warning.message) for warning in w]
        assert not any('sex' in msg for msg in warning_messages)


def test_participants_init_verify_bids_true_no_warnings():
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=True)
    assert participants.data.shape == (1, 1)


def test_participants_save_verify_bids_true_no_warnings(tmp_path):
    data = pl.DataFrame({'participant_id': ['sub-01']})
    participants = Participants(data, verify_bids=False)
    participants.save(tmp_path / 'participants.tsv', verify_bids=True)
    assert (tmp_path / 'participants.tsv').exists()
    assert (tmp_path / 'participants.json').exists()
