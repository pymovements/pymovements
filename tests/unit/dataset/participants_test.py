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
import json
import warnings
from copy import deepcopy

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Participants
from pymovements.dataset.participants import _validate_age
from pymovements.dataset.participants import _validate_handedness
from pymovements.dataset.participants import _validate_participant_id
from pymovements.dataset.participants import _validate_sex
from pymovements.dataset.participants import _validate_species
from pymovements.dataset.participants import _validate_strain
from pymovements.dataset.participants import _validate_strain_rrid


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
    ],
)
def test_participants_init_data(data):
    participants = Participants(data)
    assert_frame_equal(participants.data, data)


@pytest.mark.parametrize(
    ('data', 'expected_metadata'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['1']}),
            {'participant_id': {'Format': 'string'}},
            id='participant_id_string',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'age': [21]}),
            {'participant_id': {'Format': 'string'}, 'age': {'Format': 'number'}},
            id='age_number',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': [True]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'bool'}},
            id='bool',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': [42]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'integer'}},
            id='integer',
        ),
        pytest.param(
            pl.DataFrame(
                {'participant_id': ['1'], 'test': [42]},
                schema={'participant_id': pl.String, 'test': pl.UInt64},
            ),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'index'}},
            id='index',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': [6.7]}),
            {'participant_id': {'Format': 'string'}, 'test': {'Format': 'number'}},
            id='number',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': ['a']}),
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
        pytest.param(
            pl.DataFrame({'participant_id': ['1']}),
            id='participant_id',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'age': [21]}),
            id='age',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': ['a']}),
            id='string',
        ),
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
            pl.DataFrame(),
            ValueError,
            "data must have column named 'participant_id'",
            id='empty',
        ),
        pytest.param(
            pl.DataFrame({'a': [1]}),
            ValueError,
            "data must have column named 'participant_id'",
            id='no_participant_id',
        ),
        pytest.param(
            pl.DataFrame({'a': [1], 'participant_id': ['001']}),
            ValueError,
            "first column in data must be named 'participant_id'",
            id='participant_id_not_first_column',
        ),
        pytest.param(
            pl.DataFrame({'subject_id': [1]}),
            ValueError,
            "data must have column named 'participant_id'",
            id='subject_id_not_participant_id',
        ),
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
            pl.DataFrame({'participant_id': [1], 'test': 1}),
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
        pl.DataFrame({'participant_id': ['1']}),
        pl.DataFrame({'participant_id': ['1'], 'age': [21.0]}),
    ],
)
def test_participants_load_data_from_tsv(data, make_csv_file):
    filename = 'participants.tsv'
    path = make_csv_file(filename, data, separator='\t')

    participants = Participants.load(path)

    assert_frame_equal(participants.data, data)


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
    ],
)
def test_participants_load_data_from_csv(data, make_csv_file):
    filename = 'participants.csv'
    path = make_csv_file(filename, data)

    participants = Participants.load(path)

    assert_frame_equal(participants.data, data)


def test_participants_load_data_from_directory(make_csv_file):
    """Test that participants.tsv is used for loading if path is directory."""
    filename = 'participants.tsv'
    data = pl.DataFrame({'participant_id': ['1']})
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
        Participants.load(path)


@pytest.mark.parametrize(
    ('source_data', 'rename', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'subject_id': ['1']}),
            {'subject_id': 'participant_id'},
            pl.DataFrame({'participant_id': ['1']}),
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
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'age': ['21.3']}),
            pl.DataFrame(
                {'participant_id': ['1'], 'age': [21.3]},
                schema={'participant_id': pl.String, 'age': pl.Float64},
            ),
            id='autocast_age_column_to_float',
        ),
    ],
)
def test_participants_init_autocasts(data, expected_data):
    participants = Participants(data)
    assert_frame_equal(participants.data, expected_data)


@pytest.mark.parametrize(
    ('data', 'metadata', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'age': ['21']}),
            {'age': {'Format': 'integer'}},
            pl.DataFrame(
                {'participant_id': ['1'], 'age': [21]},
                schema={'participant_id': pl.String, 'age': pl.Int64},
            ),
            id='cast_age_column_to_int',
        ),
    ],
)
class TestParticipantsCastsFromMetadata:
    def test_participants_init_casts_from_metadata(self, data, metadata, expected_data):
        participants = Participants(data, metadata)
        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_dict(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
    ):
        path = make_csv_file('participants.tsv', data, separator='\t')
        participants = Participants.load(path, metadata=metadata)
        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_path(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        json_path = make_json_file('participants.json', metadata)

        participants = Participants.load(tsv_path, metadata=json_path)

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_filename(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'test_participants.json', metadata)

        participants = Participants.load(tsv_path, metadata='test_participants.json')

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_implicit(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'participants.json', metadata)

        participants = Participants.load(tsv_path)

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_kwargs_take_precedence(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'participants.json', metadata)

        participants = Participants.load(
            tsv_path,
            separator=',',
            read_csv_kwargs={
                'separator': '\t',
            },  # takes precedence over explicit separator
        )

        assert_frame_equal(participants.data, expected_data)


def test_participants_save_data_to_filepath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
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
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(save_path, separator=separator, verify_bids=False)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator=separator)
    assert_frame_equal(saved_data, data)


def test_participants_save_data_write_csv_kwargs_precedence_over_separator(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(
        save_path,
        separator=',',
        write_csv_kwargs={'separator': '\t'},
        verify_bids=False,
    )

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


def test_participants_save_data_to_dirpath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    participants.save(tmp_path, verify_bids=False)

    save_path = tmp_path / 'participants.tsv'

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


@pytest.mark.parametrize(
    'participants',
    [
        Participants(
            data=pl.DataFrame({'participant_id': [1], 'age': [21]}),
            metadata={
                'participant_id': {
                    'Description': 'id of the participant',
                    'Format': 'string',
                },
                'age': {
                    'Description': 'age of the participant',
                    'Format': 'integer',
                    'Units': 'year',
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    'encoding',
    ['utf-8', 'ascii'],
)
class TestParticipantsSaveMetadata:
    def test_participants_save_metadata_to_dirpath(
        self,
        participants,
        encoding,
        tmp_path,
    ):
        metadata = deepcopy(participants.metadata)
        participants.save(tmp_path, metadata_encoding=encoding, verify_bids=False)

        save_path = tmp_path / 'participants.json'
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_participants_save_metadata_to_filepath(
        self,
        participants,
        encoding,
        tmp_path,
    ):
        metadata = deepcopy(participants.metadata)
        metadata_path = tmp_path / 'test_participants.json'
        participants.save(
            tmp_path,
            metadata_path=metadata_path,
            metadata_encoding=encoding,
            verify_bids=False,
        )

        save_path = metadata_path
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_participants_save_metadata_to_relative_path(
        self,
        participants,
        encoding,
        tmp_path,
    ):
        metadata = deepcopy(participants.metadata)
        metadata_filename = 'test_participants.json'
        participants.save(
            tmp_path,
            metadata_path=metadata_filename,
            metadata_encoding=encoding,
            verify_bids=False,
        )

        save_path = tmp_path / metadata_filename
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata


class TestVerifyBids:
    @pytest.mark.parametrize(
        ('data', 'level', 'expected_warnings'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                'REQUIRED',
                [],
                id='valid_participant_id',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-02']}),
                'REQUIRED',
                [],
                id='valid_multiple_participants',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['01']}),
                'REQUIRED',
                [
                    "participant_id values must match 'sub-<label>' pattern. "
                    "Invalid values: ['01']",
                ],
                id='invalid_participant_id_format',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-01']}),
                'REQUIRED',
                ['participant_id values must be unique'],
                id='duplicate_participant_id',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [34]}),
                'RECOMMENDED',
                [],
                id='valid_age',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [100]}),
                'RECOMMENDED',
                ['age should be capped at 89, found 100.0'],
                id='age_over_89',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [100]}),
                'REQUIRED',
                [],
                id='age_not_checked_at_required',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'handedness': ['right']}),
                'RECOMMENDED',
                [],
                id='valid_handedness_right',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'handedness': ['r']}),
                'RECOMMENDED',
                [],
                id='valid_handedness_r',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'handedness': ['ambidextrous']},
                ),
                'RECOMMENDED',
                [],
                id='valid_handedness_ambidextrous',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'handedness': ['invalid']}),
                'RECOMMENDED',
                [
                    "handedness must be one of ['A', 'AMBIDEXTROUS', 'Ambidextrous', "
                    "'L', 'LEFT', 'Left', 'R', 'RIGHT', 'Right', 'a', 'ambidextrous', "
                    "'l', 'left', 'r', 'right'], found: ['invalid']",
                ],
                id='invalid_handedness',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'species': ['homo sapiens']},
                ),
                'RECOMMENDED',
                [],
                id='valid_species',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'strain': ['C57BL/6J']}),
                'RECOMMENDED',
                [],
                id='valid_strain',
            ),
            pytest.param(
                pl.DataFrame(
                    {
                        'participant_id': ['sub-01'],
                        'strain_rrid': ['RRID:IMSR_JAX:000664'],
                    },
                ),
                'RECOMMENDED',
                [],
                id='valid_strain_rrid',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'strain_rrid': ['invalid']},
                ),
                'RECOMMENDED',
                [
                    "strain_rrid must match 'RRID:<identifier>' pattern. "
                    "Invalid values: ['invalid']",
                ],
                id='invalid_strain_rrid',
            ),
            pytest.param(
                pl.DataFrame(
                    {
                        'participant_id': ['sub-01'],
                        'age': [34],
                        'sex': ['M'],
                        'handedness': ['right'],
                    },
                ),
                'RECOMMENDED',
                [],
                id='valid_full_recommended',
            ),
        ],
    )
    def test_verify_bids(self, data, level, expected_warnings):
        participants = Participants(data, verify_bids=False)
        warnings_list = participants.verify_bids(level)
        assert warnings_list == expected_warnings


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


class TestValidateParticipantId:
    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01']}),
                [],
                id='valid',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['01']}),
                [
                    "participant_id values must match 'sub-<label>' pattern. "
                    "Invalid values: ['01']",
                ],
                id='invalid_format',
            ),
            pytest.param(
                pl.DataFrame({'a': ['sub-01']}),
                ['participant_id column is missing'],
                id='missing_column',
            ),
            pytest.param(
                pl.DataFrame({'age': [34], 'participant_id': ['sub-01']}),
                ['participant_id column must be the first column'],
                id='not_first_column',
            ),
        ],
    )
    def test_validate_participant_id(self, data, expected):
        warnings_list = _validate_participant_id(data)
        assert warnings_list == expected


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
                ['age must be a numeric value, found not_a_number'],
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


class TestValidateEdgeCases:
    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': [None, None]}),
                [],
                id='participant_id_all_null',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01', 'sub-02'], 'age': [None, None]},
                ),
                [],
                id='age_all_null',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01', 'sub-02'], 'sex': [None, None]},
                ),
                [],
                id='sex_all_null',
            ),
            pytest.param(
                pl.DataFrame(
                    {
                        'age': [34],
                        'participant_id': ['sub-01'],
                        'handedness': ['right'],
                    },
                ),
                [],
                id='handedness_column_not_first',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'handedness': ['n/a']},
                    schema={'participant_id': pl.String, 'handedness': pl.String},
                ),
                [],
                id='handedness_with_na',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'species': ['homo sapiens']},
                ),
                [],
                id='species_with_values',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'species': [None]},
                ),
                [],
                id='species_all_null',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'handedness': ['NaN']},
                    schema={'participant_id': pl.String, 'handedness': pl.String},
                ),
                [],
                id='handedness_with_nan',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'handedness': ['N/A']},
                    schema={'participant_id': pl.String, 'handedness': pl.String},
                ),
                [],
                id='handedness_with_NA',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'handedness': ['']},
                    schema={'participant_id': pl.String, 'handedness': pl.String},
                ),
                [],
                id='handedness_with_empty',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'age': ['NaN']},
                    schema={'participant_id': pl.String, 'age': pl.String},
                ),
                [],
                id='age_with_nan',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'age': ['N/A']},
                    schema={'participant_id': pl.String, 'age': pl.String},
                ),
                [],
                id='age_with_NA',
            ),
            pytest.param(
                pl.DataFrame(
                    {'participant_id': ['sub-01'], 'sex': ['NA']},
                    schema={'participant_id': pl.String, 'sex': pl.String},
                ),
                [],
                id='sex_with_NA',
            ),
        ],
    )
    def test_validate_functions_edge_cases(self, data, expected):
        if 'age' in data.columns:
            warnings_list = _validate_age(data)
        elif 'sex' in data.columns:
            warnings_list = _validate_sex(data)
        elif 'handedness' in data.columns:
            warnings_list = _validate_handedness(data)
        elif 'species' in data.columns:
            warnings_list = _validate_species(data)
        else:
            warnings_list = _validate_participant_id(data)
        assert warnings_list == expected

    def test_verify_bids_with_sex_na(self, make_csv_file):
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

    def test_validate_age_all_null(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01', 'sub-02'], 'age': [None, None]},
        )
        warnings_list = _validate_age(data)
        assert not warnings_list

    def test_validate_sex_all_null(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01', 'sub-02'], 'sex': [None, None]},
        )
        warnings_list = _validate_sex(data)
        assert not warnings_list

    def test_validate_handedness_column_not_first(self):
        data = pl.DataFrame(
            {'age': [34], 'participant_id': ['sub-01'], 'handedness': ['right']},
        )
        warnings_list = _validate_handedness(data)
        assert not warnings_list

    def test_validate_handedness_with_na(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'handedness': ['n/a']},
        )
        warnings_list = _validate_handedness(data)
        assert not warnings_list

    def test_validate_species_with_values(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'species': ['homo sapiens']},
        )
        warnings_list = _validate_species(data)
        assert not warnings_list

    def test_validate_species_all_null(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'species': [None]},
        )
        warnings_list = _validate_species(data)
        assert not warnings_list

    def test_validate_strain_rrid_empty_list(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'strain_rrid': [[]]},
            schema={'participant_id': pl.String, 'strain_rrid': pl.List(pl.String)},
        )
        warnings_list = _validate_strain_rrid(data)
        assert not warnings_list

    def test_validate_strain_with_values(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'strain': ['C57BL/6J']},
        )
        warnings_list = _validate_strain(data)
        assert not warnings_list

    def test_validate_strain_all_null(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'strain': [None]},
        )
        warnings_list = _validate_strain(data)
        assert not warnings_list

    def test_validate_age_with_nan(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'age': ['NaN']},
            schema={'participant_id': pl.String, 'age': pl.String},
        )
        warnings_list = _validate_age(data)
        assert not warnings_list

    def test_validate_age_with_na_string(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'age': ['N/A']},
            schema={'participant_id': pl.String, 'age': pl.String},
        )
        warnings_list = _validate_age(data)
        assert not warnings_list

    def test_validate_sex_with_na_string(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'sex': ['NA']},
            schema={'participant_id': pl.String, 'sex': pl.String},
        )
        warnings_list = _validate_sex(data)
        assert not warnings_list

    def test_validate_handedness_with_empty_string(self):
        data = pl.DataFrame(
            {'participant_id': ['sub-01'], 'handedness': ['']},
            schema={'participant_id': pl.String, 'handedness': pl.String},
        )
        warnings_list = _validate_handedness(data)
        assert not warnings_list
