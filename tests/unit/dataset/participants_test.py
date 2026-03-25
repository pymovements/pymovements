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
from copy import deepcopy

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Participants


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
    assert participants.metadata == {}


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
            "unknown bids format descriptor 'test'",
            id='unknown_bids_format',
        ),
    ],
)
def test_participants_init_metadata_raises(data, metadata, expected_exception, expected_message):
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
    path = make_csv_file('participants.tsv', pl.DataFrame({'a': [1, 2]}), separator='\t')

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
        source_data, rename, expected_data, make_csv_file,
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
            self, data, metadata, expected_data, make_csv_file,
    ):
        path = make_csv_file('participants.tsv', data, separator='\t')
        participants = Participants.load(path, metadata=metadata)
        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_path(
            self, data, metadata, expected_data, make_csv_file, make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        json_path = make_json_file('participants.json', metadata)

        participants = Participants.load(tsv_path, metadata=json_path)

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_filename(
            self, data, metadata, expected_data, make_csv_file, make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'test_participants.json', metadata)

        participants = Participants.load(tsv_path, metadata='test_participants.json')

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_casts_from_metadata_implicit(
            self, data, metadata, expected_data, make_csv_file, make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'participants.json', metadata)

        participants = Participants.load(tsv_path)

        assert_frame_equal(participants.data, expected_data)

    def test_participants_load_kwargs_take_precedence(
            self, data, metadata, expected_data, make_csv_file, make_json_file,
    ):
        tsv_path = make_csv_file('participants.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'participants.json', metadata)

        participants = Participants.load(
            tsv_path,
            separator=',',
            read_csv_kwargs={'separator': '\t'},  # takes precedence over explicit separator
        )

        assert_frame_equal(participants.data, expected_data)


def test_participants_save_data_to_filepath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(save_path)

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
    participants.save(save_path, separator=separator)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator=separator)
    assert_frame_equal(saved_data, data)


def test_participants_save_data_write_csv_kwargs_precedence_over_separator(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    save_path = tmp_path / 'test_participants.tsv'
    participants.save(save_path, separator=',', write_csv_kwargs={'separator': '\t'})

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


def test_participants_save_data_to_dirpath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'age': [21.0]})
    participants = Participants(data)

    participants.save(tmp_path)

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
                'participant_id': {'Description': 'id of the participant', 'Format': 'string'},
                'age': {
                    'Description': 'age of the participant', 'Format': 'integer', 'Units': 'year',
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    'encoding', ['utf-8', 'ascii'],
)
class TestParticipantsSaveMetadata:
    def test_participants_save_metadata_to_dirpath(self, participants, encoding, tmp_path):
        metadata = deepcopy(participants.metadata)
        participants.save(tmp_path, metadata_encoding=encoding)

        save_path = tmp_path / 'participants.json'
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_participants_save_metadata_to_filepath(self, participants, encoding, tmp_path):
        metadata = deepcopy(participants.metadata)
        metadata_path = tmp_path / 'test_participants.json'
        participants.save(tmp_path, metadata_path=metadata_path, metadata_encoding=encoding)

        save_path = metadata_path
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_participants_save_metadata_to_relative_path(self, participants, encoding, tmp_path):
        metadata = deepcopy(participants.metadata)
        metadata_filename = 'test_participants.json'
        participants.save(tmp_path, metadata_path=metadata_filename, metadata_encoding=encoding)

        save_path = tmp_path / metadata_filename
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata
