
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Participants

@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
    ]
)
def test_participants_init_data(data):
    participants = Participants(data)
    assert_frame_equal(participants.data, data)


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
    ],
)
def test_participants_init_data_raises(data, expected_exception, expected_message):
    with pytest.raises(expected_exception, match=expected_message):
        participants = Participants(data)


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
        pl.DataFrame({'participant_id': ['1'], 'age': [21.0]}),
    ]
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
    ]
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
        participants = Participants.load(path)


@pytest.mark.parametrize(
    ('source_data', 'rename', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'subject_id': ['1']}),
            {'subject_id': 'participant_id'},
            pl.DataFrame({'participant_id': ['1']}),
            id='subject_id_to_participant_id',
        ),
    ]
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
            id='autocast_participant_id_to_string'
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
