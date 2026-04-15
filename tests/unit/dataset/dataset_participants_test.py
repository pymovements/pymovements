import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetFile
from pymovements import Participants


def test_dataset_init_empty_participants(tmp_path):
    dataset = Dataset(DatasetDefinition('.'), path=tmp_path)

    assert isinstance(dataset.participants, Participants)
    assert dataset.participants.data.schema == {'participant_id': pl.String}
    assert dataset.participants.data.shape == (0, 1)


@pytest.mark.parametrize(
    ('files', 'expected_participants'),
    [
        pytest.param(
            [],
            pl.DataFrame(schema={'participant_id': pl.String}),
            id='empty_files',
        ),
        pytest.param(
            [DatasetFile(path='1.csv', metadata={})],
            pl.DataFrame(schema={'participant_id': pl.String}),
            id='one_file_no_participant',
        ),
        pytest.param(
            [DatasetFile(path='1.csv', metadata={'participant_id': '1'})],
            pl.DataFrame({'participant_id': ['1']}),
            id='one_file_with_participant_id',
        ),
        pytest.param(
            [
                DatasetFile(path='1.csv', metadata={'participant_id': '1'}),
                DatasetFile(path='1_supplementary.csv', metadata={'participant_id': '1'}),
            ],
            pl.DataFrame({'participant_id': ['1']}),
            id='two_files_with_same_participant_id',
        ),
        pytest.param(
            [
                DatasetFile(path='1.csv', metadata={'participant_id': '1'}),
                DatasetFile(path='2.csv', metadata={'participant_id': '2'}),
            ],
            pl.DataFrame({'participant_id': ['1', '2']}),
            id='two_files_with_different_participant_id',
        ),
    ]
)
def test_dataset_scan_participants_correct_data(files, expected_participants, tmp_path):
    dataset = Dataset(DatasetDefinition('.'), path=tmp_path)
    dataset.files = files

    assert dataset.participants.data.shape == (0, 1)

    dataset.scan_participants()

    assert_frame_equal(dataset.participants.data, expected_participants)
