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
"""Test Dataset participants functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetFile
from pymovements import Participants
from pymovements import ResourceDefinition


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
    ],
)
def test_dataset_scan_participants_correct_data(files, expected_participants, tmp_path):
    dataset = Dataset(DatasetDefinition('.'), path=tmp_path)
    dataset._files = files

    assert dataset.participants.data.shape == (0, 1)

    dataset.scan_participants()

    assert_frame_equal(dataset.participants.data, expected_participants)


@pytest.mark.parametrize(
    ('participants', 'resources'),
    [
        pytest.param(
            pl.DataFrame(schema={'participant_id': pl.String}),
            [ResourceDefinition(content='participants', filename_pattern='participants.tsv')],
            id='empty_participants',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1']}),
            [ResourceDefinition(content='participants', filename_pattern='participants.tsv')],
            id='one_participant_single_column',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1', '2']}),
            [ResourceDefinition(content='participants', filename_pattern='participants.tsv')],
            id='two_participants_single_column',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1', '2'], 'age': [21, 28]}),
            [ResourceDefinition(content='participants', filename_pattern='participants.tsv')],
            id='two_participants_with_age',
        ),
    ],
)
class TestDatasetLoadParticipants:
    def test_dataset_load_participants_file(
            self, participants, resources, tmp_path, make_csv_file,
    ):
        make_csv_file(
            tmp_path / 'participants.tsv',
            data=participants,
            separator='\t',
        )
        dataset = Dataset(DatasetDefinition('.', resources=resources), path=tmp_path)
        dataset.scan()

        assert dataset.participants.data.shape == (0, 1)

        dataset.load_participants()

        assert_frame_equal(dataset.participants.data, participants)

    def test_dataset_load_loads_participants(
            self, participants, resources, tmp_path, make_csv_file,
    ):
        make_csv_file(
            tmp_path / 'participants.tsv',
            data=participants,
            separator='\t',
        )
        dataset = Dataset(DatasetDefinition('.', resources=resources), path=tmp_path)
        dataset.scan()

        assert dataset.participants.data.shape == (0, 1)

        dataset.load()

        assert_frame_equal(dataset.participants.data, participants)

    def test_dataset_load_does_not_load_participants(
            self, participants, resources, tmp_path, make_csv_file,
    ):
        make_csv_file(
            tmp_path / 'participants.tsv',
            data=participants,
            separator='\t',
        )
        dataset = Dataset(DatasetDefinition('.', resources=resources), path=tmp_path)
        dataset.scan()

        assert dataset.participants.data.shape == (0, 1)

        dataset.load(participants=False)

        assert dataset.participants.data.shape == (0, 1)


def test_dataset_load_participants_raises_with_multiple_participant_files(
        tmp_path,
        make_csv_file,
):
    participants = pl.DataFrame({'participant_id': ['1']})
    make_csv_file(
        tmp_path / 'participants.tsv',
        data=participants,
        separator='\t',
    )
    make_csv_file(
        tmp_path / 'participants2.tsv',
        data=participants,
        separator='\t',
    )

    resources = [
        ResourceDefinition(content='participants', filename_pattern='participants.tsv'),
        ResourceDefinition(content='participants', filename_pattern='participants2.tsv'),
    ]
    dataset = Dataset(DatasetDefinition('.', resources=resources), path=tmp_path)
    dataset.scan()

    message = 'there may be only a single participants resource per dataset'
    with pytest.raises(AttributeError, match=message):
        dataset.load_participants()


def test_dataset_load_participants_raises_without_participants_resource(tmp_path):
    dataset = Dataset(DatasetDefinition('.', resources=[]), path=tmp_path)
    dataset.scan()

    message = 'no participant file defined in dataset resources'
    with pytest.raises(AttributeError, match=message):
        dataset.load_participants()
