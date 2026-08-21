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
"""Unit tests of Phenotype class functionality."""
import json
from copy import deepcopy

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Phenotype


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
    ],
)
def test_phenotype_init_data(data):
    phenotype = Phenotype(data)
    assert_frame_equal(phenotype.data, data)


@pytest.mark.parametrize(
    ('data', 'expected_metadata'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['1']}),
            {'participant_id': {'Format': 'string'}},
            id='participant_id_string',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'adhd_score': [21.0]}),
            {
                'participant_id': {'Format': 'string'},
                'adhd_score': {'Format': 'number'},
            },
            id='adhd_score_number',
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
def test_phenotype_init_infers_correct_format(data, expected_metadata):
    phenotype = Phenotype(data)
    assert phenotype.metadata == expected_metadata


@pytest.mark.parametrize(
    'data',
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['1']}),
            id='participant_id',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'adhd_score': [21]}),
            id='adhd_score',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'test': ['a']}),
            id='string',
        ),
    ],
)
def test_phenotype_init_no_metadata_infer(data):
    phenotype = Phenotype(data, infer_metadata=False)
    assert isinstance(phenotype.metadata, dict)
    assert not phenotype.metadata


@pytest.mark.parametrize(
    ('data', 'verify_bids', 'expected_exception', 'expected_message'),
    [
        pytest.param(
            pl.DataFrame(),
            True,
            ValueError,
            "data must have column named 'participant_id'",
            id='empty',
        ),
        pytest.param(
            pl.DataFrame({'a': [1]}),
            True,
            ValueError,
            "data must have column named 'participant_id'",
            id='no_participant_id',
        ),
        pytest.param(
            pl.DataFrame({'a': [1], 'participant_id': ['001']}),
            True,
            ValueError,
            "first column in data must be named 'participant_id'",
            id='participant_id_not_first_column',
        ),
        pytest.param(
            pl.DataFrame({'subject_id': [1]}),
            True,
            ValueError,
            "data must have column named 'participant_id'",
            id='subject_id_not_participant_id',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': [1], 'test': [(1, 2)]}),
            False,
            TypeError,
            r'polars datatype List\(Int64\) has no mapping to bids format descriptor',
            id='list_format_not_supported',
        ),
    ],
)
def test_phenotype_init_data_raises(
    data,
    verify_bids,
    expected_exception,
    expected_message,
):
    with pytest.raises(expected_exception, match=expected_message):
        Phenotype(data, verify_bids=verify_bids)


def test_phenotype_init_no_args():
    phenotype = Phenotype()
    expected_data = pl.DataFrame(
        schema={'participant_id': pl.String},
    )
    assert_frame_equal(phenotype.data, expected_data)


def test_phenotype_init_no_data_no_infer():
    phenotype = Phenotype(infer_metadata=False)
    assert isinstance(phenotype.metadata, dict)
    assert not phenotype.metadata


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
def test_phenotype_init_metadata_raises(
    data,
    metadata,
    expected_exception,
    expected_message,
):
    with pytest.raises(expected_exception, match=expected_message):
        Phenotype(data, metadata)


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
        pl.DataFrame({'participant_id': ['1'], 'adhd_score': [21.0]}),
    ],
)
def test_phenotype_load_data_from_tsv(data, make_csv_file):
    filename = 'acds_adult.tsv'
    path = make_csv_file(filename, data, separator='\t')

    phenotype = Phenotype.load(path)

    assert_frame_equal(phenotype.data, data)


@pytest.mark.parametrize(
    'data',
    [
        pl.DataFrame({'participant_id': ['1']}),
    ],
)
def test_phenotype_load_data_from_csv(data, make_csv_file):
    filename = 'acds_adult.csv'
    path = make_csv_file(filename, data)

    phenotype = Phenotype.load(path)

    assert_frame_equal(phenotype.data, data)


def test_phenotype_load_data_from_directory(make_csv_file):
    """Test that phenotype/<name>.tsv is loaded when file path in phenotype dir is provided."""
    data = pl.DataFrame({'participant_id': ['1']})
    phenotype_file = make_csv_file('phenotype.tsv', data, separator='\t')
    phenotype_dir = phenotype_file.parent / 'phenotype'
    phenotype_dir.mkdir(parents=True, exist_ok=True)
    phenotype_file.rename(phenotype_dir / 'phenotype.tsv')

    phenotype = Phenotype.load(phenotype_dir / 'phenotype.tsv')

    assert_frame_equal(phenotype.data, data)


def test_phenotype_load_from_directory_path(tmp_path):
    """Test that phenotype data is loaded when directory path is provided."""
    data = pl.DataFrame({'participant_id': ['1']})
    dataset_dir = tmp_path / 'mydataset'
    phenotype_dir = dataset_dir / 'phenotype'
    phenotype_dir.mkdir(parents=True)
    data_file = phenotype_dir / 'mydataset.tsv'
    data.write_csv(data_file, separator='\t')

    phenotype = Phenotype.load(dataset_dir)

    assert_frame_equal(phenotype.data, data)


def test_phenotype_load_from_directory_path_with_existing_phenotype_dir(tmp_path):
    """Test loading when phenotype directory already exists."""
    data = pl.DataFrame({'participant_id': ['1']})
    dataset_dir = tmp_path / 'mydataset'
    dataset_dir.mkdir()
    phenotype_dir = dataset_dir / 'phenotype'
    phenotype_dir.mkdir()
    data_file = phenotype_dir / 'mydataset.tsv'
    data.write_csv(data_file, separator='\t')

    phenotype = Phenotype.load(dataset_dir)

    assert_frame_equal(phenotype.data, data)


def test_phenotype_load_missing_participant_id_raises(make_csv_file):
    """Test that ValueError is raised if loaded phenotype data has no participant_id column."""
    path = make_csv_file('acds_adult.tsv', pl.DataFrame({'a': [1, 2]}), separator='\t')

    with pytest.raises(ValueError, match='participant_id'):
        Phenotype.load(path, verify_bids=True)


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
def test_phenotype_load_and_rename_data_from_file(
    source_data,
    rename,
    expected_data,
    make_csv_file,
):
    filename = 'acds_adult.tsv'
    path = make_csv_file(filename, source_data, separator='\t')

    phenotype = Phenotype.load(path, rename=rename)

    assert_frame_equal(phenotype.data, expected_data)


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
def test_phenotype_init_autocasts(data, expected_data):
    phenotype = Phenotype(data)
    assert_frame_equal(phenotype.data, expected_data)


@pytest.mark.parametrize(
    ('data', 'metadata', 'expected_data'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['1'], 'adhd_score': ['21']}),
            {'adhd_score': {'Format': 'integer'}},
            pl.DataFrame(
                {'participant_id': ['1'], 'adhd_score': [21]},
                schema={'participant_id': pl.String, 'adhd_score': pl.Int64},
            ),
            id='cast_adhd_score_column_to_int',
        ),
    ],
)
class TestPhenotypeCastsFromMetadata:
    def test_phenotype_init_casts_from_metadata(self, data, metadata, expected_data):
        phenotype = Phenotype(data, metadata)
        assert_frame_equal(phenotype.data, expected_data)

    def test_phenotype_load_casts_from_metadata_dict(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
    ):
        path = make_csv_file('acds_adult.tsv', data, separator='\t')
        phenotype = Phenotype.load(path, metadata=metadata)
        assert_frame_equal(phenotype.data, expected_data)

    def test_phenotype_load_casts_from_metadata_path(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('acds_adult.tsv', data, separator='\t')
        json_path = make_json_file('acds_adult.json', metadata)

        phenotype = Phenotype.load(tsv_path, metadata=json_path)

        assert_frame_equal(phenotype.data, expected_data)

    def test_phenotype_load_casts_from_metadata_filename(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('acds_adult.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'test_phenotype.json', metadata)

        phenotype = Phenotype.load(tsv_path, metadata='test_phenotype.json')

        assert_frame_equal(phenotype.data, expected_data)

    def test_phenotype_load_casts_from_metadata_implicit(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('acds_adult.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'acds_adult.json', metadata)

        phenotype = Phenotype.load(tsv_path)

        assert_frame_equal(phenotype.data, expected_data)

    def test_phenotype_load_kwargs_take_precedence(
        self,
        data,
        metadata,
        expected_data,
        make_csv_file,
        make_json_file,
    ):
        tsv_path = make_csv_file('acds_adult.tsv', data, separator='\t')
        make_json_file(tsv_path.parent / 'acds_adult.json', metadata)

        phenotype = Phenotype.load(
            tsv_path,
            separator=',',
            read_csv_kwargs={'separator': '\t'},
        )

        assert_frame_equal(phenotype.data, expected_data)


def test_phenotype_save_data_to_filepath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'adhd_score': [21.0]})
    phenotype = Phenotype(data)

    save_path = tmp_path / 'test_phenotype.tsv'
    phenotype.save(save_path)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


@pytest.mark.parametrize(
    'separator',
    ['\t', ','],
)
def test_phenotype_save_data_to_filepath_custom_separator(separator, tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'adhd_score': [21.0]})
    phenotype = Phenotype(data)

    save_path = tmp_path / 'test_phenotype.tsv'
    phenotype.save(save_path, separator=separator)

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator=separator)
    assert_frame_equal(saved_data, data)


def test_phenotype_save_data_write_csv_kwargs_precedence_over_separator(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'adhd_score': [21.0]})
    phenotype = Phenotype(data)

    save_path = tmp_path / 'test_phenotype.tsv'
    phenotype.save(save_path, separator=',', write_csv_kwargs={'separator': '\t'})

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)


def test_phenotype_save_data_to_dirpath(tmp_path):
    data = pl.DataFrame({'participant_id': [1], 'adhd_score': [21.0]})
    phenotype = Phenotype(data)

    phenotype.save(tmp_path)

    save_path = tmp_path / 'phenotype' / f"{tmp_path.name}.tsv"

    assert save_path.is_file()
    saved_data = pl.read_csv(save_path, separator='\t')
    assert_frame_equal(saved_data, data)

    metadata_path = tmp_path / f"{tmp_path.name}.json"
    assert metadata_path.is_file()


@pytest.mark.parametrize(
    'phenotype',
    [
        Phenotype(
            data=pl.DataFrame({'participant_id': [1], 'adhd_score': [21]}),
            metadata={
                'participant_id': {
                    'Description': 'id of the participant',
                    'Format': 'string',
                },
                'adhd_score': {
                    'Description': 'adhd score of the participant',
                    'Format': 'integer',
                    'Units': 'score',
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    'encoding',
    ['utf-8', 'ascii'],
)
class TestPhenotypeSaveMetadata:
    def test_phenotype_save_metadata_to_dirpath(self, phenotype, encoding, tmp_path):
        metadata = deepcopy(phenotype.metadata)
        phenotype.save(tmp_path, metadata_encoding=encoding)

        save_path = tmp_path / f"{tmp_path.name}.json"
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_phenotype_save_metadata_to_filepath(self, phenotype, encoding, tmp_path):
        metadata = deepcopy(phenotype.metadata)
        metadata_path = tmp_path / 'test_phenotype.json'
        phenotype.save(
            tmp_path,
            metadata_path=metadata_path,
            metadata_encoding=encoding,
        )

        save_path = metadata_path
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata

    def test_phenotype_save_metadata_to_relative_path(
        self,
        phenotype,
        encoding,
        tmp_path,
    ):
        metadata = deepcopy(phenotype.metadata)
        metadata_filename = 'test_phenotype.json'
        phenotype.save(
            tmp_path / 'test_phenotype.tsv',
            metadata_path=metadata_filename,
            metadata_encoding=encoding,
        )

        save_path = tmp_path / metadata_filename
        assert save_path.is_file()
        with open(save_path, encoding=encoding) as opened_file:
            saved_metadata = json.load(opened_file)
        assert saved_metadata == metadata


def test_phenotype_load_with_measurement_tool_metadata(make_csv_file, make_json_file):
    """Test that MeasurementToolMetadata is loaded correctly."""
    data = pl.DataFrame({'participant_id': ['1'], 'adhd_b': [1]})
    tsv_path = make_csv_file('acds_adult.tsv', data, separator='\t')

    metadata = {
        'MeasurementToolMetadata': {
            'Description': 'Adult ADHD Clinical Diagnostic Scale V1.2',
            'TermURL': 'https://www.cognitiveatlas.org/task/id/trm_5586ff878155d',
        },
        'adhd_b': {
            'Description': 'B. CHILDHOOD ONSET OF ADHD (PRIOR TO AGE 7)',
            'Levels': {'1': 'YES', '2': 'NO'},
        },
    }
    make_json_file(tsv_path.parent / 'acds_adult.json', metadata)

    phenotype = Phenotype.load(tsv_path)

    assert 'MeasurementToolMetadata' in phenotype.metadata
    assert (
        phenotype.metadata['MeasurementToolMetadata']['Description']
        == 'Adult ADHD Clinical Diagnostic Scale V1.2'
    )
    assert phenotype.metadata['adhd_b']['Levels'] == {'1': 'YES', '2': 'NO'}


def test_phenotype_with_derivative_field():
    """Test that Derivative field in column metadata is preserved."""
    data = pl.DataFrame({'participant_id': ['1'], 'total_score': [42]})
    metadata = {
        'participant_id': {'Format': 'string'},
        'total_score': {
            'Description': 'Sum of all item scores',
            'Derivative': True,
            'Format': 'integer',
        },
    }
    phenotype = Phenotype(data, metadata)

    assert phenotype.metadata['total_score']['Derivative'] is True


@pytest.mark.parametrize(
    ('data', 'level', 'expect_warnings'),
    [
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01']}),
            'REQUIRED',
            False,
            id='valid',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['01']}),
            'REQUIRED',
            True,
            id='invalid_id',
        ),
        pytest.param(
            pl.DataFrame({'participant_id': ['sub-01']}),
            'INVALID',
            False,
            id='unknown_level',
        ),
    ],
)
def test_phenotype_verify_bids_method(data, level, expect_warnings):
    phenotype = Phenotype(data, verify_bids=False)
    warnings_list = phenotype.verify_bids(level)  # type: ignore[arg-type]
    assert (len(warnings_list) > 0) == expect_warnings


def test_phenotype_verify_bids_init_raises():
    with pytest.raises(ValueError, match='BIDS non-conformities found'):
        Phenotype(
            pl.DataFrame({'participant_id': ['01']}),
            verify_bids=True,
        )


def test_phenotype_verify_bids_init_warns():
    with pytest.warns(UserWarning, match="match 'sub-<label>' pattern"):
        Phenotype(
            pl.DataFrame({'participant_id': ['01']}),
            verify_bids='REQUIRED',
        )


def test_phenotype_verify_bids_init_passes():
    phenotype = Phenotype(
        pl.DataFrame({'participant_id': ['sub-01']}),
        verify_bids=True,
    )
    assert phenotype.data.shape == (1, 1)


def test_phenotype_verify_bids_save_raises(tmp_path):
    phenotype = Phenotype(
        pl.DataFrame({'participant_id': ['01']}),
        verify_bids=False,
    )
    with pytest.raises(ValueError, match='BIDS non-conformities found'):
        phenotype.save(tmp_path / 'phenotype.tsv', verify_bids=True)


def test_phenotype_verify_bids_save_warns(tmp_path):
    phenotype = Phenotype(
        pl.DataFrame({'participant_id': ['01']}),
        verify_bids=False,
    )
    with pytest.warns(UserWarning, match="match 'sub-<label>' pattern"):
        phenotype.save(tmp_path / 'phenotype.tsv', verify_bids='REQUIRED')


def test_phenotype_verify_bids_save_passes(tmp_path):
    phenotype = Phenotype(
        pl.DataFrame({'participant_id': ['sub-01']}),
        verify_bids=False,
    )
    phenotype.save(tmp_path / 'phenotype.tsv', verify_bids=True)
    assert (tmp_path / 'phenotype.tsv').exists()
