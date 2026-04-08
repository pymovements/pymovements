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
"""Unit tests for shared BIDS dataset helpers."""
import polars as pl
import pytest

from pymovements.dataset._bids_dataset import _bids_format_to_polars_datatype
from pymovements.dataset._bids_dataset import _cast_columns_to_metadata_format
from pymovements.dataset._bids_dataset import _polars_datatype_to_bids_format
from pymovements.dataset._bids_dataset import _validate_participant_id_format
from pymovements.dataset._bids_dataset import _validate_participant_id_structure


class TestValidateParticipantId:
    @pytest.mark.parametrize(
        'data',
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-02']}),
                id='valid_participant_id_first',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'age': [25]}),
                id='valid_with_additional_columns',
            ),
        ],
    )
    def test_valid_participant_id_passes(self, data):
        _validate_participant_id_structure(data)

    @pytest.mark.parametrize(
        ('data', 'error_msg'),
        [
            pytest.param(
                pl.DataFrame({'subject': ['sub-01']}),
                "data must have column named 'participant_id'",
                id='missing_participant_id_column',
            ),
            pytest.param(
                pl.DataFrame({'age': [25], 'participant_id': ['sub-01']}),
                "first column in data must be named 'participant_id'",
                id='participant_id_not_first',
            ),
        ],
    )
    def test_invalid_participant_id_raises(self, data, error_msg):
        with pytest.raises(ValueError, match=error_msg):
            _validate_participant_id_structure(data)


class TestValidateParticipantIdFormat:
    @pytest.mark.parametrize(
        'data,expected',
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-02']}),
                [],
                id='valid_format',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['01', '02']}),
                [
                    "participant_id values must match 'sub-<label>' pattern. "
                    "Invalid values: ['01', '02']",
                ],
                id='invalid_format',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-01']}),
                ['participant_id values must be unique'],
                id='duplicate_ids',
            ),
            pytest.param(
                pl.DataFrame({'a': [1], 'participant_id': ['sub-01']}),
                ['participant_id column must be the first column'],
                id='not_first_column',
            ),
            pytest.param(
                pl.DataFrame(),
                ['participant_id column is missing'],
                id='missing_column',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': [None]}),
                [],
                id='all_null',
            ),
        ],
    )
    def test_validate_participant_id_format(self, data, expected):
        result = _validate_participant_id_format(data)
        assert result == expected


class TestCastColumnsToMetadataFormat:
    @pytest.mark.parametrize(
        ('data', 'metadata', 'expected_dtypes'),
        [
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01', 'sub-02']}),
                {'participant_id': {'Format': 'string'}},
                {'participant_id': pl.String},
                id='cast_to_string',
            ),
            pytest.param(
                pl.DataFrame({'score': [1, 2, 3]}),
                {'score': {'Format': 'integer'}},
                {'score': pl.Int64},
                id='cast_to_integer',
            ),
            pytest.param(
                pl.DataFrame({'value': [1.5, 2.5, 3.5]}),
                {'value': {'Format': 'number'}},
                {'value': pl.Float64},
                id='cast_to_number',
            ),
            pytest.param(
                pl.DataFrame({'flag': [True, False, True]}),
                {'flag': {'Format': 'bool'}},
                {'flag': pl.Boolean},
                id='cast_to_bool',
            ),
            pytest.param(
                pl.DataFrame({'idx': [0, 1, 2]}),
                {'idx': {'Format': 'index'}},
                {'idx': pl.UInt64},
                id='cast_to_index',
            ),
            pytest.param(
                pl.DataFrame({'participant_id': ['sub-01'], 'score': [1]}),
                {
                    'participant_id': {'Format': 'string'},
                    'score': {'Format': 'integer'},
                },
                {'participant_id': pl.String, 'score': pl.Int64},
                id='multiple_columns',
            ),
            pytest.param(
                pl.DataFrame({'col': [1, 2, 3]}),
                {},
                {'col': pl.Int64},
                id='no_format_in_metadata',
            ),
            pytest.param(
                pl.DataFrame({'col': [1, 2, 3]}),
                {'other': {'Format': 'string'}},
                {'col': pl.Int64},
                id='no_format_for_column',
            ),
        ],
    )
    def test_cast_columns_to_metadata_format(self, data, metadata, expected_dtypes):
        result = _cast_columns_to_metadata_format(data, metadata)
        assert result.schema == expected_dtypes

    def test_unknown_bids_format_raises(self):
        data = pl.DataFrame({'col': [1, 2, 3]})
        metadata = {'col': {'Format': 'unknown_format'}}
        with pytest.raises(
            TypeError,
            match="unknown bids format descriptor 'unknown_format'",
        ):
            _cast_columns_to_metadata_format(data, metadata)


class TestBidsFormatRoundTrip:
    @pytest.mark.parametrize(
        ('polars_dtype', 'expected_bids_format'),
        [
            pytest.param(pl.UInt8, 'index', id='uint8'),
            pytest.param(pl.UInt16, 'index', id='uint16'),
            pytest.param(pl.UInt32, 'index', id='uint32'),
            pytest.param(pl.UInt64, 'index', id='uint64'),
            pytest.param(pl.Int8, 'integer', id='int8'),
            pytest.param(pl.Int16, 'integer', id='int16'),
            pytest.param(pl.Int32, 'integer', id='int32'),
            pytest.param(pl.Int64, 'integer', id='int64'),
            pytest.param(pl.Float32, 'number', id='float32'),
            pytest.param(pl.Float64, 'number', id='float64'),
            pytest.param(pl.Boolean, 'bool', id='boolean'),
            pytest.param(pl.String, 'string', id='string'),
        ],
    )
    def test_polars_to_bids_roundtrip(self, polars_dtype, expected_bids_format):
        bids_format = _polars_datatype_to_bids_format(polars_dtype)
        assert bids_format == expected_bids_format

        polars_result = _bids_format_to_polars_datatype(bids_format)
        assert isinstance(polars_result, type(polars_dtype))

    @pytest.mark.parametrize(
        'bids_format',
        [
            pytest.param('string', id='string'),
            pytest.param('number', id='number'),
            pytest.param('integer', id='integer'),
            pytest.param('bool', id='bool'),
            pytest.param('index', id='index'),
        ],
    )
    def test_bids_to_polars_roundtrip(self, bids_format):
        polars_dtype = _bids_format_to_polars_datatype(bids_format)
        bids_result = _polars_datatype_to_bids_format(polars_dtype)
        assert bids_result == bids_format

    def test_unknown_polars_dtype_raises(self):
        with pytest.raises(
            TypeError,
            match='polars datatype .* has no mapping to bids format descriptor',
        ):
            _polars_datatype_to_bids_format(pl.Date)

    def test_unknown_bids_format_raises(self):
        with pytest.raises(
            TypeError,
            match="unknown bids format descriptor 'unknown'",
        ):
            _bids_format_to_polars_datatype('unknown')
