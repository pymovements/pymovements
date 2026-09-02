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
"""Shared helpers for BIDS dataset classes."""
from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any
from typing import Literal

import polars


def _validate_participant_id_structure(data: polars.DataFrame) -> None:
    """Validate that a participant_id column exists.

    All other conformity issues, including the participant_id column not being the
    first column, are reported as warnings by _validate_participant_id_format.
    """
    if 'participant_id' not in data.columns:
        raise ValueError("data must have column named 'participant_id'")


def _validate_participant_id_format(data: polars.DataFrame) -> list[str]:
    """Validate participant_id column format per BIDS specification.

    Parameters
    ----------
    data : polars.DataFrame
        The participants DataFrame to validate.

    Returns
    -------
    list[str]
        List of warning messages for any non-conformities found.
    """
    validation_warnings: list[str] = []

    if 'participant_id' not in data.columns:
        return ['participant_id column is missing']

    if data.columns[0] != 'participant_id':
        validation_warnings.append('participant_id column must be the first column')

    if data['participant_id'].dtype != polars.String:
        validation_warnings.append(
            'participant_id column must have string (Utf8) data type',
        )

    if data['participant_id'].null_count() > 0:
        validation_warnings.append('participant_id column contains null values')

    participant_ids = data['participant_id'].drop_nulls().to_list()

    pattern = re.compile(r'^sub-[a-zA-Z0-9+]+$')
    invalid_ids = [pid for pid in participant_ids if not pattern.match(str(pid))]
    if invalid_ids:
        validation_warnings.append(
            f"participant_id values must match 'sub-<label>' pattern. "
            f"Invalid values: {invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}",
        )

    unique_ids = set(participant_ids)
    if len(unique_ids) != len(participant_ids):
        validation_warnings.append('participant_id values must be unique')

    return validation_warnings


def _bids_format_to_polars_datatype(bids_format: str) -> polars.DataType:
    """Infer polars datatype from bids format descriptor."""
    mapping = {
        'string': polars.String,
        'number': polars.Float64,
        'integer': polars.Int64,
        'bool': polars.Boolean,
        'index': polars.UInt64,
        'label': polars.String,
    }

    if bids_format in mapping:
        return mapping[bids_format]

    raise TypeError(
        f"unknown bids format descriptor '{bids_format}'. Known formats: {list(mapping.keys())}",
    )


def _polars_datatype_to_bids_format(dtype: polars.DataType) -> str:
    """Infer bids format descriptor from polars datatype."""
    if dtype.is_unsigned_integer():
        return 'index'
    if dtype.is_integer():
        return 'integer'
    if dtype.is_numeric():
        return 'number'
    if dtype == polars.Boolean:
        return 'bool'
    if dtype == polars.String:
        return 'string'
    if dtype == polars.Null:
        return 'string'

    raise TypeError(
        f"polars datatype {dtype} has no mapping to bids format descriptor. "
        f"Supported polars datatypes are: Integer, Float, String",
    )


def _check_na_conformity(data: polars.DataFrame) -> list[str]:
    """Check that missing values are coded as 'n/a' in every column.

    BIDS requires that missing and non-applicable values MUST be coded as 'n/a'.
    The convention applies to all columns, not just standard BIDS columns.
    Null values are conformant: they are written as 'n/a' on save and read back
    as nulls on load.
    """
    validation_warnings: list[str] = []
    na_alternatives = ['N/A', 'NA', 'na', 'NaN', 'nan', '']

    for column in data.columns:
        series = data[column]
        invalid_na: set[str] = set()

        if series.dtype == polars.String:
            mask = series.is_in(na_alternatives).fill_null(False)
            invalid_na.update(series.filter(mask).unique().to_list())
        elif series.dtype in (polars.Float32, polars.Float64):
            if series.drop_nulls().is_nan().any():
                invalid_na.add('NaN')

        if invalid_na:
            validation_warnings.append(
                f"Column '{column}' contains invalid null values: {invalid_na}. "
                "BIDS requires missing values to be coded as 'n/a'.",
            )
    return validation_warnings


def _infer_metadata_column_format(
        data: polars.DataFrame,
        metadata: dict[str, Any],
) -> dict[str, Any]:
    """Infer bids format of each column in data and update metadata."""
    for column in data.columns:
        if column not in metadata:
            metadata[column] = {}

        if 'Format' not in metadata[column]:
            # infer format from BIDS specification or use polars datatypes of data columns
            if column == 'participant_id':
                metadata[column]['Format'] = 'string'
            else:
                # convert polars datatype to bids format descriptor
                metadata[column]['Format'] = _polars_datatype_to_bids_format(data[column].dtype)

    return metadata


def _cast_columns_to_metadata_format(
    data: polars.DataFrame,
    metadata: dict[str, Any],
) -> polars.DataFrame:
    """Cast columns in data according to column bids format specified in metadata."""
    schema_overrides = {}
    for column in data.columns:
        bids_format = metadata.get(column, {}).get('Format', None)
        if bids_format:
            schema_overrides[column] = _bids_format_to_polars_datatype(bids_format)
    return data.cast(schema_overrides)


def _default_bids_read_csv_kwargs() -> dict[str, Any]:
    """Return default read_csv keyword arguments for BIDS tabular files.

    BIDS tabular files are tab-separated. Values encoded as 'n/a' are read
    back as nulls per BIDS specification.
    """
    return {'separator': '\t', 'null_values': ['n/a']}


def _default_bids_write_csv_kwargs() -> dict[str, Any]:
    """Return default write_csv keyword arguments for BIDS tabular files.

    BIDS tabular files are tab-separated. Null values are written as 'n/a'
    per BIDS specification.
    """
    return {'separator': '\t', 'null_value': 'n/a'}


def _verify_bids_handler(
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool,
    verify_func: Callable[[Literal['REQUIRED', 'RECOMMENDED']], list[str]],
    stacklevel: int = 3,
) -> None:
    """Handle verify_bids parameter by raising or warning on non-conformities.

    Parameters
    ----------
    verify_bids : Literal['REQUIRED', 'RECOMMENDED'] | bool
        If True, raise exception on non-conformity at REQUIRED level.
        If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that level.
        If False, do nothing.
    verify_func : Callable[[Literal['REQUIRED', 'RECOMMENDED']], list[str]]
        Function that takes a level string and returns list of warning messages.
    stacklevel : int
        Stack level for warnings.warn (default 3).
    """
    if verify_bids is not False:
        level: Literal['REQUIRED', 'RECOMMENDED'] = 'REQUIRED'
        if isinstance(verify_bids, str):
            level = verify_bids
        warnings_list = verify_func(level)
        if warnings_list:
            if verify_bids is True:
                raise ValueError(
                    f"BIDS non-conformities found: {'; '.join(warnings_list)}",
                )
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, UserWarning, stacklevel=stacklevel)
