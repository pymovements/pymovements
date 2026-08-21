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
    """Validate participant_id column exists and is first column."""
    if 'participant_id' not in data.columns:
        raise ValueError("data must have column named 'participant_id'")
    if data.columns[0] != 'participant_id':
        raise ValueError("first column in data must be named 'participant_id'")


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
