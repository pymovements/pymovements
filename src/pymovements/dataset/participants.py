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
"""Participants module."""
from __future__ import annotations

import json
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import polars

from pymovements._utils._html import repr_html


@dataclass
@repr_html()
class Participants:
    """Participant table with additional metadadata.

    Attributes
    ----------
    data: polars.DataFrame
        The participant data conforming to BIDS (i.e., first column must be named participant_id).
    metadata: dict[str, Any]
        Additional metadata on participant data conforming to BIDS side car json files.

    Parameters
    ----------
    data: polars.DataFrame
        The participant data conforming to BIDS (i.e., first column must be named participant_id).
    metadata: dict[str, Any] | None
        Additional metadata on participant data conforming to BIDS side car json files.
        If ``None``, initialize an empty dictionary.
        (default: ``None``)
    verify_bids: str | bool
        Verify BIDS conformity. If True, raise exception on non-conformity.
        If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that level.
        If False, do not verify.
        (default: ``False``)
    infer_metadata: bool
        Infer metadata column format descriptors from ``data``.
        (default: ``True``)
    """

    data: polars.DataFrame
    metadata: dict[str, Any]

    def __init__(
        self,
        data: polars.DataFrame,
        metadata: dict[str, Any] | None = None,
        *,
        verify_bids: str | bool = False,
        infer_metadata: bool = True,
    ):
        if 'participant_id' not in data.columns:
            raise ValueError("data must have column named 'participant_id'")
        if data.columns[0] != 'participant_id':
            raise ValueError("first column in data must be named 'participant_id'")

        if metadata:
            # metadata may be changed and updated, work on copy
            metadata = deepcopy(metadata)
        else:
            metadata = {}
        if infer_metadata:
            metadata = _infer_metadata_column_format(data, metadata)
        data = _cast_columns_to_metadata_format(data, metadata)

        self.data = data
        self.metadata = metadata

        if verify_bids is not False:
            level: Literal['REQUIRED', 'RECOMMENDED'] = 'REQUIRED'
            if isinstance(verify_bids, str):
                level = verify_bids  # type: ignore[assignment]
            warnings_list = self.verify_bids(level)
            if warnings_list:
                if verify_bids is True:
                    raise ValueError(
                        f'BIDS non-conformities found: {'; '.join(warnings_list)}',
                    )
                for warning_msg in warnings_list:
                    warnings.warn(warning_msg, UserWarning, stacklevel=2)

    @staticmethod
    def load(
        path: Path | str,
        metadata: Path | str | dict[str, Any] | None = None,
        *,
        verify_bids: str | bool = False,
        separator: str = '\t',
        rename: dict[str, str] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        metadata_encoding: str = 'utf-8',
    ) -> Participants:
        r"""Load participant data from participant files.

        Parameters
        ----------
        path: Path | str
            If this points to a directory, assume file with name `participants.tsv` inside that
            directory.
        metadata: Path | str | dict[str, Any] | None
            Additional metadata. Can be directly passed as a dictionary. If path or string: load
            metadata from json file. If a relative path given, path is assumed to be relative to
            parent of ``path``. If None: check for ``path`` parent directory for existing
            ``participants.json`` and load metadata if available.
            (default: None)
        verify_bids: str | bool
            Verify BIDS conformity. If True, raise exception on non-conformity.
            If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that level.
            If False, do not verify.
            (default: ``False``)
        separator: str
            Separator in the tabular data file.
            (default: ``\t``)
        rename: dict[str, str] | None
            Rename columns according to mapping in dictionary.
            (default: ``None``)
        read_csv_kwargs: dict[str, Any] | None
            Pass these additional keyword arguments to :py:func:`polars.read_csv`.
            Takes precedence over the ``separator`` argument.
            (default: ``None``)
        metadata_encoding: str
            Use this encoding for writing the metadata json file.
            (default: ``utf-8``)

        Returns
        -------
        Participants
            Participants initialized with data from loaded files.
        """
        path = Path(path)
        if path.is_dir():
            dir_path = path
            data_path = path / 'participants.tsv'
        else:
            dir_path = path.parent
            data_path = path

        if read_csv_kwargs is None:
            read_csv_kwargs = {'separator': separator}
        else:
            # **read_csv_kwargs takes precedence over explicit separator argument.
            read_csv_kwargs = {'separator': separator, **read_csv_kwargs}

        data = polars.read_csv(data_path, **read_csv_kwargs)

        if rename:
            data = data.rename(rename)

        if metadata is None:
            # Detect if there's a corresponding json file in the directory
            candidate_path = dir_path / f'{data_path.stem}.json'
            if candidate_path.is_file():
                metadata = candidate_path

        if isinstance(metadata, (Path, str)):
            metadata_path = Path(metadata)
            if metadata_path.parent == Path('.'):
                # Assume path is relative to directory of data file.
                metadata_path = dir_path / metadata_path

            # Load metadata from json file.
            with open(metadata_path, encoding=metadata_encoding) as opened_file:
                metadata_dict = json.load(opened_file)
        else:
            metadata_dict = metadata

        return Participants(data, metadata_dict, verify_bids=verify_bids)

    def save(
        self,
        path: Path | str,
        *,
        verify_bids: str | bool = 'REQUIRED',
        metadata_path: Path | str = 'participants.json',
        separator: str = '\t',
        write_csv_kwargs: dict[str, Any] | None = None,
        metadata_encoding: str = 'utf-8',
    ) -> None:
        r"""Save participants data including metadata.

        Parameters
        ----------
        path: Path | str
            Save participants data to this path. If this is a directory, use ``participants.tsv`` as
            filename.
        verify_bids: str | bool
            Verify BIDS conformity before saving. If True, raise exception on non-conformity.
            If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that level.
            If False, do not verify.
            (default: ``'REQUIRED'``)
        metadata_path:  Path | str
            Save metadata json to this path. If this is a relative path it is assumed to be relative
            to the directory specified by ``path``.
            (default: ``participants.json``)
        separator: str
            Separator in the tabular data file.
            (default: ``\t``)
        write_csv_kwargs: dict[str, Any] | None
            Pass these additional keyword arguments to :py:func:`polars.write_csv`.
            Takes precedence over the ``separator`` argument.
            (default: ``None``)
        metadata_encoding: str
            Use this encoding for loading the metadata json file.
            (default: ``utf-8``)
        """
        if verify_bids is not False:
            level: Literal['REQUIRED', 'RECOMMENDED'] = 'REQUIRED'
            if isinstance(verify_bids, str):
                level = verify_bids  # type: ignore[assignment]
            warnings_list = self.verify_bids(level)
            if warnings_list:
                if verify_bids is True:
                    raise ValueError(
                        f'BIDS non-conformities found: {'; '.join(warnings_list)}',
                    )
                for warning_msg in warnings_list:
                    warnings.warn(warning_msg, UserWarning, stacklevel=2)

        path = Path(path)
        if path.is_dir():
            dir_path = path
            data_path = path / 'participants.tsv'
        else:
            dir_path = path.parent
            data_path = path

        if write_csv_kwargs is None:
            write_csv_kwargs = {'separator': separator}
        else:
            # **write_csv_kwargs takes precedence over explicit separator argument.
            write_csv_kwargs = {'separator': separator, **write_csv_kwargs}
        self.data.write_csv(data_path, **write_csv_kwargs)

        metadata_path = Path(metadata_path)
        if metadata_path.parent == Path('.'):
            # Assume path is relative to directory of data file.
            metadata_path = dir_path / metadata_path
        # Save metadata to json file.
        with open(metadata_path, 'w', encoding=metadata_encoding) as opened_file:
            json.dump(self.metadata, opened_file)

    def verify_bids(
        self,
        level: Literal['REQUIRED', 'RECOMMENDED'] = 'REQUIRED',
    ) -> list[str]:
        r"""Verify BIDS conformity of participant data.

        Parameters
        ----------
        level : Literal['REQUIRED', 'RECOMMENDED']
            Level of BIDS compliance to verify.
            ``REQUIRED``: Check required fields only.
            ``RECOMMENDED``: Check required fields plus recommended fields.

        Returns
        -------
        list[str]
            List of warning messages for each non-conformity found.
            Empty list if data is BIDS conformant.

        Examples
        --------
        Verify BIDS compliance at REQUIRED level (default):

        >>> import polars as pl
        >>> from pymovements import Participants
        >>> data = pl.DataFrame({
        ...     "participant_id": ["sub-01", "sub-02"],
        ...     "age": [34, 12],
        ...     "sex": ["M", "F"],
        ... })
        >>> participants = Participants(data, verify_bids=False)
        >>> warnings = participants.verify_bids("REQUIRED")
        >>> print(warnings)
        []

        Verify at RECOMMENDED level with non-conformant data:

        >>> data = pl.DataFrame({
        ...     "participant_id": ["01", "sub-02"],
        ...     "age": [34, 100],  # age over 89
        ...     "sex": ["M", "invalid"],
        ... })
        >>> participants = Participants(data, verify_bids=False)
        >>> warnings = participants.verify_bids("RECOMMENDED")
        >>> for w in warnings:
        ...     print(w)
        participant_id values must match 'sub-<label>' pattern. Invalid values: ['01']
        age should be capped at 89, found 100.0
        sex must be one of ['F', 'FEMALE', 'Female', 'M', 'MALE', 'Male', 'O', 'OTHER', 'Other',
        'f', 'female', 'm', 'male', 'o', 'other'], found: ['invalid']

        Using ``verify_bids=True`` during initialisation raises an exception:

        >>> data = pl.DataFrame({"participant_id": ["01"]})
        >>> try:
        ...     participants = Participants(data, verify_bids=True)
        ... except ValueError as e:
        ...     print(str(e)[:50])
        BIDS non-conformities found: participant_id values

        Using ``verify_bids='REQUIRED'`` emits warnings but continues:

        >>> import warnings as warn
        >>> data = pl.DataFrame({"participant_id": ["01"]})
        >>> with warn.catch_warnings(record=True) as w:
        ...     warn.simplefilter("always")
        ...     participants = Participants(data, verify_bids="REQUIRED")
        ...     print(str(w[0].message))
        participant_id values must match 'sub-<label>' pattern. Invalid values: ['01']
        """
        warnings_list: list[str] = []

        if level in {'REQUIRED', 'RECOMMENDED'}:
            warnings_list.extend(_validate_participant_id(self.data))

        if level == 'RECOMMENDED':
            warnings_list.extend(_validate_age(self.data))
            warnings_list.extend(_validate_sex(self.data))
            warnings_list.extend(_validate_handedness(self.data))
            warnings_list.extend(_validate_species(self.data))
            warnings_list.extend(_validate_strain(self.data))
            warnings_list.extend(_validate_strain_rrid(self.data))

        return warnings_list


def _validate_participant_id(data: polars.DataFrame) -> list[str]:
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

    participant_ids = data['participant_id'].drop_nulls().to_list()

    pattern = re.compile(r'^sub-[a-zA-Z0-9]+$')
    invalid_ids = [pid for pid in participant_ids if not pattern.match(str(pid))]
    if invalid_ids:
        validation_warnings.append(
            f"participant_id values must match 'sub-<label>' pattern. "
            f'Invalid values: {invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}',
        )

    unique_ids = set(participant_ids)
    if len(unique_ids) != len(participant_ids):
        validation_warnings.append('participant_id values must be unique')

    return validation_warnings


def _validate_age(data: polars.DataFrame) -> list[str]:
    """Validate age column values per BIDS specification.

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

    if 'age' not in data.columns:
        return validation_warnings

    ages = data['age'].drop_nulls().to_list()
    na_values = {'n/a', 'N/A', 'NA', 'NaN', 'nan', ''}
    for age in ages:
        if str(age).lower() in na_values:
            continue
        try:
            age_val = float(age)
            if age_val > 89:
                validation_warnings.append(
                    f'age should be capped at 89, found {age_val}',
                )
        except (ValueError, TypeError):
            validation_warnings.append(f'age must be a numeric value, found {age}')

    return validation_warnings


def _validate_sex(data: polars.DataFrame) -> list[str]:
    """Validate sex column values per BIDS specification.

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

    if 'sex' not in data.columns:
        return validation_warnings

    valid_male = {'male', 'm', 'M', 'MALE', 'Male'}
    valid_female = {'female', 'f', 'F', 'FEMALE', 'Female'}
    valid_other = {'other', 'o', 'O', 'OTHER', 'Other'}
    valid_sex = valid_male | valid_female | valid_other
    na_values = {'n/a', 'N/A', 'na', 'NA', 'NaN', 'nan', ''}

    sex_values = data['sex'].drop_nulls().to_list()
    invalid_sex = [
        s
        for s in sex_values
        if str(s).lower() not in valid_sex and str(s).lower() not in na_values
    ]
    if invalid_sex:
        validation_warnings.append(
            f'sex must be one of {sorted(valid_sex)}, found: {invalid_sex[:5]}'
            f"{'...' if len(invalid_sex) > 5 else ''}",
        )

    return validation_warnings


def _validate_handedness(data: polars.DataFrame) -> list[str]:
    """Validate handedness column values per BIDS specification.

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

    if 'handedness' not in data.columns:
        return validation_warnings

    valid_left = {'left', 'l', 'L', 'LEFT', 'Left'}
    valid_right = {'right', 'r', 'R', 'RIGHT', 'Right'}
    valid_ambidextrous = {'ambidextrous', 'a', 'A', 'AMBIDEXTROUS', 'Ambidextrous'}
    valid_handedness = valid_left | valid_right | valid_ambidextrous
    na_values = {'n/a', 'N/A', 'NA', 'NaN', 'nan', ''}

    handedness_values = data['handedness'].drop_nulls().to_list()
    invalid_handedness = [
        h
        for h in handedness_values
        if str(h).lower() not in valid_handedness and str(h).lower() not in na_values
    ]
    if invalid_handedness:
        validation_warnings.append(
            f'handedness must be one of {sorted(valid_handedness)}, '
            f"found: {invalid_handedness[:5]}{'...' if len(invalid_handedness) > 5 else ''}",
        )

    return validation_warnings


def _validate_species(data: polars.DataFrame) -> list[str]:
    """Validate species column values per BIDS specification.

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

    if 'species' not in data.columns:
        return validation_warnings

    species_values = data['species'].drop_nulls().to_list()
    if not species_values:
        return validation_warnings

    return validation_warnings


def _validate_strain(data: polars.DataFrame) -> list[str]:
    """Validate strain column values per BIDS specification.

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

    if 'strain' not in data.columns:
        return validation_warnings

    strain_values = data['strain'].drop_nulls().to_list()
    if not strain_values:
        return validation_warnings

    return validation_warnings


def _validate_strain_rrid(data: polars.DataFrame) -> list[str]:
    """Validate strain_rrid column values per BIDS specification.

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

    if 'strain_rrid' not in data.columns:
        return validation_warnings

    strain_rrid_values = data['strain_rrid'].drop_nulls().to_list()
    strain_rrid_values = [rrid for rrid in strain_rrid_values if rrid]
    if not strain_rrid_values:
        return validation_warnings

    rrid_pattern = re.compile(r'^RRID:[a-zA-Z0-9_:\-]+$')
    invalid_rrids = [
        rrid for rrid in strain_rrid_values if not rrid_pattern.match(str(rrid))
    ]
    if invalid_rrids:
        validation_warnings.append(
            f"strain_rrid must match 'RRID:<identifier>' pattern. "
            f"Invalid values: {invalid_rrids[:5]}{'...' if len(invalid_rrids) > 5 else ''}",
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
            elif column == 'age':
                metadata[column]['Format'] = 'number'
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
    data = data.cast(schema_overrides)
    return data


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
        f'unknown bids format descriptor "{bids_format}". Known formats: {list(mapping.keys())}',
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

    raise TypeError(
        f'polars datatype {dtype} has no mapping to bids format descriptor. '
        f'Supported polars datatypes are: Integer, Float, String',
    )
