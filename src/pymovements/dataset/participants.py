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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars


@dataclass
class Participants:
    """Participant table with additional metdadata.

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

    @staticmethod
    def load(
            path: Path | str,
            metadata: Path | str | dict[str, Any] | None = None,
            *,
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

        return Participants(data, metadata_dict)

    def save(
        self,
        path: Path | str,
        *,
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
    if dtype == polars.String:
        return 'string'

    raise TypeError(
        f'polars datatype {dtype} has no mapping to bids format descriptor. '
        f'Supported polars datatypes are: Integer, Float, String',
    )
