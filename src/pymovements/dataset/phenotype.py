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
"""Phenotype module."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import polars

from pymovements._utils._html import repr_html
from pymovements.dataset._bids_dataset import _cast_columns_to_metadata_format
from pymovements.dataset._bids_dataset import _validate_participant_id_format
from pymovements.dataset._bids_dataset import _validate_participant_id_structure
from pymovements.dataset._bids_dataset import _verify_bids_handler
from pymovements.dataset.participants import _check_na_conformity
from pymovements.dataset.participants import _infer_metadata_column_format


@dataclass
@repr_html()
class Phenotype:
    """Phenotypic and assessment data conforming to BIDS specification.

    Attributes
    ----------
    data: polars.DataFrame
        The phenotypic data conforming to BIDS (i.e., first column must be named
        participant_id).
    metadata: dict[str, Any]
        Additional metadata on phenotypic data conforming to BIDS side car json files.
        Can include MeasurementToolMetadata at top level and Derivative field on columns.

    Parameters
    ----------
    data: polars.DataFrame | None
        The phenotypic data conforming to BIDS (i.e., first column must be named
        participant_id). If ``None``, initialize an empty DataFrame with a
        ``participant_id`` string column.
        (default: ``None``)
    metadata: dict[str, Any] | None
        Additional metadata on phenotypic data conforming to BIDS side car json files.
        If ``None``, initialize an empty dictionary.
        (default: ``None``)
    infer_metadata: bool
        Infer metadata column format descriptors from ``data``.
        (default: ``True``)
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool
        Verify BIDS conformity after initialization. If True, raise exception on
        non-conformity at REQUIRED level.
        If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that
        level.
        If False, do not verify.
        (default: ``False``)
    """

    data: polars.DataFrame
    metadata: dict[str, Any]

    def __init__(
        self,
        data: polars.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        infer_metadata: bool = True,
        verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    ):
        if data is None:
            data = polars.DataFrame(schema={'participant_id': polars.String})

        if verify_bids is not False:
            _validate_participant_id_structure(data)

        if metadata:
            metadata = deepcopy(metadata)
        else:
            metadata = {}
        if infer_metadata:
            metadata = _infer_metadata_column_format(data, metadata)
        data = _cast_columns_to_metadata_format(data, metadata)

        self.data = data
        self.metadata = metadata

        _verify_bids_handler(verify_bids, self.verify_bids, stacklevel=2)

    def verify_bids(
        self,
        level: Literal['REQUIRED', 'RECOMMENDED'] = 'REQUIRED',
    ) -> list[str]:
        """Verify BIDS conformity of phenotypic data.

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
        """
        warnings_list: list[str] = []

        if level in {'REQUIRED', 'RECOMMENDED'}:
            warnings_list.extend(_validate_participant_id_format(self.data))
            warnings_list.extend(_check_na_conformity(self.data))

        return warnings_list

    @staticmethod
    def load(
        path: Path | str,
        metadata: Path | str | dict[str, Any] | None = None,
        *,
        separator: str = '\t',
        rename: dict[str, str] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        metadata_encoding: str = 'utf-8',
        verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    ) -> Phenotype:
        r"""Load phenotypic data from phenotype files.

        Parameters
        ----------
        path: Path | str
            If this points to a directory, assume file with name matching the directory
            basename inside a ``phenotype`` subdirectory.
            If this points to a file, load from that file.
        metadata: Path | str | dict[str, Any] | None
            Additional metadata. Can be directly passed as a dictionary. If path or string:
            load metadata from json file. If a relative path given, path is assumed to be
            relative to parent of ``path``. If None: check for corresponding json file in the
            same directory and load metadata if available.
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
            Use this encoding for loading the metadata json file.
            (default: ``utf-8``)
        verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool
            Verify BIDS conformity after loading. If True, raise exception on
            non-conformity at REQUIRED level.
            If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that
            level.
            If False, do not verify.
            (default: ``False``)

        Returns
        -------
        Phenotype
            Phenotype initialised with data from loaded files.
        """
        path = Path(path)
        if path.is_dir():
            dir_path = path
            data_path = path / 'phenotype' / f"{path.name}.tsv"
            default_metadata_path = path / f"{path.name}.json"
        else:
            dir_path = path.parent
            data_path = path
            default_metadata_path = data_path.with_suffix('.json')

        if read_csv_kwargs is None:
            read_csv_kwargs = {'separator': separator}
        else:
            read_csv_kwargs = {'separator': separator, **read_csv_kwargs}

        data = polars.read_csv(data_path, **read_csv_kwargs)

        if rename:
            data = data.rename(rename)

        if metadata is None:
            if default_metadata_path.is_file():
                metadata = default_metadata_path

        if isinstance(metadata, (Path, str)):
            metadata_path = Path(metadata)
            if metadata_path.parent == Path('.'):
                metadata_path = dir_path / metadata_path

            with open(metadata_path, encoding=metadata_encoding) as opened_file:
                metadata_dict = json.load(opened_file)
        else:
            metadata_dict = metadata

        return Phenotype(data, metadata_dict, verify_bids=verify_bids)

    def save(
        self,
        path: Path | str,
        *,
        verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
        metadata_path: Path | str | None = None,
        separator: str = '\t',
        write_csv_kwargs: dict[str, Any] | None = None,
        metadata_encoding: str = 'utf-8',
    ) -> None:
        r"""Save phenotypic data including metadata.

        Parameters
        ----------
        path: Path | str
            Save phenotypic data to this path. If this is a directory, use the basename of
            the directory as filename stem and save as ``phenotype/<stem>.tsv``.
        verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool
            Verify BIDS conformity before saving. If True, raise exception on non-conformity
            at REQUIRED level.
            If 'REQUIRED' or 'RECOMMENDED', emit warnings for non-conformity at that level.
            If False, do not verify.
            (default: ``False``)
        metadata_path: Path | str | None
            Save metadata json to this path. If this is a relative path it is assumed to be
            relative to the directory specified by ``path``. If None: use stem from ``path``
            and save as ``<stem>.json`` in same directory as data file.
            (default: ``None``)
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
        _verify_bids_handler(verify_bids, self.verify_bids)

        path = Path(path)
        if path.is_dir():
            phenotype_dir = path / 'phenotype'
            phenotype_dir.mkdir(parents=True, exist_ok=True)
            data_path = phenotype_dir / f"{path.name}.tsv"
            default_metadata_path = path / f"{path.name}.json"
        else:
            data_path = path
            default_metadata_path = data_path.with_suffix('.json')

        if write_csv_kwargs is None:
            write_csv_kwargs = {'separator': separator}
        else:
            write_csv_kwargs = {'separator': separator, **write_csv_kwargs}

        # Ensure null values are encoded as n/a.
        data_to_save = self.data.fill_null('n/a')
        data_to_save.write_csv(data_path, **write_csv_kwargs)

        if metadata_path is None:
            metadata_path = default_metadata_path
        else:
            metadata_path = Path(metadata_path)
            if metadata_path.parent == Path('.'):
                metadata_path = data_path.parent / metadata_path

        with open(metadata_path, 'w', encoding=metadata_encoding) as opened_file:
            json.dump(self.metadata, opened_file)
