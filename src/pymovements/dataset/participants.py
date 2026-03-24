from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
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
    metadata: dict[str, Any]
        Additional metadata on participant data conforming to BIDS side car json files.
        (default: ``{}``)
    """

    data: polars.DataFrame
    metadata: dict[str, Any]

    def __init__(
            self,
            data: polars.DataFrame,
            metadata: dict[str, Any] | None = None,
    ):
        if 'participant_id' not in data.columns:
            raise ValueError("data must have column named 'participant_id'")
        if data.columns[0] != 'participant_id' :
            raise ValueError("first column in data must be named 'participant_id'")

        metadata = _infer_metadata_column_format(metadata, data)
        data = _cast_columns_to_metadata_format(data, metadata)

        self.data = data
        self.metadata = metadata

    @staticmethod
    def load(
            path: Path | str,
            #*,
            #metadata: Path | str | dict[str, Any] | None = None,  # if path or string: load side car json, if None: check for 'participants.json'
            separator: str = '\t',
            rename: dict[str, str] | None = None,
            read_csv_kwargs: dict[str, Any] | None = None,
    ) -> 'Participants':
        """Load participant data from participant files.

        Parameters
        ----------
        path: Path | str
            If this points to a directory, assume file to be named `participants.tsv` inside that
            directory.
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

        Returns
        -------
        Participants
            Participants initialized with data from loaded files.
        """
        path = Path(path)
        if path.is_dir():
            path = path / 'participants.tsv'

        if read_csv_kwargs is None:
            read_csv_kwargs = {'separator': separator}
        else:
            # **read_csv_kwargs takes precedence over explicit separator argument.
            read_csv_kwargs = {'separator': separator, **read_csv_kwargs}

        data = polars.read_csv(
            path,
            **read_csv_kwargs
        )

        if rename:
            data = data.rename(rename)

        return Participants(data)

    '''
    def save(
        self,
        path: Path | str,  # if directory: append`/ 'participants.tsv'`,
        *,
        write_csv_kwargs: dict[str, Any] | None = None,  # if None and tsv suffix: add separator
        metadata_path: Path | str | None = None,  # if None: 'participants.json',
        verify_bids: bool = True,
    ) -> Path:
    """Save participant data."""
        ...
    '''


def _infer_metadata_column_format(metadata, data):
    if metadata:
        # metadata may be changed and updated, work on copy
        metadata = deepcopy(metadata)
    else:
        metadata = {}

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


def _cast_columns_to_metadata_format(data, metadata):
    schema_overrides = {}
    for column in data.columns:
        format = metadata.get(column, {}).get('Format', None)
        if format:
            schema_overrides[column] = _bids_format_to_polars_datatype(format)
    data = data.cast(schema_overrides)
    return data


def _bids_format_to_polars_datatype(format: str) -> polars.DataType:
    mapping = {
        'string': polars.String,
        'number': polars.Float64,
        'integer': polars.Int64,
        'bool': polars.Boolean,
        'index': polars.UInt64,
        'label': polars.String,
    }

    if format in mapping:
        return mapping[format]

    raise ValueError(
        f"unknown bids format descriptor '{format}'. Known formats: {list(mapping.keys())}",
    )


def _polars_datatype_to_bids_format(dtype: polars.DataType) -> str:
    mapping = {
        polars.String: 'string',
        polars.Float64: 'number',
        polars.Int64: 'integer',
        polars.Boolean: 'bool',
        polars.UInt64: 'index',
    }

    if dtype in mapping:
        return mapping[dtype]

    raise ValueError(
        f"polars datatype '{dtype}' has no mapping to bids format descriptor. "
        f'Supported polars datatypes are: {list(mapping.keys())}',
    )
