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
            schema: dict[str, polars.DataType] | None = None,
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

        if schema is None:
            schema = {}
        else:
            schema = deepcopy(schema)
        if 'participant_id' not in schema and 'participant_id' in data.columns:
            schema = {'participant_id': polars.String}
        data = data.cast(schema)

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