class Participants:
    """

    Parameters
    ----------
    data: polars.DataFrame
        the participant data conforming to BIDS (e.g., column participant_id required)
    metadata: dict[str, Any]
        metadata on participant data conforming to BIDS side car json files
    """

    data: polars.DataFrame
    metadata: dict[str, Any]

    @staticmethod
    def load(
            path: Path | str,  # if directory: append `/ 'participants.tsv'`,
            *,
            metadata: Path | str | dict[str, Any] | None = None,  # if path or string: load side car json, if None: check for 'participants.json'
            read_csv_kwargs = dict[str, Any] | None = None,  # if tsv file suffix auto-add separator
            rename_columns: dict[str, str] | None = None,  # necessary because BIDS requires participant_id column
    ) -> Participants:
    """Load participant data."""
        ...

    def save(
        self,
        path: Path | str,  # if directory: append`/ 'participants.tsv'`,
        *,
        write_csv_kwargs: dict[str, Any] | None = None,  # if None and tsv suffix: add separator
        metadata_path: Path | str | None = None,  # if None: 'participants.json',
    ) -> Path:
    """Save participant data."""
        ...
