# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Utils module for extracting archives and decompressing files."""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

from tqdm import tqdm

from pymovements._utils._paths import get_filepaths
from pymovements.exceptions import UnknownFileType


def extract_archive(
        source_path: Path,
        destination_path: Path | None = None,
        *,
        recursive: bool = True,
        remove_finished: bool = False,
        remove_top_level: bool = True,
        resume: bool = True,
        verbose: int = 1,
) -> Path:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name.
    If the file is compressed but not an archive the call is dispatched to :func:`_decompress`.

    Parameters
    ----------
    source_path: Path
        Path to the file to be extracted.
    destination_path: Path | None
        Path to the directory the file will be extracted to. If omitted, the directory of the file
        is used. (default: None)
    recursive: bool
        Recursively extract archives which are included in extracted archive. (default: True)
    remove_finished: bool
        If ``True``, remove the file after the extraction. (default: False)
    remove_top_level: bool
        If ``True``, remove the top-level directory if it has only one child. (default: True)
    resume: bool
        Resume previous extraction by skipping existing files.
        Checks for correct size of existing files but not integrity. (default: True)
    verbose: int
        Verbosity levels: (1) Print messages for extracting each dataset resource without printing
        messages for recursive archives. (2) Print additional messages for each recursive archive
        extract. (default: 1)

    Returns
    -------
    Path
        Path to the directory the file was extracted to.
    """
    archive_type, compression_type = _detect_file_type(source_path)

    if not archive_type:
        return _decompress(
            source_path=source_path,
            destination_path=destination_path,
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key, since this was already done in__detect_file_type().
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    if destination_path is None:
        destination_path = source_path.parent

    if verbose:
        print(f'Extracting {source_path.name} to {destination_path}')

    # Extract file and remove archive if desired.
    extractor(
        source_path,
        destination_path,
        compression_type,
        resume=resume,
        verbose=verbose,
    )
    if remove_finished:
        source_path.unlink()

    if remove_top_level:
        # path of extracted archive
        extract_destination = destination_path / source_path.stem
        if destination_path.stem == source_path.stem:
            extract_destination = destination_path

        # check if extracted archive has single child that is a directory
        children = [str(file) for file in extract_destination.glob('*')]
        if len(children) == 1 and os.path.isdir(children[0]):
            # move contents of single child and remove it
            for f in [str(file) for file in Path(children[0]).glob('*')]:
                shutil.move(f, extract_destination)
            Path(children[0]).rmdir()

    if recursive:
        # Get filepaths of all archives in extracted directory.
        archive_extensions = [
            *_ARCHIVE_EXTRACTORS.keys(),
            *_ARCHIVE_TYPE_ALIASES.keys(),
            *_COMPRESSED_FILE_OPENERS.keys(),
        ]
        archive_filepaths = get_filepaths(path=destination_path, extension=archive_extensions)
        archive_filepaths = [filepath for filepath in archive_filepaths if filepath != source_path]

        # Extract all found archives.
        for archive_filepath in archive_filepaths:
            extract_destination = archive_filepath.parent / archive_filepath.stem

            extract_archive(
                source_path=archive_filepath,
                destination_path=extract_destination,
                recursive=recursive,
                remove_finished=remove_finished,
                remove_top_level=remove_top_level,
                verbose=0 if verbose < 2 else 2,
                resume=resume,
            )

    return destination_path


def _extract_tar(
        source_path: Path,
        destination_path: Path,
        compression: str | None,
        *,
        resume: bool,
        verbose: int,
) -> None:
    """Extract a tar archive.

    Parameters
    ----------
    source_path: Path
        Path to the file to be extracted.
    destination_path: Path
        Path to the directory the file will be extracted to.
    compression: str | None
        Compression filename suffix.
    resume: bool
        Resume if archive was already previous extracted.
    verbose: int
        If ``True``, show a progress bar and print messages for resuming each dataset resource.
    """
    mode = f'r:{compression[1:]}' if compression else 'r'

    # ignore mypy error for now, see issue #1020
    with tarfile.open(source_path, mode) as archive:  # type: ignore[call-overload]
        members = archive.getmembers()
        for member in tqdm(
                members,
                total=len(members),
                desc='Extracting archive',
                unit='file',
                ncols=80,
                disable=not verbose,
        ):
            if resume:
                member_dest_path = os.path.join(destination_path, member.name)
                if (
                        os.path.exists(member_dest_path) and
                        member.name[-4:] in _ARCHIVE_EXTRACTORS and
                        member.size == os.path.getsize(member_dest_path)
                ):
                    if verbose:
                        print(f'Skipping {member.name} due to previous extraction')
                    continue
            if sys.version_info < (3, 12):  # pragma: <3.12 cover
                archive.extract(member.name, destination_path)
            else:  # pragma: >=3.12 cover
                archive.extract(member.name, destination_path, filter='tar')


def _extract_zip(
        source_path: Path,
        destination_path: Path,
        compression: str | None,
        *,
        resume: bool,
        verbose: int,
) -> None:
    """Extract a zip archive.

    Parameters
    ----------
    source_path: Path
        Path to the file to be extracted.
    destination_path: Path
        Path to the directory the file will be extracted to.
    compression: str | None
        Compression filename suffix.
    resume: bool
        Resume if archive was already previous extracted.
    verbose: int
        If ``True``, show a progress bar and print messages for resuming each dataset resource.
    """
    compression_id = _ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    with zipfile.ZipFile(source_path, 'r', compression=compression_id) as archive:
        for member in tqdm(
                archive.filelist,
                total=len(archive.filelist),
                desc='Extracting archive',
                unit='file',
                disable=not verbose,
        ):
            if resume:
                member_dest_path = os.path.join(destination_path, member.filename)
                if (
                    os.path.exists(member_dest_path) and
                    member.filename[-4:] in _ARCHIVE_EXTRACTORS and
                    member.file_size == os.path.getsize(member_dest_path)
                ):
                    if verbose:
                        print(f'Skipping {member.filename} due to previous extraction')
                    continue
            archive.extract(member.filename, destination_path)


_ARCHIVE_EXTRACTORS = {
    '.tar': _extract_tar,
    '.zip': _extract_zip,
}

_ARCHIVE_TYPE_ALIASES = {
    '.tbz': ('.tar', '.bz2'),
    '.tbz2': ('.tar', '.bz2'),
    '.tgz': ('.tar', '.gz'),
}

_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = {
    '.bz2': bz2.open,
    '.gz': gzip.open,
    '.xz': lzma.open,
}

_ZIP_COMPRESSION_MAP = {
    '.bz2': zipfile.ZIP_BZIP2,
    '.xz': zipfile.ZIP_LZMA,
}


def _detect_file_type(filepath: Path) -> tuple[str | None, str | None]:
    """Detect the archive type and/or compression of a file.

    Parameters
    ----------
    filepath: Path
        The path of the file.

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of archive type, and compression type.

    Raises
    ------
    UnknownFileType
        If the file has no suffix or the suffix is not supported.
    """
    if not (suffixes := filepath.suffixes):
        raise UnknownFileType(
            f"File '{filepath}' has no suffixes that could be used to detect the archive type or"
            ' compression.',
        )

    # Get last suffix only.
    suffix = suffixes[-1]

    # Check if suffix is a known alias.
    if suffix in _ARCHIVE_TYPE_ALIASES:
        return _ARCHIVE_TYPE_ALIASES[suffix]

    # Check if suffix refers to an archive type.
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, None

    # Check if the suffix refers to a compression type.
    if suffix in _COMPRESSED_FILE_OPENERS:
        # Check if there are more than one suffix.
        if len(suffixes) > 1:

            # Check if the second last suffix refers to an archive type.
            if (suffix2 := suffixes[-2]) not in _ARCHIVE_EXTRACTORS:
                raise UnknownFileType(
                    f"Unsupported compression or archive type: '{suffix2}{suffix}'.\n"
                    f"Supported suffixes are: '{_get_valid_suffixes()}'.",
                )
            # We detected a compressed archive file (e.g. tar.gz).
            return suffix2, suffix

        # We detected a single compressed file not an archive.
        return None, suffix

    # Raise error as we didn't find a valid suffix.
    raise UnknownFileType(
        f"Unsupported compression or archive type: '{suffix}'.\n"
        f"Supported suffixes are: '{_get_valid_suffixes()}'.",
    )


def _decompress(
        source_path: Path,
        destination_path: Path | None = None,
        remove_finished: bool = False,
) -> Path:
    """Decompress a file.

    The compression is automatically detected from the file name.

    Parameters
    ----------
    source_path: Path
        Path to the file to be decompressed.
    destination_path: Path | None
        Path to the decompressed file. (default: None)
    remove_finished: bool
        If ``True``, remove the source file after decompression. (default: False)

    Returns
    -------
    Path
        Path to the decompressed file.

    Raises
    ------
    RuntimeError
        If the file has no suffix or the suffix is not supported for decompression.
    """
    _, compression = _detect_file_type(source_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {source_path.suffix}.")

    if destination_path is None:
        destination_path = source_path.parent / source_path.stem

    # We don't need to check for a missing key, since this was already done in _detect_file_type().
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    # Decompress by reading from compressed file and writing to destination.
    with compressed_file_opener(source_path, 'rb') as rfh, open(destination_path, 'wb') as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        source_path.unlink()

    return destination_path


def _get_valid_suffixes() -> list[str]:
    """Get valid archive file extensions."""
    return sorted(
        set(_ARCHIVE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS),
    )
