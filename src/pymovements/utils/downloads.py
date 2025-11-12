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
"""Utils module for downloading files.

.. deprecated:: v0.21.1
   Please use :py:meth:`~pymovements.Dataset.download()` instead.
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

from pathlib import Path

from deprecated.sphinx import deprecated

from pymovements.dataset._utils._archives import extract_archive as _extract_archive
from pymovements.dataset._utils._downloads import download_file as _download_file


@deprecated(
    reason='Please use Dataset.download() instead. '
           'This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def download_and_extract_archive(
        url: str,
        download_dirpath: Path,
        download_filename: str,
        extract_dirpath: Path | None = None,
        md5: str | None = None,
        recursive: bool = True,
        remove_finished: bool = False,
        remove_top_level: bool = True,
        verbose: int = 1,
) -> None:
    """Download and extract archive file.

    .. deprecated:: v0.21.1
       Please use :py:meth:`~pymovements.Dataset.download()` instead.
       This function will be removed in v0.26.0.

    Parameters
    ----------
    url: str
        URL of archive file to be downloaded.
    download_dirpath: Path
        Path to directory where file will be saved to.
    download_filename: str
        Target filename of saved file.
    extract_dirpath: Path | None
        Path to directory where archive files will be extracted to. (default: None)
    md5: str | None
        MD5 checksum of downloaded file. If None, do not check. (default: None)
    recursive: bool
        Recursively extract archives which are included in extracted archive. (default: True)
    remove_finished: bool
        Remove downloaded file after successful extraction or decompression. (default: False)
    remove_top_level: bool
        If ``True``, remove the top-level directory if it has only one child. (default: True)
    verbose: int
        Verbosity levels: (1) Show download progress bar and print info messages on downloading
        and extracting archive files without printing messages for recursive archive extraction.
        (2) Print additional messages for each recursive archive extract. (default: 1)

    Raises
    ------
    RuntimeError
        If the downloaded file has no suffix or suffix is not supported, or in case of a
        specified MD5 checksum which doesn't match the checksum of the downloaded file.
    """
    archive_path = _download_file(
        url=url,
        dirpath=download_dirpath,
        filename=download_filename,
        md5=md5,
        verbose=bool(verbose),
    )

    if extract_dirpath is None:
        extract_dirpath = download_dirpath

    _extract_archive(
        source_path=archive_path,
        destination_path=extract_dirpath,
        recursive=recursive,
        remove_finished=remove_finished,
        remove_top_level=remove_top_level,
        verbose=verbose,
    )


@deprecated(
    reason='Please use Dataset.download() instead. '
           'This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def download_file(
        url: str,
        dirpath: Path,
        filename: str,
        md5: str | None = None,
        max_redirect_hops: int = 3,
        verbose: bool = True,
) -> Path:
    """Download a file from a URL and place it in root.

    .. deprecated:: v0.21.1
       Please use :py:meth:`~pymovements.Dataset.download()` instead.
       This function will be removed in v0.26.0.

    Parameters
    ----------
    url : str
        URL of file to be downloaded.
    dirpath : Path
        Path to directory where file will be saved to.
    filename : str
        Target filename of saved file.
    md5 : str | None
        MD5 checksum of downloaded file. If None, do not check. (default: None)
    max_redirect_hops : int
        Maximum number of redirect hops allowed. (default: 3)
    verbose : bool
        If True, show progress bar and print info messages on downloading file. (default: True)

    Returns
    -------
    Path
        Filepath to downloaded file.

    Raises
    ------
    OSError
        If the download process failed.
    RuntimeError
        If the MD5 checksum of the downloaded file did not match the expected checksum.
    """
    return _download_file(
        url=url,
        dirpath=dirpath,
        filename=filename,
        md5=md5,
        max_redirect_hops=max_redirect_hops,
        verbose=verbose,
    )
