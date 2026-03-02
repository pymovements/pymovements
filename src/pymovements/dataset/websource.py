# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""WebSource definition and download helper."""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from warnings import warn

from tqdm.auto import tqdm

from pymovements._version import __version__

USER_AGENT: str = f"pymovements/{__version__}"


def _download_file(
        url: str,
        dirpath: Path,
        filename: str,
        md5: str | None = None,
        *,
        max_redirect_hops: int = 3,
        verbose: bool = True,
) -> Path:
    """Download a file from a URL and place it in root.

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
        If True, show a progress bar and print info messages on the downloading file.
        (default: True)

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
    dirpath = dirpath.expanduser()
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / filename

    # check if file is already present locally
    if _check_integrity(filepath, md5):
        if verbose:
            print('Using already downloaded and verified file:', filepath)
        return filepath

    if verbose:
        print(f'Downloading {url} to {filepath}')

    # expand redirect chain if needed
    url = _get_redirected_url(url=url, max_hops=max_redirect_hops)

    # download the file
    try:
        _download_url(url=url, destination=filepath, verbose=verbose)

    except OSError as e:
        if url[:5] == 'https':
            print('Download failed. Trying https -> http instead.')
            url = url.replace('https:', 'http:')

            if verbose:
                print(f'Downloading {url} to {filepath}')
            _download_url(url=url, destination=filepath, verbose=verbose)
        else:
            raise e

    # check integrity of downloaded file
    if verbose:
        print(f'Checking integrity of {filepath.name}')
    if not _check_integrity(filepath=filepath, md5=md5):
        raise RuntimeError(f'File {filepath} not found or download corrupted.')

    return filepath


def _get_redirected_url(url: str, max_hops: int = 3) -> str:
    """Get redirected URL.

    Parameters
    ----------
    url: str
        Initial URL to be requested for redirection.
    max_hops: int
        Maximum number of redirection hops. (default: 3)

    Returns
    -------
    str
        Final URL after all redirections.

    Raises
    ------
    RuntimeError
        If number of redirects exceed `max_hops`.
    """
    initial_url = url
    headers = {'Method': 'HEAD', 'User-Agent': USER_AGENT}
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', USER_AGENT)]
    urllib.request.install_opener(opener)

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    raise RuntimeError(
        f'Request to {initial_url} exceeded {max_hops} redirects.'
        f' The last redirect points to {url}.',
    )


class _DownloadProgressBar(tqdm):
    """Progress bar for downloads.

    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Reference: https://github.com/tqdm/tqdm#hooks-and-callbacks

    Parameters
    ----------
    **kwargs : Any
    """

    def __init__(self, **kwargs: Any):
        super().__init__(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, **kwargs)

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> bool | None:
        """Update progress bar.

        Parameters
        ----------
        b  : int
            Number of blocks transferred so far (default: 1).
        bsize  : int
            Size of each block (in tqdm units) (default: 1).
        tsize  : int | None
            Total size (in tqdm units). If None it remains unchanged (default: None).

        Returns
        -------
        bool | None
            Returns `None` if the update was successful, `False` if the update was skipped
            because the last update was too recent.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def _download_url(url: str, destination: Path, verbose: bool = True) -> None:
    """Download file from URL and save to destination.

    Parameters
    ----------
    url : str
        URL of file to be downloaded.
    destination : Path
        Destination path of downloaded file.
    verbose : bool
        If True, show progressbar.
    """
    with _DownloadProgressBar(desc=destination.name, disable=not verbose) as t:
        urllib.request.urlretrieve(url=url, filename=destination, reporthook=t.update_to)
        t.total = t.n


def _check_integrity(filepath: Path, md5: str | None = None) -> bool:
    """Check file integrity by MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to file.
    md5: str | None
        Expected MD5 checksum of file. If None, do not check. (default: None)

    Returns
    -------
    bool
        True if file checksum matches passed `md5` or if passed `md5` is None. False if file
        checksum does not match passed `md5` or `filepath` doesn't exist.
    """
    if not filepath.is_file():
        return False
    if md5 is None:
        return True

    # Calculate checksum and check for match.
    file_md5 = _calculate_md5(filepath)
    return file_md5 == md5


def _calculate_md5(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to file.
    chunk_size : int
        Byte size of processed chunks. (default: 1024 * 1024)

    Returns
    -------
    str
        Calculated MD5 checksum.
    """
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but
    # indicates that we are not using the MD5 checksum for cryptography.
    # This enables its usage in restricted environments like FIPS without raising an error.
    file_md5 = hashlib.new('md5', usedforsecurity=False)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            file_md5.update(chunk)
    return file_md5.hexdigest()


@dataclass
class WebSource:
    """Web-based source of a dataset resource.

    Attributes
    ----------
    url: str
        Primary URL of the resource to be downloaded.
    filename: str | None
        Optional target filename. If not provided, the basename of the URL path is used.
    md5: str | None
        Optional MD5 checksum for integrity verification.
    mirrors: list[str] | None
        Optional list of full mirror URLs. Tried in order if primary URL fails.
    """

    url: str
    filename: str | None = None
    md5: str | None = None
    mirrors: list[str] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WebSource:
        """Create a `WebSource` from a dictionary."""
        # Accept both with and without explicit keys; default to None when missing
        return WebSource(
            url=data.get('url'),
            filename=data.get('filename'),
            md5=data.get('md5'),
            mirrors=data.get('mirrors'),
        )

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize this `WebSource` to a dictionary.

        Omits None values if `exclude_none` is True.
        """
        payload: dict[str, Any] = asdict(self)
        if exclude_none:
            return {k: v for k, v in payload.items() if v is not None}
        return payload

    def download(
        self, target_dir: Path | str, *, exist_ok: bool = True,
        verbose: bool = True,
    ) -> Path:
        """Download this resource into `target_dir`.

        Tries the primary `url` first, then any `mirrors` in order. Integrity is
        validated via MD5 when provided. Returns the local file path.
        """
        dirpath = Path(target_dir).expanduser()
        # `_download_file` will ensure directory exists; we still optionally create it here
        if exist_ok:
            dirpath.mkdir(parents=True, exist_ok=True)

        # Determine filename; fallback to URL basename when not explicitly set
        filename = self.filename
        if filename is None:
            parsed = urlparse(self.url)
            candidate = Path(parsed.path).name
            if not candidate:
                raise ValueError('Unable to infer filename from URL; please provide `filename`.')
            filename = candidate

        # Attempt primary URL
        try:
            return _download_file(
                url=self.url, dirpath=dirpath,
                filename=filename, md5=self.md5, verbose=verbose,
            )
        # pylint: disable=overlapping-except
        except (URLError, OSError, RuntimeError) as primary_error:
            # No mirrors to try
            if not self.mirrors:
                raise RuntimeError(f"Downloading resource {self.url} failed.") from primary_error

            warn(UserWarning(f"Downloading resource {self.url} failed. Trying mirror."))

            # Try mirrors in order
            for mirror_idx, mirror_url in enumerate(self.mirrors, start=1):
                try:
                    return _download_file(
                        url=mirror_url,
                        dirpath=dirpath,
                        filename=filename,
                        md5=self.md5,
                        verbose=verbose,
                    )
                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as mirror_error:
                    msg = f"Downloading resource from mirror {mirror_url} failed."
                    if mirror_idx < len(self.mirrors):
                        msg = msg + \
                            f" Trying next mirror ({len(self.mirrors) - mirror_idx} remaining)."
                    warning = UserWarning(msg)
                    warning.__cause__ = mirror_error
                    warn(warning)
                    # Continue to next mirror

            # If we are here, all mirrors failed
            raise RuntimeError(
                f"Downloading resource {filename} failed for all mirrors.",
            ) from primary_error
