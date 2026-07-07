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

import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import KW_ONLY
from pathlib import Path
from typing import Any
from urllib.error import URLError
from warnings import warn

from tqdm.auto import tqdm

from pymovements._version import __version__
from pymovements.dataset.dataset_files import ChecksumError
from pymovements.dataset.dataset_files import DatasetFile

USER_AGENT: str = f'pymovements/{__version__}'


@dataclass(frozen=True)
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

    _: KW_ONLY

    filename: str | None = None
    md5: str | None = None
    mirrors: list[str] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WebSource:
        """Create a `WebSource` from a dictionary."""
        return WebSource(
            url=data.get('url'),  # type: ignore[arg-type]
            filename=data.get('filename'),
            md5=data.get('md5'),
            mirrors=data.get('mirrors'),
        )

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize this `WebSource` to a dictionary.

        Omits None values if `exclude_none` is True.
        """
        data = asdict(self)
        if exclude_none:
            return {key: value for key, value in data.items() if value}
        return data

    def download(
            self,
            target_dirpath: Path | str,
            *,
            verify_checksum: bool = True,
            verbose: bool = True,
    ) -> Path:
        """Download this resource into `target_dirpath`.

        Tries the primary `url` first, then any `mirrors` in order. Integrity is
        validated via MD5 when available. Returns the local file path.
        """
        dirpath = Path(target_dirpath).expanduser()

        if self.url is None:
            raise AttributeError('WebSource.url must not be None')
        if self.filename is None:
            raise AttributeError('WebSource.filename must not be None')

        # Attempt downloading from primary URL.
        try:
            return _download_file(
                url=self.url,
                dirpath=dirpath,
                filename=self.filename,
                md5=self.md5,
                verbose=verbose,
                verify_checksum=verify_checksum,
            )
        except (OSError, RuntimeError, ChecksumError) as primary_error:
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
                        filename=self.filename,
                        md5=self.md5,
                        verbose=verbose,
                        verify_checksum=verify_checksum,
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

            # If we are here, downloading failed for all mirrors.
            raise RuntimeError(
                f"Downloading resource {self.filename} failed for all mirrors.",
            ) from primary_error


def _download_file(
        url: str,
        dirpath: Path,
        filename: str,
        md5: str | None = None,
        *,
        verify_checksum: bool = True,
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
    verify_checksum : bool
        If True, check integrity by using the md5 checksum. (default: True)
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
    ChecksumError
        If the MD5 checksum of the downloaded file did not match the expected checksum.
    """
    dirpath = dirpath.expanduser()
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / filename

    if filepath.is_file():
        if verify_checksum and md5:
            if verbose:
                print('Verifying existing file:', filepath)
            try:
                DatasetFile(filepath).verify_checksum(md5)
            except ChecksumError as e:
                if verbose:
                    print('Local file failed checksum verification:')
                    print(f"expected '{e.expected}', got '{e.actual}'")
                    print('Re-downloading file.')
            else:
                if verbose:
                    print('Using existing verified file:', filepath)
                    return filepath
        else:
            if verbose:
                print('Using existing unverified file:', filepath)
                print('Please set verify_checksum to True if you require checksum verification.')
            return filepath

    if verbose:
        print(f'Downloading {url} to {filepath}')

    # expand redirect chain if needed
    url = _get_redirected_url(url=url, max_hops=max_redirect_hops)

    # download the file
    _download_url(url=url, destination=filepath, verbose=verbose)

    # check integrity of downloaded file
    if verify_checksum and md5:
        if verbose:
            print(f'Checking integrity of {filepath.name}')
        DatasetFile(filepath).verify_checksum(md5)

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
