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
"""Utils module for downloading files."""
from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import URLError

from tqdm.auto import tqdm

from pymovements._version import get_versions

USER_AGENT: str = f"pymovements/{get_versions()['version']}"


def download_file(
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
    # Use a HEAD request to cheaply follow redirects and discover the final URL
    # without downloading the full content. Avoid installing a global opener to
    # prevent side effects in other parts of the application/tests.
    opener = _build_no_http_error_opener()

    for _ in range(max_hops + 1):
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        # backwards-compatible
        request.get_method = lambda: 'HEAD'  # type: ignore[assignment]

        try:
            response = opener.open(request)
        except URLError:  # pragma: no cover
            # Network failure – just return current URL and let caller decide.
            return url

        # Ensure the response is closed via context manager
        with response:
            code = getattr(response, 'status', None) or response.getcode()
            # Manually handle redirects to avoid HTTPError creation
            if code and 300 <= code < 400:
                loc = response.headers.get('Location') if hasattr(response, 'headers') else None
                if not loc:  # pragma: no cover
                    return url
                # Resolve relative redirects
                url = urllib.parse.urljoin(url, loc)
                continue
            # No redirect - return current URL
            return url

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
    # Preflight request to avoid urlretrieve creating an HTTPError with
    # unclosed temporary file objects on Python 3.14 (PytestUnraisableExceptionWarning).
    _raise_if_http_error(url)

    # Stream download with an opener that does NOT raise HTTPError automatically.
    # This prevents creation of HTTPError objects that can hold temp files and
    # lead to unraisable warnings on Python 3.14.
    opener = _build_no_http_error_opener()
    request = urllib.request.Request(
        url,
        headers={'User-Agent': USER_AGENT, 'Accept': 'application/octet-stream'},
    )
    with _DownloadProgressBar(desc=destination.name, disable=not verbose) as t:
        # Keep network operations inside a small try/except
        try:
            response = opener.open(request)
        except URLError as e:  # pragma: no cover
            raise OSError(str(e)) from e

        # Ensure the response is closed via context manager
        with response:
            status = getattr(response, 'status', None) or response.getcode()
            if status and status >= 400:  # pragma: no cover
                raise OSError(f'HTTP Error {status} for URL: {url}')

            content_length = response.headers.get(
                'Content-Length',
            ) if hasattr(response, 'headers') else None
            total = int(content_length) if content_length and content_length.isdigit() else None
            if total is not None:  # pragma: no cover
                t.total = total

            with open(destination, 'wb') as out:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)
                    t.update(len(chunk))

        # Ensure progress bar completes
        if t.total is None:  # pragma: no cover
            t.total = t.n


def _raise_if_http_error(url: str) -> None:  # pylint: disable=inconsistent-return-statements
    """Perform a lightweight HEAD request and raise OSError on HTTP errors.

    This avoids entering the problematic ``urlretrieve`` error path on Python 3.14
    where an ``HTTPError`` may keep a temporary file object alive, causing
    ``PytestUnraisableExceptionWarning`` during test teardown.
    """
    # Use a HEAD request with an Accept header that mimics a binary download to trigger the same
    # redirection behavior GitHub uses for archives (redirects to codeload).
    # This lets us detect a 4xx early.
    opener = _build_no_http_error_opener()
    request = urllib.request.Request(
        url,
        headers={'User-Agent': USER_AGENT, 'Accept': 'application/octet-stream'},
    )
    request.get_method = lambda: 'HEAD'  # type: ignore[assignment]

    try:
        response = opener.open(request)
    except URLError as e:  # pragma: no cover
        # Network or URL issue
        raise OSError(str(e)) from e

    # Ensure the response is closed via context manager
    with response:
        code = getattr(response, 'status', None) or response.getcode()
        if code and 300 <= code < 400:
            # Follow one redirect here by recursing
            loc = response.headers.get('Location') if hasattr(response, 'headers') else None
            if loc:  # pragma: no cover
                return _raise_if_http_error(urllib.parse.urljoin(url, loc))
        if code and code >= 400:
            raise OSError(f'HTTP Error {code} for URL: {url}')


def _build_no_http_error_opener() -> urllib.request.OpenerDirector:
    """Build an opener that does not turn HTTP status >=400 into HTTPError.

    By omitting the HTTPErrorProcessor handler we ensure that responses with
    error status codes are returned as regular responses. This lets us manage
    them deterministically without creating HTTPError objects that may retain
    temporary file handles on Python 3.14.
    """
    opener = urllib.request.OpenerDirector()
    # Keep standard handlers except HTTPErrorProcessor
    opener.add_handler(urllib.request.ProxyHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    # Do not add HTTPRedirectHandler to avoid internal HTTPError use
    return opener


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
