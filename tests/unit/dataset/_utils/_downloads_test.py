# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test pymovements utils downloads."""

import hashlib
import os.path
from unittest import mock

import pytest

from pymovements.dataset._utils._downloads import _DownloadProgressBar
from pymovements.dataset._utils._downloads import _get_redirected_url
from pymovements.dataset._utils._downloads import download_file


@pytest.mark.network
@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_download_file(tmp_path, verbose):
    url = 'https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'

    filepath = download_file(url, tmp_path, filename, md5, verbose=verbose)

    assert filepath.exists()
    assert filepath.name == filename
    assert filepath.parent == tmp_path

    with open(filepath, 'rb') as f:
        file_bytes = f.read()
        assert hashlib.md5(file_bytes).hexdigest() == md5


@pytest.mark.network
def test_download_file_md5_None(tmp_path):
    url = 'https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    filepath = download_file(url, tmp_path, filename)

    assert filepath.exists()
    assert filepath.name == filename
    assert filepath.parent == tmp_path


def test_download_file_404(tmp_path):
    url = 'http://github.com/pymovements/pymovement/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'

    with pytest.raises(OSError):
        download_file(url, tmp_path, filename, md5)


@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_download_file_https_failure(tmp_path, verbose):
    url = 'https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'

    with (
        mock.patch(
            'pymovements.dataset._utils._downloads._download_url',
            side_effect=OSError(),
        ),
        pytest.raises(OSError),
    ):
        download_file(url, tmp_path, filename, md5, verbose=verbose)


def test_download_file_http_failure(tmp_path):
    url = 'http://example.com/'
    filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'

    with (
        mock.patch(
            'pymovements.dataset._utils._downloads._download_url',
            side_effect=OSError(),
        ),
        pytest.raises(OSError),
    ):
        download_file(url, tmp_path, filename, md5)


@pytest.mark.network
def test_download_file_with_invalid_md5(tmp_path):
    url = 'https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'
    md5 = '00000000000000000000000000000000'

    with pytest.raises(RuntimeError) as excinfo:
        download_file(url, tmp_path, filename, md5)

    (msg,) = excinfo.value.args
    assert (
        msg == f'File {os.path.join(tmp_path, "pymovements-0.4.0.tar.gz")} '
        'not found or download corrupted.'
    )


@pytest.mark.network
def test__get_redirected_url():
    url = 'https://codeload.github.com/pymovements/pymovements/tar.gz/refs/tags/v0.4.0'
    expected_url = 'https://codeload.github.com/pymovements/pymovements/tar.gz/refs/tags/v0.4.0'

    final_url = _get_redirected_url(url)

    assert final_url == expected_url


@pytest.mark.network
def test__get_redirected_url_with_redirects():
    url = 'https://github.com/pymovements/pymovements/archive/master.zip'
    expected_final_url = 'https://codeload.github.com/pymovements/pymovements/zip/main'

    final_url = _get_redirected_url(url)

    assert final_url == expected_final_url


@pytest.mark.network
def test__get_redirected_url_with_redirects_max_hops():
    url = 'https://github.com/pymovements/pymovements/archive/master.zip'

    with pytest.raises(RuntimeError) as excinfo:
        _get_redirected_url(url, max_hops=0)

    (msg,) = excinfo.value.args
    assert (
        msg == 'Request to '
        'https://github.com/pymovements/pymovements/archive/master.zip '
        'exceeded 0 redirects. The last redirect points to '
        'https://codeload.github.com/pymovements/pymovements/zip/main.'
    )


def test__DownloadProgressBar_tsize_not_None():
    download_progress_bar = _DownloadProgressBar()
    assert download_progress_bar.n == 0
    assert download_progress_bar.total is None
    download_progress_bar.update_to()
    assert download_progress_bar.n == 1
    assert download_progress_bar.total is None
    download_progress_bar.update_to(tsize=100)
    assert download_progress_bar.n == 1
    assert download_progress_bar.total == 100
