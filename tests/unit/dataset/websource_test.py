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
"""Tests for WebSource and download utilities."""
import hashlib
import io
from pathlib import Path
from unittest import mock
from unittest.mock import patch
from urllib.error import ContentTooShortError
from urllib.error import HTTPError

import pytest

from pymovements._version import __version__
from pymovements.dataset.websource import ChecksumError
from pymovements.dataset.websource import WebSource


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the response object returned by ``urllib.request.urlopen``."""

    def __init__(self, content: bytes, headers: dict[str, str] | None = None):
        super().__init__(content)
        self.headers = headers if headers is not None else {}


def raises_download_failed(source):
    """Return the ``pytest.raises`` context for a failed ``source.download()`` call."""
    return pytest.raises(RuntimeError, match=f'Downloading resource {source.url} failed')


def test_websource_init():
    source = WebSource(url='http://example.com/file.zip', filename='file.zip', md5='123')
    assert source.url == 'http://example.com/file.zip'
    assert source.filename == 'file.zip'
    assert source.md5 == '123'
    assert source.mirrors is None


def test_websource_from_dict():
    data = {'url': 'http://example.com/file2.zip', 'filename': 'file2.zip', 'md5': '456'}
    source = WebSource.from_dict(data)
    assert source.url == 'http://example.com/file2.zip'
    assert source.filename == 'file2.zip'
    assert source.md5 == '456'


@pytest.mark.parametrize(
    ('source', 'exclude_none', 'expected_dict'),
    [
        pytest.param(
            WebSource(url='http://example.com/file.zip'),
            False,
            {'url': 'http://example.com/file.zip', 'filename': None, 'md5': None, 'mirrors': None},
            id='url_only',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip'),
            True,
            {'url': 'http://example.com/file.zip'},
            id='url_only_exclude_none',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', filename='test.zip'),
            False,
            {
                'url': 'http://example.com/file.zip',
                'filename': 'test.zip',
                'md5': None,
                'mirrors': None,
            },
            id='url_and_filename',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', filename='test.zip'),
            True,
            {'url': 'http://example.com/file.zip', 'filename': 'test.zip'},
            id='url_and_filename_exclude_none',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', md5='abc'),
            False,
            {
                'url': 'http://example.com/file.zip',
                'filename': None,
                'md5': 'abc',
                'mirrors': None,
            },
            id='url_and_md5',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', md5='abc'),
            True,
            {'url': 'http://example.com/file.zip', 'md5': 'abc'},
            id='url_and_md5_exclude_none',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', mirrors=['http://example2.com/file.zip']),
            False,
            {
                'url': 'http://example.com/file.zip',
                'filename': None,
                'md5': None,
                'mirrors': ['http://example2.com/file.zip'],
            },
            id='url_and_mirrors',
        ),
        pytest.param(
            WebSource(url='http://example.com/file.zip', mirrors=['http://example2.com/file.zip']),
            True,
            {'url': 'http://example.com/file.zip', 'mirrors': ['http://example2.com/file.zip']},
            id='url_and_mirrors_exclude_none',
        ),
        pytest.param(
            WebSource(
                url='http://example.com/file.zip',
                filename='file.zip',
                md5='qwer',
                mirrors=['http://example3.com/file.zip'],
            ),
            False,
            {
                'url': 'http://example.com/file.zip',
                'filename': 'file.zip',
                'md5': 'qwer',
                'mirrors': ['http://example3.com/file.zip'],
            },
            id='complete',
        ),
        pytest.param(
            WebSource(
                url='http://example.com/file.zip',
                filename='file.zip',
                md5='qwer',
                mirrors=['http://example3.com/file.zip'],
            ),
            True,
            {
                'url': 'http://example.com/file.zip',
                'filename': 'file.zip',
                'md5': 'qwer',
                'mirrors': ['http://example3.com/file.zip'],
            },
            id='complete_exclude_none',
        ),
    ],
)
def test_websource_to_dict(source, exclude_none, expected_dict):
    data = source.to_dict(exclude_none=exclude_none)
    assert data == expected_dict, source


def test_websource_download_with_mirrors():
    source = WebSource(
        url='http://primary.com/file.zip',
        filename='file.zip',
        mirrors=['http://mirror1.com/file.zip', 'http://mirror2.com/file.zip'],
    )
    with patch('pymovements.dataset.websource._download_file') as mock_download:
        # Fail primary, fail mirror 1, succeed mirror 2
        mock_download.side_effect = [
            RuntimeError('fail'),
            RuntimeError('fail'),
            Path('tmp/file.zip'),
        ]

        with pytest.warns(UserWarning):
            path = source.download('tmp')
        assert path == Path('tmp/file.zip')
        assert mock_download.call_count == 3


@pytest.mark.network
@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_websource_download(tmp_path, verbose):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )
    filepath = source.download(tmp_path, verbose=verbose)

    assert filepath.exists()
    assert filepath.name == source.filename
    assert filepath.parent == tmp_path

    with open(filepath, 'rb') as f:
        file_bytes = f.read()
        assert hashlib.md5(file_bytes).hexdigest() == source.md5


@pytest.mark.network
@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
@pytest.mark.parametrize(
    'verify_checksum',
    [
        pytest.param(False, id='verify_checksum_false'),
        pytest.param(True, id='verify_checksum_true'),
    ],
)
def test_websource_download_existing_file_verified(tmp_path, verbose, verify_checksum):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )
    filepath = source.download(tmp_path, verify_checksum=verify_checksum, verbose=verbose)

    assert filepath.exists()
    last_mtime = filepath.stat().st_mtime

    new_filepath = source.download(tmp_path, verify_checksum=verify_checksum, verbose=verbose)

    new_mtime = filepath.stat().st_mtime

    assert new_filepath.exists()
    assert new_filepath == filepath
    assert new_mtime == last_mtime


@pytest.mark.network
@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_websource_download_existing_file_checksum_fail(make_text_file, verbose):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    filepath = make_text_file(source.filename)

    assert filepath.exists()
    last_mtime = filepath.stat().st_mtime

    new_filepath = source.download(filepath.parent, verify_checksum=True, verbose=verbose)

    new_mtime = filepath.stat().st_mtime

    assert new_filepath.exists()
    assert new_filepath == filepath
    assert new_mtime > last_mtime


@pytest.mark.network
def test_websource_download_md5_None(tmp_path):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
    )
    filepath = source.download(tmp_path)

    assert filepath.exists()
    assert filepath.name == source.filename
    assert filepath.parent == tmp_path


def test_websource_download_404(tmp_path):
    source = WebSource(
        url='http://github.com/pymovements/pymovement/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    with raises_download_failed(source):
        source.download(tmp_path)


@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_websource_download_os_error(verbose, tmp_path):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    with mock.patch('urllib.request.urlopen', side_effect=OSError()):
        with raises_download_failed(source):
            source.download(tmp_path, verbose=verbose)


def test_websource_download_http_failure(tmp_path):
    source = WebSource(
        url='http://example.com/',
        filename='pymovements-0.4.0.tar.gz',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    with mock.patch('urllib.request.urlopen', side_effect=OSError()):
        with raises_download_failed(source):
            source.download(tmp_path)


@pytest.mark.network
def test_websource_download_file_not_found(tmp_path):
    source = WebSource(
        url='https://github.com/pymovements/pymovements/archive/refs/tags/v0.4.0.tar.gz',
        filename='pymovements-0.4.0.tar.gz',
        md5='00000000000000000000000000000000',
    )

    with pytest.raises(RuntimeError) as excinfo:
        source.download(tmp_path)

    message = (
        f"MD5 checksum mismatch for file '{tmp_path / source.filename}'"
        f": expected '{source.md5}', got '52bbf03a7c50ee7152ccb9d357c2bb30'"
    )
    assert isinstance(excinfo.value.__cause__, ChecksumError)
    assert str(excinfo.value.__cause__) == message


@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
@pytest.mark.parametrize(
    'headers',
    [
        pytest.param({}, id='without_content_length'),
        pytest.param({'Content-Length': '12'}, id='with_content_length'),
    ],
)
def test_websource_download_single_request_streams_to_file(tmp_path, verbose, headers):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    content = b'file content'

    with mock.patch(
        'urllib.request.urlopen', return_value=FakeResponse(content, headers),
    ) as mock_urlopen:
        filepath = source.download(tmp_path, verbose=verbose)

    assert filepath == tmp_path / 'file.zip'
    assert filepath.read_bytes() == content
    assert sorted(path.name for path in tmp_path.iterdir()) == ['file.zip']

    mock_urlopen.assert_called_once()
    request, = mock_urlopen.call_args.args
    assert request.full_url == source.url
    assert request.get_header('User-agent') == f'pymovements/{__version__}'


def test_websource_download_replaces_stale_part_file(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    content = b'file content'
    (tmp_path / 'file.zip.part').write_bytes(b'leftover from an interrupted download')

    with mock.patch('urllib.request.urlopen', return_value=FakeResponse(content)):
        filepath = source.download(tmp_path, verbose=False)

    assert filepath.read_bytes() == content
    assert sorted(path.name for path in tmp_path.iterdir()) == ['file.zip']


def test_websource_download_http_error_closes_response(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    http_error = HTTPError(
        url=source.url, code=404, msg='Not Found', hdrs=None, fp=io.BytesIO(b''),
    )

    with mock.patch.object(http_error, 'close', wraps=http_error.close) as mock_close:
        with mock.patch('urllib.request.urlopen', side_effect=http_error):
            with raises_download_failed(source):
                source.download(tmp_path, verbose=False)

    mock_close.assert_called_once()
    assert not list(tmp_path.iterdir())


def test_websource_download_interrupted_transfer_leaves_no_file(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    response = FakeResponse(b'first chunk')
    response.read = mock.Mock(side_effect=[b'first chunk', ConnectionResetError()])

    with mock.patch('urllib.request.urlopen', return_value=response):
        with raises_download_failed(source):
            source.download(tmp_path, verbose=False)

    assert not list(tmp_path.iterdir())


def test_websource_download_short_transfer_leaves_no_file(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    response = FakeResponse(b'short', headers={'Content-Length': '100'})

    with mock.patch('urllib.request.urlopen', return_value=response):
        with raises_download_failed(source) as exc:
            source.download(tmp_path, verbose=False)

    assert isinstance(exc.value.__cause__, ContentTooShortError)
    assert exc.value.__cause__.reason == 'retrieval incomplete: got only 5 out of 100 bytes'
    assert not list(tmp_path.iterdir())


def test_websource_download_failed_rename_leaves_no_file(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    response = FakeResponse(b'complete content')

    with mock.patch('urllib.request.urlopen', return_value=response):
        with mock.patch('os.replace', side_effect=PermissionError('file.zip is locked')):
            with raises_download_failed(source):
                source.download(tmp_path, verbose=False)

    assert not list(tmp_path.iterdir())


def test_websource_download_invalid_content_length_closes_response(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    response = FakeResponse(b'complete content', headers={'Content-Length': 'many'})

    with mock.patch('urllib.request.urlopen', return_value=response):
        with pytest.raises(ValueError, match="'many'"):
            source.download(tmp_path, verbose=False)

    assert response.closed
    assert not list(tmp_path.iterdir())


def test_websource_download_interrupted_transfer_without_md5_redownloads(tmp_path):
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    broken_response = FakeResponse(b'')
    broken_response.read = mock.Mock(side_effect=[b'partial', ConnectionResetError()])
    responses = [broken_response, FakeResponse(b'complete content')]

    with mock.patch('urllib.request.urlopen', side_effect=responses) as mock_urlopen:
        with raises_download_failed(source):
            source.download(tmp_path, verbose=False)
        filepath = source.download(tmp_path, verbose=False)

    assert mock_urlopen.call_count == 2
    assert filepath.read_bytes() == b'complete content'


@pytest.mark.parametrize(
    ('websource', 'expected_exception', 'expected_msg'),
    [
        pytest.param(
            WebSource(url=None),  # type: ignore[arg-type]
            AttributeError,
            'WebSource.url must not be None',
            id='url_none',
        ),
        pytest.param(
            WebSource(url='https://example.com/test.gz.tar', filename=None),
            AttributeError,
            'WebSource.filename must not be None',
            id='filename_none',
        ),
        pytest.param(
            WebSource(url='test.gz.tar', filename='test.gz.tar'),
            ValueError,
            'unknown url type: ',
            id='no_http_resource_gaze',
        ),
    ],
)
def test_websource_download_raises_exception(websource, expected_exception, expected_msg, tmp_path):
    with pytest.raises(expected_exception, match=expected_msg):
        websource.download(tmp_path)


@mock.patch('pymovements.dataset.websource._download_file')
@pytest.mark.parametrize('side_effect', [OSError, RuntimeError])
def test_websource_download_fail(
        mock_download_file,
        side_effect,
        tmp_path,
):
    websource = WebSource(
        url='https://example.com/test.gz.tar',
        filename='test.gz.tar',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    mock_download_file.side_effect = side_effect

    with pytest.raises(
        RuntimeError,
        match='Downloading resource https://example.com/test.gz.tar failed.',
    ):
        websource.download(tmp_path)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
    ])


@mock.patch('pymovements.dataset.websource._download_file')
@pytest.mark.parametrize('side_effect', [OSError, RuntimeError])
@pytest.mark.filterwarnings('ignore:Downloading resource .* failed.*:UserWarning')
def test_websource_download_mirror_fail(
        mock_download_file,
        side_effect,
        tmp_path,
):
    websource = WebSource(
        url='https://example.com/test.gz.tar',
        mirrors=['https://mirror.example.com/test.gz.tar'],
        filename='test.gz.tar',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    mock_download_file.side_effect = side_effect

    with pytest.raises(
        RuntimeError,
        match='Downloading resource test.gz.tar failed for all mirrors',
    ):
        websource.download(tmp_path)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
    ])


@mock.patch('pymovements.dataset.websource._download_file')
@pytest.mark.parametrize('side_effect', [OSError, RuntimeError])
@pytest.mark.filterwarnings('ignore:Downloading resource .* failed.*:UserWarning')
def test_websource_download_fail_two_mirrors(
        mock_download_file,
        tmp_path,
        side_effect,
):
    websource = WebSource(
        url='https://example.com/test.gz.tar',
        mirrors=[
            'https://mirror1.example.com/test.gz.tar',
            'https://mirror2.example.com/test.gz.tar',
        ],
        filename='test.gz.tar',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    mock_download_file.side_effect = side_effect

    with pytest.raises(
        RuntimeError,
        match='Downloading resource test.gz.tar failed for all mirrors',
    ):
        websource.download(tmp_path)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror1.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror2.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
    ])


@mock.patch('pymovements.dataset.websource._download_file')
@pytest.mark.parametrize('side_effect', [OSError, RuntimeError])
@pytest.mark.filterwarnings('ignore:Downloading resource .* failed.*:UserWarning')
def test_websource_download_first_mirror(mock_download_file, side_effect, tmp_path):
    websource = WebSource(
        url='https://example.com/test.gz.tar',
        mirrors=['https://mirror.example.com/test.gz.tar'],
        filename='test.gz.tar',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    mock_download_file.side_effect = [side_effect(), None]

    websource.download(tmp_path)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
    ])


@mock.patch('pymovements.dataset.websource._download_file')
@pytest.mark.parametrize('side_effect', [OSError, RuntimeError])
@pytest.mark.filterwarnings('ignore:Downloading resource .* failed.*:UserWarning')
def test_websource_download_first_of_two_mirrors_gaze_fails(
        mock_download_file, side_effect, tmp_path,
):
    websource = WebSource(
        url='https://example.com/test.gz.tar',
        mirrors=[
            'https://mirror1.example.com/test.gz.tar',
            'https://mirror2.example.com/test.gz.tar',
        ],
        filename='test.gz.tar',
        md5='52bbf03a7c50ee7152ccb9d357c2bb30',
    )

    mock_download_file.side_effect = [side_effect(), side_effect(), None]

    websource.download(tmp_path)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror1.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
        mock.call(
            url='https://mirror2.example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
            verify_checksum=True,
        ),
    ])


@pytest.mark.parametrize(
    ('filename', 'expected_checksum'),
    [
        pytest.param('rda_test_file.rda', 'db8a1766878007ccfdcbd112cb084249', id='rda'),
        pytest.param('monocular_example.csv', 'b56e208f6442bab645789defe865183b', id='csv'),
    ],
)
class TestWebSourceChecksum:
    def test_websource_checksum(self, filename, expected_checksum, testfiles_dirpath):
        path = testfiles_dirpath / filename

        result = WebSource.checksum(path)

        assert result == expected_checksum

    def test_websource_verify_checksum_success(
            self, filename, expected_checksum, testfiles_dirpath,
    ):
        path = testfiles_dirpath / filename

        WebSource(url='test', md5=expected_checksum).verify_checksum(path)

    def test_websource_verify_checksum_mismatch(
            self, filename, expected_checksum, testfiles_dirpath,
    ):
        path = testfiles_dirpath / filename

        wrong_checksum = '123456'

        with pytest.raises(ChecksumError) as exc:
            WebSource(url='test', md5=wrong_checksum).verify_checksum(path)

        assert exc.value.actual == expected_checksum
        assert exc.value.expected == wrong_checksum
        assert exc.value.path == path
        assert exc.value.algorithm == 'MD5'


def test_websource_verify_checksum_no_checksum(make_text_file):
    path = make_text_file('test.txt', 'test')

    message = 'WebSource.md5 must be of type string but got NoneType'

    with pytest.raises(TypeError, match=message):
        WebSource(url='test', md5=None).verify_checksum(path)


def test_websource_verify_checksum_no_file(tmp_path):
    path = tmp_path / 'test.txt'

    message = 'No such file or directory'

    with pytest.raises(FileNotFoundError, match=message):
        WebSource(url='test', md5='123456').verify_checksum(path)
