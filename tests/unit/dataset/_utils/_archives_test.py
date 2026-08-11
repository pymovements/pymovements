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
"""Test pymovements utils archives."""
import bz2
import gzip
import io
import lzma
import os
import pathlib
import tarfile
import zipfile
from collections.abc import Callable

import pytest

from pymovements.dataset._utils._archives import _decompress
from pymovements.dataset._utils._archives import _ZIP_COMPRESSION_MAP
from pymovements.dataset._utils._archives import extract_archive


def test_extract_archive_wrong_suffix():
    """Test unsupported suffix for extract_archive()."""
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(pathlib.Path('test.jpg'))
    msg, = excinfo.value.args
    assert msg == """Unsupported compression or archive type: '.jpg'.
Supported suffixes are: '['.bz2', '.gz', '.tar', '.tbz', '.tbz2', '.tgz', '.xz', '.zip']'."""


def test_detect_file_type_no_suffixes():
    """Test extract_archive() for no files with suffix."""
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(pathlib.Path('test'))
    msg, = excinfo.value.args
    assert msg == "File 'test' has no suffixes that could be used to "\
        'detect the archive type or compression.'


@pytest.fixture(name='make_archive', scope='function')
def fixture_make_archive(
        tmp_path: pathlib.Path,
) -> Callable[[str | pathlib.Path, dict[str, bytes | str]], pathlib.Path]:
    """Make a zip or tar archive from in-memory members.

    The archive format is chosen from the filename suffix: ``.tar`` creates a tar archive,
    any other suffix creates a zip archive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory the archive is written to. Built-in fixture.

    Returns
    -------
    Callable[[str | pathlib.Path, dict[str, bytes | str]], pathlib.Path]
        Function that takes an archive filename and a ``{member_path: contents}`` mapping and
        returns the path to the created archive inside a temporary directory.
    """
    def _make_archive(filename, members):
        archive_path = tmp_path / filename
        if archive_path.suffix == '.tar':
            with tarfile.open(archive_path, 'w') as tar_open:
                for name, content in members.items():
                    data = content.encode() if isinstance(content, str) else content
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar_open.addfile(info, io.BytesIO(data))
        else:
            with zipfile.ZipFile(archive_path, 'w') as zip_open:
                for name, content in members.items():
                    zip_open.writestr(name, content)
        return archive_path
    return _make_archive


@pytest.fixture(
    name='archive',
    params=[
        (None, 'tar'),
        (None, 'zip'),
        ('tbz', None),
        ('tbz2', None),
        ('tgz', None),
        ('bz2', 'tar'),
        ('bz2', 'zip'),
        ('gz', 'tar'),
        ('xz', 'tar'),
        ('xz', 'zip'),
    ],
    ids=[
        'tar_archive',
        'zip_archive',
        'tbz_compressed_archive',
        'tbz2_compressed_archive',
        'tgz_compressed_archive',
        'bz2_compressed_tar_archive',
        'bz2_compressed_zip_archive',
        'gz_compressed_tar_archive',
        'xz_compressed_tar_archive',
        'xz_compressed_zip_archive',
    ],
)
def fixture_archive(request, make_text_file):
    compression, extension = request.param

    test_filepath = make_text_file(filename='test.file', body='test')
    rootpath = test_filepath.parent

    single_child_directory = 'singlechild'
    top_level_directory = 'toplevel'

    # add additional archive
    filepath = rootpath / 'recursive.zip'
    with zipfile.ZipFile(filepath, 'w') as zip_open:
        zip_open.write(
            test_filepath,
            arcname=os.path.join(single_child_directory, test_filepath.name),
        )

    # declare archive path
    if compression is None:
        archive_path = rootpath / f'test.{extension}'
    elif compression is not None and extension is None:
        archive_path = rootpath / f'test.{compression}'
    elif compression is not None and extension is not None:
        archive_path = rootpath / f'test.{extension}.{compression}'
    else:
        raise ValueError(f'{request.param} not supported for archive fixture')

    if compression is None and extension == 'zip':
        with zipfile.ZipFile(archive_path, 'w') as zip_open:
            zip_open.write(filepath, arcname=os.path.join(top_level_directory, filepath.name))

    elif compression is not None and extension == 'zip':
        comp_type = _ZIP_COMPRESSION_MAP[f'.{compression}']
        with zipfile.ZipFile(archive_path, 'w', compression=comp_type) as zip_open:
            zip_open.write(filepath, arcname=os.path.join(top_level_directory, filepath.name))

    elif compression is None and extension == 'tar':
        with tarfile.TarFile.open(archive_path, 'w') as fp:
            fp.add(filepath, arcname=os.path.join(top_level_directory, filepath.name))

    elif (
        (compression is not None and extension == 'tar') or
        (compression in {'tbz', 'tbz2', 'tgz'})
    ):
        if compression in {'tbz', 'tbz2'}:
            compression = 'bz2'
        if compression in {'tgz'}:
            compression = 'gz'
        with tarfile.TarFile.open(archive_path, f'w:{compression}') as fp:
            fp.add(filepath, arcname=os.path.join(top_level_directory, filepath.name))

    else:
        raise ValueError(f'{request.param} not supported for archive fixture')

    # now remove original files again
    test_filepath.unlink()
    filepath.unlink()

    yield archive_path


@pytest.fixture(
    name='compressed_file',
    params=[
        'bz2',
        'gz',
        'xz',
    ],
    ids=[
        'bz2_compressed_file',
        'gz_compressed_file',
        'xz_compressed_file',
    ],
)
def fixture_compressed_file(request, tmp_path):
    rootpath = tmp_path
    compression = request.param

    # write tmp filepath
    test_filepath = rootpath / 'test.file'
    test_filepath.write_bytes(b'test')

    # declare archive path
    compressed_filepath = rootpath / f'test.{compression}'

    if compression == 'bz2':
        with bz2.open(compressed_filepath, 'wb') as fp:
            fp.write(test_filepath.read_bytes())

    elif compression == 'gz':
        with gzip.open(compressed_filepath, 'wb') as fp:
            fp.write(test_filepath.read_bytes())

    elif compression == 'xz':
        with lzma.open(compressed_filepath, 'wb') as fp:
            fp.write(test_filepath.read_bytes())

    else:
        raise ValueError(f'{request.param} not supported for compressed file fixture')

    # now remove original file again
    test_filepath.unlink()

    yield compressed_filepath


@pytest.fixture(
    name='unsupported_archive',
    params=[
        ('xz', 'jpg'),
    ],
    ids=[
        'xz_compressed_unsupported_archive',
    ],
)
def fixture_unsupported_archive(request, make_text_file):
    compression, extension = request.param

    filepath = make_text_file(filename='test.file', body='test')
    rootpath = filepath.parent

    archive_path = rootpath / f'test.{extension}.{compression}'
    comp_type = _ZIP_COMPRESSION_MAP[f'.{compression}']
    with zipfile.ZipFile(archive_path, 'w', compression=comp_type) as zip_open:
        zip_open.write(filepath)
    yield archive_path


@pytest.mark.parametrize(
    ('recursive', 'remove_finished', 'remove_top_level', 'expected_files'),
    [
        pytest.param(
            False, False, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_finished_false',
        ),
        pytest.param(
            False, True, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_finished_true',
        ),
        pytest.param(
            True, False, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive', 'singlechild'),
                os.path.join('toplevel', 'recursive', 'singlechild', 'test.file'),
            ),
            id='recursive_true_remove_finished_false',
        ),
        pytest.param(
            True, True, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive', 'singlechild'),
                os.path.join('toplevel', 'recursive', 'singlechild', 'test.file'),
            ),
            id='recursive_true_remove_finished_true',
        ),
        pytest.param(
            False, False, True,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_top_level_true',
        ),
        pytest.param(
            True, False, True,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive', 'test.file'),
            ),
            id='recursive_true_remove_top_level_true',
        ),
    ],
)
def test_extract_archive_destination_path_None(
        recursive,
        remove_finished,
        remove_top_level,
        expected_files,
        archive,
):
    extract_archive(
        source_path=archive,
        destination_path=None,
        recursive=recursive,
        remove_finished=remove_finished,
        remove_top_level=remove_top_level,
    )
    result_files = {
        str(file.relative_to(archive.parent)) for file in archive.parent.rglob('*')
    }

    expected_files = set(expected_files)
    if not remove_finished:
        expected_files.add(archive.name)
    assert result_files == expected_files


@pytest.mark.parametrize(
    ('recursive', 'remove_finished'),
    [
        pytest.param(False, False, id='recursive_false_remove_finished_false'),
        pytest.param(False, True, id='recursive_false_remove_finished_true'),
        pytest.param(True, False, id='recursive_true_remove_finished_false'),
        pytest.param(True, True, id='recursive_true_remove_finished_true'),
    ],
)
def test_extract_compressed_file_destination_path_None(
        recursive, remove_finished, compressed_file,
):
    extract_archive(
        source_path=compressed_file,
        destination_path=None,
        recursive=recursive,
        remove_finished=remove_finished,
    )
    result_files = {
        str(file.relative_to(compressed_file.parent)) for file in compressed_file.parent.rglob('*')
    }

    expected_files = {'test'}
    if not remove_finished:
        expected_files.add(compressed_file.name)
    assert result_files == expected_files


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_unsupported_archive_destination_path_None(
        recursive,
        remove_finished,
        unsupported_archive,
):
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(
            source_path=unsupported_archive,
            destination_path=None,
            recursive=recursive,
            remove_finished=remove_finished,
        )
    msg, = excinfo.value.args
    assert msg == """Unsupported compression or archive type: '.jpg.xz'.
Supported suffixes are: '['.bz2', '.gz', '.tar', '.tbz', '.tbz2', '.tgz', '.xz', '.zip']'."""


@pytest.mark.parametrize(
    ('recursive', 'remove_finished', 'remove_top_level', 'expected_files'),
    [
        pytest.param(
            False, False, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_finished_false',
        ),
        pytest.param(
            False, True, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_finished_true',
        ),
        pytest.param(
            True, False, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive.zip'),
                os.path.join('toplevel', 'recursive', 'singlechild'),
                os.path.join('toplevel', 'recursive', 'singlechild', 'test.file'),
            ),
            id='recursive_true_remove_finished_false',
        ),
        pytest.param(
            True, True, False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive', 'singlechild'),
                os.path.join('toplevel', 'recursive', 'singlechild', 'test.file'),
            ),
            id='recursive_true_remove_finished_true',
        ),
        pytest.param(
            False, False, True,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_top_level_true',
        ),
        pytest.param(
            True, False, True,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive.zip'),
                os.path.join('toplevel', 'recursive', 'test.file'),
            ),
            id='recursive_true_remove_top_level_true',
        ),
    ],
)
def test_extract_archive_destination_path_not_None(
        recursive,
        remove_finished,
        remove_top_level,
        archive,
        tmp_path,
        expected_files,
):
    destination_path = tmp_path / pathlib.Path('tmpfoo')
    extract_archive(
        source_path=archive,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=remove_finished,
        remove_top_level=remove_top_level,
    )

    if destination_path.is_file():
        destination_path = destination_path.parent

    result_files = {str(file.relative_to(destination_path)) for file in destination_path.rglob('*')}

    assert result_files == set(expected_files)
    assert archive.is_file() != remove_finished


@pytest.mark.parametrize(
    ('recursive', 'remove_finished'),
    [
        pytest.param(False, False, id='recursive_false_remove_finished_false'),
        pytest.param(False, True, id='recursive_false_remove_finished_true'),
        pytest.param(True, False, id='recursive_true_remove_finished_false'),
        pytest.param(True, True, id='recursive_true_remove_finished_true'),
    ],
)
def test_extract_compressed_file_destination_path_not_None(
        recursive,
        remove_finished,
        compressed_file,
        tmp_path,
):
    destination_filename = 'tmpfoo'
    destination_path = tmp_path / pathlib.Path(destination_filename)
    extract_archive(
        source_path=compressed_file,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=remove_finished,
    )
    result_files = {
        str(file.relative_to(compressed_file.parent)) for file in compressed_file.parent.rglob('*')
    }

    expected_files = {destination_filename}
    if not remove_finished:
        expected_files.add(compressed_file.name)
    assert result_files == expected_files


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_unsupported_archive_destination_path_not_None(
        recursive,
        remove_finished,
        unsupported_archive,
        tmp_path,
):
    destination_path = tmp_path / pathlib.Path('tmpfoo')
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(
            source_path=unsupported_archive,
            destination_path=destination_path,
            recursive=recursive,
            remove_finished=remove_finished,
        )
    msg, = excinfo.value.args
    assert msg == """Unsupported compression or archive type: '.jpg.xz'.
Supported suffixes are: '['.bz2', '.gz', '.tar', '.tbz', '.tbz2', '.tgz', '.xz', '.zip']'."""


def test_decompress_unknown_compression_suffix():
    with pytest.raises(RuntimeError) as excinfo:
        _decompress(pathlib.Path('test.zip.zip'))
    msg, = excinfo.value.args
    assert msg == "Couldn't detect a compression from suffix .zip."


@pytest.mark.parametrize(
    ('resume'),
    [
        pytest.param(True, id='resume_True'),
        pytest.param(False, id='resume_False'),
    ],
)
@pytest.mark.parametrize(
    ('recursive', 'remove_top_level', 'expected_files'),
    [
        pytest.param(
            False,
            False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
            ),
            id='recursive_false_remove_finished_false',
        ),
        pytest.param(
            True,
            False,
            (
                'toplevel',
                os.path.join('toplevel', 'recursive.zip'),
                os.path.join('toplevel', 'recursive'),
                os.path.join('toplevel', 'recursive', 'singlechild'),
                os.path.join('toplevel', 'recursive', 'singlechild', 'test.file'),
            ),
            id='recursive_true_remove_finished_false',
        ),
    ],
)
@pytest.mark.parametrize(
    ('verbose'),
    [
        pytest.param(True, id='verbose_True'),
        pytest.param(False, id='verbose_False'),
    ],
)
def test_extract_archive_destination_path_not_None_no_remove_top_level_no_remove_finished_twice(
        verbose,
        recursive,
        remove_top_level,
        archive,
        tmp_path,
        resume,
        expected_files,
        capsys,
):
    destination_path = tmp_path / pathlib.Path('tmp')
    extract_archive(
        source_path=archive,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=False,
        remove_top_level=remove_top_level,
        resume=resume,
        verbose=verbose,
    )
    extract_archive(
        source_path=archive,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=False,
        remove_top_level=remove_top_level,
        resume=resume,
        verbose=verbose,
    )
    if resume and verbose:
        assert 'Skipping' in capsys.readouterr().out

    if destination_path.is_file():
        destination_path = destination_path.parent

    result_files = {str(file.relative_to(destination_path)) for file in destination_path.rglob('*')}

    assert result_files == set(expected_files)


def test_extract_archive_skips_macosx_metadata_files(tmp_path, make_archive):
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, 'w') as inner_archive:
        inner_archive.writestr('test.file', 'test')

    archive_path = make_archive(
        'test.zip', {
            'inner.zip': inner_buffer.getvalue(),
            # macOS stores resource forks as ._ prefixed files inside a __MACOSX directory.
            os.path.join('__MACOSX', '._inner.zip'): b'\x00\x05\x16\x07',
        },
    )

    destination_path = tmp_path / 'extracted'
    extract_archive(source_path=archive_path, destination_path=destination_path, recursive=True)

    assert (destination_path / 'inner' / 'test.file').is_file()
    # The metadata twin must be skipped entirely, not just kept out of nested extraction.
    assert not (destination_path / '__MACOSX').exists()


@pytest.mark.parametrize(
    ('archive_name', 'member', 'skipped'),
    [
        # macOS metadata that must be skipped during extraction.
        pytest.param('test.zip', '.DS_Store', True, id='zip_dsstore'),
        pytest.param('test.zip', os.path.join('__MACOSX', '._data.txt'), True, id='zip_macosx'),
        pytest.param('test.tar', '.DS_Store', True, id='tar_dsstore'),
        # ._ prefixed files outside a __MACOSX directory may be legitimate and must be kept.
        pytest.param('test.zip', '._data.txt', False, id='zip_underscore_outside_macosx'),
        pytest.param('test.tar', '._data.txt', False, id='tar_underscore_outside_macosx'),
    ],
)
def test_extract_archive_macos_metadata_filtering(
        make_archive, tmp_path, archive_name, member, skipped,
):
    archive_path = make_archive(
        archive_name, {'data.txt': b'test', member: b'\x00\x00\x00\x01Bud1'},
    )

    destination_path = tmp_path / 'extracted'
    extract_archive(source_path=archive_path, destination_path=destination_path, recursive=True)

    assert (destination_path / 'data.txt').is_file()
    assert (destination_path / member).exists() == (not skipped)
