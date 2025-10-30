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
"""Test pymovements utils archives."""
import bz2
import gzip
import lzma
import os
import pathlib
import tarfile
import zipfile

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
