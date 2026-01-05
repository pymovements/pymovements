# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Provide fixtures to securely make files from examples in ``tests/files``."""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(name='testfiles_dirpath', scope='session')
def fixture_testfiles_dirpath(request):
    """Return the path to tests/files."""
    return request.config.rootpath / 'tests' / 'files'


@pytest.fixture(name='make_example_file', scope='function')
def fixture_make_example_file(
        testfiles_dirpath: Path, tmp_path: Path,
) -> Callable[[str, str | None], Path]:
    """Make a temporary copy of a file from one of the example files in tests/files.

    This way each file can be used in tests without the risk of changing contents.

    Parameters
    ----------
    testfiles_dirpath : Path
        Path to the tests/files directory.
    tmp_path : Path
        Temporary directory where files are copied to.

    Returns
    -------
    Callable[[str, str | None], Path]
        Function that takes a example_filename and returns the Path to the copied file.

    """
    def _make_example_file(example_filename: str, target_filename: str | None = None) -> Path:
        """Make a temporary copy of a file from one of the example files in tests/files.

        The example file is automatically copied into a ``tmp_path``.

        Parameters
        ----------
        example_filename : str
            Use this file as a source to make the test file.
        target_filename : str | None
            Use this as the filename of the target file. If ``None``, use ``example_filename``.
            (default: None)

        Returns
        -------
        Path
            Path to created example test file.

        """
        if target_filename is None:
            target_filename = example_filename
        source_filepath = testfiles_dirpath / example_filename
        target_filepath = tmp_path / target_filename
        shutil.copy2(source_filepath, target_filepath)
        return target_filepath
    return _make_example_file


@pytest.fixture(name='make_text_file', scope='function')
def fixture_make_text_file(tmp_path: Path) -> Callable[[str | Path, str, str, str], Path]:
    """Make a custom file with self-written header and body.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory where files are copied to. Built-in fixture.

    Returns
    -------
    Callable[[str | Path, str, str, str], Path]
        Function that takes a filename, the file header and body and the encoding of the file.
        Returns the path to the created file.
        Header and body are simply concatenated into a single string and written to the file.
        The file is saved into a temporary directory.

    """
    def _make_text_file(
            filename: str | Path, header: str = '', body: str = '\n', encoding: str = 'utf-8',
    ) -> Path:
        relative_filepath = _validate_filename(filename)
        content = header + body
        filepath = tmp_path / relative_filepath
        # assure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # write contents to a file
        filepath.write_text(content, encoding=encoding)
        return filepath
    return _make_text_file


@pytest.fixture(name='make_csv_file', scope='function')
def fixture_make_csv_file(tmp_path: Path) -> Callable:
    """Make a csv file with optional header.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory where files are copied to. Built-in fixture.

    Returns
    -------
    Callable
        Function that takes a filename, a data frame, an optional header, and optional keyword
        arguments to be passed to :py:class:`polars.write_csv`.
        Returns the path to the created file. The file is saved into a temporary directory.

    """
    def _make_csv_file(
            filename: str | Path,
            data: pl.DataFrame,
            *,
            header: str | None = None,
            include_bom: bool = False,
            include_header: bool = True,
            separator: str = ',',
            line_terminator: str = '\n',
            quote_char: str = '"',
            datetime_format: str | None = None,
            date_format: str | None = None,
            time_format: str | None = None,
            float_scientific: bool | None = None,
            float_precision: int | None = None,
            null_value: str | None = None,
            quote_style: pl._typing.CsvQuoteStyle | None = None,
    ) -> Path:
        r"""Make a csv file with optional header.

        This is the actual function called when using the ``make_csv_file`` fixture.

        Parameters
        ----------
        filename: str | Path
            Make csv file with this filename. Can also be a relative path.
        data: pl.DataFrame
            Write this data frame into csv file.
        header: str | None
            Write this string in the line before the actual column header of the csv file.
            (default: None)
        include_bom: bool
            Whether to include UTF-8 BOM in the CSV output. (default: False)
        include_header: bool
            Whether to include header in the CSV output. (default: True)
        separator: str
            Separate CSV fields with this symbol. (default: ",")
        line_terminator: str
            String used to end each row. (default: "\n")
        quote_char: str
            Byte to use as quoting character. (default '"')
        datetime_format: str | None
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate. If no format specified, the default fractional-second
            precision is inferred from the maximum timeunit found in the frame's
            Datetime cols (if any). (default: None)
        date_format: str | None
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate. (default: None)
        time_format: str | None
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate. (default: None)
        float_scientific: bool | None
            Whether to use scientific form always (true), never (false), or
            automatically (None) for `Float32` and `Float64` datatypes. (default: None)
        float_precision: int | None
            Number of decimal places to write, applied to both `Float32` and
            `Float64` datatypes. (default: None)
        null_value: str | None
            A string representing null values (defaulting to the empty string).
        quote_style : pl._typing.CsvQuoteStyle | None
            Determines the quoting strategy used.
            Valid quote styles are: {'necessary', 'always', 'non_numeric', 'never'}

            - necessary (default): This puts quotes around fields only when necessary.
              They are necessary when fields contain a quote,
              separator or record terminator.
              Quotes are also necessary when writing an empty record
              (which is indistinguishable from a record with one empty field).
              This is the default.
            - always: This puts quotes around every field. Always.
            - never: This never puts quotes around fields, even if that results in
              invalid CSV data (e.g.: by not quoting strings containing the separator).
            - non_numeric: This puts quotes around all fields that are non-numeric.
              Namely, when writing a field that does not parse as a valid float
              or integer, then quotes will be used even if they aren`t strictly
              necessary.

        Returns
        -------
        Path
            Path to csv file.

        """
        relative_filepath = _validate_filename(filename)
        filepath = tmp_path / relative_filepath
        # assure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, mode='x', encoding='utf-8') as opened_file:
            # write header into
            if header is not None:
                opened_file.write(header + '\n')
            # write contents to a file
            data.write_csv(
                opened_file,
                include_bom=include_bom,
                include_header=include_header,
                separator=separator,
                line_terminator=line_terminator,
                quote_char=quote_char,
                datetime_format=datetime_format,
                date_format=date_format,
                time_format=time_format,
                float_scientific=float_scientific,
                float_precision=float_precision,
                null_value=null_value,
                quote_style=quote_style,
            )
        return filepath
    return _make_csv_file


def _validate_filename(filename: str | Path) -> Path:
    """Ensure filename is a relative path.

    Accepts str or Path. Rejects other types with TypeError.
    Rejects absolute paths, drive letters (Windows),
    and user home shortcuts '~' with ValueError.
    Returns a Path relative to current directory (to be joined with tmp_path).
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError(f"filename must be a str or Path, got {type(filename).__name__}")

    if isinstance(filename, str) and filename.startswith('~'):
        raise ValueError("filename must be a relative path; '~' (home) is not allowed")

    path = Path(filename)
    # On Windows, p.drive captures drive letters; p.anchor is non-empty for absolute paths
    if path.is_absolute() or path.anchor or getattr(path, 'drive', ''):
        raise ValueError('filename must be a relative path without drive or root')
    return path
