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
"""Utils module for extracting archives and decompressing files.

.. deprecated:: v0.21.1
   Please use :py:meth:`~pymovements.Dataset.extract()` instead.
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

from pathlib import Path

from deprecated.sphinx import deprecated

from pymovements.dataset._utils._archives import extract_archive as _extract_archive


@deprecated(
    reason='Please use Dataset.extract() instead. '
           'This function will be removed in v0.26.0.',
    version='v0.21.1',
)
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

    .. deprecated:: v0.21.1
       Please use :py:meth:`pymovements.Dataset.extract()` instead.
       This module will be removed in v0.26.0.

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
        Checks for correct size of existing files but not integrity. (default: False)
    verbose: int
        Verbosity levels: (1) Print messages for extracting each dataset resource without printing
        messages for recursive archives. (2) Print additional messages for each recursive archive
        extract. (default: 1)

    Returns
    -------
    Path
        Path to the directory the file was extracted to.
    """
    return _extract_archive(
        source_path=source_path,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=remove_finished,
        remove_top_level=remove_top_level,
        resume=resume,
        verbose=verbose,
    )
