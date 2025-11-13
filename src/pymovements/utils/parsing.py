# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Module for parsing input data.

.. deprecated:: v0.21.0
   Please use :py:func:`~pymovements.gaze.from_asc()` instead.
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from deprecated.sphinx import deprecated

from pymovements.gaze._utils._parsing_eyelink import parse_eyelink as _parse_eyelink


@deprecated(
    reason='Please use gaze.from_asc() instead. '
           'This module will be removed in v0.26.0.',
    version='v0.21.0',
)
def parse_eyelink(
        filepath: Path | str,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str = 'ascii',
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Parse EyeLink asc file.

    .. deprecated:: v0.21.0
       Please use :py:func:`~pymovements.gaze.from_asc()` instead.
       This module will be removed in v0.26.0.

    Parameters
    ----------
    filepath: Path | str
        file name of ascii file to convert.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        list of patterns to match for additional metadata. (default: None)
    encoding: str
        Text encoding of the file. (default: 'ascii')

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        A tuple containing the parsed sample data and the metadata in a dictionary.

    Raises
    ------
    Warning
        If no metadata is found in the file.
    """
    gaze, _, metadata, _ = _parse_eyelink(
        filepath=filepath,
        patterns=patterns,
        schema=schema,
        metadata_patterns=metadata_patterns,
        encoding=encoding,
    )
    return gaze, metadata
