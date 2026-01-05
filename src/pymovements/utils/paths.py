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
"""Provides path specific funtions.

.. deprecated:: v0.21.1
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

import re
from pathlib import Path

from deprecated.sphinx import deprecated

from pymovements._utils._paths import get_filepaths as _get_filepaths
from pymovements._utils._paths import match_filepaths as _match_filepaths


@deprecated(
    reason='This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def get_filepaths(
        path: str | Path,
        extension: str | list[str] | None = None,
        regex: re.Pattern | None = None,
) -> list[Path]:
    """Get filepaths from rootpath depending on extension or regular expression.

    Passing extension and regex is mutually exclusive.

    .. deprecated:: v0.21.1
       This module will be removed in v0.26.0.

    Parameters
    ----------
    path: str | Path
        Root path to be traversed.
    extension: str | list[str] | None
        File extension to be filtered for. (default: None)
    regex: re.Pattern | None
        Regular expression filenames will be filtered for. (default: None)

    Returns
    -------
    list[Path]

    Raises
    ------
    ValueError
        If both extension and regex is being passed.
    """
    return _get_filepaths(
        path=path,
        extension=extension,
        regex=regex,
    )


@deprecated(
    reason='This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def match_filepaths(
        path: str | Path,
        regex: re.Pattern,
        relative: bool = True,
        relative_anchor: Path | None = None,
) -> list[dict[str, str]]:
    """Traverse path and match regular expression.

    .. deprecated:: v0.21.1
       This module will be removed in v0.26.0.

    Parameters
    ----------
    path: str | Path
        Root path to be traversed.
    regex: re.Pattern
        Regular expression filenames will be matched against.
    relative: bool
        If True, specify filepath as relative to root path. (default: True)
    relative_anchor: Path | None
        Specifies root path in case of ``relative == True``. If None, ``path`` will be chosen
        as `relative_anchor` for recursive calls. (default: None)

    Returns
    -------
    list[dict[str, str]]
        Each entry contains the match group dictionary of the regular expression.

    Raises
    ------
    ValueError
        If ``path`` does not point to a directory.
    """
    return _match_filepaths(
        path=path,
        regex=regex,
        relative=relative,
        relative_anchor=relative_anchor,
    )
