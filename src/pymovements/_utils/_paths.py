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
"""Provides path specific funtions."""
from __future__ import annotations

import re
from pathlib import Path


def get_filepaths(
        path: str | Path,
        extension: str | list[str] | None = None,
        regex: re.Pattern | None = None,
) -> list[Path]:
    """Get filepaths from rootpath depending on extension or regular expression.

    Passing extension and regex are mutually exclusive.

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
    if extension is not None and regex is not None:
        raise ValueError('extension and regex are mutually exclusive')

    if extension is not None and isinstance(extension, str):
        extension = [extension]

    path = Path(path)
    if not path.is_dir():
        return []

    filepaths = []
    for childpath in path.iterdir():
        if childpath.is_dir():
            filepaths.extend(get_filepaths(path=childpath, extension=extension, regex=regex))
        else:
            # if extension specified and not matching, continue to next
            if extension and childpath.suffix not in extension:
                continue
            # if regex specified and not matching, continue to next
            if regex and not regex.match(childpath.name):
                continue
            filepaths.append(childpath)
    return filepaths


def match_filepaths(
        path: str | Path,
        regex: re.Pattern,
        relative: bool = True,
        relative_anchor: Path | None = None,
) -> list[dict[str, str]]:
    """Traverse path and match regular expression.

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
    path = Path(path)

    if not path.exists():
        raise ValueError(f'path does not exist (path = {path})')

    if not path.is_dir():
        raise ValueError(f'path must point to a directory (path = {path})')

    if relative and relative_anchor is None:
        relative_anchor = path

    match_dicts: list[dict[str, str]] = []
    for childpath in path.iterdir():
        if childpath.is_dir():
            recursive_results = match_filepaths(
                path=childpath, regex=regex,
                relative=relative, relative_anchor=relative_anchor,
            )
            match_dicts.extend(recursive_results)
        elif match := regex.match(childpath.name):
            match_dict = match.groupdict()

            filepath = childpath
            if relative:
                # mypy is unaware that 'relative_anchor' can never be None (l.116)
                assert relative_anchor is not None
                filepath = filepath.relative_to(relative_anchor)

            match_dict['filepath'] = str(filepath)
            match_dicts.append(match_dict)
    return match_dicts
