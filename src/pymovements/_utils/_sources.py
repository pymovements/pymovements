# Copyright (c) 2026 The pymovements Project Authors
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
"""Provides utilities for recording source files in metadata dictionaries.

The ``sources`` metadata entry lists the files an object was generated from,
analogous to the BIDS ``Sources`` sidecar field. Entries are POSIX-style path
strings: absolute for standalone loading, relative to the dataset root when
loaded via :py:class:`~pymovements.Dataset`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def add_source(metadata: dict[str, Any] | None, file: Any) -> dict[str, Any]:
    """Return a metadata dictionary with the loaded file recorded in ``sources``.

    The passed metadata dictionary is copied, not mutated. A ``sources`` entry
    already present in the passed metadata is respected and left unchanged.

    Parameters
    ----------
    metadata: dict[str, Any] | None
        Metadata dictionary to extend. May be ``None``.
    file: Any
        Path of the loaded source file. Values that are not :py:class:`str` or
        :py:class:`~pathlib.Path` are ignored.

    Returns
    -------
    dict[str, Any]
        Copy of the metadata dictionary with a ``sources`` entry.
    """
    metadata = dict(metadata) if metadata else {}
    if 'sources' not in metadata and isinstance(file, (str, Path)):
        metadata['sources'] = [Path(file).resolve().as_posix()]
    return metadata


def relativize_sources(metadata: dict[str, Any] | None, root: Path) -> None:
    """Rewrite absolute ``sources`` entries below ``root`` as root-relative paths.

    The metadata dictionary is modified in place. Entries outside of ``root``
    are kept as they are.

    Parameters
    ----------
    metadata: dict[str, Any] | None
        Metadata dictionary holding a ``sources`` entry. May be ``None``.
    root: Path
        Directory to relativize the ``sources`` entries against.

    Raises
    ------
    TypeError
        If the ``sources`` entry is not a list of path strings.
    """
    if not metadata or not metadata.get('sources'):
        return

    if not isinstance(metadata['sources'], list):
        raise TypeError(
            "metadata['sources'] must be a list of path strings "
            f'but is of type {type(metadata["sources"]).__name__}: {metadata["sources"]!r}',
        )

    resolved_root = root.resolve()

    sources = []
    for source in metadata['sources']:
        if not isinstance(source, (str, Path)):
            raise TypeError(
                "metadata['sources'] entries must be path strings "
                f'but found entry of type {type(source).__name__}: {source!r}',
            )
        try:
            sources.append(Path(source).relative_to(resolved_root).as_posix())
        except ValueError:
            sources.append(source if isinstance(source, str) else source.as_posix())
    metadata['sources'] = sources


def merge_sources(metadata: dict[str, Any], other: dict[str, Any] | None) -> None:
    """Append the ``sources`` entries of another metadata dictionary.

    Duplicate entries are dropped while the original order is preserved. The
    metadata dictionary is modified in place.

    Parameters
    ----------
    metadata: dict[str, Any]
        Metadata dictionary to extend.
    other: dict[str, Any] | None
        Metadata dictionary to merge the ``sources`` entries from. May be ``None``.
    """
    other_sources = (other or {}).get('sources') or []
    if not other_sources:
        return

    sources = list(metadata.get('sources') or [])
    sources.extend(source for source in other_sources if source not in sources)
    metadata['sources'] = sources
