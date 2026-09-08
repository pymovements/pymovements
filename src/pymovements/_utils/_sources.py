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

The ``sources`` metadata entry lists the files an object was directly read or
generated from, analogous to the BIDS ``Sources`` sidecar field. Like BIDS
``Sources``, it records proximate provenance only: reloading a previously
saved file records the file that was actually read, not the sources of the
original data. Full provenance chains are recovered hop by hop through the
metadata saved alongside each file. Entries are POSIX-style path strings:
absolute for standalone loading, relative to the dataset root when loaded via
:py:class:`~pymovements.Dataset`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def add_source(metadata: dict[str, Any] | None, file: Any) -> dict[str, Any]:
    """Return a metadata dictionary with the loaded file recorded in ``sources``.

    The passed metadata dictionary is copied, not mutated. A ``sources`` entry
    already present in the passed metadata is respected and left unchanged, as
    it expresses explicit user intent. Loaders reading metadata from a file
    saved alongside the data must not forward that file's ``sources`` entry
    here: it describes the saved file's own provenance, while the proximate
    source of the loaded object is the file that was actually read.

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


def _check_sources(metadata: dict[str, Any] | None) -> None:
    """Raise if a present ``sources`` entry is not a list of path strings."""
    if not metadata or not metadata.get('sources'):
        return

    if not isinstance(metadata['sources'], list):
        raise TypeError(
            "metadata['sources'] must be a list of path strings "
            f'but is of type {type(metadata["sources"]).__name__}: {metadata["sources"]!r}',
        )

    for source in metadata['sources']:
        if not isinstance(source, (str, Path)):
            raise TypeError(
                "metadata['sources'] entries must be path strings "
                f'but found entry of type {type(source).__name__}: {source!r}',
            )


def _as_posix_string(source: str | Path) -> str:
    """Return the source entry as a POSIX-style path string."""
    return source if isinstance(source, str) else source.as_posix()


def _relativize_source(source: str | Path, resolved_root: Path) -> str:
    """Return the source entry as a POSIX-style path string, root-relative if below the root."""
    try:
        return Path(source).relative_to(resolved_root).as_posix()
    except ValueError:
        return _as_posix_string(source)


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
    _check_sources(metadata)

    resolved_root = root.resolve()
    sources = (metadata or {}).get('sources') or []
    sources[:] = [_relativize_source(source, resolved_root) for source in sources]


def merge_sources(metadata: dict[str, Any], other: dict[str, Any] | None) -> None:
    """Append the ``sources`` entries of another metadata dictionary.

    Duplicate entries are dropped while the original order is preserved.
    Entries are compared as POSIX-style path strings, so equal :py:class:`str`
    and :py:class:`~pathlib.Path` entries deduplicate; appended entries are
    recorded as POSIX-style path strings. The metadata dictionary is modified
    in place.

    Parameters
    ----------
    metadata: dict[str, Any]
        Metadata dictionary to extend.
    other: dict[str, Any] | None
        Metadata dictionary to merge the ``sources`` entries from. May be ``None``.

    Raises
    ------
    TypeError
        If a ``sources`` entry of either dictionary is not a list of path strings.
    """
    _check_sources(metadata)
    _check_sources(other)

    other_sources = (other or {}).get('sources') or []
    if not other_sources:
        return

    sources = list(metadata.get('sources') or [])
    seen = {_as_posix_string(source) for source in sources}
    for source in other_sources:
        key = _as_posix_string(source)
        if key not in seen:
            sources.append(key)
            seen.add(key)
    metadata['sources'] = sources
