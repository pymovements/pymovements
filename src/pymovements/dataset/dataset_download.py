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
"""Provides private functions for downloading and extracting datasets."""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from warnings import warn

from pymovements.dataset._utils._archives import extract_archive
from pymovements.dataset._utils._downloads import download_file
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.dataset.resources import ResourceDefinition
from pymovements.dataset.resources import ResourceDefinitions
from pymovements.exceptions import UnknownFileType


def download_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        *,
        extract: bool = True,
        remove_finished: bool = False,
        resume: bool = True,
        verbose: bool = True,
) -> None:
    """Download dataset resources.

    This downloads all resources of the dataset. Per default this also extracts all archives
    into :py:meth:`Dataset.paths.raw`,
    To save space on your device you can remove the archive files after
    successful extraction with ``remove_finished=True``.

    If a corresponding file already exists in the local system, its checksum is calculated and
    checked against the expected checksum.
    Downloading will be evaded if the integrity of the existing file can be verified.
    If the existing file does not match the expected checksum it is overwritten with the
    downloaded new file.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition.
    paths: DatasetPaths
        The dataset paths.
    extract: bool
        Extract dataset archive files. (default: True)
    remove_finished: bool
        Remove archive files after extraction. (default: False)
    resume: bool
        Resume previous extraction by skipping existing files.
        Checks for correct size of existing files but not integrity. (default: True)
    verbose: bool
        If True, show progress of download and print status messages for integrity checking and
        file extraction. (default: True)

    Raises
    ------
    AttributeError
        If number of mirrors or number of resources specified for dataset is zero.
    RuntimeError
        If downloading a resource failed for all given mirrors.
    """
    if not definition.resources:
        raise AttributeError('resources must be specified to download a dataset.')

    for content in ('gaze', 'precomputed_events', 'precomputed_reading_measures'):
        if definition.resources.has_content(content):
            if not definition.mirrors:
                mirrors = None
            else:
                mirrors = definition.mirrors.get(content, None)

            _download_resources(
                mirrors=mirrors,
                resources=definition.resources.filter(content),
                target_dirpath=paths.downloads,
                verbose=verbose,
            )

    if extract:
        extract_dataset(
            definition=definition,
            paths=paths,
            remove_finished=remove_finished,
            resume=resume,
            verbose=verbose,
        )


def extract_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        *,
        remove_finished: bool = False,
        remove_top_level: bool = True,
        resume: bool = True,
        verbose: int = 1,
) -> None:
    """Extract downloaded dataset archive files.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition.
    paths: DatasetPaths
        The dataset paths.
    remove_finished: bool
        Remove archive files after extraction. (default: False)
    remove_top_level: bool
        If ``True``, remove the top-level directory if it has only one child. (default: True)
    resume: bool
        Resume previous extraction by skipping existing files.
        Checks for correct size of existing files but not integrity. (default: True)
    verbose: int
        Verbosity levels: (1) Print messages for extracting each dataset resource without printing
        messages for recursive archives. (2) Print messages for extracting each dataset resource and
        each recursive archive extract. (default: 1)
    """
    content_dirnames = {
        'gaze': 'raw',
        'precomputed_events': 'precomputed_events',
        'precomputed_reading_measures': 'precomputed_reading_measures',
    }

    for content, content_directory in content_dirnames.items():
        if definition.resources.has_content(content):
            destination_dirpath = getattr(paths, content_directory)
            destination_dirpath.mkdir(parents=True, exist_ok=True)
            for resource in definition.resources.filter(content):
                source_path = paths.downloads / resource.filename

                try:
                    extract_archive(
                        source_path=source_path,
                        destination_path=destination_dirpath,
                        recursive=True,
                        remove_finished=remove_finished,
                        remove_top_level=remove_top_level,
                        resume=resume,
                        verbose=verbose,
                    )
                except UnknownFileType:  # just copy file to target if not an archive.
                    shutil.copy(source_path, destination_dirpath / resource.filename)


def _download_resources(
        mirrors: Sequence[str] | None,
        resources: ResourceDefinitions,
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resources."""
    for resource in resources:
        if not mirrors:
            _download_resource(resource, target_dirpath, verbose)
        else:
            _download_resource_with_legacy_mirrors(mirrors, resource, target_dirpath, verbose)


def _download_resource(
        resource: ResourceDefinition,
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resource without mirrors."""
    if resource.url is None:
        raise AttributeError('Resource.url must not be None')
    if resource.filename is None:
        raise AttributeError('Resource.filename must not be None')

    try:
        download_file(
            url=resource.url,
            dirpath=target_dirpath,
            filename=resource.filename,
            md5=resource.md5,
            verbose=verbose,
        )

    # pylint: disable=overlapping-except
    except (URLError, OSError, RuntimeError) as error:
        if not resource.mirrors:
            raise RuntimeError(f"Downloading resource {resource.url} failed.") from error

        warn(UserWarning(f'Downloading resource {resource.url} failed. Trying mirror.'))

        success = _download_resource_from_mirrors(
            mirrors=resource.mirrors,
            filename=resource.filename,
            md5=resource.md5,
            target_dirpath=target_dirpath,
            verbose=verbose,
        )

        if not success:
            raise RuntimeError(
                f"Downloading resource {resource.filename} failed for all mirrors.",
            ) from error


def _download_resource_from_mirrors(
        mirrors: list[str],
        filename: str,
        md5: str | None,
        target_dirpath: Path,
        verbose: bool,
) -> bool:
    """Download resource from mirrors."""
    for mirror_idx, mirror_url in enumerate(mirrors, start=1):
        try:
            download_file(
                url=mirror_url,
                dirpath=target_dirpath,
                filename=filename,
                md5=md5,
                verbose=verbose,
            )
            return True  # Download successful, exit loop
        # pylint: disable=overlapping-except
        except (URLError, OSError, RuntimeError) as error:
            msg = f'Downloading resource from mirror {mirror_url} failed.'
            if mirror_idx < len(mirrors):
                msg = msg + f' Trying next mirror ({len(mirrors) - mirror_idx} remaining).'
            warning = UserWarning(msg)
            warning.__cause__ = error
            warn(warning)
            # Try the next mirror
    return False  # All mirrors failed


def _download_resource_with_legacy_mirrors(
        mirrors: Sequence[str],
        resource: ResourceDefinition,
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resource with mirrors."""
    if resource.url is None:
        raise AttributeError('Resource.url must not be None')
    if resource.filename is None:
        raise AttributeError('Resource.filename must not be None')

    for mirror_idx, mirror in enumerate(mirrors, start=1):
        mirror_url = f'{mirror}{resource.url}'
        try:
            download_file(
                url=mirror_url,
                dirpath=target_dirpath,
                filename=resource.filename,
                md5=resource.md5,
                verbose=verbose,
            )
            return  # Download successful
        # pylint: disable=overlapping-except
        except (URLError, OSError, RuntimeError) as error:
            # Error downloading the resource, try next mirror
            if mirror_idx < len(mirrors):
                warning = UserWarning(
                    f'Downloading resource from mirror {mirror_url} failed.'
                    f' Trying next mirror ({len(mirrors) - mirror_idx} remaining).',
                )
                warning.__cause__ = error
                warn(warning)

    raise RuntimeError(
        f"Downloading resource {resource.url} failed for all mirrors.",
    )
