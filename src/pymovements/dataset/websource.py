# Copyright (c) 2025-2026 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
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
"""WebSource definition and download helper.

Phase 1 foundation for introducing `WebSource` as a source container for
remote resources. This class encapsulates URL, optional mirrors, filename
and MD5 checksum plus a `download()` method which leverages existing
utility functions without changing their behavior.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from warnings import warn

from pymovements.dataset._utils._downloads import download_file


@dataclass
class WebSource:
    """Web-based source of a dataset resource.

    Attributes
    ----------
    url: str
        Primary URL of the resource to be downloaded.
    filename: str | None
        Optional target filename. If not provided, the basename of the URL path is used.
    md5: str | None
        Optional MD5 checksum for integrity verification.
    mirrors: list[str] | None
        Optional list of full mirror URLs. Tried in order if primary URL fails.
    """

    url: str
    filename: str | None = None
    md5: str | None = None
    mirrors: list[str] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WebSource:
        """Create a `WebSource` from a dictionary."""
        # Accept both with and without explicit keys; default to None when missing
        return WebSource(
            url=data.get("url"),
            filename=data.get("filename"),
            md5=data.get("md5"),
            mirrors=data.get("mirrors"),
        )

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize this `WebSource` to a dictionary.

        Omits None values if `exclude_none` is True.
        """
        payload: dict[str, Any] = asdict(self)
        if exclude_none:
            return {k: v for k, v in payload.items() if v is not None}
        return payload

    def download(self, target_dir: Path | str, *, exist_ok: bool = True, verbose: bool = True) -> Path:
        """Download this resource into `target_dir`.

        Tries the primary `url` first, then any `mirrors` in order. Integrity is
        validated via MD5 when provided. Returns the local file path.
        """
        dirpath = Path(target_dir).expanduser()
        # `download_file` will ensure directory exists; we still optionally create it here
        if exist_ok:
            dirpath.mkdir(parents=True, exist_ok=True)

        # Determine filename; fallback to URL basename when not explicitly set
        filename = self.filename
        if filename is None:
            parsed = urlparse(self.url)
            candidate = Path(parsed.path).name
            if not candidate:
                raise ValueError("Unable to infer filename from URL; please provide `filename`.")
            filename = candidate

        # Attempt primary URL
        try:
            return download_file(url=self.url, dirpath=dirpath, filename=filename, md5=self.md5, verbose=verbose)
        # pylint: disable=overlapping-except
        except (URLError, OSError, RuntimeError) as primary_error:
            # No mirrors to try
            if not self.mirrors:
                raise RuntimeError(f"Downloading resource {self.url} failed.") from primary_error

            warn(UserWarning(f"Downloading resource {self.url} failed. Trying mirror."))

            # Try mirrors in order
            for mirror_idx, mirror_url in enumerate(self.mirrors, start=1):
                try:
                    return download_file(
                        url=mirror_url,
                        dirpath=dirpath,
                        filename=filename,
                        md5=self.md5,
                        verbose=verbose,
                    )
                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as mirror_error:
                    msg = f"Downloading resource from mirror {mirror_url} failed."
                    if mirror_idx < len(self.mirrors):
                        msg = msg + f" Trying next mirror ({len(self.mirrors) - mirror_idx} remaining)."
                    warning = UserWarning(msg)
                    warning.__cause__ = mirror_error
                    warn(warning)
                    # Continue to next mirror

            # If we are here, all mirrors failed
            raise RuntimeError(
                f"Downloading resource {filename} failed for all mirrors.") from primary_error
