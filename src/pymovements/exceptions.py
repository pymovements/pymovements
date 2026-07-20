# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Exceptions module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, eq=False)
class ChecksumError(Exception):
    """Exception raised when a checksum integrity check fails.

    Attributes
    ----------
    expected: str
        Expected checksum.
    actual: str
        Actual checksum.
    path: Path
        Path of checked file.
    algorithm: str
        Name of the checksum algorithm. (default: 'MD5')
    """

    expected: str
    actual: str
    path: Path
    algorithm: str = 'MD5'

    def __str__(self) -> str:
        """Get exception message."""
        return (
            f"{self.algorithm} checksum mismatch for file '{self.path}'"
            f": expected '{self.expected}', got '{self.actual}'"
        )


class UnknownMeasure(Exception):
    """Raised if requested measure is unknown.

    Parameters
    ----------
    measure_name: str
        Name of the property which is invalid.

    known_measures: list[str]
        List of valid properties.
    """

    def __init__(self, measure_name: str, known_measures: list[str]):
        message = f"Measure '{measure_name}' is unknown. Known measures are: {known_measures}"
        super().__init__(message)


class UnknownFileType(RuntimeError):
    """Raised on unknown file types."""
