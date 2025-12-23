# Copyright (c) 2025 The pymovements Project Authors
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
"""Module for sample measure library."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import polars as pl


class SampleMeasureLibrary:
    """Provides access by name to sample measures.

    Attributes
    ----------
    measures: dict[str, Callable[..., pl.Expr]]
        Dictionary of measures. The key correpsonds to the name of each measure.
    """

    measures: dict[str, Callable[..., pl.Expr]] = {}

    @classmethod
    def add(cls, measure: Callable[..., pl.Expr]) -> None:
        """Add a measure to the library.

        Parameters
        ----------
        measure: Callable[..., pl.Expr]
            The measure to add to the library.
        """
        cls.measures[measure.__name__] = measure

    @classmethod
    def get(cls, name: str) -> Callable[..., pl.Expr]:
        """Get measure py name.

        Parameters
        ----------
        name: str
            Name of the measure in the library.

        Returns
        -------
        Callable[..., pl.Expr]
            The requested measure.
        """
        return cls.measures[name]

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Check if class contains measure of given name.

        Parameters
        ----------
        name: str
            Name of the measure to check.

        Returns
        -------
        bool
            True if SampleMeasureLibrary contains measure with given name, else False.
        """
        return name in cls.measures


SampleMeasure = TypeVar('SampleMeasure', bound=Callable[..., pl.Expr])


def register_sample_measure(measure: SampleMeasure) -> SampleMeasure:
    """Register a sample measure.

    Parameters
    ----------
    measure: SampleMeasure
        The measure to register.

    Returns
    -------
    SampleMeasure
        The registered sample measure.
    """
    SampleMeasureLibrary.add(measure)
    return measure
