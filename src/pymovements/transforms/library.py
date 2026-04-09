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
"""Module for py:func:`pymovements.gaze.transforms."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import polars as pl


TransformMethod = TypeVar('TransformMethod', bound=Callable[..., pl.Expr])


class TransformLibrary:
    """Provides access by name to transformation methods.

    Attributes
    ----------
    methods: dict[str, Callable[..., pl.Expr]]
        Dictionary of transformation methods.
    """

    methods: dict[str, Callable[..., pl.Expr]] = {}

    @classmethod
    def add(cls, method: Callable[..., pl.Expr]) -> None:
        """Add a transformation method to the library.

        Parameters
        ----------
        method: Callable[..., pl.Expr]
            The transformation method to add to the library.
        """
        cls.methods[method.__name__] = method

    @classmethod
    def get(cls, name: str) -> Callable[..., pl.Expr]:
        """Get transformation method py name.

        Parameters
        ----------
        name: str
            Name of the transformation method in the library.

        Returns
        -------
        Callable[..., pl.Expr]
            The transformation method.
        """
        return cls.methods[name]

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Check if class contains method of given name.

        Parameters
        ----------
        name: str
            Name of the method to check.

        Returns
        -------
        bool
            True if TransformsLibrary contains a method with a given name, else False.
        """
        return name in cls.methods


def register_transform(method: TransformMethod) -> TransformMethod:
    """Register a transform method.

    Parameters
    ----------
    method: TransformMethod
        The transform method to register.

    Returns
    -------
    TransformMethod
        The registered transform method.
    """
    TransformLibrary.add(method)
    return method
