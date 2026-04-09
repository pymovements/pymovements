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
"""Module for py:func:`pymovements.transforms.norm`."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pymovements._utils import _checks
from pymovements.transforms.library import register_transform


@register_transform
def norm(
        column: str | None = None,
        *,
        components: tuple[int, int] | tuple[str, str] = (0, 1),
        columns: tuple[str, str] | None = None,
) -> pl.Expr:
    r"""Take the norm of a 2D series.

    The norm is defined by :math:`\sqrt{x^2 + y^2}` with :math:`x` being the yaw component and
    :math:`y` being the pitch component of a coordinate.

    Parameters
    ----------
    column: str | None
        Take the norm of the ``components`` of this column with nested data (list or struct).
        This argumment is mutually exclusive with the ``columns`` argument.
        (default: ``None``)
    components: tuple[int, int] | tuple[str, str]
        If ``column`` is provided, take the norm of these two components in the specified nested
        ``column``. If the tuple elements are of type ``int`` the nested column dtype is to be
        assumed as ``polars.List`` and child elements with that indices are used for taking the
        norm. If the tuple elements are of type ``str`` the nested column dtype is to be assumed as
        ``polars.Struct`` and the fields with these names are used for taking the norm.
        (default: ``(0, 1)``)
    columns: tuple[str, str] | None
        Two columns to take the norm of. This is mutually exclusive with the ``column`` argument.
        (default: ``None``)

    Returns
    -------
    pl.Expr
        The respective polars expression.
    """
    _checks.check_is_mutual_exclusive(column=column, columns=columns)

    if columns is not None:  # norm of two columns
        x = pl.col(columns[0])
        y = pl.col(columns[1])
    elif column is not None and isinstance(components, Sequence):
        if len(components) != 2:
            raise ValueError(f'components must be of length 2 but is {len(components)}')

        if all(isinstance(component, int) for component in components):  # assume pl.List column
            x = pl.col(column).list.get(components[0])
            y = pl.col(column).list.get(components[1])
        elif all(isinstance(component, str) for component in components):  # assume pl.Struct column
            x = pl.col(column).struct.field(components[0])
            y = pl.col(column).struct.field(components[1])
        else:
            raise TypeError(
                "elements of 'components' must be either of type int or str but they are "
                f'({type(components[0]).__name__}, {type(components[1]).__name__})',
            )
    elif column is not None:  # not a sequence, unexpected type
        raise TypeError(
            f"'components' must be a sequence but is of type {type(components).__name__}",
        )
    else:
        raise TypeError('either column or columns must be provided but both are None')
    return (x.pow(2) + y.pow(2)).sqrt()
