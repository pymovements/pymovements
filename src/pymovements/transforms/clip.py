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

import polars as pl

from pymovements.transforms.library import register_transform


@register_transform
def clip(
        lower_bound: int | float | None,
        upper_bound: int | float | None,
        *,
        input_column: str,
        output_column: str,
        n_components: int,
) -> pl.Expr:
    """Clip gaze signal to a lower and upper bound.

    Parameters
    ----------
    lower_bound : int | float | None
        Lower bound of the clipped column.
    upper_bound : int | float | None
        Upper bound of the clipped column.
    input_column : str
        Name of the input column.
    output_column : str
        Name of the output column.
    n_components : int
        Number of components in input column.

    Returns
    -------
    pl.Expr
        The respective polars expression.
    """
    return pl.concat_list(
        [
            pl.col(input_column).list.get(component).clip(lower_bound, upper_bound)
            for component in range(n_components)
        ],
    ).alias(output_column)
