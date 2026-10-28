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
"""Module for py:func:`pymovements.transforms.nullify_out_of_bounds`."""
from __future__ import annotations

import polars as pl

from pymovements.transforms.library import register_transform


@register_transform
def nullify_out_of_bounds(
        lower_bound: int | float | None,
        upper_bound: int | float | None,
        *,
        input_column: str,
        output_column: str,
        n_components: int,
) -> pl.Expr:
    """Set gaze samples with any component outside a lower and upper bound to null.

    In contrast to :func:`~pymovements.transforms.clip`, which replaces out-of-bound components by
    the bound, a sample with any component outside the bounds is set to null as a whole.
    Samples with null components keep their existing nulls. Only ``output_column`` is affected.
    Other signal columns of the same sample are untouched, see
    :py:meth:`~pymovements.Gaze.nullify_event_samples` for nulling every signal column.

    Parameters
    ----------
    lower_bound : int | float | None
        Lower bound of the valid range. ``None`` means no lower bound.
    upper_bound : int | float | None
        Upper bound of the valid range. ``None`` means no upper bound.
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
    out_of_bounds = []
    for component in range(n_components):
        value = pl.col(input_column).list.get(component)
        if lower_bound is not None:
            out_of_bounds.append(value < lower_bound)
        if upper_bound is not None:
            out_of_bounds.append(value > upper_bound)

    if not out_of_bounds:
        return pl.col(input_column).alias(output_column)

    return (
        pl.when(pl.any_horizontal(out_of_bounds))
        .then(None)
        .otherwise(pl.col(input_column))
        .alias(output_column)
    )
