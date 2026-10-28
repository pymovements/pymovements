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
"""Provides helpers for dropping rows with null values from dataframes."""
from __future__ import annotations

from typing import Literal

import polars as pl


def row_is_null(
        schema: pl.Schema,
        subset: list[str],
        how: Literal['all', 'any'],
) -> pl.Expr:
    """Build a boolean expression that is true for rows counting as null.

    A scalar column counts as null if its value is null. A nested list column (e.g. ``pixel`` or
    ``position``) counts as null if any of its components is null under ``how='any'``, and only if
    all of its components are null under ``how='all'``. For an empty subset, no row counts as
    null, matching :py:meth:`polars.DataFrame.drop_nulls` with an empty subset.

    Parameters
    ----------
    schema: pl.Schema
        Schema of the dataframe to build the expression for, used to look up column dtypes.
    subset: list[str]
        List of column names to check for null values.
    how: Literal['all', 'any']
        If 'any', the expression is true for rows where any of the specified columns counts as
        null. If 'all', the expression is true for rows where all of the specified columns count
        as null.

    Returns
    -------
    pl.Expr
        Boolean expression to be used in a filter or remove context.

    Raises
    ------
    ValueError
        If ``how`` is neither 'any' nor 'all'.
    """
    if how not in ('any', 'all'):
        raise ValueError(f"how must be either 'any' or 'all' but is '{how}'")

    if not subset:
        return pl.lit(False)

    column_null_expressions = []
    for column in subset:
        if isinstance(schema[column], pl.List):
            component_nulls = pl.col(column).list.eval(pl.element().is_null())
            if how == 'any':
                column_counts_as_null = component_nulls.list.any()
            else:
                column_counts_as_null = component_nulls.list.all()
            column_null_expressions.append(pl.col(column).is_null() | column_counts_as_null)
        else:
            column_null_expressions.append(pl.col(column).is_null())

    if how == 'any':
        return pl.any_horizontal(column_null_expressions)
    return pl.all_horizontal(column_null_expressions)
