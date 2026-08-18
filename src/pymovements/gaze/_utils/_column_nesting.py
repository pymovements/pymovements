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
"""Utility functions for working with nested columns in gaze dataframes."""
from __future__ import annotations

from warnings import warn

import polars

from pymovements._utils._checks import check_is_mutual_exclusive


def unnest_list_columns(
        df: polars.DataFrame,
        input_columns: list[str] | str | None = None,
        *,
        output_suffixes: list[str] | None = None,
        output_columns: list[str] | None = None,
) -> polars.DataFrame:
    """Explode a column of type ``polars.List`` into one column for each list component.

    The unnested input columns are dropped from the returned dataframe.

    Parameters
    ----------
    df: polars.DataFrame
        Unnest columns from that dataframe.
    input_columns: list[str] | str | None
        Name(s) of input column(s) to be unnested into several component columns.
        If None, all list columns will be unnested if existing. (default: None)
    output_suffixes: list[str] | None
        Suffixes to append to the column names. (default: None)
    output_columns: list[str] | None
        Name of the resulting tuple columns. (default: None)

    Returns
    -------
    polars.DataFrame
        Dataframe with unnested columns. Unnested columns are dropped.

    Raises
    ------
    ValueError
        If both output_columns and output_suffixes are specified.
        If number of output columns / suffixes does not match number of components.
        If output columns / suffixes are not unique.
        If no columns to unnest exist and none are specified.
        If output columns are specified and more than one input column is specified.
    AttributeError
        If number of components is not 2, 4 or 6.
    Warning
        If no columns to unnest exist and none are specified.
    """
    if input_columns is None:
        input_columns = [column for column in df.columns if df[column].dtype == polars.List]

        if len(input_columns) == 0:
            warn(
                'No columns to unnest. '
                'Please specify columns to unnest via the "input_columns" argument.',
            )

    if isinstance(input_columns, str):
        input_columns = [input_columns]

    check_is_mutual_exclusive(
        output_columns=output_columns,
        output_suffixes=output_suffixes,
    )

    column_map = {}
    if output_columns:
        # no support for custom output columns if more than one input column will be unnested
        if not len(input_columns) == 1:
            raise ValueError(
                'You cannot specify output columns if you want to unnest more than '
                'one input column. Please specify output suffixes or use a single '
                'input column instead.',
            )
        if len({*output_columns}) != len(output_columns):
            raise ValueError('Output columns must be unique')
        column_map = {input_columns[0]: output_columns}
    elif output_suffixes is None:
        # Dynamically infer component suffixes.
        column_map = {
            input_column: [
                input_column + output_suffix
                for output_suffix in _infer_list_unnest_suffixes(df[input_column])
            ]
            for input_column in input_columns
        }
    else:  # explicit output_suffixes
        if len({*output_suffixes}) != len(output_suffixes):
            raise ValueError('Output suffixes must be unique')
        column_map = {
            input_column: [input_column + output_suffix for output_suffix in output_suffixes]
            for input_column in input_columns
        }

    for input_column, _output_columns in column_map.items():
        n_components = _infer_list_n_components(df[input_column])
        if len(_output_columns) != n_components:
            raise ValueError(
                f"Number of output columns for column '{input_column}' ({_output_columns}) "
                f'must match number of components ({n_components})',
            )

        df = df.with_columns(
            [
                polars.col(input_column).list.get(component_id).alias(output_column)
                for component_id, output_column in enumerate(_output_columns)
            ],
        )
    df = df.drop(input_columns)
    return df


def get_nested_columns(df: polars.DataFrame) -> list[str]:
    """Get column names of nested columns."""
    return [column for column in df.columns if df[column].dtype == polars.List]


def _infer_list_n_components(series: polars.Series) -> int:
    """Dynamically infer number of list components in series."""
    n_component_candidates = series.list.len().unique()
    if len(n_component_candidates) != 1:
        raise ValueError(
            'number of components inconsistent in column '
            f"'{series.name}': {n_component_candidates}",
        )
    return n_component_candidates[0]


def _infer_list_unnest_suffixes(series: polars.Series) -> list[str]:
    """Dynamically infer component suffixes from series.

    Number of components must be either 2, 4 or 6:

    - 2 components: ``_x``, ``_y``
    - 4 components: ``_xl``, ``_yl``, ``_xr``, ``_yr``
    - 6 components: ``_xl``, ``_yl``, ``_xr``, ``_yr``, ``_xa``, ``_ya``
    """
    n_components = _infer_list_n_components(series)
    if n_components not in {2, 4, 6}:
        raise ValueError(
            'Inferring suffixes only possible for list lengths of 2, 4 or 6,'
            f" but list length of column '{series.name}' is: {n_components}.",
        )
    if n_components == 2:
        return ['_x', '_y']
    if n_components == 4:
        return ['_xl', '_yl', '_xr', '_yr']
    # This must be 6 as we already have checked our n_components.
    return ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya']
