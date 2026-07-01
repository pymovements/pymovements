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

from __future__ import annotations

import polars

from pymovements.transforms.library import register_transform


@register_transform
def value_impossible(
        maxi: int | float,
        *,
        mode: str = 'null',
        input_column: str,
        output_column: str,
        num: int,
) -> polars.Expr:

"""Transform gaze data by handling impossible values exceeding a maximum threshold.

	Parameters
	----------
	maxi : int | float
		The maximum allowed absolute value.
	mode : str, optional
		The handling mode. 'null' replaces out-of-bounds values with None,
		'clip' truncates values to [-maxi, maxi]. Default is 'null'.
	input_column : str
		The name of the input column containing the list components.
	output_column : str
		The name of the resulting output column.
	num : int
		The number of elements/dimensions in the input list column to process.

	Returns
	-------
	polars.Expr
		A Polars expression performing the impossible value transformation.

	Raises
	------
	ValueError
		If the provided mode is not 'null' or 'clip'.
	"""

   if mode != 'null' and mode != 'clip':
        raise ValueError("Invalid mode. Expected 'null' or 'clip'.")

    expr = []
    for e in range(num):
        col_e = polars.col(input_column).list.get(e)

        if mode == 'clip':
            res = col_e.clip(-maxi, maxi)
        else:
            res = polars.when(col_e.abs() <= maxi).then(col_e).otherwise(None)

        expr.append(res)
    return polars.concat_list(expr).alias(output_column)
