# Copyright (c) 2024-2025 The pymovements Project Authors
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
"""AOI utilities.

.. deprecated:: v0.21.1
   Please use :py:meth:`~pymovements.TextStimulus.get_aoi()` instead.
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

import polars as pl
from deprecated.sphinx import deprecated

import pymovements as pm


@deprecated(
    reason='Please use TextStimulus.get_aoi() instead. '
           'This function will be removed in v0.26.0.',
    version='v0.21.1',
)
def get_aoi(
        aoi_dataframe: pm.stimulus.TextStimulus,
        row: pl.DataFrame.row,
        x_eye: str,
        y_eye: str,
) -> str:
    """Given eye movement and aoi dataframe, return aoi.

    If `width` is used, calculation: start_x_column <= x_eye < start_x_column + width.
    If `end_x_column` is used, calculation: start_x_column <= x_eye < end_x_column.
    Analog for y coordinate and height.

    .. deprecated:: v0.21.1
       Please use :py:meth:`~pymovements.TextStimulus.get_aoi()` instead.
       This function will be removed in v0.26.0.

    Parameters
    ----------
    aoi_dataframe: pm.stimulus.TextStimulus
        Text dataframe to containing area of interests.
    row: pl.DataFrame.row
        Eye movement row.
    x_eye: str
        Name of x eye coordinate.
    y_eye: str
        Name of y eye coordinate.

    Returns
    -------
    str
        Looked at area of interest.

    Raises
    ------
    ValueError
        If width and end_TYPE_column is None.
    """
    return aoi_dataframe.get_aoi(row=row, x_eye=x_eye, y_eye=y_eye)
