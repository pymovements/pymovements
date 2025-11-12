# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Module for the TextDataFrame."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from pymovements._utils import _checks
from pymovements._utils._html import repr_html


@repr_html(['aois'])
class TextStimulus:
    """A DataFrame for the text stimulus that the gaze data was recorded on.

    Parameters
    ----------
    aois: pl.DataFrame
        A stimulus dataframe.
    aoi_column: str
        Name of the column that contains the content of the aois.
    start_x_column: str
        Name of the column which contains the x coordinate's start position of the
        areas of interest.
    start_y_column: str
        Name of the column which contains the y coordinate's start position of the
        areas of interest.
    width_column: str | None
        Name of the column which contains the width of the area of interest. (default: None)
    height_column: str | None
        Name of the column which contains the height of the area of interest. (default: None)
    end_x_column: str | None
        Name of the column which contains the x coordinate's end position of the areas of interest.
        (default: None)
    end_y_column: str | None
        Name of the column which contains the y coordinate's end position of the areas of interest.
        (default: None)
    page_column: str | None
        Name of the column which contains the page information of the area of interest.
        (default: None)
    """

    def __init__(
            self,
            aois: pl.DataFrame,
            *,
            aoi_column: str,
            start_x_column: str,
            start_y_column: str,
            width_column: str | None = None,
            height_column: str | None = None,
            end_x_column: str | None = None,
            end_y_column: str | None = None,
            page_column: str | None = None,
    ) -> None:

        self.aois = aois.clone()
        self.aoi_column = aoi_column
        self.width_column = width_column
        self.height_column = height_column
        self.start_x_column = start_x_column
        self.start_y_column = start_y_column
        self.end_x_column = end_x_column
        self.end_y_column = end_y_column
        self.page_column = page_column

    def split(
            self,
            by: str | Sequence[str],
    ) -> list[TextStimulus]:
        """Split the AOI df.

        Parameters
        ----------
        by: str | Sequence[str]
            Splitting criteria.

        Returns
        -------
        list[TextStimulus]
            A list of TextStimulus objects.
        """
        return [
            TextStimulus(
                aois=df,
                aoi_column=self.aoi_column,
                width_column=self.width_column,
                height_column=self.height_column,
                start_x_column=self.start_x_column,
                start_y_column=self.start_y_column,
                end_x_column=self.end_x_column,
                end_y_column=self.end_y_column,
                page_column=self.page_column,
            )
            for df in self.aois.partition_by(by=by, as_dict=False)
        ]

    def get_aoi(
            self,
            *,
            row: pl.DataFrame.row,
            x_eye: str,
            y_eye: str,
    ) -> pl.DataFrame:
        """Given eye movement and aoi dataframe, return aoi.

        If `width` is used, calculation: start_x_column <= x_eye < start_x_column + width.
        If `end_x_column` is used, calculation: start_x_column <= x_eye < end_x_column.
        Analog for y coordinate and height.

        Parameters
        ----------
        row: pl.DataFrame.row
            Eye movement row.
        x_eye: str
            Name of x eye coordinate.
        y_eye: str
            Name of y eye coordinate.

        Returns
        -------
        pl.DataFrame
            Looked at area of interest.

        Raises
        ------
        ValueError
            If width and end_TYPE_column is None.
        """
        return _get_aoi(self, row=row, x_eye=x_eye, y_eye=y_eye)


def from_file(
        aoi_path: str | Path,
        *,
        aoi_column: str,
        start_x_column: str,
        start_y_column: str,
        width_column: str | None = None,
        height_column: str | None = None,
        end_x_column: str | None = None,
        end_y_column: str | None = None,
        page_column: str | None = None,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> TextStimulus:
    """Load text stimulus from file.

    Parameters
    ----------
    aoi_path:  str | Path
        Path to file to be read.
    aoi_column: str
        Name of the column that contains the content of the aois.
    start_x_column: str
        Name of the column which contains the x coordinate's start position of the
        areas of interest.
    start_y_column: str
        Name of the column which contains the y coordinate's start position of the
        areas of interest.
    width_column: str | None
        Name of the column which contains the width of the area of interest. (default: None)
    height_column: str | None
        Name of the column which contains the height of the area of interest. (default: None)
    end_x_column: str | None
        Name of the column which contains the x coordinate's end position of the areas of interest.
        (default: None)
    end_y_column: str | None
        Name of the column which contains the y coordinate's end position of the areas of interest.
        (default: None)
    page_column: str | None
        Name of the column which contains the page information of the area of interest.
        (default: None)
    custom_read_kwargs: dict[str, Any] | None
        Custom read keyword arguments for polars. (default: None)


    Returns
    -------
    TextStimulus
        Returns the text stimulus file.
    """
    if isinstance(aoi_path, str):
        aoi_path = Path(aoi_path)
    if custom_read_kwargs is None:
        custom_read_kwargs = {}

    valid_extensions = {'.csv', '.tsv', '.txt', '.ias'}
    if aoi_path.suffix in valid_extensions:
        stimulus_df = pl.read_csv(
            aoi_path,
            **custom_read_kwargs,
        )
        stimulus_df = stimulus_df.fill_null(' ')
    else:
        raise ValueError(
            f'unsupported file format "{aoi_path.suffix}".'
            f'Supported formats are: {sorted(valid_extensions)}',
        )

    return TextStimulus(
        aois=stimulus_df,
        aoi_column=aoi_column,
        start_x_column=start_x_column,
        start_y_column=start_y_column,
        width_column=width_column,
        height_column=height_column,
        end_x_column=end_x_column,
        end_y_column=end_y_column,
        page_column=page_column,
    )


def _get_aoi(
        aoi_dataframe: TextStimulus,
        row: pl.DataFrame.row,
        x_eye: str,
        y_eye: str,
) -> pl.DataFrame:
    """Given eye movement and aoi dataframe, return aoi.

    If `width` is used, calculation: start_x_column <= x_eye < start_x_column + width.
    If `end_x_column` is used, calculation: start_x_column <= x_eye < end_x_column.
    Analog for y coordinate and height.

    .. deprecated:: v0.21.1
       Please use :py:meth:`~pymovements.TextStimulus.get_aoi()` instead.
       This function will be removed in v0.26.0.

    Parameters
    ----------
    aoi_dataframe: TextStimulus
        Text dataframe to containing area of interests.
    row: pl.DataFrame.row
        Eye movement row.
    x_eye: str
        Name of x eye coordinate.
    y_eye: str
        Name of y eye coordinate.

    Returns
    -------
    pl.DataFrame
        Looked at area of interest.

    Raises
    ------
    ValueError
        If width and end_TYPE_column is None.
    """
    if aoi_dataframe.width_column is not None:
        _checks.check_is_none_is_mutual(
            height_column=aoi_dataframe.width_column,
            width_column=aoi_dataframe.height_column,
        )
        aoi = aoi_dataframe.aois.filter(
            (aoi_dataframe.aois[aoi_dataframe.start_x_column] <= row[x_eye]) &
            (
                row[x_eye] <
                aoi_dataframe.aois[aoi_dataframe.start_x_column] +
                aoi_dataframe.aois[aoi_dataframe.width_column]
            ) &
            (aoi_dataframe.aois[aoi_dataframe.start_y_column] <= row[y_eye]) &
            (
                row[y_eye] <
                aoi_dataframe.aois[aoi_dataframe.start_y_column] +
                aoi_dataframe.aois[aoi_dataframe.height_column]
            ),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))
        return aoi

    if aoi_dataframe.end_x_column is not None:
        _checks.check_is_none_is_mutual(
            end_x_column=aoi_dataframe.end_x_column,
            end_y_column=aoi_dataframe.end_y_column,
        )
        aoi = aoi_dataframe.aois.filter(
            # x-coordinate: within bounding box
            (aoi_dataframe.aois[aoi_dataframe.start_x_column] <= row[x_eye]) &
            (row[x_eye] < aoi_dataframe.aois[aoi_dataframe.end_x_column]) &
            # y-coordinate: within bounding box
            (aoi_dataframe.aois[aoi_dataframe.start_y_column] <= row[y_eye]) &
            (row[y_eye] < aoi_dataframe.aois[aoi_dataframe.end_y_column]),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))

        return aoi
    raise ValueError(
        'either TextStimulus.width or TextStimulus.end_x_column must be defined',
    )
