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

import math
import warnings
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
    trial_column: str | None
        Name for the column that specifies the unique trial id.
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
            trial_column: str | None = None,
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
        self.trial_column = trial_column

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
                trial_column=self.trial_column,
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
        """Return the AOI that contains the given gaze row.

        This function checks spatial bounds using the interval `start <= coord < start + size` if
        `width`/`height` are provided, or `start <= coord < end` if `end_x_column`/`end_y_column`
        are provided.
        In both cases, the end boundary is exclusive (half-open interval `[start, end)`).
        When `trial_column` and/or `page_column` are configured,
        AOIs are first filtered to match the current row's values for these columns,
        which are then dropped to avoid duplicate columns during concatenation.

        Parameters
        ----------
        row: pl.DataFrame.row
            Eye movement row with fields for the eye coordinates and any trial/page identifiers.
        x_eye: str
            Name of the x eye coordinate field in ``row``.
        y_eye: str
            Name of the y eye coordinate field in ``row``.

        Returns
        -------
        pl.DataFrame
            A one-row DataFrame representing the matched AOI. If no AOI matches, the result is a
            single row with ``None`` values in the AOI columns.

        Raises
        ------
        ValueError
            If neither width/height nor end_x/end_y columns are defined to specify AOI bounds.

        Notes
        -----
        If multiple AOIs overlap and match the same point, a `UserWarning` is emitted.
        For invalid or missing coordinates (e.g. `None` or strings), a `UserWarning` is emitted,
        and the lookup returns a single row of `None` values.

        """
        return _get_aoi(self, row=row, x_eye=x_eye, y_eye=y_eye)


def _is_number(v: Any) -> bool:
    """Return True if v is an int/float and not NaN."""
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _empty_aoi_like(df: pl.DataFrame) -> pl.DataFrame:
    """Create a single-row AOI DataFrame with None for each column of df."""
    return pl.from_dict({col: None for col in df.columns})


def _extract_valid_xy_or_none(
    row: pl.DataFrame.row,
    x_eye: str,
    y_eye: str,
) -> tuple[float, float] | None:
    """Extract numeric x,y from row or return None after warning if invalid.

    Emits a UserWarning exactly once per invalid input pair.
    """
    x_val = row.get(x_eye)
    y_val = row.get(y_eye)
    if not (_is_number(x_val) and _is_number(y_val)):
        warnings.warn(
            f'Invalid eye coordinates (x={x_val}, y={y_val}) for AOI lookup. '
            'Returning no match.',
            UserWarning,
        )
        return None
    return float(x_val), float(y_val)


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
        trial_column: str | None = None,
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
    trial_column: str | None
        Name fo the column that specifies the unique trial id.
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
        trial_column=trial_column,
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
    row_aois = aoi_dataframe.aois
    # Filter AOIs to the same trial/page as the current row (if those columns are defined).
    # After filtering, drop these key columns from the temporary AOI selection to avoid
    # duplicate columns later when concatenating AOI properties back to event/gaze frames.
    if aoi_dataframe.trial_column is not None:
        trial_val = row.get(aoi_dataframe.trial_column)
        if trial_val is not None:
            row_aois = row_aois.filter(
                row_aois[aoi_dataframe.trial_column] == trial_val,
            )
    if aoi_dataframe.page_column is not None:
        page_val = row.get(aoi_dataframe.page_column)
        if page_val is not None:
            row_aois = row_aois.filter(
                row_aois[aoi_dataframe.page_column] == page_val,
            )

    if aoi_dataframe.width_column is not None:
        _checks.check_is_none_is_mutual(
            height_column=aoi_dataframe.width_column,
            width_column=aoi_dataframe.height_column,
        )
        # Validate and extract numeric coordinates once.
        xy = _extract_valid_xy_or_none(row, x_eye, y_eye)
        if xy is None:
            return _empty_aoi_like(row_aois)
        x_val, y_val = xy

        aoi = row_aois.filter(
            (row_aois[aoi_dataframe.start_x_column] <= x_val) &
            (
                x_val <
                row_aois[aoi_dataframe.start_x_column] +
                row_aois[aoi_dataframe.width_column]
            ) &
            (row_aois[aoi_dataframe.start_y_column] <= y_val) &
            (
                y_val <
                row_aois[aoi_dataframe.start_y_column] +
                row_aois[aoi_dataframe.height_column]
            ),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))
            return aoi
        # If multiple AOIs overlap, warn
        if aoi.height > 1:
            warnings.warn(
                'Multiple AOIs matched this point '
                f'(x={x_val}, y={y_val}).',
                UserWarning,
            )
        return aoi

    if aoi_dataframe.end_x_column is not None:
        _checks.check_is_none_is_mutual(
            end_x_column=aoi_dataframe.end_x_column,
            end_y_column=aoi_dataframe.end_y_column,
        )
        # Validate and extract numeric coordinates once.
        xy = _extract_valid_xy_or_none(row, x_eye, y_eye)
        if xy is None:
            return _empty_aoi_like(row_aois)
        x_val, y_val = xy

        aoi = row_aois.filter(
            # x-coordinate: within bounding box
            (row_aois[aoi_dataframe.start_x_column] <= x_val) &
            (x_val < row_aois[aoi_dataframe.end_x_column]) &
            # y-coordinate: within bounding box
            (row_aois[aoi_dataframe.start_y_column] <= y_val) &
            (y_val < row_aois[aoi_dataframe.end_y_column]),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))
            return aoi

        # If multiple AOIs overlap, warn
        if aoi.height > 1:
            warnings.warn(
                'Multiple AOIs matched this point '
                f'(x={x_val}, y={y_val}).',
                UserWarning,
            )

        return aoi
    raise ValueError(
        'either TextStimulus.width or TextStimulus.end_x_column must be defined',
    )
