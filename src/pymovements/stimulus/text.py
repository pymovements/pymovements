# Copyright (c) 2023-2026 The pymovements Project Authors
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
from dataclasses import dataclass
from dataclasses import KW_ONLY
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Literal

import polars as pl

from pymovements._utils import _checks
from pymovements._utils._html import repr_html
from pymovements.stimulus import _lines


@dataclass(frozen=True)
@repr_html()
class WritingSystem:
    """Writing system specification for text stimuli.

    Attributes
    ----------
    directionality: Literal['left-to-right', 'right-to-left', 'top-to-bottom']
        Direction in which text flows within a line.
        For horizontal text, this is typically 'left-to-right' or 'right-to-left'.
        For vertical text, this is typically 'top-to-bottom'.
        Bidirectional/Boustrophedon scripts (e.g., Arabic with embedded English)
        is currently not supported.
        (default: 'left-to-right')
    axis: Literal['horizontal', 'vertical']
        Primary axis along which text is laid out.
        (default: 'horizontal')
    lining: Literal['top-to-bottom', 'left-to-right', 'right-to-left']
        Direction in which lines of text are stacked.
        For horizontal text, this is typically 'top-to-bottom'.
        For vertical text, this is typically 'left-to-right' or 'right-to-left'.
        (default: 'top-to-bottom')
    DESCRIPTORS: ClassVar[tuple[str, ...]]
        Valid descriptor strings for :meth:`from_descriptor`:

        - ``'left-to-right'``, ``ltr``
        - ``'right-to-left'``, ``rtl``

    Examples
    --------
    Typical configurations are:

    * Horizontal left-to-right (LTR)::

        WritingSystem(
            directionality='left-to-right',
            axis='horizontal',
            lining='top-to-bottom',
        )

    * Horizontal right-to-left (RTL)::

        WritingSystem(
            directionality='right-to-left',
            axis='horizontal',
            lining='top-to-bottom',
        )

    * Vertical right-to-left columns::

        WritingSystem(
            directionality='top-to-bottom',
            axis='vertical',
            lining='right-to-left',
        )

    * Vertical left-to-right columns::

        WritingSystem(
            directionality='top-to-bottom',
            axis='vertical',
            lining='left-to-right',
        )
    """

    directionality: Literal['left-to-right', 'right-to-left', 'top-to-bottom'] = 'left-to-right'
    _: KW_ONLY
    axis: Literal['horizontal', 'vertical'] = 'horizontal'
    lining: Literal['top-to-bottom', 'left-to-right', 'right-to-left'] = 'top-to-bottom'

    DESCRIPTORS: ClassVar[tuple[str, ...]] = (
        'left-to-right',
        'ltr',
        'right-to-left',
        'rtl',
    )

    @staticmethod
    def from_descriptor(descriptor: str) -> WritingSystem:
        """Create a WritingSystem instance from a descriptor string.

        Mapping is as follows:

        - ``left-to-right``, ``ltr``: vertical left to right
        - ``right-to-left``, ``rtl``: vertical right to left

        Parameters
        ----------
        descriptor: str
            The descriptor string. Valid values are found in ``WritingSystem.DESCRIPTORS``.

        Returns
        -------
        WritingSystem
            The corresponding WritingSystem instance.

        Examples
        --------
        Vertical left to right:

        >>> WritingSystem.from_descriptor('left-to-right')
        WritingSystem(directionality='left-to-right', axis='horizontal', lining='top-to-bottom')

        Abbreviations are also supported:

        >>> WritingSystem.from_descriptor('ltr')
        WritingSystem(directionality='left-to-right', axis='horizontal', lining='top-to-bottom')

        and

        >>> WritingSystem.from_descriptor('rtl')
        WritingSystem(directionality='right-to-left', axis='horizontal', lining='top-to-bottom')
        """
        descriptor = descriptor.lower()
        if descriptor in {'left-to-right', 'ltr'}:
            return WritingSystem(
                directionality='left-to-right', axis='horizontal', lining='top-to-bottom',
            )
        if descriptor in {'right-to-left', 'rtl'}:
            return WritingSystem(
                directionality='right-to-left', axis='horizontal', lining='top-to-bottom',
            )
        raise ValueError(
            f"Unknown descriptor '{descriptor}'. "
            f'Valid descriptors are: {WritingSystem.DESCRIPTORS}',
        )


@repr_html(['aois', 'metadata'])
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
    writing_system: WritingSystem | str
        Writing system of the text. If ``writing_system`` is a string,
        :py:meth:`~pymovements.stimulus.WritingSystem.from_descriptor()` is used for initialization.
        (default: ``'left-to-right'``)
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata. (default: None)
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
            writing_system: WritingSystem | str = 'left-to-right',
            metadata: dict[str, Any] | None = None,
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
        self.metadata = metadata if metadata is not None else {}

        if isinstance(writing_system, str):
            self.writing_system = WritingSystem.from_descriptor(writing_system)
        else:
            self.writing_system = writing_system

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
                writing_system=self.writing_system,
            )
            for df in self.aois.partition_by(by, as_dict=False)
        ]

    def with_line_idx(
            self,
            *,
            line_column: str = 'line_idx',
            tolerance: float = 0.5,
            overwrite: bool = False,
    ) -> TextStimulus:
        """Return a copy of this stimulus whose AOIs carry a text line index.

        Lines are numbered ``0..k-1`` from the top. AOIs are grouped by their vertical centre:
        a new line starts where the gap to the previous centre exceeds ``tolerance`` line
        heights, the line height being the median AOI height. Grouping on the centre rather
        than on the top edge tolerates boxes of different height inside one line; whether a
        corpus has such boxes is another matter, and the two we checked do not.

        The practical reason for this method is a different one: nothing in pymovements writes
        a ``line_idx`` column today, so
        :py:func:`~pymovements.events.correction.correct_fixations` always falls back to
        grouping AOIs by an exactly equal top edge.

        If ``line_column`` already exists it is validated and kept, unless ``overwrite`` is
        set. Datasets that publish a line number of their own usually count from one; convert
        it to a zero-based index before passing it in.

        Adding this column also affects
        :py:func:`~pymovements.events.correction.correct_fixations`, which groups AOIs into
        lines by ``line_idx`` when the column is present and falls back to exact top-edge
        equality when it is not.

        Parameters
        ----------
        line_column: str
            Name of the line index column. (default: 'line_idx')
        tolerance: float
            Maximum gap within one line, in line heights. The upper bound that matters is the
            distance between two lines divided by the line height; above it, neighbouring lines
            merge. Measured over the 600 character-AOI layouts of PoTeC and EMTeC, every value
            from 0.001 to 0.9 gives the same line count as the corpus itself, 0.95 merges lines
            in 30 EMTeC layouts and 1.0 in all of them. (default: 0.5)
        overwrite: bool
            Recompute ``line_column`` even if it already exists. (default: False)

        Returns
        -------
        TextStimulus
            A new stimulus whose ``aois`` carry an integer ``line_column``.

        Raises
        ------
        ValueError
            If the stimulus holds AOIs of more than one page or trial, if ``tolerance`` is not
            positive, if the AOIs are empty, or if an existing ``line_column`` is not numbered
            ``0..k-1`` without gaps.

        Examples
        --------
        >>> import polars as pl
        >>> from pymovements.stimulus import TextStimulus
        >>> aois = pl.DataFrame({
        ...     'char': ['a', 'b', 'c', 'd'],
        ...     'x0': [0, 10, 0, 10], 'y0': [0, 0, 20, 20],
        ...     'x1': [10, 20, 10, 20], 'y1': [10, 10, 30, 30],
        ... })
        >>> stimulus = TextStimulus(
        ...     aois, aoi_column='char',
        ...     start_x_column='x0', start_y_column='y0',
        ...     end_x_column='x1', end_y_column='y1',
        ... )
        >>> stimulus.with_line_idx().aois['line_idx'].to_list()
        [0, 0, 1, 1]
        """
        self._check_single_page()

        aois = self.aois
        if line_column in aois.columns and not overwrite:
            present = aois[line_column].drop_nulls().cast(pl.Int64).unique().sort().to_list()
            if present != list(range(len(present))):
                shown = present[:5]
                raise ValueError(
                    f"column '{line_column}' must be numbered 0..k-1 without gaps, got "
                    f"{shown}{'...' if len(present) > 5 else ''}. Convert a dataset-specific "
                    'line numbering before passing it in, or use overwrite=True to recompute.',
                )
            aois = aois.with_columns(pl.col(line_column).cast(pl.Int64))
        else:
            bounds = _lines.normalized_bounds(self)
            dropped = aois.height - bounds.height
            if dropped:
                warnings.warn(
                    f'{dropped} of {aois.height} areas of interest have a non-finite '
                    'coordinate or a height of zero or less; they get no line index. Check '
                    f'the {self.start_y_column}/{self.end_y_column} columns of the stimulus.',
                    UserWarning,
                    stacklevel=2,
                )
            # Written back by the row each rectangle came from, not by position: the dropped
            # ones would otherwise shift every line index below them.
            derived = bounds.select(
                pl.col(_lines.AOI_ROW_COLUMN),
                _lines.derive_line_idx(bounds, tolerance).alias(line_column),
            )
            aois = (
                aois
                .drop(line_column, strict=False)  # overwrite=True: the old column goes
                .with_row_index(_lines.AOI_ROW_COLUMN)
                .join(derived, on=_lines.AOI_ROW_COLUMN, how='left', maintain_order='left')
                .drop(_lines.AOI_ROW_COLUMN)
            )

        return self._with_aois(aois)

    def _check_single_page(self) -> None:
        """Raise if this stimulus mixes AOIs of several pages or trials.

        Raises
        ------
        ValueError
            If ``page_column`` or ``trial_column`` holds more than one distinct value.
        """
        for column in (self.trial_column, self.page_column):
            if column is not None and column in self.aois.columns:
                if self.aois[column].n_unique() > 1:
                    raise ValueError(
                        f"stimulus holds AOIs of several values of '{column}'; "
                        f"use split('{column}') and handle each page separately",
                    )

    def _with_aois(self, aois: pl.DataFrame) -> TextStimulus:
        """Return a copy of this stimulus with a different AOI dataframe.

        Parameters
        ----------
        aois: pl.DataFrame
            The AOI dataframe of the copy.

        Returns
        -------
        TextStimulus
            A copy carrying ``aois`` and this stimulus' column configuration.
        """
        return TextStimulus(
            aois=aois,
            aoi_column=self.aoi_column,
            width_column=self.width_column,
            height_column=self.height_column,
            start_x_column=self.start_x_column,
            start_y_column=self.start_y_column,
            end_x_column=self.end_x_column,
            end_y_column=self.end_y_column,
            page_column=self.page_column,
            trial_column=self.trial_column,
            writing_system=self.writing_system,
            metadata=dict(self.metadata) if self.metadata else None,
        )

    def get_aoi(
            self,
            *,
            row: pl.DataFrame.row,
            x_eye: str,
            y_eye: str,
            max_matches: int | None = None,
    ) -> pl.DataFrame:
        """Return the AOI that contains the given gaze row.

        This function checks spatial bounds using the interval `start <= coord < start + size` if
        `width`/`height` is provided, or `start <= coord < end` if `end_x_column`/`end_y_column`
        is provided.
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
        max_matches: int | None
            If not ``None``, reduce the number of matches to this amount. This may happen in case of
            overlapping AOIs.
            (default: None)

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
        aois = _get_aoi(self, row=row, x_eye=x_eye, y_eye=y_eye)
        if max_matches:
            aois = aois.head(max_matches)
        return aois

    @staticmethod
    def from_csv(
            path: str | Path,
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
            writing_system: WritingSystem | str = 'left-to-right',
            metadata: dict[str, Any] | None = None,
            read_csv_kwargs: dict[str, Any] | None = None,
    ) -> TextStimulus:
        """Load text stimulus from file.

        Parameters
        ----------
        path:  str | Path
            Path to file to be read.
        aoi_column: str
            Name of column that contains the content of the aois.
        start_x_column: str
            Name of column which contains the x coordinate's start position of the
            areas of interest.
        start_y_column: str
            Name of column which contains the y coordinate's start position of the
            areas of interest.
        width_column: str | None
            Name of the column which contains the width of the area of interest. (default: None)
        height_column: str | None
            Name of column which contains the height of the area of interest. (default: None)
        end_x_column: str | None
            Name of column which contains the x coordinate's end position of the areas of interest.
            (default: None)
        end_y_column: str | None
            Name of column which contains the y coordinate's end position of the areas of interest.
            (default: None)
        page_column: str | None
            Name of column which contains the page information of the area of interest.
            (default: None)
        trial_column: str | None
            Name of column that specifies the unique trial id.
            (default: None)
        writing_system: WritingSystem | str
            Writing system of the text. If ``writing_system`` is a string,
            :py:meth:`~pymovements.stimulus.WritingSystem.from_descriptor()` for initialization.
            (default: ``'left-to-right'``)
        metadata: dict[str, Any] | None
            Dictionary containing additional metadata. (default: None)
        read_csv_kwargs: dict[str, Any] | None
            Custom read keyword arguments for polars. (default: None)


        Returns
        -------
        TextStimulus
            Returns the text stimulus file.
        """
        if isinstance(path, str):
            path = Path(path)
        if read_csv_kwargs is None:
            read_csv_kwargs = {}

        try:
            stimulus_df = pl.read_csv(path, **read_csv_kwargs)
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'Stimulus file not found: {path}') from exception
        except pl.exceptions.ComputeError as exception:
            raise ValueError(f'Stimulus file is not a valid CSV file: {path}') from exception

        stimulus_df = stimulus_df.fill_null(' ')

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
            writing_system=writing_system,
            metadata=metadata,
        )


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
        writing_system: WritingSystem | str = 'left-to-right',
        metadata: dict[str, Any] | None = None,
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
        Name for the column that specifies the unique trial id.
        (default: None)
    custom_read_kwargs: dict[str, Any] | None
        Custom read keyword arguments for polars. (default: None)
    writing_system: WritingSystem | str
        Text writing system. See TextStimulus.__init__ for details.
        (default: WritingSystem(horizontal, top-to-bottom, left-to-right))
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata. (default: None)


    Returns
    -------
    TextStimulus
        Returns the text stimulus file.
    """
    return TextStimulus.from_csv(
        path=aoi_path,
        aoi_column=aoi_column,
        start_x_column=start_x_column,
        start_y_column=start_y_column,
        width_column=width_column,
        height_column=height_column,
        end_x_column=end_x_column,
        end_y_column=end_y_column,
        page_column=page_column,
        trial_column=trial_column,
        read_csv_kwargs=custom_read_kwargs,
        writing_system=writing_system,
        metadata=metadata,
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
