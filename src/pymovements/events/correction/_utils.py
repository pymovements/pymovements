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
"""Provides shared helpers for the drift correction algorithms."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import polars as pl
import math
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
from pandas import DataFrame
import os
from pathlib import Path
from typing import Any


import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

from pymovements._utils._expressions import as_expr


def location_expr(location: str | pl.Expr) -> pl.Expr:
    """Resolve a location argument to an expression of [x, y] lists."""
    return as_expr(location).cast(pl.List(pl.Float64))


def location_x(location: str | pl.Expr) -> pl.Expr:
    """Extract the x-coordinate from [x, y] locations."""
    return location_expr(location).list.get(0)


def location_y(location: str | pl.Expr) -> pl.Expr:
    """Extract the y-coordinate from [x, y] locations."""
    return location_expr(location).list.get(1)


def to_line_values(line_ys: pl.Series | Sequence[float]) -> list[float]:
    """Normalize line y-coordinates to a list of floats."""
    if isinstance(line_ys, pl.Series):
        return [float(line_y) for line_y in line_ys.to_list()]
    return [float(line_y) for line_y in line_ys]


def nearest_line_index(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the index of the nearest text line for each y-value."""
    distances = pl.concat_list([(y_expr - line_y).abs() for line_y in line_values])
    return distances.list.arg_min()


def nearest_line_y(y_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression giving the y-coordinate of the nearest text line."""
    return nearest_line_index(y_expr, line_values).replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def line_index_to_y(index_expr: pl.Expr, line_values: list[float]) -> pl.Expr:
    """Return an expression mapping line indices to line y-coordinates."""
    return index_expr.replace_strict(
        dict(enumerate(line_values)), return_dtype=pl.Float64,
    )


def locations_to_lists(locations: pl.Series) -> tuple[list[float], list[float]]:
    """Split a series of [x, y] locations into lists of x and y values."""
    points = locations.cast(pl.List(pl.Float64)).to_list()
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return x_values, y_values


def map_locations(
    location: str | pl.Expr,
    core: Callable[[pl.Series], pl.Series],
    alias: str,
) -> pl.Expr:
    """Return an expression piping the [x, y] locations through core as one batch.

    The core treats its input as a single reading sequence, so the caller must evaluate
    the returned expression per trial.
    """
    return (
        location_expr(location)
        .map_batches(core, return_dtype=pl.Float64)
        .alias(alias)
    )


def nearest_index(values: Sequence[float], target: float) -> int:
    """Return the index of the value closest to target, ties favoring the first."""
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def is_right_to_left(directionality: str) -> bool:
    """Validate a directionality value and resolve it to a right-to-left flag.

    Parameters
    ----------
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left'.

    Returns
    -------
    bool
        True if the directionality is 'right-to-left', False if 'left-to-right'.

    Raises
    ------
    ValueError
        If the directionality is 'top-to-bottom' or not a known value.
    """
    if directionality == 'top-to-bottom':
        raise ValueError(
            "directionality 'top-to-bottom' is not supported by the drift correction "
            'algorithms, which assume horizontal lines of text.',
        )
    if directionality not in ('left-to-right', 'right-to-left'):
        raise ValueError(
            f"Unknown directionality '{directionality}'. "
            "Valid values are: 'left-to-right', 'right-to-left'.",
        )
    return directionality == 'right-to-left'



def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Return the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_closest_top_box(
    px: int, py: int,
    list_of_centers: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return the closest word to the top."""
    closest_top_box = (px, py)
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy < py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_top_box = center
    if closest_distance == float('inf'):
        closest_top_box = (px, py)
    return closest_top_box


def find_closest_left_box(
    px: int, py: int,
    list_of_centers: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return the closest word to the left."""
    closest_left_box = (px, py)
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx < px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_left_box = center
    if closest_distance == float('inf'):
        closest_left_box = (px, py)
    return closest_left_box


def find_closest_bottom_box(
    px: int, py: int,
    list_of_centers: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return the closest word to the bottom."""
    closest_bottom_box = (px, py)
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy > py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_bottom_box = center
    if closest_distance == float('inf'):
        closest_bottom_box = (px, py)
    return closest_bottom_box


def find_closest_right_box(
    px: int, py: int,
    list_of_centers: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return the closest word to the right."""
    closest_right_box = (px, py)
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx > px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_right_box = center
    if closest_distance == float('inf'):
        closest_right_box = (px, py)
    return closest_right_box


class OCR_Reader:
    """Read an image and extract the center positions of text boxes using OCR.

    Parameters
    ----------
    path_to_image : str
        File path to the image.

    """

    def __init__(self, path_to_image: str):
        self.path_to_image: str = path_to_image
        self.list_of_centers: list[tuple[int, int]] = []

    def get_list_of_centers(self) -> dict:
        """Extract and store the centers of valid OCR-detected text boxes.

        Returns
        -------
        dict
            The full OCR result dictionary from pytesseract.
        """
        img = cv2.imread(self.path_to_image)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if (
                int(d['conf'][i]) > 60
                and d['width'][i] / d['height'][i] > 0.3
                and d['height'][i] > 2
            ):
                x, y, w, h = (
                    d['left'][i], d['top'][i],
                    d['width'][i], d['height'][i],
                )
                center = (int(x + w / 2), int(y + h / 2))
                self.list_of_centers.append(center)
        return d

    def read_image(self) -> dict:
        """Visualize OCR results by drawing bounding boxes on the image.

        This method is not used by the fixationcorrection module.

        Returns
        -------
        dict
            The full OCR result dictionary from pytesseract.
        """
        img = cv2.imread(self.path_to_image)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if (
                int(d['conf'][i]) > 60
                and d['width'][i] / d['height'][i] > 0.3
                and d['height'][i] > 2
            ):
                x, y, w, h = (
                    d['left'][i], d['top'][i],
                    d['width'][i], d['height'][i],
                )
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 0), 2,
                )

        cv2.imshow('image', img)
        cv2.waitKey(0)
        return d


class ColumnMappingDialog(simpledialog.Dialog):
    """Dialog for defining how CSV columns map to `FixationCorrection` fields.

    The user is asked to enter

    * X- and Y-coordinate column names (required)
    * the column that identifies the corresponding image (required)
    * optional column names to group the fixations by
    * optional extra filters in the form
      ``col=value1|value2, other_col=foo``

    After the user clicks **OK** the dialog stores a dictionary in
    ``self.result``::


        {
            "pixel_x":                <str>,
            "pixel_y":                <str>,
            "image_column":           <str>,
            "grouping_parameters":    list[str],
            "filter_columns":         {<str>: list[str], ...|None}
        }

    If the user cancels or validation fails, ``self.result`` is ``None``.

    Attributes
    ----------
    result : dict[str, str | list[str] | dict[str, list[str]]] | None
        The mapping returned by the dialog, or None if the user cancelled.

    Parameters
    ----------
    parent : tk.Misc | None
        The parent window (can be withdrawn).
    title : str | None
        Window title. (Default: None)

    Notes
    -----
    `simpledialog.Dialog` shows the window immediately during construction;
    When done, read ``ColumnMappingDialog(...).result`` to get the mapping.
    """

    result: dict[str, str | list[str] | dict[str, list[str]]] | None

    def __init__(self, parent: tk.Misc | None, title: str | None = None):
        super().__init__(parent, title)
        self.pixel_x_entry: tk.Entry | None = None
        self.pixel_y_entry: tk.Entry | None = None
        self.image_column_entry: tk.Entry | None = None
        self.grouping_entry: tk.Entry | None = None
        self.filters_entry: tk.Entry | None = None

    def body(self, master: tk.Frame) -> tk.Entry:
        """Build and lay out the dialog widgets; return the widget to focus."""
        self.title('Configure Column Mapping')

        (
            ttk.Label(master, text='X-coordinate column; e.g. CURRENT_FIX_X:')
            .grid(row=0, column=0, sticky='w', pady=2)
        )
        self.pixel_x_entry = ttk.Entry(master, width=30)
        self.pixel_x_entry.grid(row=0, column=1, pady=2)

        (
            ttk.Label(master, text='Y-coordinate column; e.g. CURRENT_FIX_Y:')
            .grid(row=1, column=0, sticky='w', pady=2)
        )
        self.pixel_y_entry = ttk.Entry(master, width=30)
        self.pixel_y_entry.grid(row=1, column=1, pady=2)

        (
            ttk.Label(
                master,
                text='Image name column; e.g. page_name:',
            )
            .grid(row=2, column=0, sticky='w', pady=2)
        )
        self.image_column_entry = ttk.Entry(master, width=30)
        self.image_column_entry.grid(row=2, column=1, pady=2)

        (
            ttk.Label(
                master, text=(
                    'Grouping parameters (optional) (Comma-separated);'
                    ' e.g. RECORDING_SESSION_LABEL, trial_number:'
                ),
            ).grid(row=4, column=0, sticky='w', pady=2)
        )
        self.grouping_entry = ttk.Entry(master, width=30)
        self.grouping_entry.grid(row=4, column=1, pady=2)

        (
            ttk.Label(
                master,
                text=(
                    'Filter columns '
                    "(comma-separated, use '=' for column "
                    "and '|' for alternatives; "
                    'e.g.  RECORDING_SESSION_LABEL=msd002|msd003, '
                    'page_name=reading-dickens-1):'
                ),
            ).grid(row=6, column=0, columnspan=2, sticky='w', pady=2)
        )
        self.filters_entry = ttk.Entry(master, width=50)
        self.filters_entry.grid(row=7, column=0, columnspan=2, pady=2)

        return self.pixel_x_entry

    def validate(self) -> bool:
        """Validate inputs before closing the dialog."""
        pixel_x = self.get_non_optional(
            self.pixel_x_entry, 'pixel_x_entry',
        ).get().strip()
        pixel_y = self.get_non_optional(
            self.pixel_y_entry, 'pixel_y_entry',
        ).get().strip()
        image = self.get_non_optional(
            self.image_column_entry, 'image_column_entry',
        ).get().strip()
        raw_grouping = self.get_non_optional(
            self.grouping_entry, 'grouping_entry',
        ).get().strip()
        raw_filters = self.get_non_optional(
            self.filters_entry, 'filters_entry',
        ).get().strip()

        if not (pixel_x and pixel_y):
            messagebox.showerror(
                'Error',
                'X and Y coordinate column names are required.',
            )
            return False

        if not image:
            messagebox.showerror(
                'Error',
                'Image column name is required.',
            )
            return False

        # Check grouping format
        if raw_grouping:
            try:
                _ = [v.strip() for v in raw_grouping.split('|')]
            except ValueError as err:
                messagebox.showerror('Grouping Format Error', str(err))
                return False

        # Check filter format
        if raw_filters:
            try:
                for pair in (
                    p.strip() for p in raw_filters.split(',')
                    if p.strip()
                ):
                    if '=' not in pair:
                        raise ValueError(
                            f"Missing '=' in filter pair: '{pair}'",
                        )
                    col, val = pair.split('=', 1)
                    values = [v.strip() for v in val.split('|') if v.strip()]
                    if not values:
                        raise ValueError(
                            f"No value specified for column '{col.strip()}'",
                        )
            except ValueError as err:
                messagebox.showerror('Filter Format Error', str(err))
                return False

        return True

    def apply(self) -> None:
        """Save the mapping to self.result."""
        pixel_x = self.get_non_optional(
            self.pixel_x_entry, 'pixel_x_entry',
        ).get().strip()
        pixel_y = self.get_non_optional(
            self.pixel_y_entry, 'pixel_y_entry',
        ).get().strip()
        image = self.get_non_optional(
            self.image_column_entry, 'image_column_entry',
        ).get().strip()
        raw_grouping = self.get_non_optional(
            self.grouping_entry, 'grouping_entry',
        ).get().strip()
        raw_filters = self.get_non_optional(
            self.filters_entry, 'filters_entry',
        ).get().strip()

        grouping = [
            v.strip()
            for v in raw_grouping.split('|')
        ] if raw_grouping else []
        grouping.append(image)

        filters: dict[str, list[str]] = {}
        if raw_filters:
            for pair in (
                p.strip() for p in raw_filters.split(',')
                if p.strip()
            ):
                col, val = pair.split('=', 1)
                values = [v.strip() for v in val.split('|') if v.strip()]
                filters[col.strip()] = values

        self.result = {
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'image_column': image,
            'grouping': grouping,
            'filter_columns': filters,
        }

    def get_non_optional(self, entry: tk.Entry | None, name: str) -> tk.Entry:
        """Ensure the given entry is not None and return it."""
        if entry is None:
            raise RuntimeError(f"{name} was not initialized")
        return entry

class DataProcessing:
    """Handle loading, filtering, and grouping of CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing fixation data.
    image_folder : str
        Directory containing corresponding stimulus images.
    mapping : dict[str, str | list[str] | dict[str, list[str]]] | None
        Optional column mapping for fixation coordinates.
        If None, a column-mapping dialogue will be shown to the user.
        The mapping must include keys for pixel_x, pixel_x, image_column
        and optionally grouping and filter_columns. (Default: None)
    custom_read_kwargs : dict[str, Any] | None
        Optional keyword arguments to pass to the pandas.read_csv function.
        (Default: None)

    Raises
    ------
    ValueError
        If the user cancels the column-mapping dialog.
    """

    def __init__(
        self, csv_file: str, image_folder: str,
        mapping: dict[str, str | list[str] | dict[str, list[str]]] | None,
        custom_read_kwargs: dict[str, Any] | None = None,
    ):
        self.csv_file = csv_file
        self.dataframes: list[pd.DataFrame] = []
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)

        default_read_kwargs = {
            'sep': None,
            'engine': 'python',
            'encoding': 'utf-8-sig',
        }
        self.custom_read_kwargs = {
            **default_read_kwargs, **(custom_read_kwargs or {}),
        }

        if mapping is None:
            root = tk.Tk()
            root.withdraw()
            mapping = ColumnMappingDialog(
                root, title='Column Mapping',
            ).result
            root.destroy()

            if mapping is None:
                raise ValueError(
                    'Column mapping configuration cancelled by user.',
                )

        self.column_mapping = {
            'pixel_x': mapping['pixel_x'],
            'pixel_y': mapping['pixel_y'],
            'image_column': mapping['image_column'],
            'grouping': mapping['grouping'],
            'filter_columns': mapping['filter_columns'],
        }

    def prepare_data(self) -> list[DataFrame]:
        """Prepare data for fixation correction.

        Load the CSV file, remove rows without matching image files,
        and apply grouping and filtering as specified by the user.
        """
        try:
            raw_data = pd.read_csv(self.csv_file, **self.custom_read_kwargs)

            if raw_data.shape[1] <= 1:
                raise ValueError(
                    'Parsing failed. The file appears to only have one column.'
                    'Probably the wrong delimiter was specified.',
                )
        except UnicodeDecodeError as e:
            raise ValueError(
                'Encoding error. This may be due to an incorrect file format'
                f'or an incorrect encoding - {e}',
            ) from e

        except pd.errors.ParserError as e:
            raise ValueError(
                f'Parsing error, probably the wrong delimiter '
                f'was specified - {e}',
            ) from e

        except FileNotFoundError as e:
            raise ValueError(
                f'File not found - {e}',
            ) from e

        # Drop the entries where there is no corresponding image
        clean_list = {self.normalize(p) for p in self.image_list}
        mask = raw_data[self.column_mapping['image_column']].astype(
            str,
        ).apply(self.normalize).isin(clean_list)
        dropped = raw_data[mask]

        self.dataframes = self.filter_and_group(dropped)
        return self.dataframes

    def normalize(self, name: str) -> str:
        """Convert a filename to lowercase and strip its extension."""
        return Path(name).stem.lower()

    def filter_and_group(self, dataframe: pd.DataFrame) -> list[pd.DataFrame]:
        """Filter and group the dataframe based on selected values."""
        self.make_title()
        if self.column_mapping['filter_columns'] is not None\
                and isinstance(self.column_mapping['filter_columns'], dict):
            for key, val in self.column_mapping['filter_columns'].items():
                if key not in dataframe:
                    print(
                        f"WARNING: Filter column '{key}' not found; "
                        f"ignoring filter.",
                    )
                    continue

                unique_values = dataframe[key].dropna().unique()

                if isinstance(val, list):
                    missing_vals = [v for v in val if v not in unique_values]
                    if missing_vals:
                        print(
                            f"WARNING: Values '{missing_vals}' "
                            f"not found in column '{key}'; "
                            f"ignoring filter.",
                        )
                    dataframe = dataframe[dataframe[key].isin(val)]
                else:
                    if val not in unique_values:
                        print(
                            f"WARNING: Values '{val}' "
                            f"not found in column '{key}'; "
                            f"ignoring filter.",
                        )
                    dataframe = dataframe[dataframe[key] == val]
        else:
            print(
                "WARNING: 'filter_columns' is not a dictionary; "
                'skipping filters.',
            )

        if self.column_mapping['grouping'] is not None \
                and isinstance(self.column_mapping['grouping'], list):
            missing = [
                col for col in self.column_mapping['grouping']
                if col not in dataframe.columns
            ]
            valid = [
                col for col in self.column_mapping['grouping']
                if col in dataframe.columns
            ]
            if missing:
                print(
                    f"WARNING: Grouping column(s) {missing} "
                    f"not found in data; "
                    f"skipping this grouping.",
                )

            if valid:
                grouped = dataframe.groupby(valid)
                return [group.copy() for _, group in grouped]

        return [dataframe.copy()]

    def make_title(self) -> str:
        """Construct a title string from the selected filter values."""
        all_filters = []
        if isinstance(self.column_mapping['filter_columns'], dict):
            for value in self.column_mapping['filter_columns'].values():
                all_filters.append(value)
        flattened = [item for sublist in all_filters for item in sublist]
        title = '_'.join(flattened)
        return title
