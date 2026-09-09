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
"""Provides the slice drift correction algorithm."""
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from statistics import fmean

import polars as pl

from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import map_per_trial
from pymovements.events.correction._utils import nearest_index
from pymovements.events.correction._utils import to_line_values


# pylint: disable=redefined-builtin
def slice(
    line_ys: pl.Series | Sequence[float],
    *,
    x_thresh: float = 192,
    y_thresh: float = 32,
    w_thresh: float = 32,
    n_thresh: float = 90,
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Slice algorithm to assign fixations in multi-line reading tasks.

    Reference: :cite:p:`Glandorf2021`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    x_thresh: float
        Horizontal run segmentation threshold. (default: 192)
    y_thresh: float
        Vertical run segmentation threshold. (default: 32)
    w_thresh: float
        Proto-line merger threshold. (default: 32)
    n_thresh: float
        Adjacent proto-line merger threshold. (default: 90)
    location: str | pl.Expr
        Column name or expression of [x, y] fixation locations. The returned expression
        operates on the full fixation sequence of a single trial, so it must be evaluated
        per trial. (default: 'location')

    Returns
    -------
    pl.Expr
        Expression computing the corrected y-coordinates.
    """
    line_values = to_line_values(line_ys)
    if len(line_values) > 1:
        line_height = fmean(
            next_line - line for line, next_line in zip(line_values, line_values[1:])
        )
    else:
        # The fallback only extends behavior to single-line texts, on which the reference
        # implementation of Carr et al. degenerates (the mean spacing of one line is NaN);
        # 32 pixels matches the default y_thresh, so phantom proto lines are placed one
        # plausible line height away.
        line_height = 32.0

    core = partial(
        _slice_core,
        line_values=line_values,
        x_thresh=x_thresh,
        y_thresh=y_thresh,
        w_thresh=w_thresh,
        n_thresh=n_thresh,
        line_height=line_height,
    )
    return map_per_trial(location, core, 'y_slice')


def _run_offset(
    run: list[int],
    proto_line: list[tuple[float, float]],
    x_values: list[float],
    y_values: list[float],
) -> float:
    """Mean vertical offset of a run to the horizontally closest proto line points."""
    proto_x = [point[0] for point in proto_line]
    return fmean(
        y_values[index] - proto_line[nearest_index(proto_x, x_values[index])][1]
        for index in run
    )


def _proto_line_points(
    proto_line_index: int,
    proto_lines: dict[int, list[int]],
    phantom_proto_lines: dict[int, list[tuple[float, float]]],
    x_values: list[float],
    y_values: list[float],
) -> list[tuple[float, float]]:
    """Points of a proto line, falling back to its phantom points while it is empty."""
    if proto_lines[proto_line_index]:
        return [
            (x_values[index], y_values[index])
            for index in proto_lines[proto_line_index]
        ]
    return phantom_proto_lines[proto_line_index]


def _slice_core(
    locations: pl.Series,
    *,
    line_values: list[float],
    x_thresh: float,
    y_thresh: float,
    w_thresh: float,
    n_thresh: float,
    line_height: float,
) -> pl.Series:
    """Grow proto lines from the fixation runs of a single trial and map them to lines."""
    x_values, y_values = locations_to_lists(locations)
    n = len(x_values)

    # 1. Segment runs of horizontally and vertically close fixations.
    boundaries = [
        index + 1 for index in range(n - 1)
        if abs(x_values[index + 1] - x_values[index]) > x_thresh
        or abs(y_values[index + 1] - y_values[index]) > y_thresh
    ]
    runs = [
        list(range(start, end))
        for start, end in zip([0] + boundaries, boundaries + [n])
    ]

    # 2. The horizontally longest run starts the first proto line.
    longest_run = max(
        range(len(runs)),
        key=lambda index: x_values[runs[index][-1]] - x_values[runs[index][0]],
    )
    proto_lines: dict[int, list[int]] = {0: runs.pop(longest_run)}
    phantom_proto_lines: dict[int, list[tuple[float, float]]] = {}
    points_of = partial(
        _proto_line_points,
        proto_lines=proto_lines,
        phantom_proto_lines=phantom_proto_lines,
        x_values=x_values,
        y_values=y_values,
    )

    # 3. Grow proto lines above and below by merging runs within the thresholds; where
    # nothing merges, a phantom proto line one line height away keeps the search going.
    while runs:
        merged_on_this_iteration = False
        for proto_line_index, direction in (
            (min(proto_lines), -1), (max(proto_lines), 1),
        ):
            proto_lines[proto_line_index + direction] = []
            points = points_of(proto_line_index)

            offsets = [_run_offset(run, points, x_values, y_values) for run in runs]
            merge_into_current = [
                index for index, offset in enumerate(offsets)
                if abs(offset) < w_thresh
            ]
            merge_into_adjacent = [
                index for index, offset in enumerate(offsets)
                if w_thresh <= offset * direction < n_thresh
            ]

            for index in merge_into_current:
                proto_lines[proto_line_index].extend(runs[index])
            for index in merge_into_adjacent:
                proto_lines[proto_line_index + direction].extend(runs[index])

            if not merge_into_adjacent:
                average_x = fmean(point[0] for point in points)
                average_y = fmean(point[1] for point in points)
                phantom_proto_lines[proto_line_index + direction] = [
                    (average_x, average_y + line_height * direction),
                ]

            for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                del runs[index]
                merged_on_this_iteration = True

        if not merged_on_this_iteration:
            break

    # 4. Assign leftover runs to the vertically closest proto line.
    for run in runs:
        best_distance = math.inf
        best_proto_line = next(iter(proto_lines))
        for proto_line_index in proto_lines:
            distance = abs(_run_offset(run, points_of(proto_line_index), x_values, y_values))
            if distance < best_distance:
                best_distance = distance
                best_proto_line = proto_line_index
        proto_lines[best_proto_line].extend(run)

    # 5. Merge the smaller of the outermost proto lines inwards until the number of
    # proto lines matches the number of text lines.
    while len(proto_lines) > len(line_values):
        top, bottom = min(proto_lines), max(proto_lines)
        if len(proto_lines[top]) < len(proto_lines[bottom]):
            proto_lines[top + 1].extend(proto_lines.pop(top))
        else:
            proto_lines[bottom - 1].extend(proto_lines.pop(bottom))

    # 6. Proto lines map to the text lines top to bottom.
    corrected_y = [0.0] * n
    for line_index, proto_line_index in enumerate(sorted(proto_lines)):
        for fixation_index in proto_lines[proto_line_index]:
            corrected_y[fixation_index] = line_values[line_index]
    return pl.Series(corrected_y)
