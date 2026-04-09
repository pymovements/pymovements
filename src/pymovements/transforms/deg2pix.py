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
"""Module for py:func:`pymovements.transforms.deg2pix`."""
from __future__ import annotations

import polars as pl

from pymovements.transforms._utils import _check_distance
from pymovements.transforms._utils import _check_screen_resolution
from pymovements.transforms._utils import _check_screen_size
from pymovements.transforms.library import register_transform


@register_transform
def deg2pix(
        *,
        screen_resolution: tuple[int, int],
        screen_size: tuple[float, float],
        distance: float | str,
        n_components: int,
        pixel_origin: str = 'upper left',
        position_column: str = 'position',
        pixel_column: str = 'pixel',
) -> pl.Expr:
    """Convert degrees of visual angle to pixel screen coordinates.

    Parameters
    ----------
    screen_resolution: tuple[int, int]
        Pixel screen resolution as tuple (width, height).
    screen_size: tuple[float, float]
        Screen size in centimeters as tuple (width, height).
    distance: float | str
        Must be either a scalar or a string. If a scalar is passed, it is interpreted as the
        Eye-to-screen distance in centimeters. If a string is passed, it is interpreted as the name
        of a column containing the Eye-to-screen distance in millimeters for each sample.
    n_components: int
        Number of components in the input column.
    pixel_origin: str
        The desired location of the pixel origin. Supported values: ``center``, ``upper left``.
        (default: 'upper left')
    position_column: str
        The input position column name. (default: 'position')
    pixel_column: str
        The output pixel column name. (default: 'pixel')

    Returns
    -------
    pl.Expr
        The respective polars expression.
    """
    _check_screen_resolution(screen_resolution)
    _check_screen_size(screen_size)

    if isinstance(distance, (float, int)):
        _check_distance(distance)
        distance_series = pl.lit(distance)
    elif isinstance(distance, str):
        # True division by 10 is needed to convert distance from mm to cm
        distance_series = pl.col(distance).truediv(10)
    else:
        raise TypeError(
            f'`distance` must be of type `float`, `int` or `str`, but is of type'
            f'`{type(distance).__name__}`',
        )

    distance_pixels = pl.concat_list([
        distance_series.mul(screen_resolution[component % 2] / screen_size[component % 2])
        for component in range(n_components)
    ])

    centered_pixels = [
        pl.col(position_column).list.get(component).radians().tan() *
        distance_pixels.list.get(component)
        for component in range(n_components)
    ]

    if pixel_origin == 'center':
        origin_offset = (0.0, 0.0)
    elif pixel_origin == 'upper left':
        origin_offset = ((screen_resolution[0] - 1) / 2, (screen_resolution[1] - 1) / 2)
    else:
        supported_origins = ['center', 'upper left']
        raise ValueError(
            f'value `{pixel_origin}` for argument `pixel_origin` is invalid. '
            f' Valid values are: {supported_origins}',
        )

    pixel_series = pl.concat_list(
        [
            centered_pixels[component] + origin_offset[component % 2]
            for component in range(n_components)
        ],
    )
    return pixel_series.alias(pixel_column)
