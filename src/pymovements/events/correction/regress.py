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
"""Provides the regress drift correction algorithm."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

from pymovements.events.correction._utils import location_expr
from pymovements.events.correction._utils import locations_to_lists
from pymovements.events.correction._utils import to_line_values


def regress(
    line_ys: pl.Series | Sequence[float],
    *,
    k_bounds: tuple[float, float] = (-0.1, 0.1),
    o_bounds: tuple[float, float] = (-50, 50),
    s_bounds: tuple[float, float] = (1, 20),
    location: str | pl.Expr = 'location',
) -> pl.Expr:
    """Fit linear regression parameters (slope, offset, std) to align fixations to lines.

    Reference: :cite:p:`Cohen2013,Carr2022`.

    Parameters
    ----------
    line_ys: pl.Series | Sequence[float]
        Vertical y-coordinates (midlines) of lines of text.
    k_bounds: tuple[float, float]
        Slope bounds. (default: (-0.1, 0.1))
    o_bounds: tuple[float, float]
        Offset bounds. (default: (-50, 50))
    s_bounds: tuple[float, float]
        Standard deviation bounds. (default: (1, 20))
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

    def _regress_core(locations: pl.Series) -> pl.Series:
        x_values, y_values = locations_to_lists(locations)

        def line_log_densities(params: Sequence[float]) -> list[np.ndarray]:
            """Per-line log-densities of observing the fixation y-values."""
            slope = k_bounds[0] + (k_bounds[1] - k_bounds[0]) * norm.cdf(params[0])
            offset = o_bounds[0] + (o_bounds[1] - o_bounds[0]) * norm.cdf(params[1])
            deviation = s_bounds[0] + (s_bounds[1] - s_bounds[0]) * norm.cdf(params[2])
            predicted_y = [x * slope for x in x_values]
            return [
                norm.logpdf(
                    y_values,
                    [predicted + line_y + offset for predicted in predicted_y],
                    deviation,
                )
                for line_y in line_values
            ]

        def negative_log_likelihood(params: Sequence[float]) -> float:
            densities = line_log_densities(params)
            return -sum(max(fixation) for fixation in zip(*densities))

        best_fit = minimize(negative_log_likelihood, [0, 0, 0])
        densities = line_log_densities(best_fit.x)
        corrected_y = [
            line_values[max(range(len(line_values)), key=list(fixation).__getitem__)]
            for fixation in zip(*densities)
        ]
        return pl.Series(corrected_y)

    return (
        location_expr(location)
        .map_batches(_regress_core, return_dtype=pl.Float64)
        .alias('y_regress')
    )
