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
"""Module for py:func:`pymovements.transforms.downsample`."""
from __future__ import annotations

import polars as pl

from pymovements._utils import _checks
from pymovements.transforms.library import register_transform


@register_transform
def downsample(
        *,
        factor: int,
) -> pl.Expr:
    """Downsample gaze data by an integer factor.

    Downsampling is done by taking every `nth` sample specified by the downsampling factor.

    Parameters
    ----------
    factor: int
        Downsample factor.

    Returns
    -------
    pl.Expr
        The respective polars expression.
    """
    _checks.check_is_int(factor=factor)
    _checks.check_is_positive_value(factor=factor)

    return pl.all().gather_every(n=factor)
