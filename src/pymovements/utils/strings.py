# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Provides string specific funtions.

.. deprecated:: v0.21.1
   This module will be removed in v0.26.0.
"""
from __future__ import annotations

import re

from deprecated.sphinx import deprecated

from pymovements._utils._strings import curly_to_regex as _curly_to_regex


@deprecated(
    reason='This module will be removed in v0.26.0.',
    version='v0.21.1',
)
def curly_to_regex(s: str) -> re.Pattern:
    """Return regex pattern converted from a provided python formatting style pattern.

    By default, all parameters are strings, if you want to specify number, you can do: {num:d}
    If you want to specify parameter's length, you can do: {two_symbols:2} or {four_digits:4d}
    Characters { and } can be escaped the same way as in python: {{ and }}
    For example:
                r'{subject_id:d}_{session_name}.csv'
    converts to r'(?P<subject_id>[0-9]+)_(?P<session_name>.+).csv'

    .. deprecated:: v0.21.1
       This module will be removed in v0.26.0.

    Parameters
    ----------
    s: str
        Pattern in python formatting style.

    Returns
    -------
    re.Pattern
        Converted regex patterns.
    """
    return _curly_to_regex(s=s)
