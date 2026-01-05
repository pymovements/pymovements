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
"""Provides string specific funtions."""
from __future__ import annotations

import re

CURLY_TO_REGEX = re.compile(
    r'(?:^|(?<=[^{])|(?<={{)){(?P<name>[^{][0-9a-zA-z_]*?)(?:\:(?P<quantity>\d*)(?P<type>[sd])?)?}',
)


def curly_to_regex(s: str) -> re.Pattern:
    """Return regex pattern converted from a provided python formatting style pattern.

    By default, all parameters are strings, if you want to specify number you can do: {num:d}
    If you want to specify parameter's length you can do: {two_symbols:2} or {four_digits:4d}
    Characters { and } can be escaped the same way as in python: {{ and }}
    For example:
                r'{subject_id:d}_{session_name}.csv'
    converts to r'(?P<subject_id>[0-9]+)_(?P<session_name>.+).csv'

    Parameters
    ----------
    s: str
        Pattern in python formatting style.

    Returns
    -------
    re.Pattern
        Converted regex patterns.
    """

    def replace_aux(match: re.Match) -> str:     # Auxiliary replacement function
        pattern = r'.'
        if match.group('type') == 'd':
            pattern = r'[0-9]'
        elif match.group('type') == 's':
            pattern = r'.'
        quantity = r'+'
        if match.group('quantity'):
            quantity = f'{{{match.group("quantity")}}}'
        return fr'(?P<{match.group("name")}>{pattern}{quantity})'

    result = CURLY_TO_REGEX.sub(replace_aux, s)
    result = result.replace('{{', '{').replace('}}', '}')
    return re.compile(result)
