# Copyright (c) 2026 The pymovements Project Authors
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

import polars
import pytest

from pymovements.transforms import value_impossible

# check if wrong mode raises a ValueError


def test_value_impossible_invalid():
    with pytest.raises(ValueError):
        value_impossible(maxi=230, mode='bad', input_column='hello', output_column='bye', num=2)

# check for the clip mode


def test_value_impossible_clip():
    res = value_impossible(maxi=230, mode='clip', input_column='hello', output_column='bye', num=2)

    expr = polars.concat_list([
        polars.col('hello').list.get(0).clip(-230, 230),
        polars.col('hello').list.get(1).clip(-230, 230),
    ]).alias('bye')

    assert str(expr) == str(res)


# check for the null mode
def test_value_impossible_null():
    res = value_impossible(maxi=230, mode='null', input_column='hello', output_column='bye', num=2)

    expr = polars.concat_list([
        polars.when(polars.col('hello').list.get(0).abs() <= 230).then(
            polars.col('hello').list.get(0)).otherwise(None),
        polars.when(polars.col('hello').list.get(1).abs() <= 230).then(
            polars.col('hello').list.get(1)).otherwise(None),
    ]).alias('bye')

    assert str(expr) == str(res)
