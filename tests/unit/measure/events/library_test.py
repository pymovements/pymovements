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
"""Test module pymovements.events.event_properties."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.events import duration
from pymovements.measure.events import EVENT_MEASURES


@pytest.mark.parametrize(
    ('measure_function', 'measure_name'),
    [
        pytest.param(duration, 'duration', id='duration'),
    ],
)
def test_measure_registered(measure_function, measure_name):
    assert measure_name in EVENT_MEASURES
    assert EVENT_MEASURES[measure_name] == measure_function
    assert EVENT_MEASURES[measure_name].__name__ == measure_name


@pytest.mark.parametrize('measure_function', EVENT_MEASURES.values())
def test_measure_returns_polars_expression(measure_function):
    assert isinstance(measure_function(), pl.Expr)
