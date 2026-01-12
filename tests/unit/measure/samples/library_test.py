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
"""Test measure library."""
from __future__ import annotations

import pytest

from pymovements import SampleMeasureLibrary
from pymovements.measure import samples


@pytest.mark.parametrize(
    ('measure', 'name'),
    [
        pytest.param(samples.amplitude, 'amplitude', id='amplitude'),
        pytest.param(samples.dispersion, 'dispersion', id='dispersion'),
        pytest.param(samples.disposition, 'disposition', id='disposition'),
        pytest.param(samples.location, 'location', id='location'),
        pytest.param(samples.null_ratio, 'null_ratio', id='null_ratio'),
        pytest.param(samples.peak_velocity, 'peak_velocity', id='peak_velocity'),
    ],
)
def test_measure_registered(measure, name):
    assert name in SampleMeasureLibrary()
    assert SampleMeasureLibrary.get(name) == measure
    assert SampleMeasureLibrary.get(name).__name__ == name
