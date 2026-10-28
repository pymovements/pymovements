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
"""Test event detection library."""
from __future__ import annotations

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('method', 'name'),
    [
        pytest.param(pm.events.idt, 'idt', id='idt'),
        pytest.param(pm.events.ivt, 'ivt', id='ivt'),
        pytest.param(pm.events.microsaccades, 'microsaccades', id='microsaccades'),
        pytest.param(pm.events.fill, 'fill', id='fill'),
        pytest.param(pm.events.out_of_screen, 'out_of_screen', id='out_of_screen'),
    ],
)
def test_transform_registered(method, name):
    assert name in pm.events.EventDetectionLibrary.methods
    assert pm.events.EventDetectionLibrary.get(name) == method
    assert pm.events.EventDetectionLibrary.get(name).__name__ == name
