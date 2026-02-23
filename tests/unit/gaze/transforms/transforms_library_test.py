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
"""Test public dataset definitions."""

from __future__ import annotations

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('transform_function', 'transform_name'),
    [
        pytest.param(pm.gaze.transforms.center_origin, 'center_origin', id='center_origin'),
        pytest.param(pm.gaze.transforms.deg2pix, 'deg2pix', id='deg2pix'),
        pytest.param(pm.gaze.transforms.downsample, 'downsample', id='downsample'),
        pytest.param(pm.gaze.transforms.norm, 'norm', id='norm'),
        pytest.param(pm.gaze.transforms.pix2deg, 'pix2deg', id='pix2deg'),
        pytest.param(pm.gaze.transforms.pos2acc, 'pos2acc', id='pos2acc'),
        pytest.param(pm.gaze.transforms.pos2vel, 'pos2vel', id='pos2vel'),
        pytest.param(pm.gaze.transforms.savitzky_golay, 'savitzky_golay', id='savitzky_golay'),
    ],
)
def test_transform_registered(transform_function, transform_name):
    assert transform_name in pm.gaze.transforms.TransformLibrary.methods
    assert pm.gaze.transforms.TransformLibrary.get(transform_name) == transform_function
    assert pm.gaze.transforms.TransformLibrary.get(transform_name).__name__ == transform_name
