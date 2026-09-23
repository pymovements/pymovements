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
"""Tests for the display-space canvas."""
import matplotlib.pyplot as plt
import pytest

from pymovements import plotting


@pytest.mark.parametrize(
    ('origin', 'expected_xlim', 'expected_ylim'),
    [
        pytest.param('upper left', (0, 1920), (1080, 0), id='upper-left'),
        pytest.param('center', (-960, 960), (540, -540), id='center'),
    ],
)
def test_screen_extent_and_orientation(origin, expected_xlim, expected_ylim):
    fig, ax = plotting.screen(1920, 1080, origin=origin)

    assert ax.figure is fig
    assert ax.get_xlim() == expected_xlim
    assert ax.get_ylim() == expected_ylim
    assert ax.get_aspect() == 1
    assert not ax.lines
    assert not ax.collections
    assert not ax.images
    assert not ax.patches


def test_screen_reuses_axes():
    expected_fig, expected_ax = plt.subplots()

    fig, ax = plotting.screen(800, 600, ax=expected_ax)

    assert fig is expected_fig
    assert ax is expected_ax


@pytest.mark.parametrize(
    ('width_px', 'height_px'),
    [
        pytest.param(None, 1080, id='missing-width'),
        pytest.param(1920, None, id='missing-height'),
        pytest.param(None, None, id='missing-both'),
    ],
)
def test_screen_unset_resolution_raises(width_px, height_px):
    with pytest.raises(ValueError, match='width_px and height_px must be set'):
        plotting.screen(width_px, height_px)


@pytest.mark.parametrize('origin', [None, 'lower left', 'upper right'])
def test_screen_invalid_origin_raises(origin):
    with pytest.raises(ValueError, match='origin must be'):
        plotting.screen(1920, 1080, origin=origin)
