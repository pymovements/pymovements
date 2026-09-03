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
"""Test pymovements.plotting.screen."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pymovements.plotting import screen


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_screen_returns_figure_and_axes():
    fig, ax = screen(1280, 1024)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_figure() is fig


def test_screen_upper_left_limits_and_aspect():
    _, ax = screen(1280, 1024)

    assert ax.get_xlim() == (0, 1280)
    assert ax.get_ylim() == (1024, 0)
    assert ax.get_aspect() == 1


def test_screen_center_limits_and_aspect():
    _, ax = screen(1280, 1024, origin='center')

    assert ax.get_xlim() == (-640, 640)
    assert ax.get_ylim() == (512, -512)
    assert ax.get_aspect() == 1


@pytest.mark.parametrize('origin', ['upper left', 'center'])
def test_screen_y_increases_downward(origin):
    """Both supported origins keep screen y increasing downward."""
    _, ax = screen(1280, 1024, origin=origin)

    bottom, top = ax.get_ylim()
    assert bottom > top


@pytest.mark.parametrize(
    ('width_px', 'height_px'),
    [
        pytest.param(None, None, id='both_none'),
        pytest.param(None, 1024, id='width_none'),
        pytest.param(1280, None, id='height_none'),
    ],
)
def test_screen_unset_resolution_raises(width_px, height_px):
    with pytest.raises(ValueError, match='screen width and height must be set'):
        screen(width_px, height_px)


@pytest.mark.parametrize(
    'origin',
    [
        pytest.param(None, id='none'),
        pytest.param('lower left', id='lower_left'),
        pytest.param('upper right', id='upper_right'),
        pytest.param('', id='empty'),
    ],
)
def test_screen_unsupported_origin_raises(origin):
    with pytest.raises(ValueError, match='screen origin must be one of'):
        screen(1280, 1024, origin=origin)


def test_screen_reuses_passed_axes():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = screen(1280, 1024, ax=ax)

    assert returned_ax is ax
    assert returned_fig is fig


def test_screen_draws_no_content():
    """The axes frame is the only boundary; screen() draws nothing itself."""
    _, ax = screen(1280, 1024)

    assert len(ax.lines) == 0
    assert len(ax.patches) == 0
    assert len(ax.images) == 0
    assert len(ax.collections) == 0
