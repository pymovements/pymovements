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
"""Tests for unified axes/closefig plotting API behavior and warnings."""
from __future__ import annotations

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pymovements as pm


@pytest.fixture(name='gaze', scope='session')
def gaze_fixture():
    x = np.arange(-50, 50)
    y = np.arange(-50, 50)
    arr = np.column_stack((x, y)).transpose()

    experiment = pm.Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = pm.gaze.from_numpy(
        samples=arr,
        schema=['x_pix', 'y_pix'],
        experiment=experiment,
        pixel_columns=['x_pix', 'y_pix'],
    )

    gaze.pix2deg()
    gaze.pos2vel()

    return gaze


def test_traceplot_with_external_ax_emits_warnings_and_does_not_show_or_close(gaze, monkeypatch):
    fig, ax = plt.subplots()

    show_mock = Mock()
    monkeypatch.setattr(plt, 'show', show_mock)
    close_mock = Mock()
    monkeypatch.setattr(plt, 'close', close_mock)

    with pytest.warns(UserWarning):
        ret_fig, ret_ax = pm.plotting.traceplot(gaze=gaze, ax=ax, show=True, closefig=True)
    assert ret_ax is ax
    assert ret_fig is fig

    # when external ax is provided, show should not be called and figure should not be closed
    show_mock.assert_not_called()
    close_mock.assert_not_called()


def test_heatmap_own_figure_closes_by_default(gaze, monkeypatch):
    # For heatmap we ensure default close happens when it owns the figure
    close_mock = Mock()
    monkeypatch.setattr(plt, 'close', close_mock)
    # We do not show to avoid backend usage here
    pm.plotting.heatmap(gaze, position_column='pixel', show=False)
    close_mock.assert_called()


def test_heatmap_with_external_ax_no_show_no_close_and_warnings(gaze, monkeypatch):
    fig, ax = plt.subplots()

    show_mock = Mock()
    monkeypatch.setattr(plt, 'show', show_mock)
    close_mock = Mock()
    monkeypatch.setattr(plt, 'close', close_mock)

    with pytest.warns(UserWarning):
        ret_fig, ret_ax = pm.plotting.heatmap(
            gaze,
            position_column='pixel',
            ax=ax,
            show=True,
            closefig=True,
        )
    assert ret_ax is ax
    assert ret_fig is fig
    show_mock.assert_not_called()
    close_mock.assert_not_called()


def test_tsplot_external_ax_ignored_when_multi_channel(gaze):
    # prepare fresh gaze with two channels unnested
    g = gaze
    g.unnest('pixel', output_columns=['x_pix', 'y_pix'])

    fig, ax = plt.subplots()
    with pytest.warns(UserWarning):
        # Using external ax but with two channels -> expect warning and a new figure
        ret_fig, ret_ax = pm.plotting.tsplot(
            g,
            channels=['x_pix', 'y_pix'],
            ax=ax,
            show=False,
        )
    assert ret_ax is not ax
    assert ret_fig is not fig
