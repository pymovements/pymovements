# Copyright (c) 2025 The pymovements Project Authors
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
"""Unit tests targeting internal figure utils branches.

These focus on covering specific branches that were previously missed:
- prepare_figure with figsize=None when creating a new figure
- finalize_figure with own_figure=False and closefig=True (ignored + warns)
- _setup_axes_and_colormap warning when external ax is provided together with a non-None figsize
"""
from __future__ import annotations

import warnings
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pymovements.plotting._matplotlib import _setup_axes_and_colormap
from pymovements.plotting._matplotlib import finalize_figure
from pymovements.plotting._matplotlib import prepare_figure


def test_prepare_figure_figsize_none_creates_default():
    # When figsize=None and no external ax is given, prepare_figure should create a figure
    fig, ax, own = prepare_figure(ax=None, figsize=None, func_name='test_prepare_figure')
    try:
        assert own is True
        assert fig is ax.figure
    finally:
        plt.close(fig)


def test_finalize_figure_closefig_true_ignored_with_external_ax_warns(monkeypatch):
    # When an external ax is used (own_figure=False) and closefig=True is requested,
    # finalize_figure should warn and not close the figure.
    fig, _ = plt.subplots()
    close_mock = Mock()
    monkeypatch.setattr(plt, 'close', close_mock)

    with pytest.warns(UserWarning):
        finalize_figure(
            fig,
            show=False,
            savepath=None,
            closefig=True,
            own_figure=False,
            func_name='dummy_func',
        )

    close_mock.assert_not_called()
    plt.close(fig)


def test_setup_axes_and_colormap_warns_on_external_ax_with_figsize():
    # If an external ax is supplied and figsize is not None,
    # _setup_axes_and_colormap should emit a warning that figsize is ignored.
    fig, ax = plt.subplots()
    x = np.arange(10)
    y = np.arange(10)

    with pytest.warns(UserWarning):
        ret_fig, ret_ax, *_ = _setup_axes_and_colormap(
            x_signal=x,
            y_signal=y,
            figsize=(4, 3),
            cmap=None,
            cmap_norm=None,
            cmap_segmentdata=None,
            cval=None,
            show_cbar=False,
            add_stimulus=False,
            path_to_image_stimulus=None,
            stimulus_origin='upper',
            padding=None,
            pad_factor=0.05,
            ax=ax,
        )

    assert ret_ax is ax
    assert ret_fig is fig
    plt.close(fig)


def test_setup_axes_and_colormap_external_ax_figsize_none_no_warning():
    # External ax with figsize=None should not warn.
    fig, ax = plt.subplots()
    x = np.arange(10)
    y = np.arange(10)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        ret_fig, ret_ax, *_ = _setup_axes_and_colormap(
            x_signal=x,
            y_signal=y,
            figsize=None,
            cmap=None,
            cmap_norm=None,
            cmap_segmentdata=None,
            cval=None,
            show_cbar=False,
            add_stimulus=False,
            path_to_image_stimulus=None,
            stimulus_origin='upper',
            padding=None,
            pad_factor=0.05,
            ax=ax,
        )
    assert len(record) == 0
    assert ret_ax is ax
    assert ret_fig is fig
    plt.close(fig)


def test_prepare_figure_warns_on_external_ax_with_figsize():
    # prepare_figure should warn when an external ax is provided and figsize is not None.
    fig, ax = plt.subplots()
    try:
        with pytest.warns(UserWarning):
            ret_fig, ret_ax, own = prepare_figure(ax=ax, figsize=(6, 4), func_name='dummy')
        assert ret_ax is ax
        assert ret_fig is fig
        assert own is False
    finally:
        plt.close(fig)


def test_prepare_figure_external_ax_figsize_none_no_warning():
    # When external ax is supplied and figsize is None, no warning should be emitted.
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            ret_fig, ret_ax, own = prepare_figure(ax=ax, figsize=None, func_name='dummy')
        # No warnings captured
        assert len(record) == 0
        assert ret_ax is ax
        assert ret_fig is fig
        assert own is False
    finally:
        plt.close(fig)
