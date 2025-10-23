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
"""Tests for main_sequence_plot branches.

- Using external axes triggers the ax.scatter path and returns the provided fig/ax.
- Warning about ignored figsize when external ax is given (from prepare_figure).
"""
from __future__ import annotations

from unittest.mock import Mock

import matplotlib.pyplot as plt
import polars as pl
import pytest

import pymovements as pm


def _make_events() -> pm.Events:
    # Build a minimal Events with saccades and necessary columns
    df = pl.DataFrame(
        {
            'trial': [1, 1, 1],
            'name': ['saccade', 'saccade', 'saccade'],
            'onset': [0, 10, 20],
            'offset': [5, 15, 25],
            'duration': [5, 5, 5],
            'amplitude': [2.0, 3.5, 1.0],
            'peak_velocity': [100.0, 250.0, 80.0],
        },
    )
    return pm.Events(df)


def test_main_sequence_plot_with_external_ax_uses_ax_and_warns_on_figsize(monkeypatch):
    events = _make_events()
    fig, ax = plt.subplots()

    # Ensure we don't accidentally show/close
    show_mock = Mock()
    monkeypatch.setattr(plt, 'show', show_mock)
    close_mock = Mock()
    monkeypatch.setattr(plt, 'close', close_mock)

    # With an external ax and default figsize parameter present,
    # a UserWarning should be raised that figsize is ignored.
    with pytest.warns(UserWarning):
        ret_fig, ret_ax = pm.plotting.main_sequence_plot(
            events=events,
            ax=ax,
            show=False,
            closefig=False,
        )

    # It should return the same fig/ax and plot into the provided ax (through ax.scatter path)
    assert ret_ax is ax
    assert ret_fig is fig

    # A scatter call produces a PathCollection on the axes
    assert len(ax.collections) > 0

    # No show or close when using external ax
    show_mock.assert_not_called()
    close_mock.assert_not_called()

    plt.close(fig)


def make_events(rows: list[dict]) -> pm.Events:
    return pm.Events(pl.DataFrame(rows))


def test_main_sequence_plot_deprecated_event_df_path_warns_and_plots(monkeypatch):
    # event_df path triggers the deprecation branch (97->98)
    df = pl.DataFrame({
        'trial': [1, 1],
        'name': ['saccade', 'saccade'],
        'onset': [0, 10],
        'offset': [5, 15],
        'duration': [5, 5],
        'amplitude': [2.0, 4.0],
        'peak_velocity': [100.0, 200.0],
    })
    event_df = pm.Events(df)

    show_called = []
    monkeypatch.setattr(plt, 'show', lambda: show_called.append(True))

    with pytest.warns(DeprecationWarning):
        fig, ax = pm.plotting.main_sequence_plot(event_df=event_df, show=False)
    assert fig is ax.figure
    plt.close(fig)


def test_main_sequence_plot_raises_on_empty_events():
    # Covers 108->109: not events -> ValueError
    empty_events = make_events([])
    with pytest.raises(ValueError):
        pm.plotting.main_sequence_plot(events=empty_events, show=False)


def test_main_sequence_plot_raises_when_no_saccades():
    # Covers 117->118: dataframe present but no 'saccade' rows
    df = pl.DataFrame({
        'trial': [1, 2],
        'name': ['fixation', 'blink'],
        'onset': [0, 10],
        'offset': [5, 15],
        'duration': [5, 5],
        # Include columns to pass column presence checks if needed later
        'amplitude': [1.0, 1.0],
        'peak_velocity': [10.0, 20.0],
    })
    events = pm.Events(df)
    with pytest.raises(ValueError):
        pm.plotting.main_sequence_plot(events=events, show=False)


def test_main_sequence_plot_keyerror_when_missing_peak_velocity():
    # Has saccades, but no 'peak_velocity' -> triggers lines 126–127
    df = pl.DataFrame({
        'trial': [1, 1],
        'name': ['saccade', 'saccade'],
        'onset': [0, 10],
        'offset': [5, 15],
        'duration': [5, 5],
        'amplitude': [2.0, 4.0],
        # 'peak_velocity' intentionally missing
    })
    events = pm.Events(df)
    with pytest.raises(KeyError):
        pm.plotting.main_sequence_plot(events=events, show=False)


def test_main_sequence_plot_keyerror_when_missing_amplitude():
    # Has saccades, but no 'amplitude' -> triggers lines 135–136
    df = pl.DataFrame({
        'trial': [1, 1],
        'name': ['saccade', 'saccade'],
        'onset': [0, 10],
        'offset': [5, 15],
        'duration': [5, 5],
        # 'amplitude' intentionally missing
        'peak_velocity': [100.0, 200.0],
    })
    events = pm.Events(df)
    with pytest.raises(KeyError):
        pm.plotting.main_sequence_plot(events=events, show=False)


def test_main_sequence_plot_sets_title():
    df = pl.DataFrame({
        'trial': [1, 1],
        'name': ['saccade', 'saccade'],
        'onset': [0, 10],
        'offset': [5, 15],
        'duration': [5, 5],
        'amplitude': [2.0, 4.0],
        'peak_velocity': [100.0, 200.0],
    })
    events = pm.Events(df)
    fig, ax = pm.plotting.main_sequence_plot(events=events, title='Main Sequence', show=False)
    try:
        assert ax.get_title() == 'Main Sequence'
    finally:
        plt.close(fig)
