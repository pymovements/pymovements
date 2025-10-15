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
"""Test heatmap."""
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib import figure

import pymovements as pm
import pymovements.plotting._matplotlib as pm_matplotlib  # type: ignore[attr-defined]
from pymovements import Experiment
from pymovements import Gaze
from pymovements.plotting import heatmap


@pytest.fixture(name='experiment_fixture')
def fixture_experiment():
    return Experiment(1024, 768, 38, 30, 60, 'upper left', 1000)


@pytest.fixture(name='args', params=['pix', 'pos'])
def args_fixture(experiment_fixture, request):
    if request.param == 'pix':
        column_names = ['x_pix', 'y_pix']
        pixel_columns = column_names
        position_columns = None
    else:
        column_names = ['x_pos', 'y_pos']
        pixel_columns = None
        position_columns = column_names

    # Init a dataframe with 2 columns and 100 rows
    df = pl.DataFrame(
        {
            column_names[0]: np.arange(0, 100),
            column_names[1]: np.arange(0, 100),
        },
    )

    # Init a Gaze
    gaze = Gaze(
        samples=df,
        experiment=experiment_fixture,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
    )

    return gaze, request.param


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {'cmap': 'jet'}, id='str_cmap',
        ),
        pytest.param(
            {'cmap': matplotlib.colors.ListedColormap(['red', 'blue', 'green'])}, id='custom_cmap',
        ),
        pytest.param(
            {'gridsize': (10, 10)}, id='default_gridsize',
        ),
        pytest.param(
            {'gridsize': (15, 20)}, id='custom_gridsize',
        ),
        pytest.param(
            {'interpolation': 'gaussian'}, id='default_interpolation',
        ),
        pytest.param(
            {'interpolation': 'bilinear'}, id='custom_interpolation',
        ),
        pytest.param(
            {'origin': 'lower'}, id='default_origin',
        ),
        pytest.param(
            {'origin': 'upper'}, id='custom_origin',
        ),
        pytest.param(
            {
                'title': None,
                'xlabel': None,
                'ylabel': None,
                'cbar_label': None,
            }, id='default_labels',
        ),
        pytest.param(
            {
                'title': 'Custom Title',
                'xlabel': 'Custom X Label',
                'ylabel': 'Custom Y Label',
                'cbar_label': 'Custom Colorbar Label',
            },
            id='custom_labels',
        ),
        pytest.param(
            {'show_cbar': True}, id='show_cbar_true',
        ),
        pytest.param(
            {'show_cbar': False}, id='show_cbar_false',
        ),
        pytest.param(
            {
                'add_stimulus': True,
                'path_to_image_stimulus': './tests/files/pexels-zoorg-1000498.jpg',
                'stimulus_origin': 'lower',
            }, id='add_stimulus_true',
        ),
        pytest.param(
            {'add_stimulus': False}, id='add_stimulus_false',
        ),
    ],
)
def test_heatmap_show(args, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    if args[1] == 'pix':
        kwargs['position_column'] = 'pixel'
    else:
        kwargs['position_column'] = 'position'

    heatmap(args[0], **kwargs)
    plt.close()
    mock.assert_called_once()


def test_heatmap_noshow(args, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    if args[1] == 'pix':
        position_column = 'pixel'
    else:
        position_column = 'position'

    heatmap(args[0], position_column=position_column, show=False)
    plt.close()
    mock.assert_not_called()


def test_heatmap_noshow_no_pixel_or_position_column(args, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    if args[1] == 'pix':
        position_column = 'pixel'
    else:
        position_column = 'position'

    gaze = args[0]
    gaze.samples = gaze.samples.rename({position_column: 'custom_column'})

    heatmap(gaze, position_column='custom_column', show=False)
    plt.close()
    mock.assert_not_called()


def test_heatmap_save(args, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)

    if args[1] == 'pix':
        position_column = 'pixel'
    else:
        position_column = 'position'

    heatmap(
        args[0], position_column=position_column, show=False, savepath=str(tmp_path / 'test.svg'),
    )
    plt.close()
    mock.assert_called_once()


def test_heatmap_invalid_position_columns(args):
    if args[1] == 'pix':
        position_column = 'position'
    else:
        position_column = 'pixel'

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        heatmap(gaze=args[0], position_column=position_column, show=False)


def test_heatmap_no_experiment_property():
    df = pl.DataFrame(
        {
            'x_pix': np.arange(0, 100),
            'y_pix': np.arange(0, 100),
        },
    )

    gaze = Gaze(samples=df, pixel_columns=['x_pix', 'y_pix'], experiment=None)

    with pytest.raises(ValueError):
        heatmap(gaze, show=False)


@pytest.fixture(name='gaze')
def gaze_fixture():
    """Provide a minimal valid Gaze object for plotting tests."""
    df = pl.DataFrame({
        'x_pix': np.arange(100),
        'y_pix': np.arange(100),
    })

    experiment = pm.Experiment(
        screen_width_px=1024,
        screen_height_px=768,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=60,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = pm.Gaze(
        samples=df,
        experiment=experiment,
        pixel_columns=['x_pix', 'y_pix'],
    )

    return gaze


def test_heatmap_sets_screen_axes_correctly(gaze):
    fig, ax = pm.plotting.heatmap(gaze, show=False)
    screen = gaze.experiment.screen
    assert ax.get_xlim() == (0, screen.width_px)
    assert ax.get_ylim() == (screen.height_px, 0)
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_heatmap_invalid_screen_origin_raises(gaze):
    gaze.experiment.screen.origin = 'lower left'
    with pytest.raises(ValueError, match='screen origin must be "upper left"'):
        pm.plotting.heatmap(gaze, show=False)


def test_heatmap_with_missing_experiment_raises(gaze):
    gaze.experiment = None
    with pytest.raises(ValueError, match='Experiment property of Gaze is None'):
        pm.plotting.heatmap(gaze, show=False)


def test_heatmap_skips_screen_axes_if_no_gaze(monkeypatch):
    """Ensure _set_screen_axes is skipped when gaze is None."""
    mock = Mock()
    monkeypatch.setattr(pm_matplotlib, '_set_screen_axes', mock)

    # Call heatmap directly with gaze=None
    # (we expect it to raise an AttributeError or ValueError later, but not call _set_screen_axes)
    with pytest.raises((AttributeError, TypeError)):
        pm.plotting.heatmap(None, show=False)

    mock.assert_not_called()
