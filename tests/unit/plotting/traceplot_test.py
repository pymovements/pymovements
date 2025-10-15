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
"""Test traceplot."""
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import figure

import pymovements as pm


@pytest.fixture(name='gaze', scope='session')
def gaze_fixture():
    # pylint: disable=duplicate-code
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
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


@pytest.fixture(name='gaze_no_exp', scope='session')
def gaze_no_exp_fixture():
    # pylint: disable=duplicate-code
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    arr = np.column_stack((x, y)).transpose()

    gaze_no_exp = pm.gaze.from_numpy(
        samples=arr,
        schema=['x_pix', 'y_pix'],
        experiment=None,
        pixel_columns=['x_pix', 'y_pix'],
    )

    return gaze_no_exp


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {'cval': np.arange(-100, 100)},
            id='cval_array',
        ),
        pytest.param(
            {'cval': np.arange(-100, 100), 'cmap_norm': 'twoslope'},
            id='cmap_norm_twoslope',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'cmap_norm': 'nonorm'},
            id='cmap_norm_nonorm',
        ),
        pytest.param(
            {'cval': np.arange(0, 200)},
            id='cmap_norm_nonorm_implicit',
        ),
        pytest.param(
            {'cval': np.arange(-100, 100), 'cmap_norm': 'normalize'},
            id='cmap_norm_normalize',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'cmap_norm': 'linear'},
            id='cmap_norm_linear',
        ),
        pytest.param(
            {
                'cval': np.arange(0, 200),
                'cmap_norm': matplotlib.colors.NoNorm(),
            },
            id='cmap_norm_class',
        ),
        pytest.param(
            {
                'cmap': matplotlib.colors.LinearSegmentedColormap(
                    name='test', segmentdata={},
                ),
            },
            id='cmap_class',
        ),
        pytest.param(
            {'cmap_segmentdata': {}},
            id='cmap_segmentdata',
        ),
        pytest.param(
            {'padding': 0.1},
            id='padding',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'show_cbar': True},
            id='show_cbar_true',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'show_cbar': False},
            id='show_cbar_false',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'title': 'foo'},
            id='set_title',
        ),
        pytest.param(
            {
                'add_stimulus': True,
                'path_to_image_stimulus': './tests/files/pexels-zoorg-1000498.jpg',
            },
            id='set_stimulus',
        ),
    ],
)
def test_traceplot_show(gaze, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    pm.plotting.traceplot(gaze=gaze, **kwargs)
    plt.close()
    mock.assert_called_once()


def test_traceplot_noshow(gaze, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    pm.plotting.traceplot(gaze=gaze, show=False)
    plt.close()
    mock.assert_not_called()


def test_traceplot_save(gaze, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    pm.plotting.traceplot(
        gaze=gaze,
        show=False,
        savepath=str(
            tmp_path /
            'test.svg',
        ),
    )
    plt.close()
    mock.assert_called_once()


@pytest.mark.parametrize(
    ('kwargs', 'exception'),
    [
        pytest.param(
            {
                'cval': np.arange(0, 200),
                'cmap_norm': 'invalid',
            },
            ValueError,
            id='cmap_norm_unsupported',
        ),
    ],
)
def test_traceplot_exceptions(gaze, kwargs, exception, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    with pytest.raises(exception):
        pm.plotting.traceplot(gaze=gaze, **kwargs)


def test_traceplot_no_experiment(gaze_no_exp):
    # Should not raise any exception
    pm.plotting.traceplot(gaze_no_exp, show=False)
