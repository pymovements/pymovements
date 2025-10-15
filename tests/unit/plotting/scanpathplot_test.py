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
"""Test scanpathplot."""
import re
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib import figure

from pymovements import __version__
from pymovements import Events
from pymovements import Experiment
from pymovements.gaze import from_numpy
from pymovements.plotting import scanpathplot


@pytest.fixture(name='events', scope='session')
def event_fixture():
    yield Events(
        pl.DataFrame(
            data={
                'trial': [1, 1],
                'name': ['foo', 'foo'],
                'onset': [0, 2],
                'offset': [1, 3],
                'duration': [1, 1],
                'location': [(1, 2), (2, 3)],
            },
        ),
    ).clone()


@pytest.fixture(name='gaze', scope='session')
def gaze_fixture(events):
    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='upper left',
        sampling_rate=1000.0,
    )
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    arr = np.column_stack((x, y)).transpose()
    gaze = from_numpy(
        samples=arr,
        schema=['x_pix', 'y_pix'],
        experiment=experiment,
        pixel_columns=['x_pix', 'y_pix'],
    )

    gaze.events = events

    gaze.pix2deg()
    gaze.pos2vel()

    yield gaze.clone()


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
            {'cval': np.arange(0, 200), 'cmap_norm': matplotlib.colors.NoNorm()},
            id='cmap_norm_class',
        ),
        pytest.param(
            {'cmap': matplotlib.colors.LinearSegmentedColormap(name='test', segmentdata={})},
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
            {'add_traceplot': True},
            id='set_traceplot',
        ),
        pytest.param(
            {'add_traceplot': True, 'cval': np.arange(0, 200), 'show_cbar': True},
            id='set_traceplot_and_cbar',
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
def test_scanpathplot_show(gaze, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    scanpathplot(gaze=gaze, **kwargs)
    plt.close()
    mock.assert_called_once()


def test_scanpathplot_noshow(gaze, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    scanpathplot(gaze=gaze, show=False)
    plt.close()
    mock.assert_not_called()


def test_scanpathplot_save(gaze, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    scanpathplot(
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
def test_scanpathplot_exceptions(gaze, kwargs, exception, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    with pytest.raises(exception):
        scanpathplot(gaze=gaze, **kwargs)


def test_scanpathplot_gaze_events_all_none_exception():
    with pytest.raises(TypeError, match='must not be both None'):
        scanpathplot(gaze=None, events=None)


def test_scanpathplot_traceplot_gaze_samples_none_exception(gaze):
    gaze = gaze.clone()
    gaze.samples = None
    with pytest.raises(TypeError, match='must not be None'):
        scanpathplot(events=None, gaze=gaze, add_traceplot=True)


def test_scanpathplot_gaze_events_none_exception(gaze):
    gaze = gaze.clone()
    gaze.events = None
    with pytest.raises(TypeError, match='must not be None'):
        scanpathplot(gaze=gaze)


def test_scanpathplot_events_is_deprecated(gaze):
    with pytest.raises(DeprecationWarning) as info:
        scanpathplot(events=gaze.events)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'scnpatplot argument "events" was scheduled to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )


def test_scanpathplot_no_experiment(gaze):
    # test if gaze is not None and gaze.experiment is not None:
    gaze = gaze.clone()
    gaze.experiment = None
    # Should not raise any exception
    scanpathplot(gaze=gaze, show=False)
