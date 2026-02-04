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
"""Test scanpathplot."""
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib import figure

from pymovements import Events
from pymovements import Experiment
from pymovements import Gaze
from pymovements.gaze import from_numpy
from pymovements.plotting import scanpathplot


@pytest.fixture(name='make_events', scope='function')
def make_events_fixture():
    def _make_events_fixture(name: str) -> Events:
        if name == '0_events':
            events = pl.DataFrame(
                schema={
                    'trial': pl.Int64,
                    'name': pl.String,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                    'location': pl.List(pl.Int64),
                },
            )
        elif name == '1_fixation':
            events = pl.DataFrame(
                data={
                    'trial': [1],
                    'name': ['fixation'],
                    'onset': [0],
                    'offset': [1],
                    'duration': [1],
                    'location': [(1, 2)],
                },
            )
        elif name == '2_fixations':
            events = pl.DataFrame(
                data={
                    'trial': [1, 1],
                    'name': ['fixation', 'fixation'],
                    'onset': [0, 2],
                    'offset': [1, 3],
                    'duration': [1, 1],
                    'location': [(1, 2), (2, 3)],
                },
            )
        elif name == '3_events':
            events = pl.DataFrame(
                data={
                    'trial': [1, 1, 1],
                    'name': ['fixation', 'saccade', 'foo'],
                    'onset': [0, 2, 5],
                    'offset': [1, 3, 9],
                    'duration': [1, 1, 4],
                    'location': [(1, 2), (2, 3), (5, 5)],
                },
            )
        else:
            raise ValueError(f'{name} not supported as events fixture name')

        return Events(events)

    return _make_events_fixture


@pytest.fixture(
    name='events',
    params=[
        '0_events',
        '1_fixation',
        '2_fixations',
        '3_events',
    ],
    scope='function',
)
def event_fixture(request, make_events):
    yield make_events(request.param)


@pytest.fixture(name='make_gaze', scope='function')
def make_gaze_fixture(make_events):
    def _make_gaze_fixture(name: str) -> Gaze:
        events = make_events(name)

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

        return gaze

    return _make_gaze_fixture


@pytest.fixture(
    name='gaze',
    params=[
        '0_events',
        '1_fixation',
        '2_fixations',
        '3_events',
    ],
    scope='function',
)
def gaze_fixture(request, make_gaze):
    yield make_gaze(request.param)


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
            {
                'add_traceplot': True,
                'add_arrows': False,
            },
            id='set_traceplot',
        ),
        pytest.param(
            {
                'add_traceplot': True,
                'add_arrows': False,
                'cval': np.arange(0, 200),
                'show_cbar': True,
            },
            id='set_traceplot_and_cbar',
        ),
        pytest.param(
            {
                'add_stimulus': True,
                'path_to_image_stimulus': './tests/files/stimuli/pexels-zoorg-1000498.jpg',
            },
            id='set_stimulus',
        ),
        pytest.param(
            {
                'add_arrows': True,
                'arrow_color': 'blue',
                'arrow_rad': 0.0,
                'arrowstyle': '->',
            },
            id='param_arrows',
        ),
    ],
)
def test_scanpathplot_show(gaze, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    scanpathplot(gaze=gaze, **kwargs)

    mock.assert_called_once()


def test_scanpathplot_noshow(gaze, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    scanpathplot(gaze=gaze, show=False)

    mock.assert_not_called()


@pytest.mark.parametrize(
    ('make_gaze_param', 'event_name', 'expected_n_circles'),
    [
        pytest.param('0_events', 'fixation', 0, id='fixation'),
        pytest.param('0_events', 'saccade', 0, id='fixation'),
        pytest.param('1_fixation', 'fixation', 1, id='fixation'),
        pytest.param('1_fixation', 'saccade', 0, id='fixation'),
        pytest.param('2_fixations', 'fixation', 2, id='fixation'),
        pytest.param('3_events', 'fixation', 1, id='fixation'),
        pytest.param('3_events', 'saccade', 1, id='saccade'),
        pytest.param('3_events', 'foo', 1, id='foo'),
    ],
)
def test_scanpathplot_filter_events_plots_expected_circles(
        make_gaze_param, event_name, expected_n_circles, make_gaze,
):
    gaze = make_gaze(make_gaze_param)
    _, ax = scanpathplot(gaze=gaze, event_name=event_name, show=False)

    assert all(isinstance(patch, plt.Circle) for patch in ax.patches)
    assert len(ax.patches) == expected_n_circles


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
    gaze.samples = None
    with pytest.raises(TypeError, match='must not be None'):
        scanpathplot(events=None, gaze=gaze, add_traceplot=True)


def test_scanpathplot_gaze_events_none_exception(gaze):
    gaze.events = None
    with pytest.raises(TypeError, match='must not be None'):
        scanpathplot(gaze=gaze)


def test_scanpathplot_events_is_deprecated(gaze, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        scanpathplot(events=gaze.events)

    assert_deprecation_is_removed(
        function_name='scanpathplot() argument events',
        warning_message=info.value.args[0],
        scheduled_version='0.28.0',

    )


def test_scanpathplot_no_experiment(gaze):
    # test if gaze is not None and gaze.experiment is not None:
    gaze.experiment = None
    # Should not raise any exception
    scanpathplot(gaze=gaze, show=False)


def test_set_screen_axes_valid(gaze):
    _, ax = scanpathplot(
        gaze=gaze,
        show=False,
    )
    assert ax.get_xlim() == (0, gaze.experiment.screen.width_px)
    assert ax.get_ylim() == (gaze.experiment.screen.height_px, 0)
    assert ax.get_aspect() == 1


@pytest.mark.parametrize('origin', ['lower left', 'center'])
def test_set_screen_axes_invalid_origin(origin, gaze):
    gaze.experiment.screen.origin = origin
    with pytest.raises(ValueError, match='screen origin must be "upper left"'):
        scanpathplot(gaze=gaze, show=False)


@pytest.mark.parametrize(
    'width,height',
    [
        (None, None),
        (None, 768),
        (1024, None),
    ],
)
def test_set_screen_axes_none_dimensions_returns(width, height, gaze):
    """Should not raise or override axes when screen dimensions are None."""
    gaze.experiment.screen.width_px = width
    gaze.experiment.screen.height_px = height

    _, ax = plt.subplots()

    # Call scanpathplot; should return silently, without ValueError
    # _set_screen_axes() should return early without modifying axes
    scanpathplot(gaze=gaze, show=False, ax=ax, figsize=None)

    # Axes limits should be finite numbers, not NaN/None
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert np.isfinite(xlim).all()
    assert np.isfinite(ylim).all()

    # Aspect ratio should not be 'equal' (not forced by _set_screen_axes)
    assert ax.get_aspect() != 'equal'
