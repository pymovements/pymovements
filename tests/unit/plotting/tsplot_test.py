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
"""Test tsplot."""
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from pymovements import Events
from pymovements import Experiment
from pymovements.gaze import from_numpy
from pymovements.gaze import Gaze
from pymovements.plotting import tsplot


@pytest.fixture(
    name='gaze',
    params=[
        '200',
        '0',
        '1',
    ],
    scope='function',
)
def gaze_fixture(request):
    # pylint: disable=duplicate-code
    if request.param == '200':
        x = np.arange(-100, 100)
        y = np.arange(-100, 100)
    elif request.param == '1':
        x = np.array([1])
        y = np.array([2])
    elif request.param == '0':
        x = np.empty((1,))
        y = np.empty((1,))
    else:
        raise ValueError(f'{request.param} not supported as gaze fixture param')

    arr = np.column_stack((x, y)).transpose()

    events = Events(
        pl.DataFrame(
            {
                'name': ['fixation', 'saccade'],
                'onset': [100, 200],
                'offset': [150, 250],
            },
        ),
    )

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='upper left',
        sampling_rate=1000.0,
    )

    gaze = from_numpy(
        samples=arr,
        schema=['x_pix', 'y_pix'],
        experiment=experiment,
        pixel_columns=['x_pix', 'y_pix'],
        events=events,
    )

    gaze.pix2deg()
    gaze.pos2vel()

    return gaze


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='no_kwargs'),
        pytest.param({'share_y': False}, id='share_y_false'),
        pytest.param({'zero_centered_yaxis': True}, id='zero_centered_yaxis_true'),
        pytest.param({'zero_centered_yaxis': False}, id='zero_centered_yaxis_false'),
        pytest.param(
            {
                'zero_centered_yaxis': False,
                'share_y': False,
            }, id='zero_centered_yaxis_false_share_y_false',
        ),
        pytest.param({'show_yticks': False}, id='show_yticks_false'),
        pytest.param({'channels': ['x_pix']}, id='single_channel'),
        pytest.param({'channels': 'x_pix'}, id='single_channel_string'),
        pytest.param({'channels': ['x_pix', 'y_pix']}, id='two_channels'),
        pytest.param(
            {
                'channels': ['x_pix', 'y_pix'], 'n_rows': 1, 'n_cols': 2,
            },
            id='two_channels_explicit_rows_cols',
        ),
        pytest.param(
            {'channels': ['x_pix', 'y_pix'], 'rotate_ylabels': False},
            id='channels_no_rotate',
        ),
    ],
)
def test_tsplot_returns_fig_and_axes(gaze, kwargs):
    gaze.unnest('pixel', output_columns=['x_pix', 'y_pix'])
    fig, ax = tsplot(gaze=gaze, **kwargs)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_tsplot_noshow(gaze, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    gaze.unnest('pixel', ['x_pix', 'y_pix'])
    tsplot(gaze=gaze)

    mock.assert_not_called()


def test_tsplot_save(gaze, tmp_path):
    filepath = tmp_path / 'test.svg'
    assert not filepath.is_file()

    gaze.unnest('pixel', ['x_pix', 'y_pix'])
    tsplot(gaze=gaze, savepath=str(filepath))

    assert filepath.is_file()


def test_tsplot_sets_title(gaze):
    _, ax = tsplot(gaze, title='My Title')
    assert ax.get_title() == 'My Title'


@pytest.mark.parametrize(
    'bad_x, bad_y', [
        (np.inf, 0.0),
        (np.nan, 0.0),
        (np.inf, np.nan),
        (np.nan, np.inf),
    ],
)
def test_tsplot_handles_nan_inf_variations(gaze, bad_x, bad_y):
    # create a polars series with the length of samples["position"]
    replacement_position = pl.Series(
        'position',
        [
            [bad_x, bad_y],
        ] + gaze.samples['position'].to_list()[1:],
    )
    # get index of 'position' column
    pos_index = gaze.samples.get_column_index('position')
    # replace the 'position' column in gaze.samples with the new series
    gaze.samples = gaze.samples.with_columns(
        replacement_position,
        at_index=pos_index,
    )

    fig, ax = tsplot(gaze=gaze)

    assert fig is not None
    assert ax is not None


def test_tsplot_external_ax_ignored_when_multi_channel(gaze):
    # prepare fresh gaze with two channels unnested
    gaze.unnest('pixel', output_columns=['x_pix', 'y_pix'])

    fig, ax = plt.subplots()
    with pytest.warns(UserWarning):
        # Using external ax but with two channels -> expect warning and a new figure
        ret_fig, ret_ax = tsplot(
            gaze,
            channels=['x_pix', 'y_pix'],
            ax=ax,
        )
    assert ret_ax is not ax
    assert ret_fig is not fig


def test_tsplot_events(gaze):
    gaze.unnest('pixel', output_columns=['x_pix', 'y_pix'])
    fig, ax = tsplot(gaze=gaze, plot_events=True)

    assert len(ax.patches) == 2
    # tab10[0] (blue) and tab10[1] (orange) with alpha 0.5
    assert ax.patches[0].get_facecolor() == (
        0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.5,
    )
    assert ax.patches[1].get_facecolor() == (
        1.0, 0.4980392156862745, 0.054901960784313725, 0.5,
    )

    legend = fig.legend()
    assert [text.get_text() for text in legend.get_texts()] == ['fixation', 'saccade']


def test_tsplot_events_empty_events_frame():
    gaze = Gaze(
        samples=pl.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [3.0, 4.0, 5.0]}),
        events=Events(),
        pixel_columns=['x', 'y'],
    )
    gaze.unnest('pixel', output_columns=['x', 'y'])

    fig, ax = tsplot(gaze=gaze, plot_events=True)

    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) == 0


@pytest.mark.parametrize(
    ('plot_events', 'expected_n_patches'),
    [
        pytest.param(False, 0, id='plot_events_false'),
        pytest.param(True, 1, id='plot_events_true'),
    ],
)
def test_tsplot_without_time_column_uses_sample_index(plot_events, expected_n_patches):
    events = Events(
        pl.DataFrame({'name': ['fixation'], 'onset': [0], 'offset': [2]}),
    )
    gaze = Gaze(
        samples=pl.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [3.0, 4.0, 5.0]}),
        events=events,
        pixel_columns=['x', 'y'],
    )
    gaze.unnest('pixel', output_columns=['x', 'y'])
    assert 'time' not in gaze.samples.columns

    _, ax = tsplot(gaze=gaze, plot_events=plot_events)

    assert list(ax.get_lines()[0].get_xdata()) == [0, 1, 2]
    assert len(ax.patches) == expected_n_patches


def test_tsplot_events_cycles_colors_beyond_ten_event_names():
    event_names = [f'event_{i:02d}' for i in range(11)]
    events = Events(
        pl.DataFrame(
            {
                'name': event_names,
                'onset': [i * 10 for i in range(11)],
                'offset': [i * 10 + 5 for i in range(11)],
            },
        ),
    )
    gaze = Gaze(
        samples=pl.DataFrame(
            {'x': [float(i) for i in range(110)], 'y': [float(i) for i in range(110)]},
        ),
        events=events,
        pixel_columns=['x', 'y'],
    )
    gaze.unnest('pixel', output_columns=['x', 'y'])

    _, ax = tsplot(gaze=gaze, plot_events=True)

    assert len(ax.patches) == 11
    # tab10[0] (blue) with alpha 0.5
    assert ax.patches[0].get_facecolor() == (
        0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.5,
    )
    # the eleventh event name cycles back to tab10[0]
    assert ax.patches[10].get_facecolor() == (
        0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.5,
    )
