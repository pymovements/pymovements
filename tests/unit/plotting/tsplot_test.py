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

from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import pymovements as pm


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
            },
            id='zero_centered_yaxis_false_share_y_false',
        ),
        pytest.param({'show_yticks': False}, id='show_yticks_false'),
        pytest.param({'channels': ['x_pix']}, id='single_channel'),
        pytest.param({'channels': 'x_pix'}, id='single_channel_string'),
        pytest.param({'channels': ['x_pix', 'y_pix']}, id='two_channels'),
        pytest.param(
            {
                'channels': ['x_pix', 'y_pix'],
                'n_rows': 1,
                'n_cols': 2,
            },
            id='two_channels_explicit_rows_cols',
        ),
        pytest.param(
            {'channels': ['x_pix', 'y_pix'], 'rotate_ylabels': False},
            id='channels_no_rotate',
        ),
    ],
)
def test_tsplot_show(gaze, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    gaze.unnest('pixel', output_columns=['x_pix', 'y_pix'])
    pm.plotting.tsplot(gaze=gaze, **kwargs)

    mock.assert_called_once()


def test_tsplot_noshow(gaze, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    gaze.unnest('pixel', ['x_pix', 'y_pix'])
    pm.plotting.tsplot(gaze=gaze, show=False)

    mock.assert_not_called()


def test_tsplot_save(gaze, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    gaze.unnest('pixel', ['x_pix', 'y_pix'])
    pm.plotting.tsplot(gaze=gaze, show=False, savepath=str(tmp_path / 'test.svg'))

    mock.assert_called_once()


def test_tsplot_sets_title(gaze):
    _, ax = pm.plotting.tsplot(gaze, title='My Title', show=False)
    assert ax.get_title() == 'My Title'


@pytest.mark.parametrize(
    'bad_x, bad_y',
    [
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
        ]
        + gaze.samples['position'].to_list()[1:],
    )
    # get index of 'position' column
    pos_index = gaze.samples.get_column_index('position')
    # replace the 'position' column in gaze.samples with the new series
    gaze.samples = gaze.samples.with_columns(
        replacement_position,
        at_index=pos_index,
    )

    fig, ax = pm.plotting.tsplot(gaze=gaze, show=False)

    assert fig is not None
    assert ax is not None
