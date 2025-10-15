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
"""Test from gaze.from_pandas."""
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events
from pymovements import Experiment
from pymovements.gaze import from_pandas


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
def test_from_pandas_additional_time_column():
    pandas_df = pd.DataFrame(
        {
            'x_pix': [0, 1, 2, 3],
            'y_pix': [0, 1, 2, 3],
            'x_pos': [0, 1, 2, 3],
            'y_pos': [0, 1, 2, 3],
        },
    )

    experiment = Experiment(1280, 1024, 38, 30, 68, 'upper left', 1000.0)

    gaze = from_pandas(
        samples=pandas_df,
        experiment=experiment,
    )

    assert gaze.samples.shape == (4, 5)
    assert gaze.columns == list(pandas_df.columns) + ['time']


def test_from_pandas_explicit_columns():
    pandas_df = pd.DataFrame(
        {
            't': [101, 102, 103, 104],
            'd': [100, 100, 100, 100],
            'x_pix': [0, 1, 2, 3],
            'y_pix': [4, 5, 6, 7],
            'x_pos': [9, 8, 7, 6],
            'y_pos': [5, 4, 3, 2],
        },
    )

    experiment = Experiment(1280, 1024, 38, 30, 68, 'upper left', 1000.0)

    gaze = from_pandas(
        samples=pandas_df,
        experiment=experiment,
        time_column='t',
        distance_column='d',
        pixel_columns=['x_pix', 'y_pix'],
        position_columns=['x_pos', 'y_pos'],
    )

    expected = pl.DataFrame({
        'time': [101, 102, 103, 104],
        'distance': [100, 100, 100, 100],
        'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
        'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
    })

    assert_frame_equal(gaze.samples, expected)


def test_from_pandas_with_trial_columnms():
    pandas_df = pd.DataFrame(
        {
            'trial_id': [1, 1, 2, 2],
            't': [101, 102, 103, 104],
            'x_pix': [0, 1, 2, 3],
            'y_pix': [4, 5, 6, 7],
        },
    )

    experiment = Experiment(1280, 1024, 38, 30, 68, 'upper left', 1000.0)

    gaze = from_pandas(
        samples=pandas_df,
        experiment=experiment,
        trial_columns='trial_id',
        time_column='t',
        pixel_columns=['x_pix', 'y_pix'],
    )

    expected = pl.DataFrame({
        'trial_id': [1, 1, 2, 2],
        'time': [101, 102, 103, 104],
        'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
    })

    assert_frame_equal(gaze.samples, expected)
    assert gaze.trial_columns == ['trial_id']


@pytest.mark.parametrize(
    ('samples', 'events'),
    [
        pytest.param(
            pd.DataFrame(),
            None,
            id='events_none',
        ),

        pytest.param(
            pd.DataFrame(),
            Events(),
            id='events_empty',
        ),

        pytest.param(
            pd.DataFrame(),
            Events(name='fixation', onsets=[123], offsets=[345]),
            id='fixation',
        ),

        pytest.param(
            pd.DataFrame(),
            Events(name='saccade', onsets=[34123], offsets=[67345]),
            id='saccade',
        ),

    ],
)
def test_from_pandas_events(samples, events):
    if events is None:
        expected_events = Events().frame
    else:
        expected_events = events.frame

    gaze = from_pandas(samples=samples, events=events)

    assert_frame_equal(gaze.events.frame, expected_events)
    # We don't want the events point to the same reference.
    assert gaze.events.frame is not expected_events


@pytest.mark.filterwarnings('ignore:Gaze contains samples but no.*:UserWarning')
def test_from_pandas_data_argument_is_deprecated():
    pandas_df = pd.DataFrame(
        {
            'x_pix': [0, 1, 2, 3],
            'y_pix': [0, 1, 2, 3],
            'x_pos': [0, 1, 2, 3],
            'y_pos': [0, 1, 2, 3],
        },
    )

    with pytest.warns(DeprecationWarning):
        gaze = from_pandas(samples=None, data=pandas_df)

    assert gaze.samples.shape == (4, 4)


def test_from_pandas_data_argument_is_removed(assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        from_pandas(samples=None, data=pd.DataFrame())
    assert_deprecation_is_removed('keyword argument data', info.value.args[0], __version__)
