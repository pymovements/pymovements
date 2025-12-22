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
"""Test Gaze detect method."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('method', 'kwargs', 'gaze', 'expected'),
    [
        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'cyclops',
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(1, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            ),
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(1, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                events=pm.events.Events(
                    name='fixation',
                    onsets=[0],
                    offsets=[99],
                ),
            ),
            id='ivt_constant_position_monocular_fixation_six_components_eye_cyclops',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1e-5,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100,
                    steps=[20, 30, 70, 80],
                    values=[(9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            ),
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100,
                    steps=[20, 30, 70, 80],
                    values=[(9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                events=pm.Events(
                    name='saccade',
                    onsets=[20, 70],
                    offsets=[29, 79],
                ),
            ),
            id='microsaccades_four_steps_two_saccades',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                position=np.zeros((2, 100)),
                events=pm.Events(
                    name=['fixation', 'saccade'], onsets=[0, 50], offsets=[40, 100],
                ),
            ),
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                position=np.zeros((2, 100)),
                events=pm.Events(
                    name=['fixation', 'saccade', 'unclassified'],
                    onsets=[0, 50, 40],
                    offsets=[40, 100, 49],
                ),
            ),
            id='fill_fixation_10_ms_break_then_saccade_until_end_single_fill',
        ),

        pytest.param(
            'downsample',
            {'factor': 2},
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': np.arange(1000, 1010, 1),
                        'x_pix': np.arange(0, 1, 0.1),
                        'y_pix': np.arange(20, 21, 0.1),
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': np.arange(1000, 1010, 2),
                        'x_pix': np.arange(0, 1, 0.2),
                        'y_pix': [20.0, 20.2, 20.4, 20.6, 20.8],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='downsample_factor_2',
        ),

        pytest.param(
            'pix2deg',
            {},
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                    },
                ),
                experiment=pm.Experiment(100, 100, 100, 100, 100, 'center', 1000),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_center',
        ),

        pytest.param(
            'deg2pix',
            {'pixel_origin': 'center'},
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_dva': [26.335410003881346, 26.335410003881346],
                        'y_dva': [0.0, 0.0],
                    },
                ),
                experiment=pm.Experiment(100, 100, 100, 100, 100, 'center', 1000),
                position_columns=['x_dva', 'y_dva'],
            ),
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.335410003881346, 26.335410003881346],
                        'y_dva': [0.0, 0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='deg2pix_origin_center',
        ),

        pytest.param(
            'pos2vel',
            {'method': 'preceding'},
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                    },
                ),
                experiment=pm.Experiment(100, 100, 100, 100, 100, 'center', 1000),
                trial_columns='trial_id',
                position_columns=['x', 'y'],
            ),
            pm.Gaze(
                samples=pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                        'x_vel': [None, 0.0, 0.0, None, 0.0, 0.0],
                        'y_vel': [None, 100.0, 100.0, None, 100.0, 100.0],
                    },
                ),
                trial_columns='trial_id',
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_preceding_trialize_single_column_str',
        ),
    ],
)
def test_gaze_apply(method, kwargs, gaze, expected):
    gaze.apply(method, **kwargs)

    # the deg2pix test case results in a column order different to the default
    check_column_order = not method == 'deg2pix'

    assert_frame_equal(gaze.samples, expected.samples, check_column_order=check_column_order)
    assert_frame_equal(gaze.events.frame, expected.events.frame)


@pytest.mark.parametrize(
    ('method', 'kwargs', 'gaze', 'exception', 'exception_msg'),
    [
        pytest.param(
            'foobar',
            {},
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(1, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            ValueError,
            "unsupported method 'foobar'",
            id='unknown_method',
        ),

    ],
)
def test_gaze_apply_raises_exception(method, kwargs, gaze, exception, exception_msg):
    with pytest.raises(exception) as exc_info:
        gaze.apply(method, **kwargs)

    msg, = exc_info.value.args
    assert msg == exception_msg
