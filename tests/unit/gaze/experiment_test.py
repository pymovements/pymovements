# Copyright (c) 2024-2025 The pymovements Project Authors
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
"""Test for Experiment class."""
import pytest

from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Screen


def test_sampling_rate_setter():
    experiment = Experiment(1280, 1024, 38, 30, sampling_rate=1000.0)
    assert experiment.sampling_rate == 1000.0

    experiment.sampling_rate = 100.0
    assert experiment.sampling_rate == 100.0


@pytest.mark.parametrize(
    'experiment_init_kwargs',
    [
        pytest.param(
            {},
            id='empty',
        ),
        pytest.param(
            {'sampling_rate': 1000},
            id='only_sampling_rate',
        ),
        pytest.param(
            {'screen': Screen(1024, 768)},
            id='only_screen',
        ),
        pytest.param(
            {'screen': Screen(1024, 768), 'eyetracker': EyeTracker(sampling_rate=1000)},
            id='screen_and_eyetracker',
        ),

    ],
)
def test_sampling_rate_trivial_equality(experiment_init_kwargs):
    experiment1 = Experiment(**experiment_init_kwargs)
    experiment2 = Experiment(**experiment_init_kwargs)
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('experiment1', 'experiment2'),
    [
        pytest.param(
            Experiment(sampling_rate=1000),
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='explicit_sampling_rate_and_eyetracker',
        ),
        pytest.param(
            Experiment(1024, 768),
            Experiment(screen=Screen(1024, 768)),
            id='explicit_screen_size_and_screen',
        ),
    ],
)
def test_sampling_rate_equality(experiment1, experiment2):
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('dictionary', 'expected_experiment'),
    [
        pytest.param(
            {'sampling_rate': 1000},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='sampling_rate',
        ),

        pytest.param(
            {'eyetracker': {'sampling_rate': 1000}},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='eyetracker_sampling_rate',
        ),

        pytest.param(
            {'screen_width_px': 1024, 'screen_height_px': 768},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_screen_height_px',
        ),

        pytest.param(
            {'screen': {'width_px': 1024, 'height_px': 768}},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_height_px',
        ),
    ],
)
def test_experiment_from_dict(dictionary, expected_experiment):
    experiment = Experiment.from_dict(dictionary)
    assert experiment == expected_experiment


@pytest.mark.parametrize(
    ('experiment', 'exclude_none', 'expected_dict'),
    [
        pytest.param(
            Experiment(),
            True,
            {},
            id='true_default',
        ),
        pytest.param(
            Experiment(origin=None),
            True,
            {},
            id='true_origin_none',
        ),
        pytest.param(
            Experiment(sampling_rate=18.5),
            True,
            {'eyetracker': {'sampling_rate': 18.5}},
            id='true_sampling_rate_18.5',
        ),
        pytest.param(
            Experiment(sampling_rate=18.5, origin=None),
            True,
            {'eyetracker': {'sampling_rate': 18.5}},
            id='true_sampling_rate_18.5_origin_none',
        ),
        pytest.param(
            Experiment(screen=Screen(height_px=1080), eyetracker=EyeTracker(left=True)),
            True,
            {
                'screen': {
                    'height_px': 1080,
                },
                'eyetracker': {
                    'left': True,
                },
            },
            id='true_screen_eyetracker',
        ),
        pytest.param(
            Experiment(
                screen=Screen(height_px=1080, origin=None),
                eyetracker=EyeTracker(left=True),
            ),
            True,
            {
                'screen': {
                    'height_px': 1080,
                },
                'eyetracker': {
                    'left': True,
                },
            },
            id='true_screen_origin_none_eyetracker',
        ),
        pytest.param(
            Experiment(),
            False,
            {
                'screen': {
                    'width_px': None,
                    'height_px': None,
                    'width_cm': None,
                    'height_cm': None,
                    'distance_cm': None,
                    'origin': None,
                },
                'eyetracker': {
                    'sampling_rate': None,
                    'vendor': None,
                    'model': None,
                    'version': None,
                    'mount': None,
                    'left': None,
                    'right': None,
                },
            },
            id='false_default',
        ),
        pytest.param(
            Experiment(origin=None),
            False,
            {
                'screen': {
                    'width_px': None,
                    'height_px': None,
                    'width_cm': None,
                    'height_cm': None,
                    'distance_cm': None,
                    'origin': None,
                },
                'eyetracker': {
                    'sampling_rate': None,
                    'vendor': None,
                    'model': None,
                    'version': None,
                    'mount': None,
                    'left': None,
                    'right': None,
                },
            },
            id='false_all_none',
        ),
    ],
)
def test_experiment_to_dict_exclude_none(experiment, exclude_none, expected_dict):
    assert experiment.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    ('experiment', 'expected_bool'),
    [
        pytest.param(
            Experiment(),
            False,
            id='default',
        ),

        pytest.param(
            Experiment(origin=None),
            False,
            id='origin_none',
        ),

        pytest.param(
            Experiment(origin='center'),
            True,
            id='origin_center',
        ),

        pytest.param(
            Experiment(distance_cm=60),
            True,
            id='distance_60',
        ),
    ],
)
def test_experiment_bool(experiment, expected_bool):
    assert bool(experiment) == expected_bool
