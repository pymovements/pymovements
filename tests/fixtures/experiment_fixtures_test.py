# Copyright (c) 2026 The pymovements Project Authors
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
"""Test experiment fixtures."""
from operator import attrgetter

import pytest

from pymovements import Experiment


def test_make_experiment_returns_experiment(make_experiment):
    assert isinstance(make_experiment(), Experiment)


def test_make_experiment_default(make_experiment):
    experiment = make_experiment()
    assert experiment.screen.width_px == 1024
    assert experiment.screen.height_px == 768
    assert experiment.screen.width_cm == 38.0
    assert experiment.screen.height_cm == 30.0
    assert experiment.screen.distance_cm == 60.0
    assert experiment.screen.origin == 'center'
    assert experiment.sampling_rate == 1000.0


@pytest.mark.parametrize(
    ('kwargs', 'attribute', 'expected'),
    [
        pytest.param({'screen_width_px': 1280}, 'screen.width_px', 1280, id='screen_width_px'),
        pytest.param({'screen_height_px': 1024}, 'screen.height_px', 1024, id='screen_height_px'),
        pytest.param({'screen_width_cm': 52.0}, 'screen.width_cm', 52.0, id='screen_width_cm'),
        pytest.param({'screen_height_cm': 29.0}, 'screen.height_cm', 29.0, id='screen_height_cm'),
        pytest.param({'distance_cm': 68.0}, 'screen.distance_cm', 68.0, id='distance_cm'),
        pytest.param({'origin': 'upper left'}, 'screen.origin', 'upper left', id='origin'),
        pytest.param({'sampling_rate': 100.0}, 'sampling_rate', 100.0, id='sampling_rate'),
    ],
)
def test_make_experiment_overrides_parameter(make_experiment, kwargs, attribute, expected):
    experiment = make_experiment(**kwargs)
    assert attrgetter(attribute)(experiment) == expected


@pytest.mark.parametrize(
    ('parameter', 'attribute'),
    [
        ('screen_width_px', 'screen.width_px'),
        ('screen_height_px', 'screen.height_px'),
        ('screen_width_cm', 'screen.width_cm'),
        ('screen_height_cm', 'screen.height_cm'),
        ('distance_cm', 'screen.distance_cm'),
        ('origin', 'screen.origin'),
        ('sampling_rate', 'sampling_rate'),
    ],
)
def test_make_experiment_none_leaves_parameter_unset(make_experiment, parameter, attribute):
    experiment = make_experiment(**{parameter: None})
    assert attrgetter(attribute)(experiment) is None


def test_make_experiment_returns_new_object_on_each_call(make_experiment):
    assert make_experiment() is not make_experiment()
