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
from pymovements import Experiment


def test_make_experiment_returns_experiment(make_experiment):
    assert isinstance(make_experiment(), Experiment)


def test_make_experiment_default(make_experiment):
    experiment = make_experiment()
    assert experiment.screen.width_px == 1280
    assert experiment.screen.height_px == 1024
    assert experiment.sampling_rate == 1000.0


def test_make_experiment_custom_resolution_and_rate(make_experiment):
    experiment = make_experiment(width_px=1024, height_px=768, sampling_rate=62.5)
    assert experiment.screen.width_px == 1024
    assert experiment.screen.height_px == 768
    assert experiment.sampling_rate == 62.5


def test_make_experiment_has_fixed_physical_defaults(make_experiment):
    experiment = make_experiment(sampling_rate=500.0)
    assert experiment.screen.width_cm == 38.0
    assert experiment.screen.height_cm == 30.0
    assert experiment.screen.distance_cm == 68.0
