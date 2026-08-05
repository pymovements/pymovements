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
import pytest

from pymovements import Experiment


@pytest.mark.parametrize(
    ('spec', 'width_px', 'height_px', 'sampling_rate'), [
        ('1280x1024@1000Hz', 1280, 1024, 1000.0),
        ('1024x768@500Hz', 1024, 768, 500.0),
        ('800x600@62.5Hz', 800, 600, 62.5),
        ('1920x1080@240Hz', 1920, 1080, 240.0),
    ],
)
def test_make_experiment_parses_spec(make_experiment, spec, width_px, height_px, sampling_rate):
    experiment = make_experiment(spec)
    assert experiment.screen.width_px == width_px
    assert experiment.screen.height_px == height_px
    assert experiment.sampling_rate == sampling_rate


def test_make_experiment_returns_experiment(make_experiment):
    assert isinstance(make_experiment(), Experiment)


def test_make_experiment_default(make_experiment):
    experiment = make_experiment()
    assert experiment.screen.width_px == 1280
    assert experiment.screen.height_px == 1024
    assert experiment.sampling_rate == 1000.0


def test_make_experiment_case_insensitive_hz(make_experiment):
    assert make_experiment('640x480@100hz').sampling_rate == 100.0


def test_make_experiment_has_fixed_physical_defaults(make_experiment):
    experiment = make_experiment('1024x768@1000Hz')
    assert experiment.screen.width_cm == 38.0
    assert experiment.screen.height_cm == 30.0
    assert experiment.screen.distance_cm == 68.0


@pytest.mark.parametrize(
    'spec', ['1280x1024', '@1000Hz', '1280x1024@1000', 'foo', '1280x1024@Hz', ''],
)
def test_make_experiment_invalid_spec_raises(make_experiment, spec):
    with pytest.raises(ValueError, match='invalid experiment spec'):
        make_experiment(spec)
