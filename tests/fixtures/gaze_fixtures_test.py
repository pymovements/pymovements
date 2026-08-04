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
"""Test gaze fixtures."""
import polars as pl
import pytest

from pymovements import Events
from pymovements import Gaze


def test_gaze_all_has_experiment(gaze_all):
    assert gaze_all.experiment is not None
    assert gaze_all.experiment.screen is not None
    assert gaze_all.experiment.eyetracker is not None


def test_gaze_all_has_samples(gaze_all):
    assert gaze_all.samples is not None
    assert len(gaze_all.samples) > 0


def test_gaze_all_has_events(gaze_all):
    assert gaze_all.events is not None
    assert len(gaze_all.events.frame) > 0


def test_gaze_all_has_position_columns(gaze_all):
    assert gaze_all.n_components is not None
    assert gaze_all.n_components > 0


def test_gaze_minimal_has_samples(gaze_minimal):
    assert gaze_minimal.samples is not None
    assert len(gaze_minimal.samples) > 0


def test_gaze_minimal_has_events(gaze_minimal):
    assert gaze_minimal.events is not None
    assert len(gaze_minimal.events.frame) > 0


def test_gaze_minimal_has_pixel_columns(gaze_minimal):
    assert gaze_minimal.n_components is not None
    assert gaze_minimal.n_components > 0


def test_gaze_minimal_no_experiment(gaze_minimal):
    assert gaze_minimal.experiment is None


@pytest.mark.parametrize('fixture_name', ['gaze_all', 'gaze_minimal'])
def test_gaze_fixtures_are_gaze_objects(fixture_name, request):
    gaze = request.getfixturevalue(fixture_name)
    assert isinstance(gaze, Gaze)


def test_make_unvalidated_gaze_returns_gaze(make_unvalidated_gaze):
    gaze = make_unvalidated_gaze(pl.DataFrame({'time': [0, 1]}))
    assert isinstance(gaze, Gaze)


def test_make_unvalidated_gaze_defaults(make_unvalidated_gaze):
    gaze = make_unvalidated_gaze(pl.DataFrame({'time': [0, 1]}))
    assert gaze.trial_columns is None
    assert gaze.experiment is None
    assert gaze.events == Events()


def test_make_unvalidated_gaze_stores_samples(make_unvalidated_gaze):
    samples = pl.DataFrame({'time': [0, 1, 2]})
    gaze = make_unvalidated_gaze(samples)
    assert gaze.samples.equals(samples)


def test_make_unvalidated_gaze_allows_missing_trial_columns(make_unvalidated_gaze):
    gaze = make_unvalidated_gaze(pl.DataFrame({'time': [0]}), trial_columns=['missing'])
    assert gaze.trial_columns == ['missing']


def test_make_unvalidated_gaze_sets_experiment(make_unvalidated_gaze, make_experiment):
    experiment = make_experiment()
    gaze = make_unvalidated_gaze(pl.DataFrame({'time': [0]}), experiment=experiment)
    assert gaze.experiment is experiment
