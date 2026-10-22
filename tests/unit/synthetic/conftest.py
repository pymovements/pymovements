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
"""Shared fixtures for the synthetic data generation tests.

The toy generators here exist for the tests only; they implement no published method.
"""
import dataclasses

import numpy as np
import polars as pl
import pytest

from pymovements import Events
from pymovements import Gaze
from pymovements.synthetic import EventGenerator
from pymovements.synthetic import EventGeneratorLibrary
from pymovements.synthetic import SampleGenerator
from pymovements.synthetic import SampleGeneratorLibrary


@dataclasses.dataclass
class NoiseSampleGenerator(SampleGenerator):
    """Draws white-noise positions, ten samples per conditioning event or ten without events."""

    name = 'test_noise'

    scale: float = 1.0

    def _generate(self, *, events, gaze, sampling_rate, experiment, rng):
        n_samples = 10 if events is None else 10 * events.frame.height
        return Gaze(
            samples=pl.DataFrame({
                'time': np.arange(n_samples) * 1000.0 / sampling_rate,
                'x': rng.normal(0.0, self.scale, n_samples),
                'y': rng.normal(0.0, self.scale, n_samples),
            }),
            time_column='time',
            time_unit='ms',
            position_columns=['x', 'y'],
        )


@dataclasses.dataclass
class UniformEventGenerator(EventGenerator):
    """Fills the duration with fixations whose durations are drawn uniformly, in ms."""

    name = 'test_uniform'

    min_duration: float = 100.0
    max_duration: float = 300.0

    def _generate(self, *, duration, rng):
        onsets, offsets = [], []
        elapsed = 0.0
        while elapsed < duration * 1000.0:
            onsets.append(elapsed)
            elapsed = min(
                elapsed + rng.uniform(self.min_duration, self.max_duration), duration * 1000.0,
            )
            offsets.append(elapsed)

        return Events(
            name=['fixation'] * len(onsets), onsets=onsets, offsets=offsets, time_unit='ms',
        )


@pytest.fixture(name='noise_sample_generator')
def fixture_noise_sample_generator():
    """Return the toy sample generator class."""
    return NoiseSampleGenerator


@pytest.fixture(name='uniform_event_generator')
def fixture_uniform_event_generator():
    """Return the toy event generator class."""
    return UniformEventGenerator


@pytest.fixture(name='fixations')
def fixture_fixations():
    """Return a sequence of two fixations to condition on."""
    return Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560], time_unit='ms')


@pytest.fixture(name='isolated_registry')
def fixture_isolated_registry():
    """Restore both generator libraries after the test, so registrations do not leak."""
    libraries = (EventGeneratorLibrary, SampleGeneratorLibrary)
    originals = [library.generators.copy() for library in libraries]
    yield
    for library, original in zip(libraries, originals):
        library.generators.clear()
        library.generators.update(original)
