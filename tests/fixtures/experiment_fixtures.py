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
"""Provide shared fixtures for experiment tests."""
from __future__ import annotations

from collections.abc import Callable

import pytest

from pymovements import Experiment


@pytest.fixture(name='make_experiment', scope='function')
def fixture_make_experiment() -> Callable[..., Experiment]:
    """Return a factory that builds an Experiment with a 1280x1024 screen at 1000 Hz.

    Physical screen dimensions, viewing distance and origin use fixed defaults.
    Pass keyword arguments to change the resolution or the sampling rate, e.g.
    ``make_experiment(sampling_rate=100.0)``.

    Returns
    -------
    Callable[..., Experiment]
        Function that takes width_px, height_px and sampling_rate and returns an Experiment.

    """
    def _make_experiment(
            width_px: int = 1280,
            height_px: int = 1024,
            sampling_rate: float = 1000.0,
    ) -> Experiment:
        return Experiment(
            screen_width_px=width_px,
            screen_height_px=height_px,
            screen_width_cm=38.0,
            screen_height_cm=30.0,
            distance_cm=68.0,
            origin='upper left',
            sampling_rate=sampling_rate,
        )
    return _make_experiment
