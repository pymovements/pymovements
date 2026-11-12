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
    """Return a factory that builds a complete Experiment.

    The defaults are the experiment used most often in this suite: a 1024x768 screen of 38x30 cm
    at 60 cm distance with the origin at the center, sampled at 1000 Hz. Each of the seven
    positional parameters of :py:class:`~pymovements.Experiment` can be passed by keyword to
    override its default, e.g. ``make_experiment(sampling_rate=100.0)``. Passing ``None`` leaves
    that parameter unset.

    Returns
    -------
    Callable[..., Experiment]
        Function that takes the keyword arguments of Experiment and returns an Experiment.

    """
    def _make_experiment(
            screen_width_px: int | None = 1024,
            screen_height_px: int | None = 768,
            screen_width_cm: float | None = 38.0,
            screen_height_cm: float | None = 30.0,
            distance_cm: float | None = 60.0,
            origin: str | None = 'center',
            sampling_rate: float | None = 1000.0,
    ) -> Experiment:
        return Experiment(
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            screen_width_cm=screen_width_cm,
            screen_height_cm=screen_height_cm,
            distance_cm=distance_cm,
            origin=origin,
            sampling_rate=sampling_rate,
        )
    return _make_experiment
