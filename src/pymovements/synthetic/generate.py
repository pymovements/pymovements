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
"""Provides the entry points for generating synthetic data by generator name."""
from __future__ import annotations

from typing import Any

from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.library import EventGeneratorLibrary
from pymovements.synthetic.library import SampleGeneratorLibrary


def generate_events(
        name: str,
        *,
        duration: float,
        seed: int | None = None,
        **hyperparameters: Any,
) -> Events:
    """Generate a synthetic event sequence with the event generator registered under a name.

    Equivalent to ``EventGeneratorLibrary.get(name)(**hyperparameters).generate(...)``.

    Parameters
    ----------
    name: str
        The registered name of the event generator.
    duration: float
        The duration of the sequence in seconds.
    seed: int | None
        The random seed, or ``None`` to draw one. (default: None)
    **hyperparameters: Any
        The parameters the generator is constructed with.

    Returns
    -------
    Events
        The generated events, with the run recorded under the ``'synthetic'`` key of their
        metadata.

    Raises
    ------
    KeyError
        If no event generator is registered under that name.
    """
    generator = EventGeneratorLibrary.get(name)(**hyperparameters)
    return generator.generate(duration=duration, seed=seed)


def generate_gaze(
        name: str,
        events: Events | None = None,
        *,
        gaze: Gaze | None = None,
        sampling_rate: float,
        seed: int | None = None,
        experiment: Experiment | None = None,
        **hyperparameters: Any,
) -> Gaze:
    """Generate synthetic gaze samples with the sample generator registered under a name.

    Equivalent to ``SampleGeneratorLibrary.get(name)(**hyperparameters).generate(...)``.

    Parameters
    ----------
    name: str
        The registered name of the sample generator.
    events: Events | None
        An event sequence to condition on. (default: None)
    gaze: Gaze | None
        A gaze sequence to condition on. (default: None)
    sampling_rate: float
        The sampling rate of the generated samples in Hz.
    seed: int | None
        The random seed, or ``None`` to draw one. (default: None)
    experiment: Experiment | None
        The experiment the output belongs to. (default: None)
    **hyperparameters: Any
        The parameters the generator is constructed with.

    Returns
    -------
    Gaze
        The generated gaze samples, with the run recorded under the ``'synthetic'`` key of their
        metadata.

    Raises
    ------
    KeyError
        If no sample generator is registered under that name.
    """
    generator = SampleGeneratorLibrary.get(name)(**hyperparameters)
    return generator.generate(
        events,
        gaze=gaze,
        sampling_rate=sampling_rate,
        seed=seed,
        experiment=experiment,
    )
