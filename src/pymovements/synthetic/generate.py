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

import dataclasses
from typing import Any

from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.library import EventGeneratorLibrary
from pymovements.synthetic.library import SampleGeneratorLibrary


def generate_events(name: str, *, seed: int | None = None, **kwargs: Any) -> Events:
    """Generate a synthetic event sequence with the event generator registered under a name.

    Keyword arguments that match a field of the generator are its hyperparameters and go to the
    constructor, the others are run parameters and go to ``generate()``. A hyperparameter
    therefore cannot share its name with a run parameter or with ``seed``.

    Parameters
    ----------
    name: str
        The registered name of the event generator.
    seed: int | None
        The random seed, or ``None`` to draw one. (default: None)
    **kwargs: Any
        The hyperparameters and run parameters of the generator.

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
    generator_class = EventGeneratorLibrary.get(name)
    hyperparameters, run = _split_kwargs(generator_class, kwargs)
    return generator_class(**hyperparameters).generate(seed=seed, **run)


def generate_gaze(
        name: str,
        *,
        sampling_rate: float,
        seed: int | None = None,
        experiment: Experiment | None = None,
        **kwargs: Any,
) -> Gaze:
    """Generate synthetic gaze samples with the sample generator registered under a name.

    Keyword arguments that match a field of the generator are its hyperparameters and go to the
    constructor, the others are run parameters and go to ``generate()``. A hyperparameter
    therefore cannot share its name with a run parameter, ``sampling_rate``, ``seed`` or
    ``experiment``.

    Parameters
    ----------
    name: str
        The registered name of the sample generator.
    sampling_rate: float
        The sampling rate of the generated samples in Hz.
    seed: int | None
        The random seed, or ``None`` to draw one. (default: None)
    experiment: Experiment | None
        The experiment the output belongs to. (default: None)
    **kwargs: Any
        The hyperparameters and run parameters of the generator, such as ``events``.

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
    generator_class = SampleGeneratorLibrary.get(name)
    hyperparameters, run = _split_kwargs(generator_class, kwargs)
    return generator_class(**hyperparameters).generate(
        sampling_rate=sampling_rate, seed=seed, experiment=experiment, **run,
    )


def _split_kwargs(
        generator_class: type,
        kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the keyword arguments split into hyperparameters and run parameters."""
    fields = (
        {field.name for field in dataclasses.fields(generator_class) if field.init}
        if dataclasses.is_dataclass(generator_class) else set()
    )
    hyperparameters = {key: value for key, value in kwargs.items() if key in fields}
    run = {key: value for key, value in kwargs.items() if key not in fields}
    return hyperparameters, run
