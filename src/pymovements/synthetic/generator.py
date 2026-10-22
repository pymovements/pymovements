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
"""Provides the base classes for synthetic event and sample generators."""
from __future__ import annotations

import abc
import dataclasses
import secrets
from typing import Any
from typing import ClassVar
from warnings import warn

import numpy as np

from pymovements._version import __version__
from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic._utils import check_positive
from pymovements.synthetic.provenance import generation_provenance
from pymovements.warnings import ExperimentalWarning


class _Generator(abc.ABC):
    """Shared behaviour of event and sample generators."""

    name: ClassVar[str]

    level: ClassVar[str]

    def hyperparameters(self) -> dict[str, Any]:
        """Return the parameters the generator was constructed with.

        For a dataclass generator these are its fields, except those declared with
        ``repr=False``, such as trained weights. Other generators record none.

        Returns
        -------
        dict[str, Any]
            The hyperparameters, keyed by name.
        """
        if not dataclasses.is_dataclass(self):
            return {}

        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.repr
        }

    def _provenance(self, run: dict[str, Any], seed: int) -> dict[str, Any]:
        """Return the provenance record of a run of this generator."""
        return generation_provenance(
            generator=self.name,
            level=self.level,
            hyperparameters=self.hyperparameters(),
            run=run,
            seed=seed,
        )


class EventGenerator(_Generator):
    """Base class for generators that produce an event sequence.

    An event generator decides which events occur and how long they last, for example a sequence
    of fixations and saccades. Subclasses take their hyperparameters in the constructor, set
    ``name``, and put their logic in ``_generate``.

    Attributes
    ----------
    level: ClassVar[str]
        ``'events'``, the kind of data the generator produces.
    """

    level: ClassVar[str] = 'events'

    def generate(self, *, duration: float, seed: int | None = None) -> Events:
        """Generate a synthetic event sequence.

        Parameters
        ----------
        duration: float
            The duration of the sequence in seconds.
        seed: int | None
            The random seed, or ``None`` to draw one. The seed used is recorded in the output's
            metadata, so the run can be reproduced either way. (default: None)

        Returns
        -------
        Events
            The generated events, with the run recorded under the ``'synthetic'`` key of their
            metadata.

        Raises
        ------
        TypeError
            If the duration is not a number or the seed is not an integer.
        ValueError
            If the duration is not greater than zero or the seed is negative.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        _warn_experimental()
        duration = check_positive(duration, 'duration')
        seed = _resolve_seed(seed)

        generated = self._generate(duration=duration, rng=np.random.default_rng(seed))

        generated.metadata = {
            **(generated.metadata or {}),
            **self._provenance({'duration': duration}, seed),
        }
        return generated

    @abc.abstractmethod
    def _generate(self, *, duration: float, rng: np.random.Generator) -> Events:
        """Synthesize the events. All randomness must be drawn from ``rng``.

        Parameters
        ----------
        duration: float
            The duration of the sequence in seconds.
        rng: np.random.Generator
            The run's random generator.

        Returns
        -------
        Events
            The generated events.
        """


class SampleGenerator(_Generator):
    """Base class for generators that produce gaze samples.

    A sample generator turns its conditioning input, an event sequence, a gaze sequence or
    nothing, into gaze samples. Subclasses take their hyperparameters in the constructor, set
    ``name``, and put their logic in ``_generate``. Supplied events are attached to the output
    unchanged, so the output carries its ground truth.

    Attributes
    ----------
    level: ClassVar[str]
        ``'samples'``, the kind of data the generator produces.
    """

    level: ClassVar[str] = 'samples'

    def generate(
            self,
            events: Events | None = None,
            *,
            gaze: Gaze | None = None,
            sampling_rate: float,
            seed: int | None = None,
            experiment: Experiment | None = None,
    ) -> Gaze:
        """Generate synthetic gaze samples.

        Parameters
        ----------
        events: Events | None
            An event sequence to condition on. (default: None)
        gaze: Gaze | None
            A gaze sequence to condition on. (default: None)
        sampling_rate: float
            The sampling rate of the generated samples in Hz.
        seed: int | None
            The random seed, or ``None`` to draw one. The seed used is recorded in the output's
            metadata, so the run can be reproduced either way. (default: None)
        experiment: Experiment | None
            The experiment the output belongs to. Without one, an experiment with only the
            sampling rate is attached. (default: None)

        Returns
        -------
        Gaze
            The generated gaze samples, with the run recorded under the ``'synthetic'`` key of
            their metadata.

        Raises
        ------
        TypeError
            If the sampling rate is not a number or the seed is not an integer.
        ValueError
            If the sampling rate is not greater than zero, the seed is negative, or the
            experiment has another sampling rate.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        _warn_experimental()
        sampling_rate = check_positive(sampling_rate, 'sampling_rate')
        seed = _resolve_seed(seed)
        experiment = _resolve_experiment(experiment, sampling_rate)

        generated = self._generate(
            events=events,
            gaze=gaze,
            sampling_rate=sampling_rate,
            experiment=experiment,
            rng=np.random.default_rng(seed),
        )

        generated.experiment = experiment
        if events is not None:
            generated.events = events.clone()

        run = {
            'sampling_rate': sampling_rate,
            'n_events': None if events is None else events.frame.height,
            'n_gaze_samples': None if gaze is None else gaze.samples.height,
        }
        generated.metadata = {**(generated.metadata or {}), **self._provenance(run, seed)}
        return generated

    @abc.abstractmethod
    def _generate(
            self,
            *,
            events: Events | None,
            gaze: Gaze | None,
            sampling_rate: float,
            experiment: Experiment,
            rng: np.random.Generator,
    ) -> Gaze:
        """Synthesize the samples. All randomness must be drawn from ``rng``.

        Parameters
        ----------
        events: Events | None
            The event sequence to condition on, if any.
        gaze: Gaze | None
            The gaze sequence to condition on, if any.
        sampling_rate: float
            The sampling rate of the generated samples in Hz.
        experiment: Experiment
            The experiment the output belongs to.
        rng: np.random.Generator
            The run's random generator.

        Returns
        -------
        Gaze
            The generated samples.
        """


def _warn_experimental() -> None:
    """Warn that synthetic data generation is experimental."""
    warn(
        'Synthetic gaze generation is experimental. '
        'Names and behavior may change without being considered a breaking change. '
        'Please set the used pymovements version explicitly to prevent unexpected changes. '
        f'The used pymovements version is v{__version__}.',
        ExperimentalWarning,
    )


def _resolve_seed(seed: Any) -> int:
    """Return the seed, or one drawn from the operating system's entropy if it is ``None``."""
    if seed is None:
        return secrets.randbits(128)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f'seed must be an integer but is of type {type(seed).__name__}: {seed}')

    if seed < 0:
        raise ValueError(f'seed must not be negative but is: {seed}')

    return seed


def _resolve_experiment(experiment: Experiment | None, sampling_rate: float) -> Experiment:
    """Return the experiment to attach, rejecting one recorded at another sampling rate."""
    if experiment is None:
        return Experiment(sampling_rate=sampling_rate)

    if experiment.sampling_rate != sampling_rate:
        raise ValueError(
            f'experiment has a sampling rate of {experiment.sampling_rate} Hz but '
            f'{sampling_rate} Hz was requested, velocities computed from the output would be '
            'wrong',
        )

    return experiment
