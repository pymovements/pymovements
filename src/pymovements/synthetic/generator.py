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
from collections.abc import Callable
from typing import Any
from typing import ClassVar
from warnings import warn

import numpy as np

from pymovements._version import __version__
from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.provenance import generation_provenance
from pymovements.warnings import ExperimentalWarning


class _Generator(abc.ABC):
    """Shared behaviour of event and sample generators."""

    name: ClassVar[str]

    level: ClassVar[str]

    # Each level declares the abstract method; concrete generators narrow its run parameters.
    _generate: Callable[..., Any]

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

    def _run(
            self,
            seed: int | None,
            run: dict[str, Any],
            inputs: dict[str, Any],
            record: dict[str, Any],
    ) -> Any:
        """Run ``_generate`` and record the run, the template both levels share.

        ``run`` holds the run parameters the generator declares in its ``_generate`` signature,
        where a misspelled one raises a ``TypeError``. ``inputs`` holds those the base class
        defines, and ``record`` the part of them written into the provenance.
        """
        _warn_experimental()
        seed = _resolve_seed(seed)

        generated = self._generate(rng=np.random.default_rng(seed), **inputs, **run)

        run_record = {**record, **self._record_run(run)}
        generated.metadata = {**(generated.metadata or {}), **self._provenance(run_record, seed)}
        return generated

    def _record_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Return the declared run parameters as recorded; events are recorded by their count.

        Override it for inputs that are not plain values.

        Parameters
        ----------
        run: dict[str, Any]
            The run parameters the generator declares, as passed to ``generate()``.

        Returns
        -------
        dict[str, Any]
            The record of the run parameters.
        """
        return {
            (f'n_{key}' if isinstance(value, Events) else key): (
                value.frame.height if isinstance(value, Events) else value
            )
            for key, value in run.items()
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
    ``name``, and put their logic in ``_generate``, whose keyword arguments besides ``rng`` are
    the generator's run parameters, such as a duration or an event sequence to complete.

    Attributes
    ----------
    level: ClassVar[str]
        ``'events'``, the kind of data the generator produces.
    """

    level: ClassVar[str] = 'events'

    def generate(self, *, seed: int | None = None, **run: Any) -> Events:
        """Generate a synthetic event sequence.

        Parameters
        ----------
        seed: int | None
            The random seed, or ``None`` to draw one. The seed used is recorded in the output's
            metadata, so the run can be reproduced either way. (default: None)
        **run: Any
            The run parameters the generator declares in its ``_generate``.

        Returns
        -------
        Events
            The generated events, with the run recorded under the ``'synthetic'`` key of their
            metadata.

        Raises
        ------
        TypeError
            If the seed is not an integer, or a run parameter is unknown to the generator or
            missing.
        ValueError
            If the seed is negative.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        generated: Events = self._run(seed, run, inputs={}, record={})
        return generated

    @abc.abstractmethod
    def _generate(self, *, rng: np.random.Generator, **run: Any) -> Events:
        """Synthesize the events. All randomness must be drawn from ``rng``.

        Parameters
        ----------
        rng: np.random.Generator
            The run's random generator.
        **run: Any
            The run parameters. Subclasses replace this with explicit keyword arguments.

        Returns
        -------
        Events
            The generated events.
        """


class SampleGenerator(_Generator):
    """Base class for generators that produce gaze samples.

    A sample generator turns its conditioning input into gaze samples. Subclasses take their
    hyperparameters in the constructor, set ``name``, and put their logic in ``_generate``, whose
    keyword arguments besides ``sampling_rate``, ``experiment`` and ``rng`` are the generator's
    run parameters. A run parameter named ``events`` is copied before ``_generate`` receives it,
    and that copy is attached to the output, so the output carries its ground truth.

    Attributes
    ----------
    level: ClassVar[str]
        ``'samples'``, the kind of data the generator produces.
    """

    level: ClassVar[str] = 'samples'

    def generate(
            self,
            *,
            sampling_rate: float,
            seed: int | None = None,
            experiment: Experiment | None = None,
            **run: Any,
    ) -> Gaze:
        """Generate synthetic gaze samples.

        Parameters
        ----------
        sampling_rate: float
            The sampling rate of the generated samples in Hz.
        seed: int | None
            The random seed, or ``None`` to draw one. The seed used is recorded in the output's
            metadata, so the run can be reproduced either way. (default: None)
        experiment: Experiment | None
            The experiment the output belongs to. Without one, an experiment with only the
            sampling rate is attached. (default: None)
        **run: Any
            The run parameters the generator declares in its ``_generate``, such as ``events``.

        Returns
        -------
        Gaze
            The generated gaze samples, with the run recorded under the ``'synthetic'`` key of
            their metadata.

        Raises
        ------
        TypeError
            If the sampling rate is not a number, the seed is not an integer, or a run parameter
            is unknown to the generator or missing.
        ValueError
            If the sampling rate is not greater than zero, the seed is negative, or the
            experiment has another sampling rate.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        sampling_rate = _check_positive_number(sampling_rate, 'sampling_rate')
        experiment = _resolve_experiment(experiment, sampling_rate)
        if isinstance(run.get('events'), Events):
            run['events'] = run['events'].clone()

        generated: Gaze = self._run(
            seed,
            run,
            inputs={'sampling_rate': sampling_rate, 'experiment': experiment},
            record={'sampling_rate': sampling_rate},
        )

        generated.experiment = experiment
        if isinstance(run.get('events'), Events):
            generated.events = run['events']
        return generated

    @abc.abstractmethod
    def _generate(
            self,
            *,
            sampling_rate: float,
            experiment: Experiment,
            rng: np.random.Generator,
            **run: Any,
    ) -> Gaze:
        """Synthesize the samples. All randomness must be drawn from ``rng``.

        Parameters
        ----------
        sampling_rate: float
            The sampling rate of the generated samples in Hz.
        experiment: Experiment
            The experiment the output belongs to.
        rng: np.random.Generator
            The run's random generator.
        **run: Any
            The run parameters. Subclasses replace this with explicit keyword arguments.

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


def _check_positive_number(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers and values of zero or less."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f'{name} must be a number but is of type {type(value).__name__}: {value}',
        )

    if value <= 0:
        raise ValueError(f'{name} must be greater than zero but is: {float(value)}')

    return float(value)


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
