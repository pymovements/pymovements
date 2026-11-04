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
import inspect
import numbers
import secrets
from collections.abc import Callable
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import TypeVar
from warnings import warn

import numpy as np

from pymovements._version import __version__
from pymovements.events.events import Events
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.provenance import generation_provenance
from pymovements.warnings import ExperimentalWarning

OutputT = TypeVar('OutputT', Events, Gaze)


class _Generator(abc.ABC, Generic[OutputT]):
    """Shared behaviour of event and sample generators."""

    name: ClassVar[str]

    level: ClassVar[Literal['events', 'samples']]

    # Each level declares the abstract method; concrete generators narrow its run parameters.
    _generate: Callable[..., OutputT]

    def hyperparameters(self) -> dict[str, Any]:
        """Return the parameters the generator was constructed with.

        These are the generator's dataclass fields, except those declared with ``repr=False``,
        such as trained weights.

        Returns
        -------
        dict[str, Any]
            The hyperparameters, keyed by name.

        Raises
        ------
        TypeError
            If the generator is not a dataclass.
        """
        return {
            field.name: getattr(self, field.name)
            for field in self._fields()
            if field.repr
        }

    def generate(self, *, seed: int | None = None, **run: Any) -> OutputT:
        """Generate synthetic data.

        Parameters
        ----------
        seed: int | None
            The random seed, or ``None`` to draw one. The seed used is recorded in the output's
            metadata, so the run can be reproduced either way. (default: None)
        **run: Any
            The run parameters the generator declares in its ``_generate``, such as a duration
            for an event generator or ``events`` for a sample generator.

        Returns
        -------
        OutputT
            The generated events or gaze samples, with the run recorded under the
            ``'synthetic'`` key of their metadata, defaults of ``_generate`` filled in. A sample
            generator attaches an :py:class:`~pymovements.Experiment` carrying its sampling rate,
            and a copy of a run parameter named ``events`` as the ground truth.

        Raises
        ------
        TypeError
            If the generator is not a dataclass or has no ``name``, the seed is not an integer,
            or a run parameter is unknown to the generator or missing.
        ValueError
            If the seed is negative.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        _warn_experimental()
        self._check_class()
        run = self._resolve_run(run)
        seed = _resolve_seed(seed)

        generated = self._run(np.random.default_rng(seed), run)

        generated.metadata = {
            **(generated.metadata or {}), **self._provenance(self._record_run(run), seed),
        }
        return generated

    def _run(self, rng: np.random.Generator, run: dict[str, Any]) -> OutputT:
        """Call ``_generate``; a level overrides this to prepare its inputs and its output."""
        return self._generate(rng=rng, **run)

    def _check_class(self) -> None:
        """Reject a generator whose run could not be recorded, before it generates anything."""
        fields = self._fields()
        generator_class = type(self).__name__

        name = getattr(self, 'name', None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{generator_class} must define the class attribute 'name' as a non-empty string, "
                "for example name = 'my_generator', so that its output records which generator "
                'made it',
            )

        if 'name' in {field.name for field in fields}:
            raise TypeError(
                f"{generator_class} declares 'name' with a type annotation, which makes it a "
                f"dataclass field and a hyperparameter; write name = '{name}' without annotation",
            )

    def _resolve_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Return the run parameters with the defaults of ``_generate`` filled in.

        A run that relies on a default is recorded with it, so it stays reproducible. A run
        parameter that ``_generate`` does not declare raises, naming the accepted ones.
        """
        parameters = inspect.signature(self._generate).parameters
        declared = {name: parameter for name, parameter in parameters.items() if name != 'rng'}

        unknown = sorted(set(run) - set(declared))
        if unknown:
            raise TypeError(
                f'{type(self).__name__}.generate() got unknown run parameters {unknown}; '
                f'it accepts: {sorted(declared)}',
            )

        defaults = {
            name: parameter.default for name, parameter in declared.items()
            if parameter.default is not parameter.empty
        }
        return {**defaults, **run}

    def _fields(self) -> tuple[dataclasses.Field[Any], ...]:
        """Return the dataclass fields, rejecting a generator whose hyperparameters are unknown."""
        if not dataclasses.is_dataclass(self):
            raise TypeError(
                f'{type(self).__name__} must be a dataclass, so that its hyperparameters are '
                'recorded and the run can be reproduced; decorate it with @dataclasses.dataclass',
            )

        return dataclasses.fields(self)

    def _record_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Return the declared run parameters as recorded; events are recorded by their count.

        Override it for inputs that are not plain values.

        Parameters
        ----------
        run: dict[str, Any]
            The run parameters the generator declares, as passed to ``generate()`` with the
            defaults of ``_generate`` filled in.

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


class EventGenerator(_Generator[Events]):
    """Base class for generators that produce an event sequence.

    An event generator decides which events occur and how long they last, for example a sequence
    of fixations and saccades. Subclasses are dataclasses whose fields are their hyperparameters.
    They set ``name`` and put their logic in ``_generate``, whose keyword arguments besides
    ``rng`` are the generator's run parameters, such as a duration or an event sequence to
    complete. ``name`` is written without a type annotation, since an annotated ``name`` becomes a
    dataclass field and a hyperparameter.

    Attributes
    ----------
    level: ClassVar[Literal['events', 'samples']]
        ``'events'``, the kind of data the generator produces.

    Examples
    --------
    .. testsetup::

        >>> getfixture('doctest_synthetic')

    A generator that fills the requested duration with fixations of equal length:

    >>> import dataclasses
    >>> import pymovements as pm
    >>> @pm.synthetic.register_generator
    ... @dataclasses.dataclass
    ... class EvenFixations(pm.synthetic.EventGenerator):
    ...     name = 'even_fixations'
    ...     fixation_duration: float = 250.0
    ...
    ...     def _generate(self, *, rng, duration):
    ...         count = int(duration * 1000 // self.fixation_duration)
    ...         onsets = [index * self.fixation_duration for index in range(count)]
    ...         offsets = [onset + self.fixation_duration for onset in onsets]
    ...         return pm.Events(name=['fixation'] * count, onsets=onsets, offsets=offsets)
    >>> events = EvenFixations().generate(duration=1.0, seed=42)
    >>> events.frame['name'].to_list()
    ['fixation', 'fixation', 'fixation', 'fixation']
    >>> events.metadata['synthetic']['run']
    {'duration': 1.0}
    """

    level: ClassVar[Literal['events', 'samples']] = 'events'

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


@dataclasses.dataclass(kw_only=True)
class SampleGenerator(_Generator[Gaze]):
    """Base class for generators that produce gaze samples.

    A sample generator turns its conditioning input into gaze samples at its sampling rate.
    Subclasses are dataclasses whose fields, with ``sampling_rate``, are their hyperparameters.
    They set ``name``, without a type annotation as for event generators, and put their logic in
    ``_generate``, whose keyword arguments besides ``rng`` are the generator's run parameters. A
    run parameter named ``events`` is copied before ``_generate`` receives it, and that copy is
    attached to the output, so the output carries its ground truth. The output also gets an
    :py:class:`~pymovements.Experiment` carrying the sampling rate, as a recording would.

    A generator built for one sampling rate redefines the field with that rate as its default. A
    subclass that defines ``__post_init__`` calls ``super().__post_init__()`` first, so the
    sampling rate is checked.

    Attributes
    ----------
    sampling_rate: float
        The sampling rate of the generated samples in Hz. A keyword-only field, so subclass
        fields may follow it with or without defaults.
    level: ClassVar[Literal['events', 'samples']]
        ``'samples'``, the kind of data the generator produces.

    Raises
    ------
    TypeError
        If the sampling rate is not a number.
    ValueError
        If the sampling rate is not greater than zero.

    Examples
    --------
    .. testsetup::

        >>> getfixture('doctest_synthetic')

    A generator that renders each event as gaze resting at the origin, with noise. It declares
    ``events`` as its run parameter:

    >>> import dataclasses
    >>> import numpy as np
    >>> import polars as pl
    >>> import pymovements as pm
    >>> @pm.synthetic.register_generator
    ... @dataclasses.dataclass
    ... class StillGaze(pm.synthetic.SampleGenerator):
    ...     name = 'still_gaze'
    ...     noise: float = 0.1
    ...
    ...     def _generate(self, *, rng, events):
    ...         end = events.frame['offset'].dt.total_microseconds().max() / 1000
    ...         time = np.arange(0, end, 1000 / self.sampling_rate)
    ...         return pm.Gaze(
    ...             samples=pl.DataFrame({
    ...                 'time': time,
    ...                 'x': rng.normal(0, self.noise, time.size),
    ...                 'y': rng.normal(0, self.noise, time.size),
    ...             }),
    ...             time_column='time',
    ...             time_unit='ms',
    ...             position_columns=['x', 'y'],
    ...         )
    >>> events = pm.Events(name=['fixation'], onsets=[0], offsets=[500])
    >>> gaze = StillGaze(sampling_rate=1000).generate(events=events, seed=42)
    >>> gaze.samples.height
    500
    >>> gaze.experiment.sampling_rate
    1000.0
    >>> gaze.events.frame.equals(events.frame)
    True
    """

    sampling_rate: float

    level: ClassVar[Literal['events', 'samples']] = 'samples'

    def __post_init__(self) -> None:
        """Check the sampling rate and store it as a float."""
        self.sampling_rate = _check_positive_number(self.sampling_rate, 'sampling_rate')

    def _run(self, rng: np.random.Generator, run: dict[str, Any]) -> Gaze:
        """Generate from a copy of the events, and attach that copy and the experiment."""
        if isinstance(run.get('events'), Events):
            run['events'] = run['events'].clone()

        generated = self._generate(rng=rng, **run)

        generated.experiment = Experiment(sampling_rate=self.sampling_rate)
        if isinstance(run.get('events'), Events):
            generated.events = run['events']
        return generated

    @abc.abstractmethod
    def _generate(self, *, rng: np.random.Generator, **run: Any) -> Gaze:
        """Synthesize the samples. All randomness must be drawn from ``rng``.

        Parameters
        ----------
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
    """Warn that synthetic data generation is experimental, pointing at the caller of generate()."""
    warn(
        'Synthetic gaze generation is experimental. '
        'Names and behavior may change without being considered a breaking change. '
        'Please set the used pymovements version explicitly to prevent unexpected changes. '
        f'The used pymovements version is v{__version__}.',
        ExperimentalWarning,
        stacklevel=3,
    )


def _check_positive_number(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers, including booleans, and values <= 0."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f'{name} must be a number but is of type {type(value).__name__}: {value}',
        )

    if value <= 0:
        raise ValueError(f'{name} must be greater than zero but is: {float(value)}')

    return float(value)


def _resolve_seed(seed: Any) -> int:
    """Return the seed as an int, or draw one that any JSON reader keeps exact if it is ``None``."""
    if seed is None:
        return secrets.randbits(53)

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f'seed must be an integer but is of type {type(seed).__name__}: {seed}')

    if seed < 0:
        raise ValueError(f'seed must not be negative but is: {seed}')

    return int(seed)
