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
from pymovements.synthetic._utils import check_positive
from pymovements.warnings import ExperimentalWarning

#: The key the provenance record is stored under in the metadata of the generated data.
PROVENANCE_KEY = 'synthetic'

OutputT = TypeVar('OutputT', Events, Gaze)


class _Generator(abc.ABC, Generic[OutputT]):
    """Shared behaviour of event and sample generators."""

    name: ClassVar[str]

    level: ClassVar[Literal['events', 'samples']]

    # Each level declares the abstract method; concrete generators narrow its run parameters.
    _generate: Callable[..., OutputT]

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters the generator was constructed with.

        These are the dataclass fields that ``__init__`` takes, so that
        ``type(generator)(**generator.to_dict())`` is an equal generator. A value derived
        in ``__post_init__`` is declared with ``init=False`` and left out. A neural generator
        loads its weights from a path, and the path is the hyperparameter.

        Returns
        -------
        dict[str, Any]
            The hyperparameters, keyed by name.

        Raises
        ------
        TypeError
            If the generator is not a dataclass, or one of its classes annotates fields without
            being one.
        """
        return {
            field.name: getattr(self, field.name)
            for field in self._fields()
            if field.init
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
            generator attaches a copy of a run parameter named ``events`` as the ground truth,
            and an :py:class:`~pymovements.Experiment` carrying its sampling rate unless the
            samples carry one already.

        Raises
        ------
        TypeError
            If the generator is not a dataclass, one of its classes annotates fields without being
            one, it has no ``name``, the seed is not an integer,
            or a run parameter is unknown to the generator or missing.
        ValueError
            If the seed is negative, or the samples a sample generator made carry an experiment
            with another sampling rate.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic data generation is experimental.
        """
        return self._generate_at(seed, run, depth=2)

    def _generate_at(self, seed: int | None, run: dict[str, Any], *, depth: int) -> OutputT:
        """Do the work of ``generate()``, warning at the line ``depth`` calls above this method.

        Parameters
        ----------
        seed: int | None
            The random seed, or ``None`` to draw one.
        run: dict[str, Any]
            The run parameters as passed to ``generate()``.
        depth: int
            The number of calls above this method that the warning points at. ``generate()`` and
            the entry points, which call this method in its place, pass 2: the line of the user.

        Returns
        -------
        OutputT
            The generated data, as ``generate()`` returns it.
        """
        _warn_experimental(depth)
        self._check_class()
        run = self._resolve_run(run)
        seed = _resolve_seed(seed)

        generated = self._run(np.random.default_rng(seed), run)

        generated.metadata = {
            **(generated.metadata or {}), **self._provenance(self._record_run(run), seed),
        }
        return generated

    def _run(self, rng: np.random.Generator, run: dict[str, Any]) -> OutputT:
        """Call ``_generate``. A level overrides this to prepare its inputs and its output."""
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
        parameter that ``_generate`` does not declare raises, naming the accepted ones, unless
        ``_generate`` takes ``**kwargs``.
        """
        parameters = inspect.signature(self._generate).parameters
        declared = {
            name: parameter for name, parameter in parameters.items()
            if name != 'rng' and parameter.kind in (
                parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY,
            )
        }
        accepts_any = any(
            parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
        )

        unknown = sorted(set(run) - set(declared))
        if unknown and not accepts_any:
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

        # A subclass inherits the dataclass check above, so a class without the decorator passes
        # it, and the fields it annotates would be left out of the constructor and the record.
        for generator_class in type(self).__mro__:
            # A plain mixin is no generator, and the base classes of this module annotate class
            # attributes, not fields.
            if (
                not issubclass(generator_class, _Generator)
                or generator_class.__module__ == __name__
                or '__dataclass_fields__' in vars(generator_class)
            ):
                continue

            undeclared = [
                name for name, annotation in inspect.get_annotations(generator_class).items()
                if 'ClassVar' not in str(annotation)
            ]
            if undeclared:
                raise TypeError(
                    f'{generator_class.__name__} annotates {undeclared} without being a dataclass '
                    'itself, so they are no hyperparameters and the run cannot be reproduced; '
                    'decorate it with @dataclasses.dataclass',
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
        record = {}
        for key, value in run.items():
            if isinstance(value, Events):
                record[f'n_{key}'] = value.frame.height
            else:
                record[key] = value
        return record

    def _provenance(self, run: dict[str, Any], seed: int) -> dict[str, Any]:
        """Return the provenance record of a run, so the output is not mistaken for a recording.

        Dataclass values, such as a range, are recorded as dicts.

        Parameters
        ----------
        run: dict[str, Any]
            The record of the run parameters, apart from the seed.
        seed: int
            The seed the run's random generator was created from.

        Returns
        -------
        dict[str, Any]
            The record under :py:data:`PROVENANCE_KEY`, ready to be merged into the metadata.
        """
        return {
            PROVENANCE_KEY: {
                'generator': self.name,
                'level': self.level,
                'hyperparameters': {
                    key: _plain(value) for key, value in self.to_dict().items()
                },
                'run': {key: _plain(value) for key, value in run.items()},
                'seed': seed,
                'pymovements_version': __version__,
            },
        }


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
    >>> events = EvenFixations().generate(duration=1.0, seed=42)  # doctest: +IGNORE_WARNINGS
    >>> events.frame['name'].to_list()
    ['fixation', 'fixation', 'fixation', 'fixation']

    The output records how it was made:

    >>> sorted(events.metadata['synthetic'])
    ['generator', 'hyperparameters', 'level', 'pymovements_version', 'run', 'seed']
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
    :py:class:`~pymovements.Experiment` carrying the sampling rate, as a recording would, unless
    ``_generate`` attached one, which then carries the same rate or none.

    A generator built for one sampling rate redefines the field with that rate as its default,
    under ``@dataclasses.dataclass(kw_only=True)`` so that the field stays keyword-only, and
    rejects other values in its ``__post_init__``. A subclass that defines ``__post_init__`` calls
    ``super().__post_init__()`` first, so the sampling rate is checked.

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
    >>> generator = StillGaze(sampling_rate=1000)
    >>> gaze = generator.generate(events=events, seed=42)  # doctest: +IGNORE_WARNINGS
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
        self.sampling_rate = check_positive(self.sampling_rate, 'sampling_rate')

    def _run(self, rng: np.random.Generator, run: dict[str, Any]) -> Gaze:
        """Generate from a copy of the events, and attach that copy and the experiment."""
        events = run.get('events')
        if isinstance(events, Events):
            run['events'] = events = events.clone()

        generated = self._generate(rng=rng, **run)

        self._attach_experiment(generated)
        if isinstance(events, Events):
            generated.events = events
        return generated

    def _attach_experiment(self, generated: Gaze) -> None:
        """Attach an experiment with the sampling rate, or check the one ``_generate`` attached."""
        experiment = generated.experiment
        if experiment is None:
            generated.experiment = Experiment(sampling_rate=self.sampling_rate)
        elif experiment.sampling_rate is None:
            experiment.sampling_rate = self.sampling_rate
        elif experiment.sampling_rate != self.sampling_rate:
            raise ValueError(
                f'{type(self).__name__} generated samples whose experiment has a sampling rate '
                f'of {experiment.sampling_rate} Hz, but its own is {self.sampling_rate} Hz',
            )

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


def _plain(value: Any) -> Any:
    """Return a dataclass instance as a dict, so the record can be written outside Python."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _warn_experimental(depth: int) -> None:
    """Warn that synthetic data generation is experimental.

    Parameters
    ----------
    depth: int
        The number of calls above the caller of this function that the warning points at.
    """
    warn(
        'Synthetic gaze generation is experimental. '
        'Names and behavior may change without being considered a breaking change. '
        'Please set the used pymovements version explicitly to prevent unexpected changes. '
        f'The used pymovements version is v{__version__}.',
        ExperimentalWarning,
        stacklevel=depth + 2,
    )


def _resolve_seed(seed: Any) -> int:
    """Return the seed as an int, or draw one that any JSON reader keeps exact if it is ``None``."""
    if seed is None:
        return secrets.randbits(53)

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f'seed must be an integer but is of type {type(seed).__name__}: {seed}')

    if seed < 0:
        raise ValueError(f'seed must not be negative but is: {seed}')

    return int(seed)
