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
"""Provides the libraries of event and sample generators."""
from __future__ import annotations

from typing import Generic
from typing import TypeVar

from pymovements.synthetic.generator import EventGenerator
from pymovements.synthetic.generator import SampleGenerator

GeneratorT = TypeVar('GeneratorT', EventGenerator, SampleGenerator)


class _GeneratorLibrary(Generic[GeneratorT]):
    """Shared behaviour of the generator libraries."""

    generators: dict[str, type[GeneratorT]]

    @classmethod
    def add(cls, generator: type[GeneratorT]) -> None:
        """Add a generator to the library.

        Parameters
        ----------
        generator: type[GeneratorT]
            The generator class to add.

        Raises
        ------
        ValueError
            If the generator has no non-empty ``name`` class attribute.
        """
        name = getattr(generator, 'name', None)

        if not isinstance(name, str) or not name:
            raise ValueError(
                f'generator {generator.__name__} must define a non-empty string class attribute '
                f"'name' to be registered in the {cls.__name__}",
            )

        cls.generators[name] = generator

    @classmethod
    def get(cls, name: str) -> type[GeneratorT]:
        """Get a generator by name.

        Parameters
        ----------
        name: str
            Name of the generator.

        Returns
        -------
        type[GeneratorT]
            The generator class.

        Raises
        ------
        KeyError
            If no generator is registered under that name.
        """
        if name not in cls.generators:
            raise KeyError(
                f"generator '{name}' not found in {cls.__name__}. "
                f'Available generators: {sorted(cls.generators.keys())}',
            )
        return cls.generators[name]

    @classmethod
    def names(cls) -> list[str]:
        """Return the names of all registered generators.

        Returns
        -------
        list[str]
            Sorted list of registered names.
        """
        return sorted(cls.generators.keys())

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Check if a generator is registered under a name.

        Parameters
        ----------
        name: str
            Name of the generator.

        Returns
        -------
        bool
            True if a generator is registered under that name, else False.
        """
        return name in cls.generators


class EventGeneratorLibrary(_GeneratorLibrary[EventGenerator]):
    """Provides access by name to event generators.

    Generators register themselves via :py:func:`~pymovements.synthetic.register_generator`, so
    adding one requires no change to this library.

    Attributes
    ----------
    generators: dict[str, type[EventGenerator]]
        Registered generator classes, keyed by their name.
    """

    generators: dict[str, type[EventGenerator]] = {}


class SampleGeneratorLibrary(_GeneratorLibrary[SampleGenerator]):
    """Provides access by name to sample generators.

    Generators register themselves via :py:func:`~pymovements.synthetic.register_generator`, so
    adding one requires no change to this library.

    Attributes
    ----------
    generators: dict[str, type[SampleGenerator]]
        Registered generator classes, keyed by their name.
    """

    generators: dict[str, type[SampleGenerator]] = {}


def register_generator(generator: type[GeneratorT]) -> type[GeneratorT]:
    """Register a generator in the library of its level. Intended as a class decorator.

    Parameters
    ----------
    generator: type[GeneratorT]
        The event or sample generator class to register.

    Returns
    -------
    type[GeneratorT]
        The same class, unchanged.

    Raises
    ------
    TypeError
        If the class is neither an event nor a sample generator.

    Examples
    --------
    .. testsetup::

        >>> getfixture('doctest_synthetic')

    The decorator sorts the class into the library of its level:

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
    >>> 'even_fixations' in pm.synthetic.EventGeneratorLibrary()
    True
    >>> 'even_fixations' in pm.synthetic.SampleGeneratorLibrary()
    False
    """
    if issubclass(generator, EventGenerator):
        EventGeneratorLibrary.add(generator)
    elif issubclass(generator, SampleGenerator):
        SampleGeneratorLibrary.add(generator)
    else:
        raise TypeError(
            f'{generator.__name__} must be a subclass of EventGenerator or SampleGenerator '
            'to be registered',
        )
    return generator
