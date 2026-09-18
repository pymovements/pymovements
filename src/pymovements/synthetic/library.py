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
"""Provides the GeneratorLibrary."""
from __future__ import annotations

from pymovements.synthetic.generator import GazeGenerator


class GeneratorLibrary:
    """Provides access by name to synthetic gaze generators.

    Generators register themselves via
    :py:func:`~pymovements.synthetic.register_generator`, so adding one requires no change to
    this library or any other framework module.

    Attributes
    ----------
    generators: dict[str, type[GazeGenerator]]
        Registered generator classes, keyed by their name.
    """

    generators: dict[str, type[GazeGenerator]] = {}

    @classmethod
    def add(cls, generator: type[GazeGenerator]) -> None:
        """Add a synthetic gaze generator to the library.

        Parameters
        ----------
        generator: type[GazeGenerator]
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
                "'name' to be registered in the GeneratorLibrary",
            )

        cls.generators[name] = generator

    @classmethod
    def get(cls, name: str) -> type[GazeGenerator]:
        """Get a synthetic gaze generator by name.

        Parameters
        ----------
        name: str
            Name of the generator.

        Returns
        -------
        type[GazeGenerator]
            The generator class.

        Raises
        ------
        KeyError
            If no generator is registered under that name.
        """
        if name not in cls.generators:
            raise KeyError(
                f"generator '{name}' not found in GeneratorLibrary. "
                f'Available generators: {sorted(cls.generators.keys())}',
            )
        return cls.generators[name]

    @classmethod
    def names(cls) -> list[str]:
        """Return the names of all registered synthetic gaze generators.

        Returns
        -------
        list[str]
            Sorted list of registered names.
        """
        return sorted(cls.generators.keys())


def register_generator(generator: type[GazeGenerator]) -> type[GazeGenerator]:
    """Register a synthetic gaze generator. Intended as a class decorator.

    Parameters
    ----------
    generator: type[GazeGenerator]
        The generator class to register.

    Returns
    -------
    type[GazeGenerator]
        The same class, unchanged.
    """
    GeneratorLibrary.add(generator)
    return generator
