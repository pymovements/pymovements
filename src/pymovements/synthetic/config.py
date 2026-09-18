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
"""Provides configuration objects for synthetic gaze generators."""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class GeneratorConfig:
    """Holds the parameters of a generation run, separate from the generation logic.

    Carries the parameters every generator needs. A generator with parameters of its own
    declares them by subclassing this class, which keeps them typed and inspectable rather than
    hidden in the generation logic. Both the serialisation and the seeding machinery work on
    subclasses unchanged.

    Attributes
    ----------
    sampling_rate: float
        The sampling rate of the generated signal in Hz. Must be greater than zero.
    duration_s: float | None
        The duration of the generated sequence in seconds, or ``None`` to let the conditioning
        input determine it. Must be greater than zero.
    seed: int | None
        The random seed, or ``None`` to draw one. A drawn seed is recorded in the generated
        :py:class:`~pymovements.Gaze` object's metadata, so a run is reproducible either way.
        Must not be negative.

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a rate or duration is not greater than zero, or the seed is negative.

    Examples
    --------
    >>> config = GeneratorConfig(sampling_rate=1000.0, duration_s=5.0, seed=42)
    >>> config.to_dict()
    {'sampling_rate': 1000.0, 'duration_s': 5.0, 'seed': 42}
    >>> GeneratorConfig.from_dict(config.to_dict()) == config
    True
    """

    sampling_rate: float
    duration_s: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the parameters and cast the numeric ones to float."""
        self.sampling_rate = _validate_positive_number(self.sampling_rate, 'sampling_rate')

        if self.duration_s is not None:
            self.duration_s = _validate_positive_number(self.duration_s, 'duration_s')

        if self.seed is not None:
            self.seed = _validate_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a dictionary.

        Includes the parameters declared by subclasses. The result is JSON-serialisable as long
        as the subclass declares only JSON-serialisable parameters.

        Returns
        -------
        dict[str, Any]
            The configuration, keyed by parameter name.
        """
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GeneratorConfig:
        """Create a configuration from a dictionary as returned by ``to_dict``.

        Parameters
        ----------
        config: dict[str, Any]
            Parameters keyed by name. Keys with a default may be omitted.

        Returns
        -------
        GeneratorConfig
            The reconstructed configuration.

        Raises
        ------
        ValueError
            If a key is not a parameter of this configuration.
        """
        field_names = {field.name for field in dataclasses.fields(cls)}
        unknown_keys = sorted(set(config) - field_names)
        if unknown_keys:
            raise ValueError(
                f'unknown keys in generator configuration: {unknown_keys}. '
                f'Valid keys: {sorted(field_names)}',
            )

        return cls(**config)


def _validate_positive_number(value: Any, name: str) -> float:
    """Return the value as a float, rejecting non-numbers and values of zero or less."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f'{name} must be a number but is of type {type(value).__name__}: {value}',
        )

    if value <= 0:
        raise ValueError(f'{name} must be greater than zero but is: {float(value)}')

    return float(value)


def _validate_seed(value: Any) -> int:
    """Return the seed unchanged, rejecting non-integers and negative values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f'seed must be an integer but is of type {type(value).__name__}: {value}',
        )

    if value < 0:
        raise ValueError(f'seed must not be negative but is: {value}')

    return value
