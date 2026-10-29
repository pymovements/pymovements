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
"""Provides capability declarations for synthetic gaze generators."""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any


class ConditioningMode(enum.Enum):
    """What a generator conditions its output on.

    Members: ``UNCONDITIONAL``, ``EVENTS`` (an event sequence, such as fixations) and ``GAZE``
    (an existing gaze sequence). Stimulus conditioning is absent on purpose -- it is an
    extensibility target, not a supported mode.
    """

    UNCONDITIONAL = 'unconditional'
    EVENTS = 'events'
    GAZE = 'gaze'


class SignalRepresentation(enum.Enum):
    """What a generator emits.

    Members: ``POSITION``, ``VELOCITY_2D`` (per-axis, directional) and ``VELOCITY_MAGNITUDE``
    (no direction). Recorded so a directionless signal is never integrated into positions,
    which would invent directions the generator never produced.
    """

    POSITION = 'position'
    VELOCITY_2D = 'velocity_2d'
    VELOCITY_MAGNITUDE = 'velocity_magnitude'


@dataclasses.dataclass(frozen=True)
class GeneratorCapabilities:
    """Declares what a synthetic gaze generator supports.

    Lets the framework reject an invocation a generator cannot serve, and lets a caller see what
    two generators have in common. Collections may be passed as any iterable and are stored as
    frozensets, so a declaration is hashable and cannot be modified afterwards.

    Attributes
    ----------
    conditioning_modes: frozenset[ConditioningMode]
        What the generator can condition on. Must not be empty.
    signal_representations: frozenset[SignalRepresentation]
        What the generator emits. Declare several if it emits several. Must not be empty.
    event_types: frozenset[str]
        Event types covered, e.g. ``'fixation'``. Empty means no event structure.
    sampling_rates: frozenset[float] | None
        Supported rates in Hz, or ``None`` for arbitrary. Empty is rejected -- it is not the
        same as arbitrary.
    sequence_duration_s: float | None
        Fixed sequence duration in seconds, or ``None`` for arbitrary.
    requires_artifacts: bool
        Whether trained weights or similar are needed.

    Raises
    ------
    TypeError
        If a declared value has the wrong type.
    ValueError
        If a required collection is empty, or a rate or duration is not greater than zero.

    Examples
    --------
    >>> capabilities = GeneratorCapabilities(
    ...     conditioning_modes={ConditioningMode.GAZE},
    ...     signal_representations={SignalRepresentation.VELOCITY_2D},
    ...     sampling_rates={1000.0},
    ...     sequence_duration_s=5.0,
    ...     requires_artifacts=True,
    ... )
    >>> capabilities.to_dict()['sequence_duration_s']
    5.0
    """

    conditioning_modes: frozenset[ConditioningMode]
    signal_representations: frozenset[SignalRepresentation]
    event_types: frozenset[str] = frozenset()
    sampling_rates: frozenset[float] | None = None
    sequence_duration_s: float | None = None
    requires_artifacts: bool = False

    def __post_init__(self) -> None:
        """Freeze the collections and validate the declaration."""
        object.__setattr__(
            self, 'conditioning_modes',
            _freeze_enum_members(
                self.conditioning_modes, ConditioningMode, 'conditioning_modes',
            ),
        )
        object.__setattr__(
            self, 'signal_representations',
            _freeze_enum_members(
                self.signal_representations, SignalRepresentation, 'signal_representations',
            ),
        )
        object.__setattr__(
            self, 'event_types', _freeze_event_types(self.event_types),
        )
        object.__setattr__(
            self, 'sampling_rates', _freeze_sampling_rates(self.sampling_rates),
        )

        if self.sequence_duration_s is not None:
            if not _is_number(self.sequence_duration_s):
                raise TypeError(
                    f'sequence_duration_s must be a number but is of type '
                    f'{type(self.sequence_duration_s).__name__}: {self.sequence_duration_s}',
                )
            if self.sequence_duration_s <= 0:
                raise ValueError(
                    f'sequence_duration_s must be greater than zero but is: '
                    f'{float(self.sequence_duration_s)}',
                )
            object.__setattr__(self, 'sequence_duration_s', float(self.sequence_duration_s))

        if not isinstance(self.requires_artifacts, bool):
            raise TypeError(
                f'requires_artifacts must be of type bool but is of type '
                f'{type(self.requires_artifacts).__name__}: {self.requires_artifacts}',
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the declaration as a JSON-serialisable dictionary.

        Collections are sorted, so the same declaration always yields the same dictionary.

        Returns
        -------
        dict[str, Any]
            The declaration, with enum members replaced by their values.
        """
        return {
            'conditioning_modes': sorted(mode.value for mode in self.conditioning_modes),
            'signal_representations': sorted(
                representation.value for representation in self.signal_representations
            ),
            'event_types': sorted(self.event_types),
            'sampling_rates': (
                None if self.sampling_rates is None else sorted(self.sampling_rates)
            ),
            'sequence_duration_s': self.sequence_duration_s,
            'requires_artifacts': self.requires_artifacts,
        }

    @classmethod
    def from_dict(cls, declaration: dict[str, Any]) -> GeneratorCapabilities:
        """Create a declaration from a dictionary as returned by ``to_dict``.

        Parameters
        ----------
        declaration: dict[str, Any]
            Enum members may be given by value or as members. Keys with a default may be
            omitted.

        Returns
        -------
        GeneratorCapabilities
            The reconstructed declaration.

        Raises
        ------
        ValueError
            If a key or an enum value is unknown.
        """
        field_names = {field.name for field in dataclasses.fields(cls)}
        unknown_keys = sorted(set(declaration) - field_names)
        if unknown_keys:
            raise ValueError(
                f'unknown keys in capability declaration: {unknown_keys}. '
                f'Valid keys: {sorted(field_names)}',
            )

        kwargs = dict(declaration)
        kwargs['conditioning_modes'] = _members_from_values(
            kwargs.get('conditioning_modes', ()), ConditioningMode, 'conditioning_modes',
        )
        kwargs['signal_representations'] = _members_from_values(
            kwargs.get('signal_representations', ()),
            SignalRepresentation,
            'signal_representations',
        )
        return cls(**kwargs)


def _freeze_enum_members(
        values: Iterable[Any],
        member_type: type[enum.Enum],
        name: str,
) -> frozenset[Any]:
    """Freeze values, rejecting anything that is not a ``member_type`` member."""
    members = frozenset(values)

    for member in members:
        if not isinstance(member, member_type):
            raise TypeError(
                f'{name} must contain only {member_type.__name__} members but contains '
                f'{type(member).__name__}: {member}. '
                f'Valid members: {sorted(valid.value for valid in member_type)}',
            )

    if not members:
        raise ValueError(f'{name} must not be empty, declare at least one member')

    return members


def _freeze_event_types(values: Iterable[str]) -> frozenset[str]:
    """Freeze event types, rejecting non-strings."""
    event_types = frozenset(values)

    for event_type in event_types:
        if not isinstance(event_type, str):
            raise TypeError(
                f'event_types must contain only values of type str but contains '
                f'{type(event_type).__name__}: {event_type}',
            )

    return event_types


def _freeze_sampling_rates(values: Iterable[float] | None) -> frozenset[float] | None:
    """Freeze sampling rates as floats. ``None`` passes through and means arbitrary."""
    if values is None:
        return None

    rates = frozenset(values)

    for rate in rates:
        if not _is_number(rate):
            raise TypeError(
                f'sampling_rates must contain only numbers but contains '
                f'{type(rate).__name__}: {rate}',
            )
        if rate <= 0:
            raise ValueError(
                f'sampling_rates must contain only values greater than zero but contains: '
                f'{float(rate)}',
            )

    if not rates:
        raise ValueError(
            'sampling_rates must not be empty, pass none to declare arbitrary sampling rates',
        )

    return frozenset(float(rate) for rate in rates)


def _members_from_values(
        values: Iterable[Any],
        member_type: type[enum.Enum],
        name: str,
) -> frozenset[Any]:
    """Look up enum members by value, passing members through unchanged."""
    members = set()

    for value in values:
        if isinstance(value, member_type):
            members.add(value)
            continue

        try:
            members.add(member_type(value))
        except ValueError as exception:
            raise ValueError(
                f'unknown {name} value: {value}. '
                f'Valid values: {sorted(valid.value for valid in member_type)}',
            ) from exception

    return frozenset(members)


def _is_number(value: Any) -> bool:
    """Return whether the value is a number. ``bool`` does not count."""
    return not isinstance(value, bool) and isinstance(value, (int, float))
