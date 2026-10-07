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
"""Provides contract validation for synthetic gaze generation."""
from __future__ import annotations

from typing import Any

import polars as pl

from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.capabilities import ConditioningMode
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.capabilities import SignalRepresentation
from pymovements.synthetic.config import GeneratorConfig

# Which column group carries which declared representation. VELOCITY_MAGNITUDE is not a Gaze
# component group -- a magnitude has no axes -- so it is carried in a plain column of its own,
# which is what makes it checkable at all.
_REPRESENTATION_COLUMNS = {
    SignalRepresentation.POSITION: ('position', 'pixel'),
    SignalRepresentation.VELOCITY_2D: ('velocity',),
    SignalRepresentation.VELOCITY_MAGNITUDE: ('velocity_magnitude',),
}

_POSITION_COLUMNS = ('position', 'pixel')


class GenerationContractError(ValueError):
    """Raised when a generation run violates the framework's shared contract."""


class InputContractError(GenerationContractError):
    """Raised when the conditioning input does not match what the generator declares."""


class OutputContractError(GenerationContractError):
    """Raised when generated output does not match what the generator declares."""


def validate_input(
        *,
        capabilities: GeneratorCapabilities | None,
        config: GeneratorConfig | None,
        events: Events | None = None,
        gaze: Gaze | None = None,
        artifacts: dict[str, Any] | None = None,
        generator: str | None = None,
) -> None:
    """Check a conditioning input against what a generator declares it supports.

    Checks that cannot be made are skipped rather than guessed: a generator declaring no
    capabilities is only checked for well-formed input.

    Parameters
    ----------
    capabilities: GeneratorCapabilities | None
        The generator's declaration, or ``None`` if it declares nothing.
    config: GeneratorConfig | None
        The parameters of the run, or ``None`` if the generator was called without one.
    events: Events | None
        The event sequence to condition on. (default: None)
    gaze: Gaze | None
        The gaze sequence to condition on. (default: None)
    artifacts: dict[str, Any] | None
        External artifacts, such as trained weights. (default: None)
    generator: str | None
        The generator's name, used in error messages. (default: None)

    Raises
    ------
    TypeError
        If a conditioning input is not of the expected type.
    InputContractError
        If the input does not match the declaration.
    """
    prefix = _prefix(generator)

    _validate_conditioning_types(events, gaze, prefix)
    _validate_conditioning_mode(conditioning_mode(events=events, gaze=gaze), capabilities, prefix)

    if events is not None:
        _validate_events(events, capabilities, prefix)
        _validate_duration_source(config, prefix)

    if gaze is not None:
        _validate_conditioning_gaze(gaze, prefix)

    _validate_declared_config(config, capabilities, prefix)
    _validate_artifacts(artifacts, capabilities, prefix)


def validate_output(
        *,
        gaze: Gaze,
        capabilities: GeneratorCapabilities | None,
        config: GeneratorConfig | None,
        generator: str | None = None,
) -> None:
    """Check generated output against what the generator declares it emits.

    Parameters
    ----------
    gaze: Gaze
        The generated output.
    capabilities: GeneratorCapabilities | None
        The generator's declaration, or ``None`` if it declares nothing.
    config: GeneratorConfig | None
        The parameters of the run, or ``None`` if the generator was called without one.
    generator: str | None
        The generator's name, used in error messages. (default: None)

    Raises
    ------
    OutputContractError
        If the output does not match the declaration.
    """
    prefix = _prefix(generator)

    _validate_output_shape(gaze, prefix)
    _validate_representations(gaze, capabilities, prefix)
    _validate_output_events(gaze, capabilities, prefix)
    _validate_sample_count(gaze, capabilities, config, prefix)


def conditioning_mode(*, events: Events | None, gaze: Gaze | None) -> ConditioningMode:
    """Return the conditioning mode implied by the supplied input.

    Parameters
    ----------
    events: Events | None
        The event sequence to condition on.
    gaze: Gaze | None
        The gaze sequence to condition on.

    Returns
    -------
    ConditioningMode
        The mode the supplied input stands for.
    """
    if gaze is not None:
        return ConditioningMode.GAZE
    if events is not None:
        return ConditioningMode.EVENTS
    return ConditioningMode.UNCONDITIONAL


def _prefix(generator: str | None) -> str:
    """Return the message prefix naming the generator a violation belongs to."""
    return '' if generator is None else f"generator '{generator}': "


def _validate_conditioning_types(events: Events | None, gaze: Gaze | None, prefix: str) -> None:
    """Check that at most one conditioning input is supplied, and that it is what it claims."""
    if events is not None and gaze is not None:
        raise InputContractError(
            f'{prefix}both events and gaze were supplied as conditioning input, '
            'pass exactly one of them or neither',
        )

    if events is not None and not isinstance(events, Events):
        raise TypeError(
            f'{prefix}events must be of type Events but is of type {type(events).__name__}',
        )

    if gaze is not None and not isinstance(gaze, Gaze):
        raise TypeError(
            f'{prefix}gaze must be of type Gaze but is of type {type(gaze).__name__}',
        )


def _validate_conditioning_mode(
        mode: ConditioningMode,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check that the mode the supplied input stands for is one the generator declares."""
    if capabilities is None or mode in capabilities.conditioning_modes:
        return

    supported = sorted(member.value for member in capabilities.conditioning_modes)
    raise InputContractError(
        f'{prefix}{mode.value} generation is not supported, supported conditioning modes '
        f'are: {supported}',
    )


def _validate_events(
        events: Events,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check that an event sequence is usable and covered by the declaration."""
    if len(events) == 0:
        raise InputContractError(f'{prefix}events are empty, nothing to condition on')

    if capabilities is None or not capabilities.event_types:
        return

    supplied = set(events.frame['name'].unique().to_list())
    uncovered = sorted(supplied - capabilities.event_types)
    if uncovered:
        raise InputContractError(
            f'{prefix}event types {uncovered} are not covered by the generator, covered event '
            f'types are: {sorted(capabilities.event_types)}',
        )


def _validate_duration_source(config: GeneratorConfig | None, prefix: str) -> None:
    """Reject a duration alongside events, which already determine how long the output is."""
    if config is not None and config.duration_s is not None:
        raise InputContractError(
            f'{prefix}duration_s ({config.duration_s} s) was set although events were supplied, '
            'which determine the duration, set duration_s to None when conditioning on events',
        )


def _validate_conditioning_gaze(gaze: Gaze, prefix: str) -> None:
    """Check that a gaze sequence is usable as conditioning input."""
    if gaze.samples.is_empty():
        raise InputContractError(f'{prefix}gaze is empty, nothing to condition on')

    if not any(column in gaze.samples.columns for column in _POSITION_COLUMNS):
        raise InputContractError(
            f'{prefix}gaze carries none of the column groups {list(_POSITION_COLUMNS)} needed '
            f'to condition on it, its columns are: {gaze.samples.columns}',
        )


def _validate_declared_config(
        config: GeneratorConfig | None,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check the requested sampling rate and sequence duration against the declaration."""
    if capabilities is None or config is None:
        return

    rates = capabilities.sampling_rates
    if rates is not None and config.sampling_rate not in rates:
        raise InputContractError(
            f'{prefix}sampling rate {config.sampling_rate} is not supported, supported '
            f'sampling rates are: {sorted(rates)}',
        )

    duration_s = capabilities.sequence_duration_s
    if duration_s is not None and config.duration_s not in (None, duration_s):
        raise InputContractError(
            f'{prefix}sequence duration {config.duration_s} s is not supported, the only '
            f'supported duration is: {duration_s} s',
        )


def _validate_artifacts(
        artifacts: dict[str, Any] | None,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check that artifacts are supplied when the generator declares it needs them."""
    if capabilities is not None and capabilities.requires_artifacts and not artifacts:
        raise InputContractError(
            f'{prefix}the generator requires external artifacts, pass them as artifacts',
        )


def _validate_output_shape(gaze: Gaze, prefix: str) -> None:
    """Check that the output is a Gaze carrying samples and no null values."""
    if not isinstance(gaze, Gaze):
        raise OutputContractError(
            f'{prefix}generated output must be of type Gaze but is of type '
            f'{type(gaze).__name__}',
        )

    if gaze.samples.is_empty():
        raise OutputContractError(f'{prefix}generated output has no samples')

    null_columns = [column for column in gaze.samples.columns if _null_count(gaze.samples[column])]
    if null_columns:
        raise OutputContractError(
            f'{prefix}generated output has null values in columns: {sorted(null_columns)}',
        )


def _validate_representations(
        gaze: Gaze,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check that every declared representation that maps to a column group is populated."""
    if capabilities is None:
        return

    for representation in sorted(capabilities.signal_representations, key=_representation_value):
        columns = _REPRESENTATION_COLUMNS[representation]

        if not any(column in gaze.samples.columns for column in columns):
            raise OutputContractError(
                f'{prefix}the declared signal representation {representation.value} is not '
                f'populated, expected one of the column groups {list(columns)} but the output '
                f'carries: {gaze.samples.columns}',
            )


def _validate_output_events(
        gaze: Gaze,
        capabilities: GeneratorCapabilities | None,
        prefix: str,
) -> None:
    """Check that events are carried when the generator declares event coverage."""
    if capabilities is None or not capabilities.event_types:
        return

    if gaze.events is None or len(gaze.events) == 0:
        raise OutputContractError(
            f'{prefix}generated output carries no events although the generator declares '
            f'coverage of event types: {sorted(capabilities.event_types)}',
        )


def _validate_sample_count(
        gaze: Gaze,
        capabilities: GeneratorCapabilities | None,
        config: GeneratorConfig | None,
        prefix: str,
) -> None:
    """Check the number of samples against the requested rate and duration, if both are known."""
    duration_s = _expected_duration(capabilities, config)

    if config is None or duration_s is None:
        return

    expected = round(config.sampling_rate * duration_s)
    if len(gaze.samples) != expected:
        raise OutputContractError(
            f'{prefix}generated output has {len(gaze.samples)} samples but {expected} were '
            f'expected for {duration_s} s at {config.sampling_rate} Hz',
        )


def _representation_value(representation: SignalRepresentation) -> str:
    """Return a representation's value, so declarations are reported in a stable order."""
    return representation.value


def _null_count(series: pl.Series) -> int:
    """Count nulls, looking inside the list columns the component groups are stored as."""
    if isinstance(series.dtype, pl.List):
        return series.null_count() + int(series.list.count_matches(None).sum() or 0)
    return series.null_count()


def _expected_duration(
        capabilities: GeneratorCapabilities | None,
        config: GeneratorConfig | None,
) -> float | None:
    """Return the duration the output is expected to cover, if one is known."""
    if config is not None and config.duration_s is not None:
        return config.duration_s
    if capabilities is not None:
        return capabilities.sequence_duration_s
    return None
