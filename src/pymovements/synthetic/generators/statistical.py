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
"""Provides a statistical synthetic gaze generator.

The generator implements the eye movement simulator described in section 3 of:

    Fuhl, W., Santini, T., Kuebler, T., Castner, N., Rosenstiel, W., & Kasneci, E. (2018).
    Eye movement simulation and detector creation to reduce laborious parameter adjustments.
    arXiv:1804.00970.

Generation runs in four steps. A sequence of event types is chosen, each event is given a
velocity profile drawn from its own parameters, the profiles are laid end to end and interpolated
down to the requested sampling rate, and measurement noise is added last.

What comes out is a velocity magnitude without direction. The simulated velocity profiles carry
no axes, and directions are drawn separately from them: randomly for saccades and smooth
pursuits, because they depend on the stimulus and the task, and by a random walker for fixations.
Nothing here turns the magnitudes into positions, which would attribute directions to the signal
that were never generated.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import polars as pl

from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.capabilities import ConditioningMode
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.capabilities import SignalRepresentation
from pymovements.synthetic.config import GeneratorConfig
from pymovements.synthetic.generator import GazeGenerator
from pymovements.synthetic.library import register_generator

#: The column the velocity magnitude is written to. It is deliberately not one of the Gaze
#: component groups, because a magnitude has no x and y.
VELOCITY_MAGNITUDE_COLUMN = 'velocity_magnitude'

#: The distributions a value can be drawn from.
DISTRIBUTIONS = ('normal', 'uniform')

#: The shapes a smooth pursuit can take after its onset.
PURSUIT_PROFILES = ('constant', 'growing', 'decreasing')

#: The event types the simulator produces.
EVENT_TYPES = ('fixation', 'saccade', 'smooth_pursuit')


@dataclasses.dataclass
class StatisticalConfig(GeneratorConfig):
    """Parameters of the statistical generator.

    Every event type is parametrised the same way: a duration range with a distribution, and a
    consistency range describing the fluctuation along its velocity profile. Saccades add a peak
    velocity range and the skewness of their profile; smooth pursuits add a velocity range, an
    onset length and the shape the pursuit takes after the onset.

    Field names follow a convention the validation relies on: a ``_min`` field is paired with the
    matching ``_max``, a ``_distribution`` field names one of ``DISTRIBUTIONS``, and a standard
    deviation is used only when its distribution is normal.

    The sampling rate, sequence duration and seed are inherited from
    :py:class:`~pymovements.synthetic.GeneratorConfig`. **No default is a measured value.** Two
    are taken from settings the source used to render its figures -- the saccade peak velocity
    range, from a figure showing two deliberately large saccades, and the fixation consistency,
    from a figure whose text calls that setting very high. Both are illustrations, not
    recommendations, so expect to tune them against whatever you are modeling.

    Attributes
    ----------
    fixation_duration_min_ms: float
        Shortest fixation. (default: 100.0)
    fixation_duration_max_ms: float
        Longest fixation. (default: 400.0)
    fixation_duration_distribution: str
        How the duration is drawn. (default: 'uniform')
    fixation_duration_std_ms: float
        Spread of the duration when drawn normally. (default: 50.0)
    fixation_consistency_min_dps: float
        Smallest velocity fluctuation during a fixation. (default: 0.0)
    fixation_consistency_max_dps: float
        Largest velocity fluctuation during a fixation. (default: 1.0)
    fixation_consistency_distribution: str
        How the consistency is drawn. (default: 'uniform')
    saccade_duration_min_ms: float
        Shortest saccade. (default: 20.0)
    saccade_duration_max_ms: float
        Longest saccade. (default: 80.0)
    saccade_duration_distribution: str
        How the duration is drawn. (default: 'uniform')
    saccade_duration_std_ms: float
        Spread of the duration when drawn normally. (default: 10.0)
    saccade_peak_velocity_min_dps: float
        Lowest peak velocity a saccade can reach. Suits large saccades; small ones peak far
        lower. (default: 300.0)
    saccade_peak_velocity_max_dps: float
        Highest peak velocity a saccade can reach. (default: 500.0)
    saccade_skewness_min: float
        Least skewed velocity profile. Must be above zero. (default: 0.5)
    saccade_skewness_max: float
        Most skewed velocity profile. Must stay below two, where the profile loses the onset that
        makes it a saccade. (default: 1.5)
    saccade_consistency_min_dps: float
        Smallest jitter along a saccade profile. (default: 0.0)
    saccade_consistency_max_dps: float
        Largest jitter along a saccade profile. (default: 5.0)
    saccade_consistency_distribution: str
        How the consistency is drawn. (default: 'uniform')
    smooth_pursuit_duration_min_ms: float
        Shortest smooth pursuit. (default: 200.0)
    smooth_pursuit_duration_max_ms: float
        Longest smooth pursuit. (default: 600.0)
    smooth_pursuit_duration_distribution: str
        How the duration is drawn. (default: 'uniform')
    smooth_pursuit_duration_std_ms: float
        Spread of the duration when drawn normally. (default: 80.0)
    smooth_pursuit_velocity_min_dps: float
        Lowest pursuit velocity. (default: 10.0)
    smooth_pursuit_velocity_max_dps: float
        Highest pursuit velocity. (default: 30.0)
    smooth_pursuit_onset_duration_min_ms: float
        Shortest onset. (default: 40.0)
    smooth_pursuit_onset_duration_max_ms: float
        Longest onset. (default: 120.0)
    smooth_pursuit_profile: str
        The shape after the onset, one of ``PURSUIT_PROFILES``. (default: 'constant')
    smooth_pursuit_consistency_min_dps: float
        Smallest jitter along a pursuit profile. (default: 0.0)
    smooth_pursuit_consistency_max_dps: float
        Largest jitter along a pursuit profile. (default: 2.0)
    smooth_pursuit_consistency_distribution: str
        How the consistency is drawn. (default: 'uniform')
    profile_sampling_rate: float
        The rate the profiles are built at before being interpolated to ``sampling_rate``. Must
        not be below it. (default: 10000.0)
    noise_std_dps: float
        Spread of the measurement noise added last. (default: 0.0)
    event_sequence: tuple[str, ...] | None
        The event types to generate in order, repeated until the duration is filled, or ``None``
        to draw each one uniformly. Ignored when conditioning on events. (default: None)

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a name is unknown, a range is inverted, or a value is outside its valid interval.

    Examples
    --------
    >>> config = StatisticalConfig(sampling_rate=1000.0, duration_s=1.0, seed=42)
    >>> config.saccade_peak_velocity_max_dps
    500.0
    """

    fixation_duration_min_ms: float = 100.0
    fixation_duration_max_ms: float = 400.0
    fixation_duration_distribution: str = 'uniform'
    fixation_duration_std_ms: float = 50.0
    fixation_consistency_min_dps: float = 0.0
    fixation_consistency_max_dps: float = 1.0
    fixation_consistency_distribution: str = 'uniform'

    saccade_duration_min_ms: float = 20.0
    saccade_duration_max_ms: float = 80.0
    saccade_duration_distribution: str = 'uniform'
    saccade_duration_std_ms: float = 10.0
    saccade_peak_velocity_min_dps: float = 300.0
    saccade_peak_velocity_max_dps: float = 500.0
    saccade_skewness_min: float = 0.5
    saccade_skewness_max: float = 1.5
    saccade_consistency_min_dps: float = 0.0
    saccade_consistency_max_dps: float = 5.0
    saccade_consistency_distribution: str = 'uniform'

    smooth_pursuit_duration_min_ms: float = 200.0
    smooth_pursuit_duration_max_ms: float = 600.0
    smooth_pursuit_duration_distribution: str = 'uniform'
    smooth_pursuit_duration_std_ms: float = 80.0
    smooth_pursuit_velocity_min_dps: float = 10.0
    smooth_pursuit_velocity_max_dps: float = 30.0
    smooth_pursuit_onset_duration_min_ms: float = 40.0
    smooth_pursuit_onset_duration_max_ms: float = 120.0
    smooth_pursuit_profile: str = 'constant'
    smooth_pursuit_consistency_min_dps: float = 0.0
    smooth_pursuit_consistency_max_dps: float = 2.0
    smooth_pursuit_consistency_distribution: str = 'uniform'

    profile_sampling_rate: float = 10_000.0
    noise_std_dps: float = 0.0
    event_sequence: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the parameters on top of what GeneratorConfig already checks."""
        super().__post_init__()

        for field in dataclasses.fields(self):
            if field.name.endswith('_distribution'):
                _check_name(self, field.name, DISTRIBUTIONS)
            elif field.name.endswith(('_std_ms', '_std_dps')):
                _check_not_negative(self, field.name)
            elif '_min' in field.name:
                _check_ordered(self, field.name, field.name.replace('_min', '_max', 1))

        _check_name(self, 'smooth_pursuit_profile', PURSUIT_PROFILES)
        _check_skewness(self)

        if self.profile_sampling_rate < self.sampling_rate:
            raise ValueError(
                f'profile_sampling_rate ({self.profile_sampling_rate}) must not be below '
                f'sampling_rate ({self.sampling_rate}), otherwise the profiles are built more '
                f'coarsely than the signal interpolated from them',
            )

        if self.event_sequence is not None:
            self.event_sequence = _check_event_sequence(self.event_sequence)


@register_generator
class StatisticalGenerator(GazeGenerator):
    """Generates velocity magnitudes from parametrised event profiles.

    Fixations fluctuate around zero by their consistency, saccades follow a Gamma-shaped profile
    whose skewness is a parameter, and smooth pursuits open with a sigmoid onset before a
    constant, growing or decreasing pursuit.

    Either a duration to fill or an event sequence to follow must be given. Conditioned on
    events, the type and length of every segment comes from the input instead of being drawn.
    """

    name = 'statistical'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL, ConditioningMode.EVENTS}),
        signal_representations=frozenset({SignalRepresentation.VELOCITY_MAGNITUDE}),
        event_types=frozenset(EVENT_TYPES),
    )

    def _generate(
            self,
            config: GeneratorConfig | None,
            rng: np.random.Generator,
            events: Events | None = None,
            **kwargs: Any,
    ) -> Gaze:
        """Synthesize a velocity magnitude sequence.

        Parameters
        ----------
        config: GeneratorConfig | None
            A StatisticalConfig.
        rng: np.random.Generator
            The run's random generator.
        events: Events | None
            A sequence to take the event types and durations from. (default: None)
        **kwargs: Any
            Ignored.

        Returns
        -------
        Gaze
            Samples carrying a time column and the velocity magnitude, plus the event sequence
            the signal was built from.

        Raises
        ------
        ValueError
            If the configuration is not a StatisticalConfig, or if neither events nor a duration
            were given.
        """
        if not isinstance(config, StatisticalConfig):
            raise ValueError(
                f"generator '{self.name}' needs a StatisticalConfig but got "
                f'{type(config).__name__}',
            )

        segments, start_ms = _event_sequence(config, rng, events)
        profile, profile_times_ms = _build_profiles(segments, config, rng, start_ms)
        magnitudes = _resample(profile, profile_times_ms, segments, config)

        if config.noise_std_dps:
            magnitudes = np.abs(
                magnitudes + rng.normal(0.0, config.noise_std_dps, magnitudes.size),
            )

        times_ms = start_ms + np.arange(magnitudes.size) * (1000.0 / config.sampling_rate)

        # A magnitude is not one of the Gaze component groups, so Gaze warns that it could infer
        # none. Its advice, to declare position or velocity columns, cannot apply to a signal
        # without axes; the capability declaration records the representation instead.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gaze contains samples but no components')
            return Gaze(
                samples=pl.DataFrame({'time': times_ms, VELOCITY_MAGNITUDE_COLUMN: magnitudes}),
                events=_segment_events(segments, start_ms),
                time_column='time',
                time_unit='ms',
            )


def _event_sequence(
        config: StatisticalConfig,
        rng: np.random.Generator,
        events: Events | None,
) -> tuple[list[tuple[str, float]], float]:
    """Return the (event type, duration in ms) pairs to build, and the time they start at."""
    if events is not None:
        return _conditioned_sequence(events)

    if config.duration_s is None:
        raise ValueError(
            f"generator '{StatisticalGenerator.name}' needs either events to condition on or a "
            f'duration_s to fill, otherwise the length of the sequence is undefined',
        )

    total_ms = config.duration_s * 1000.0
    segments: list[tuple[str, float]] = []
    elapsed_ms = 0.0

    while elapsed_ms < total_ms:
        if config.event_sequence is None:
            event_type = EVENT_TYPES[int(rng.integers(len(EVENT_TYPES)))]
        else:
            event_type = config.event_sequence[len(segments) % len(config.event_sequence)]

        duration_ms = min(_draw(event_type, 'duration', '_ms', config, rng), total_ms - elapsed_ms)
        segments.append((event_type, duration_ms))
        elapsed_ms += duration_ms

    return segments, 0.0


def _conditioned_sequence(events: Events) -> tuple[list[tuple[str, float]], float]:
    """Make a conditioning sequence contiguous by turning each gap into a saccade.

    A fixation sequence has gaps, because the saccades between the fixations were never part of
    it. Dropping them would shorten the signal and move every later event, so the output would no
    longer agree with the input it was conditioned on.
    """
    onsets = events.frame['onset'].dt.total_milliseconds().to_list()
    offsets = events.frame['offset'].dt.total_milliseconds().to_list()
    names = events.frame['name'].to_list()

    start_ms = float(onsets[0])
    segments: list[tuple[str, float]] = []
    previous_offset_ms = start_ms

    for name, onset, offset in zip(names, onsets, offsets):
        if onset > previous_offset_ms:
            segments.append(('saccade', onset - previous_offset_ms))

        segments.append((name, float(offset - onset)))
        previous_offset_ms = float(offset)

    return segments, start_ms


def _build_profiles(
        segments: list[tuple[str, float]],
        config: StatisticalConfig,
        rng: np.random.Generator,
        start_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build each event's profile at the internal rate and lay them end to end.

    The times are carried alongside the values rather than assumed from the internal rate: an
    event shorter than one internal sample still gets one, so its samples sit closer together
    than that rate implies.
    """
    profiles = []
    times = []
    elapsed_ms = start_ms

    for event_type, duration_ms in segments:
        profile = _velocity_profile(event_type, duration_ms, config, rng)
        profiles.append(profile)
        times.append(elapsed_ms + np.linspace(0.0, duration_ms, profile.size, endpoint=False))
        elapsed_ms += duration_ms

    return np.concatenate(profiles), np.concatenate(times)


def _velocity_profile(
        event_type: str,
        duration_ms: float,
        config: StatisticalConfig,
        rng: np.random.Generator,
) -> np.ndarray:
    """Build one event's velocity magnitude profile at the internal sampling rate."""
    n_samples = max(round(duration_ms * config.profile_sampling_rate / 1000.0), 1)

    if event_type == 'fixation':
        profile = np.zeros(n_samples)
    elif event_type == 'saccade':
        profile = _saccade_profile(n_samples, duration_ms, config, rng)
    else:
        profile = _pursuit_profile(n_samples, duration_ms, config, rng)

    consistency = _draw(event_type, 'consistency', '_dps', config, rng)

    if consistency:
        profile = profile + rng.normal(0.0, consistency, n_samples)

    return np.abs(profile)


def _saccade_profile(
        n_samples: int,
        duration_ms: float,
        config: StatisticalConfig,
        rng: np.random.Generator,
) -> np.ndarray:
    """Build a Gamma-shaped saccade profile, scaled to a peak its length permits.

    The length and a second number, both in [0, 1], are multiplied, so a shorter saccade is
    limited to a lower peak velocity. The length's number is where this saccade's duration falls
    in its own range, which keeps the two coupled even when the duration came from conditioning
    input rather than from a draw.

    The skewness fixes the Gamma's shape through the distribution's own identity,
    skewness = 2 / sqrt(shape).
    """
    duration_span = config.saccade_duration_max_ms - config.saccade_duration_min_ms

    if duration_span > 0:
        length_fraction = np.clip(
            (duration_ms - config.saccade_duration_min_ms) / duration_span, 0.0, 1.0,
        )
    else:
        length_fraction = 1.0

    peak = config.saccade_peak_velocity_min_dps + length_fraction * rng.uniform() * (
        config.saccade_peak_velocity_max_dps - config.saccade_peak_velocity_min_dps
    )

    skewness = rng.uniform(config.saccade_skewness_min, config.saccade_skewness_max)
    shape = (2.0 / skewness) ** 2

    # Only the Gamma density's shape matters, normalised to its own maximum, so the normalising
    # constant cancels. It is evaluated in log space because the shape grows as the skewness
    # shrinks and a direct power would overflow. The support starts one step past zero, where the
    # logarithm is undefined.
    support = np.linspace(0.0, 2.0 * shape, n_samples + 1)[1:]
    log_density = (shape - 1.0) * np.log(support) - support

    return peak * np.exp(log_density - log_density.max())


def _pursuit_profile(
        n_samples: int,
        duration_ms: float,
        config: StatisticalConfig,
        rng: np.random.Generator,
) -> np.ndarray:
    """Build a smooth pursuit: a sigmoid onset, then a constant, growing or decreasing pursuit."""
    minimum = config.smooth_pursuit_velocity_min_dps
    maximum = config.smooth_pursuit_velocity_max_dps

    if config.smooth_pursuit_profile == 'growing':
        pursuit = np.linspace(minimum, maximum, n_samples)
    elif config.smooth_pursuit_profile == 'decreasing':
        pursuit = np.linspace(maximum, minimum, n_samples)
    else:
        pursuit = np.full(n_samples, rng.uniform(minimum, maximum))

    onset_ms = rng.uniform(
        config.smooth_pursuit_onset_duration_min_ms,
        config.smooth_pursuit_onset_duration_max_ms,
    )
    onset_samples = min(
        max(round(min(onset_ms, duration_ms) * config.profile_sampling_rate / 1000.0), 1),
        n_samples,
    )

    # A sigmoid over [-6, 6] is saturated at both ends, so the onset meets the pursuit smoothly.
    ramp = np.ones(n_samples)
    ramp[:onset_samples] = 1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, onset_samples)))

    return pursuit * ramp


def _resample(
        profile: np.ndarray,
        profile_times_ms: np.ndarray,
        segments: list[tuple[str, float]],
        config: StatisticalConfig,
) -> np.ndarray:
    """Interpolate the internal profile down to the requested sampling rate.

    This is what lets any sampling rate be served: the profiles are built at a rate far above the
    target, and the target is obtained by interpolating them.
    """
    duration_ms = sum(duration_ms for _, duration_ms in segments)
    n_samples = max(round(duration_ms / 1000.0 * config.sampling_rate), 1)
    step_ms = 1000.0 / config.sampling_rate
    target_times_ms = profile_times_ms[0] + np.arange(n_samples) * step_ms

    return np.interp(target_times_ms, profile_times_ms, profile)


def _segment_events(segments: list[tuple[str, float]], start_ms: float) -> Events:
    """Return the event sequence the signal was built from, with onsets and offsets in ms."""
    offsets_ms = start_ms + np.cumsum([duration_ms for _, duration_ms in segments])

    return Events(
        name=[event_type for event_type, _ in segments],
        onsets=[start_ms, *offsets_ms[:-1]],
        offsets=list(offsets_ms),
        time_unit='ms',
    )


def _draw(
        event_type: str,
        quantity: str,
        unit: str,
        config: StatisticalConfig,
        rng: np.random.Generator,
) -> float:
    """Draw one value for an event type's named quantity from its configured range.

    A uniform draw spans the range. A normal draw is centred on the range and clipped to it,
    using the standard deviation belonging to that quantity, or the width of the range where the
    quantity has none of its own.
    """
    minimum = getattr(config, f'{event_type}_{quantity}_min{unit}')
    maximum = getattr(config, f'{event_type}_{quantity}_max{unit}')
    distribution = getattr(config, f'{event_type}_{quantity}_distribution')

    if distribution == 'uniform':
        return float(rng.uniform(minimum, maximum))

    std = getattr(config, f'{event_type}_{quantity}_std{unit}', maximum - minimum)

    return float(np.clip(rng.normal((minimum + maximum) / 2.0, std), minimum, maximum))


def _check_name(config: StatisticalConfig, field: str, allowed: tuple[str, ...]) -> None:
    """Reject a value outside a fixed set of names."""
    value = getattr(config, field)

    if value not in allowed:
        raise ValueError(f'{field} must be one of {list(allowed)} but is: {value!r}')


def _check_not_negative(config: StatisticalConfig, field: str) -> None:
    """Reject a negative spread."""
    value = getattr(config, field)

    if value < 0:
        raise ValueError(f'{field} must not be negative but is: {value}')


def _check_ordered(config: StatisticalConfig, minimum_field: str, maximum_field: str) -> None:
    """Reject an inverted range, which would otherwise produce a silently empty draw."""
    minimum = getattr(config, minimum_field)
    maximum = getattr(config, maximum_field)

    if minimum > maximum:
        raise ValueError(
            f'{minimum_field} ({minimum}) must not be greater than {maximum_field} ({maximum})',
        )


def _check_skewness(config: StatisticalConfig) -> None:
    """Reject a skewness outside the interval where the Gamma profile rises to a peak.

    At a skewness of two the Gamma shape reaches one and the profile peaks at its first sample,
    losing the onset that makes it a saccade. At or below zero there is no Gamma shape at all.
    """
    if config.saccade_skewness_min <= 0:
        raise ValueError(
            f'saccade_skewness_min must be greater than zero but is: '
            f'{config.saccade_skewness_min}',
        )

    if config.saccade_skewness_max >= 2:
        raise ValueError(
            f'saccade_skewness_max must be below two but is: {config.saccade_skewness_max}. '
            f'At two the velocity profile peaks at its first sample and loses the onset that '
            f'makes it a saccade',
        )


def _check_event_sequence(sequence: Any) -> tuple[str, ...]:
    """Reject an empty sequence or an event type the simulator does not produce."""
    sequence = tuple(sequence)

    if not sequence:
        raise ValueError('event_sequence must not be empty, pass None to draw types instead')

    unknown = sorted(set(sequence) - set(EVENT_TYPES))
    if unknown:
        raise ValueError(
            f'event_sequence contains event types that are not produced: {unknown}. '
            f'Produced event types are: {list(EVENT_TYPES)}',
        )

    return sequence
