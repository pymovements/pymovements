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
"""Provides a statistical sample generator.

The generator implements the velocity profiles of the eye movement simulator in section 3 of:

    Fuhl, W., Santini, T., Kuebler, T., Castner, N., Rosenstiel, W., & Kasneci, E. (2018).
    Eye movement simulation and detector creation to reduce laborious parameter adjustments.
    arXiv:1804.00970.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import cast
from typing import ClassVar

import numpy as np
import polars as pl

from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic._utils import check_contiguous
from pymovements.synthetic._utils import check_name
from pymovements.synthetic._utils import check_number
from pymovements.synthetic._utils import check_positive
from pymovements.synthetic._utils import per_trial
from pymovements.synthetic._utils import sample_times
from pymovements.synthetic.generator import SampleGenerator
from pymovements.synthetic.registry import register_generator
from pymovements.synthetic.value_range import Range

#: The event types a velocity profile is defined for.
EVENT_TYPES = ('fixation', 'saccade', 'smooth_pursuit')

RangeLike = Range | tuple[float, float] | float


@register_generator
@dataclasses.dataclass
class StatisticalSampleGenerator(SampleGenerator):
    """Generates velocity magnitudes from parametrised profiles of fixations, saccades and pursuits.

    Conditioned on a gap-free event sequence, each event gets a velocity profile, built at
    ``profile_sampling_rate``: fixations are a small fluctuation above zero, saccades follow a
    Gamma-shaped profile, and smooth pursuits open with a sigmoid onset before a constant, growing
    or decreasing pursuit. Each sample is then the mean of the profile from its own time up to the
    next sample, and measurement noise is added last. Events with trial columns are generated
    trial by trial. An event with a duration of zero gets no samples and stays in the attached
    events, while a null duration is rejected, as the end is unknown, and so is a negative one.

    Each random quantity is a :py:class:`~pymovements.synthetic.Range`, given as a range, a
    ``(minimum, maximum)`` pair for a uniform range, or a single value. Durations are in
    milliseconds, velocities and consistencies in degrees per second. The defaults are
    illustrations, not measured values.

    The consistency is the standard deviation of the normal jitter at each point of a profile.
    The absolute value of the profile is taken, so a fixation's mean lies above zero, at about
    0.8 times its consistency. Since samples are means, their spread is smaller than the
    consistency, and shrinks as ``profile_sampling_rate`` rises or ``sampling_rate`` falls. A
    consistency tuned at one pair of rates does not carry over. It is drawn once per event, so
    its distribution changes how strong the jitter is from event to event, not the look of the
    fluctuation within one event.

    The output is a velocity magnitude without direction, so detectors and transforms that need
    positions cannot consume it. Not modelled are constraints on the order of the events, which
    belong to an event generator, fluctuating sampling rates, a normal distribution for where
    noise is placed, and post-saccadic movement. A skewed saccade has not decayed when its event
    ends, so it stops above zero.

    Attributes
    ----------
    name: ClassVar[str]
        ``'statistical'``, the name the generator is registered under.
    fixation_consistency: RangeLike
        The jitter during a fixation. (default: Range(0.0, 1.0))
    saccade_duration: RangeLike
        The range the saccade durations are read against. A saccade's position in it limits its
        peak velocity. (default: Range(20.0, 80.0))
    saccade_peak_velocity: RangeLike
        The peak velocity of a saccade of the longest duration. (default: Range(300.0, 500.0))
    saccade_skewness: RangeLike
        The skewness of the Gamma-shaped profile. Must lie between zero and two, where the
        profile stops rising to a peak. It also sets the width of a saccade: a low skewness gives
        a narrow peak, not a wide and nearly symmetric saccade. (default: Range(1.0, 1.5))
    saccade_consistency: RangeLike
        The jitter along a saccade. (default: Range(0.0, 5.0))
    smooth_pursuit_velocity: RangeLike
        The velocity of a pursuit. A growing or decreasing pursuit spans the whole range.
        (default: Range(10.0, 30.0))
    smooth_pursuit_onset_duration: RangeLike
        The duration of the onset of a pursuit. (default: Range(40.0, 120.0))
    smooth_pursuit_profile: str
        The shape after the onset, ``'constant'``, ``'growing'`` or ``'decreasing'``.
        (default: 'constant')
    smooth_pursuit_consistency: RangeLike
        The jitter along a pursuit. (default: Range(0.0, 2.0))
    noise_percentage: float
        The share of all samples that receive noise, from 0 to 100. (default: 0.0)
    noise_velocity: RangeLike
        The velocity change of a noisy sample. (default: Range(0.0, 0.0))
    profile_sampling_rate: float | None
        The rate in Hz the profiles are built at. Must not be below the sampling rate. ``None``
        means 10000 Hz, or the sampling rate if that is higher. (default: None)

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a name is unknown, a value is outside its valid interval, or a value would have no
        effect, such as a distribution for ``saccade_duration`` or a ``noise_velocity`` without
        a ``noise_percentage``.

    Examples
    --------
    Render a fixation, a saccade and a fixation at 1000 Hz:

    >>> import pymovements as pm
    >>> events = pm.Events(
    ...     name=['fixation', 'saccade', 'fixation'],
    ...     onsets=[0, 200, 240],
    ...     offsets=[200, 240, 500],
    ... )
    >>> generator = pm.synthetic.StatisticalSampleGenerator(
    ...     sampling_rate=1000, saccade_peak_velocity=(300, 450),
    ... )
    >>> gaze = generator.generate(events=events, seed=42)  # doctest: +IGNORE_WARNINGS
    >>> gaze.samples.columns
    ['time', 'velocity_magnitude']
    >>> gaze.samples.height
    500

    A pair is read as a uniform range:

    >>> generator.saccade_peak_velocity
    Range(minimum=300.0, maximum=450.0, distribution='uniform', std=None)
    """

    name: ClassVar[str] = 'statistical'

    fixation_consistency: RangeLike = Range(0.0, 1.0)
    saccade_duration: RangeLike = Range(20.0, 80.0)
    saccade_peak_velocity: RangeLike = Range(300.0, 500.0)
    saccade_skewness: RangeLike = Range(1.0, 1.5)
    saccade_consistency: RangeLike = Range(0.0, 5.0)
    smooth_pursuit_velocity: RangeLike = Range(10.0, 30.0)
    smooth_pursuit_onset_duration: RangeLike = Range(40.0, 120.0)
    smooth_pursuit_profile: str = 'constant'
    smooth_pursuit_consistency: RangeLike = Range(0.0, 2.0)
    noise_percentage: float = 0.0
    noise_velocity: RangeLike = Range(0.0, 0.0)
    profile_sampling_rate: float | None = None

    def __post_init__(self) -> None:
        """Read each range from its accepted forms and validate the parameters."""
        super().__post_init__()

        for name in (
            'fixation_consistency',
            'saccade_duration',
            'saccade_peak_velocity',
            'saccade_skewness',
            'saccade_consistency',
            'smooth_pursuit_velocity',
            'smooth_pursuit_onset_duration',
            'smooth_pursuit_consistency',
            'noise_velocity',
        ):
            try:
                value = _range(getattr(self, name))
            except ValueError as error:
                raise ValueError(f'{name}: {error}') from error

            if name != 'noise_velocity' and value.minimum < 0:
                raise ValueError(
                    f'{name} must not be negative but has a minimum of: {value.minimum}',
                )
            setattr(self, name, value)

        skewness = _range(self.saccade_skewness)
        if skewness.minimum <= 0:
            raise ValueError(
                f'saccade_skewness must be greater than zero but has a minimum of: '
                f'{skewness.minimum}',
            )
        if skewness.maximum >= 2:
            raise ValueError(
                f'saccade_skewness must stay below two but has a maximum of: {skewness.maximum}. '
                'At two the profile peaks at its start, and close to two a short saccade already '
                'loses the onset that makes it one, so stay well below',
            )

        check_name(
            self.smooth_pursuit_profile,
            'smooth_pursuit_profile',
            ('constant', 'growing', 'decreasing'),
        )

        self.noise_percentage = check_number(self.noise_percentage, 'noise_percentage')
        if not 0.0 <= self.noise_percentage <= 100.0:
            raise ValueError(
                f'noise_percentage must be between 0 and 100 but is: {self.noise_percentage}',
            )

        if self.profile_sampling_rate is None:
            self.profile_sampling_rate = max(10_000.0, self.sampling_rate)
        self.profile_sampling_rate = check_positive(
            self.profile_sampling_rate, 'profile_sampling_rate',
        )
        if self.profile_sampling_rate < self.sampling_rate:
            raise ValueError(
                f'profile_sampling_rate ({self.profile_sampling_rate}) must not be below '
                f'sampling_rate ({self.sampling_rate}), otherwise the profiles are built more '
                'coarsely than the samples taken from them',
            )

        self._check_effective()

    def _check_effective(self) -> None:
        """Reject a hyperparameter whose value would have no effect, as it hides a mistake."""
        durations = _range(self.saccade_duration)
        if durations.distribution != 'uniform':
            raise ValueError(
                'saccade_duration only gives the bounds a saccade length is read against, so its '
                f'distribution {durations.distribution!r} would have no effect; give it as '
                '(minimum, maximum)',
            )

        velocities = _range(self.smooth_pursuit_velocity)
        if self.smooth_pursuit_profile != 'constant' and velocities.distribution != 'uniform':
            raise ValueError(
                f'a {self.smooth_pursuit_profile!r} pursuit spans smooth_pursuit_velocity from '
                f'end to end, so its distribution {velocities.distribution!r} would have no '
                'effect; give it as (minimum, maximum)',
            )

        noise = _range(self.noise_velocity)
        has_noise_velocity = noise.minimum != 0.0 or noise.maximum != 0.0
        if has_noise_velocity and not self.noise_percentage:
            raise ValueError(
                'noise_velocity would have no effect while noise_percentage is 0; set '
                'noise_percentage to the share of samples that receive noise',
            )
        if self.noise_percentage and not has_noise_velocity:
            raise ValueError(
                f'noise_percentage ({self.noise_percentage}) would have no effect while '
                'noise_velocity is 0; set noise_velocity to the velocity change of a noisy sample',
            )

    def _generate(  # pylint: disable=arguments-differ
            self,
            *,
            rng: np.random.Generator,
            events: Events,
    ) -> Gaze:
        """Synthesize the velocity magnitudes of the events.

        Parameters
        ----------
        rng: np.random.Generator
            The run's random generator.
        events: Events
            The gap-free event sequence to condition on. It decides the type and timing of every
            velocity profile.

        Returns
        -------
        Gaze
            Samples with a time column in milliseconds and a velocity magnitude column.

        Raises
        ------
        TypeError
            If ``events`` is not an ``Events`` object.
        ValueError
            If the events have gaps, overlap, are empty, all have a duration of zero, contain
            a null onset, a null or negative duration or an event type without a profile, or
            if a trial column has the name of a generated or ``Gaze`` column.
        """
        if not isinstance(events, Events):
            raise TypeError(
                f"generator '{self.name}' needs events of type Events to condition on, but got "
                f'{type(events).__name__}',
            )

        unknown = sorted(set(events.frame['name'].to_list()) - set(EVENT_TYPES), key=str)
        if unknown:
            raise ValueError(
                f"generator '{self.name}' has no velocity profile for the event types {unknown}. "
                f'Supported event types are: {list(EVENT_TYPES)}',
            )

        unknown_end = events.frame.filter(pl.col('duration').is_null())
        if unknown_end.height:
            first = unknown_end.with_columns(
                pl.col('onset').dt.total_milliseconds(fractional=True),
            ).row(0, named=True)
            raise ValueError(
                f"generator '{self.name}': a {first['name']} at {first['onset']} ms has no known "
                'duration, so the time after it cannot be generated; set its duration or drop it',
            )

        negative = events.frame.filter(pl.col('duration') < pl.duration(microseconds=0))
        if negative.height:
            first = negative.with_columns(
                pl.col('onset', 'duration').dt.total_milliseconds(fractional=True),
            ).row(0, named=True)
            raise ValueError(
                f"generator '{self.name}': a {first['name']} at {first['onset']} ms has a "
                f"negative duration of {first['duration']} ms; correct its duration or drop it",
            )

        # An event of zero duration selects no samples. A separate copy is rendered, since the
        # events are attached to the output as they came.
        rendered = events.clone()
        rendered.frame = rendered.frame.filter(pl.col('duration') > pl.duration(microseconds=0))
        if rendered.frame.is_empty() and not events.frame.is_empty():
            raise ValueError(
                f"generator '{self.name}': all events have a duration of zero, so there is "
                'nothing to generate samples for',
            )

        check_contiguous(rendered, generator=self.name)

        def synthesize(trial: pl.DataFrame) -> pl.DataFrame:
            return self._synthesize_trial(trial, rng)

        samples = per_trial(rendered, synthesize, generator=self.name)
        samples = samples.with_columns(
            pl.Series(
                'velocity_magnitude',
                self._add_noise(samples['velocity_magnitude'].to_numpy(), rng),
            ),
        )

        # A magnitude fills none of the Gaze component columns, which Gaze warns about. Its
        # advice, to declare position or velocity columns, cannot apply to a signal without axes.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gaze contains samples but no components')
            return Gaze(
                samples=samples,
                trial_columns=events.trial_columns,
                time_column='time',
                time_unit='ms',
            )

    def _synthesize_trial(self, trial: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
        """Return the samples of one trial's ordered, gap-free events."""
        onsets = trial['onset'].dt.total_milliseconds(fractional=True).to_numpy()
        durations = trial['duration'].dt.total_milliseconds(fractional=True).to_numpy()
        ends = onsets + durations

        profiles = []
        profile_times = []
        for event_type, onset, duration, end in zip(trial['name'], onsets, durations, ends):
            profile = self._velocity_profile(event_type, duration, rng)
            profiles.append(profile)
            profile_times.append(np.linspace(onset, end, profile.size, endpoint=False))

        times = sample_times(onsets[0], ends[-1], self.sampling_rate)
        magnitudes = _resample(np.concatenate(profiles), np.concatenate(profile_times), times)

        return pl.DataFrame({
            'time': times,
            'velocity_magnitude': magnitudes,
        })

    def _velocity_profile(
            self,
            event_type: str,
            duration: float,
            rng: np.random.Generator,
    ) -> np.ndarray:
        """Build one event's velocity profile at the profile sampling rate."""
        n_samples = max(self._profile_points(duration), 1)

        if event_type == 'fixation':
            profile = np.zeros(n_samples)
            consistency = self.fixation_consistency
        elif event_type == 'saccade':
            profile = self._saccade_profile(n_samples, duration, rng)
            consistency = self.saccade_consistency
        else:
            profile = self._pursuit_profile(n_samples, rng)
            consistency = self.smooth_pursuit_consistency

        jitter = _range(consistency).draw(rng)
        if jitter:
            profile = profile + rng.normal(0.0, jitter, n_samples)

        return np.abs(profile)

    def _saccade_profile(
            self,
            n_samples: int,
            duration: float,
            rng: np.random.Generator,
    ) -> np.ndarray:
        """Build a Gamma-shaped saccade, its peak limited by where its duration falls in range.

        Fuhl et al. §3.3 multiply two numbers in [0, 1], one for the length and one for the peak
        velocity, so that shorter saccades reach lower peaks. The skewness fixes the Gamma's
        shape through skewness = 2 / sqrt(shape).
        """
        durations = _range(self.saccade_duration)
        if durations.maximum > durations.minimum:
            length_fraction = np.clip(
                (duration - durations.minimum) / (durations.maximum - durations.minimum), 0.0, 1.0,
            )
        else:
            length_fraction = 1.0

        velocities = _range(self.saccade_peak_velocity)
        peak = velocities.minimum + length_fraction * (velocities.draw(rng) - velocities.minimum)

        shape = (2.0 / _range(self.saccade_skewness).draw(rng)) ** 2

        # The density is normalised to its maximum, so its constant cancels. It is evaluated in
        # log space, since a direct power overflows for large shapes, and starts one step past
        # zero, where the logarithm is undefined.
        support = np.linspace(0.0, 2.0 * shape, n_samples + 1)[1:]
        log_density = (shape - 1.0) * np.log(support) - support

        return peak * np.exp(log_density - log_density.max())

    def _pursuit_profile(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Build a smooth pursuit: a sigmoid onset, then a constant, growing or falling pursuit."""
        velocities = _range(self.smooth_pursuit_velocity)

        if self.smooth_pursuit_profile == 'growing':
            pursuit = np.linspace(velocities.minimum, velocities.maximum, n_samples)
        elif self.smooth_pursuit_profile == 'decreasing':
            pursuit = np.linspace(velocities.maximum, velocities.minimum, n_samples)
        else:
            pursuit = np.full(n_samples, velocities.draw(rng))

        onset = _range(self.smooth_pursuit_onset_duration).draw(rng)
        onset_samples = min(self._profile_points(onset), n_samples)

        # A sigmoid over [-6, 6] is saturated at both ends, so the onset meets the pursuit smoothly.
        # Its lower end is left out, so that an onset of one point is not lost at almost zero.
        ramp = np.ones(n_samples)
        ramp[:onset_samples] = 1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, onset_samples + 1)[1:]))

        return pursuit * ramp

    def _profile_points(self, duration: float) -> int:
        """Return the number of profile points of a duration in milliseconds."""
        return round(duration * cast(float, self.profile_sampling_rate) / 1000.0)

    def _add_noise(self, magnitudes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Change a share of the samples, picked uniformly, by a drawn velocity."""
        n_noisy = round(magnitudes.size * self.noise_percentage / 100.0)
        if not n_noisy:
            return magnitudes

        noisy = rng.choice(magnitudes.size, size=n_noisy, replace=False)
        changes = [_range(self.noise_velocity).draw(rng) for _ in range(n_noisy)]

        noisy_magnitudes = magnitudes.copy()
        noisy_magnitudes[noisy] = np.abs(noisy_magnitudes[noisy] + changes)
        return noisy_magnitudes


def _range(value: RangeLike) -> Range:
    """Return the value as a range. Fields hold ranges once constructed."""
    return Range.from_value(value)


def _resample(profile: np.ndarray, profile_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return the mean of the profile from each sample's time up to the next sample's.

    Fuhl et al. §3.5 average the values between two sampling positions. Each window starts at its
    own sample, so a sample stamped at an onset belongs to that event. A window holding no
    profile value falls back to the profile at the sample's time.
    """
    resampled = np.interp(times, profile_times, profile)

    starts = np.searchsorted(profile_times, times, side='left')
    ends = np.append(starts[1:], profile.size)
    cumulative = np.concatenate(([0.0], np.cumsum(profile)))
    counts = ends - starts
    sums = cumulative[ends] - cumulative[starts]

    np.divide(sums, counts, out=resampled, where=counts > 0)
    return resampled
