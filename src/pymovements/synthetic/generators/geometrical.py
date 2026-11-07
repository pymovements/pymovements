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
"""Provides a geometrical sample generator.

The generator holds each fixation at its location and interpolates the gaze between fixations,
linearly or with a cubic spline, the two methods recommended for filling gaps in recorded gaze
positions by:

    Grootjen, J. W., Weingärtner, H., & Mayer, S. (2024). Uncovering and Addressing Blink-Related
    Challenges in Using Eye Tracking for Interactive Systems. CHI '24.

Using the fixation centres as the positions the gaze moves between follows the mapping step in
section 3.7 of:

    Fuhl, W., & Kasneci, E. (2018). Eye movement velocity and gaze data generator for evaluation,
    robustness testing and assess of eye tracking software and visualization tools.
    arXiv:1808.09296.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
from typing import cast
from typing import ClassVar

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic._utils import check_name
from pymovements.synthetic._utils import per_trial
from pymovements.synthetic._utils import sample_times
from pymovements.synthetic.generator import SampleGenerator
from pymovements.synthetic.registry import register_generator
from pymovements.synthetic.value_range import Range

#: The event types the generator can render. Everything that is not a fixation is interpolated.
EVENT_TYPES = ('fixation', 'saccade')

#: The event column the fixation locations are read from, as the ``location`` event property
#: computes it.
LOCATION_COLUMN = 'location'

RangeLike = Range | tuple[float, float] | float


@register_generator
@dataclasses.dataclass
class GeometricalSampleGenerator(SampleGenerator):
    """Generates gaze positions that hold each fixation's location and move between them.

    Conditioned on fixations with a ``location``, every sample inside a fixation takes its
    location, and every other sample is interpolated between the locations around it, per axis.
    These are the gaps between fixations and the saccades ``bridge_gaps`` fills them with. A
    saccade's own location is not used. Before the first and after the last fixation of a trial,
    the nearest location is held. Events with trial columns are generated trial by trial. An
    event of zero duration gets no samples and stays in the attached events. A fixation that no
    sample falls inside is rejected, since its location would be lost.

    A small scatter of the fixation samples can be added with ``fixation_deviation``, in the unit
    of the locations: per fixation, a deviation is drawn from the range and used as the standard
    deviation of a normal scatter per sample and axis. It is added after the interpolation, so
    the gaps follow the locations themselves. The default of zero gives noise-free output.

    The movement between fixations is interpolated, not modelled, and its duration is the gap the
    events leave. Linear interpolation gives a constant velocity. A cubic spline, fitted over all
    fixations of a trial, gives a smooth movement whose velocity peaks around mid-gap, and it can
    overshoot a location at low rates or around very short fixations. Not modelled are velocity
    profiles along the path, saccades that deviate from the straight line, and saliency maps for
    choosing fixation targets.

    Attributes
    ----------
    name: ClassVar[str]
        ``'geometrical'``, the name the generator is registered under.
    interpolation: str
        How the gaze between two fixations is interpolated, ``'linear'`` or ``'cubic_spline'``.
        (default: 'cubic_spline')
    location_column_group: str
        The column group the positions are written to, ``'position'`` or ``'pixel'``. It must
        match the unit of the locations. (default: 'position')
    fixation_deviation: RangeLike
        The scatter of a fixation's samples around its location. (default: Range(0.0, 0.0))

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a name is unknown or the deviation is negative or not finite.

    Examples
    --------
    Move between two fixations at 1000 Hz:

    >>> import polars as pl
    >>> import pymovements as pm
    >>> events = pm.Events(pl.DataFrame({
    ...     'name': ['fixation', 'fixation'],
    ...     'onset': [0, 300],
    ...     'offset': [250, 560],
    ...     'location': [[0.0, 0.0], [10.0, 0.0]],
    ... }))
    >>> generator = pm.synthetic.GeometricalSampleGenerator(sampling_rate=1000)
    >>> gaze = generator.generate(events=events, seed=42)  # doctest: +IGNORE_WARNINGS
    >>> gaze.samples.columns
    ['time', 'position']
    >>> gaze.samples.height
    560

    The fixations hold their locations, and the gap between them is interpolated:

    >>> gaze.samples['position'][[249, 300]].to_list()
    [[0.0, 0.0], [10.0, 0.0]]
    >>> 0.0 < gaze.samples['position'][275][0] < 10.0
    True
    """

    name: ClassVar[str] = 'geometrical'

    interpolation: str = 'cubic_spline'
    location_column_group: str = 'position'
    fixation_deviation: RangeLike = Range(0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate the names and read the deviation from its accepted forms."""
        super().__post_init__()

        check_name(self.interpolation, 'interpolation', ('linear', 'cubic_spline'))
        check_name(
            self.location_column_group, 'location_column_group', ('position', 'pixel'),
        )

        try:
            self.fixation_deviation = Range.from_value(self.fixation_deviation)
        except ValueError as error:
            raise ValueError(f'fixation_deviation: {error}') from error

        if self.fixation_deviation.minimum < 0:
            raise ValueError(
                'fixation_deviation must not be negative but has a minimum of: '
                f'{self.fixation_deviation.minimum}',
            )

    def _generate(  # pylint: disable=arguments-differ
            self,
            *,
            rng: np.random.Generator,
            events: Events,
    ) -> Gaze:
        """Synthesize the gaze positions that follow the fixations.

        Parameters
        ----------
        rng: np.random.Generator
            The run's random generator.
        events: Events
            The fixations to follow, with a ``location`` of two components each, and optionally
            the saccades between them.

        Returns
        -------
        Gaze
            Samples with a time column in milliseconds and the positions.

        Raises
        ------
        TypeError
            If ``events`` is not an ``Events`` object.
        ValueError
            If the events are empty, overlap, all have a duration of zero, contain a null onset,
            a null or negative duration, an event type other than fixations and saccades, or a
            fixation without a usable location, if a trial has no fixation, if no sample falls
            inside a fixation, or if a trial column has the name of a generated or ``Gaze``
            column.
        """
        if not isinstance(events, Events):
            raise TypeError(
                f"generator '{self.name}' needs events of type Events to condition on, but got "
                f'{type(events).__name__}',
            )

        unknown = sorted(set(events.frame['name'].to_list()) - set(EVENT_TYPES), key=str)
        if unknown:
            raise ValueError(
                f"generator '{self.name}' cannot render the event types {unknown}. "
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

        def synthesize(trial: pl.DataFrame) -> pl.DataFrame:
            return self._synthesize_trial(trial, rng)

        samples = per_trial(rendered, synthesize, generator=self.name)

        in_pixels = self.location_column_group == 'pixel'
        return Gaze(
            samples=samples,
            trial_columns=events.trial_columns,
            time_column='time',
            time_unit='ms',
            pixel_columns=['_x', '_y'] if in_pixels else None,
            position_columns=None if in_pixels else ['_x', '_y'],
        )

    def _locations(self, fixations: pl.DataFrame) -> np.ndarray:
        """Return the locations of the fixations, rejecting any that is not two finite numbers."""
        if LOCATION_COLUMN not in fixations.columns:
            raise ValueError(
                f"generator '{self.name}' needs a '{LOCATION_COLUMN}' column holding the position "
                f'of each fixation, for example from the location event property, but the '
                f'columns are: {fixations.columns}',
            )

        for location in fixations[LOCATION_COLUMN].to_list():
            if not isinstance(location, list) or len(location) != 2:
                raise ValueError(
                    f"generator '{self.name}' needs a '{LOCATION_COLUMN}' of two components for "
                    f'every fixation, but one has: {location}',
                )
            if not all(
                isinstance(value, numbers.Real) and not isinstance(value, bool)
                and math.isfinite(value)
                for value in location
            ):
                raise ValueError(
                    f"generator '{self.name}' needs a finite number for both components of a "
                    f"'{LOCATION_COLUMN}', but one fixation has: {location}",
                )

        return np.array(fixations[LOCATION_COLUMN].to_list(), dtype=float).reshape(-1, 2)

    def _synthesize_trial(self, trial: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
        """Return the samples of one trial's ordered events."""
        onsets = trial['onset'].dt.total_milliseconds(fractional=True).to_numpy()
        ends = onsets + trial['duration'].dt.total_milliseconds(fractional=True).to_numpy()
        times = sample_times(onsets[0], ends[-1], self.sampling_rate)

        is_fixation = (trial['name'] == 'fixation').to_numpy()
        if not is_fixation.any():
            raise ValueError(
                f"generator '{self.name}': the trial starting at {onsets[0]} ms has no fixation "
                'of a non-zero duration, so there is no location to follow',
            )

        locations = self._locations(trial.filter(is_fixation))

        # Rounding can put a sample just below the onset or end it falls on, so events are compared
        # with a margin far below their precision of a microsecond.
        exact_times = times + 1e-6

        # The fixation each sample falls into, if any: the last one starting at or before it.
        # A sample before the first fixation gets -1, which picks the appended end of -inf.
        fixation = np.searchsorted(onsets[is_fixation], exact_times, side='right') - 1
        held = exact_times < np.append(ends[is_fixation], -np.inf)[fixation]

        unsampled = np.setdiff1d(np.arange(len(locations)), fixation[held])
        if unsampled.size:
            raise ValueError(
                f"generator '{self.name}': no sample falls inside the fixation at "
                f'{onsets[is_fixation][unsampled[0]]} ms at {self.sampling_rate} Hz, so its '
                'location would be lost; raise the sampling rate or drop the fixation',
            )

        positions = self._interpolate(times, locations[fixation], held)
        positions = self._scatter(positions, fixation[held], held, rng)

        return pl.DataFrame({'time': times, '_x': positions[:, 0], '_y': positions[:, 1]})

    def _interpolate(
            self,
            times: np.ndarray,
            locations: np.ndarray,
            held: np.ndarray,
    ) -> np.ndarray:
        """Return the positions, interpolating the samples outside fixations per axis.

        Outside the held samples, the nearest location is held. The cubic spline is fitted on all
        held samples of the trial and used only between them.
        """
        known_times = times[held]
        known = locations[held]

        positions = np.column_stack([
            np.interp(times, known_times, known[:, axis]) for axis in range(2)
        ])

        if self.interpolation == 'cubic_spline' and known_times.size > 1:
            between = ~held & (times > known_times[0]) & (times < known_times[-1])
            positions[between] = CubicSpline(known_times, known)(times[between])

        return positions

    def _scatter(
            self,
            positions: np.ndarray,
            held_fixation: np.ndarray,
            held: np.ndarray,
            rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the positions with each fixation's samples scattered by a drawn deviation.

        The held samples are in time order, so each fixation's samples are one run of them.
        """
        deviations = cast(Range, self.fixation_deviation)
        scattered = positions.copy()

        boundaries = np.flatnonzero(np.diff(held_fixation)) + 1
        for samples in np.split(np.flatnonzero(held), boundaries):
            deviation = deviations.draw(rng)

            if deviation:
                scattered[samples] += rng.normal(0.0, deviation, (samples.size, 2))

        return scattered
