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
"""Test the sampling_rate keyword shared by all detection algorithms."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from pymovements import Events
from pymovements.events import blink
from pymovements.events import fill
from pymovements.events import idt
from pymovements.events import ihmm
from pymovements.events import ivt
from pymovements.events import microsaccades
from pymovements.events import out_of_screen


def _bracketed(n_samples, inner, outer):
    """Build a signal whose samples 1 to n_samples - 2 form the event."""
    return [outer] + [inner] * (n_samples - 2) + [outer]


def _ivt_kwargs(timesteps):
    return {
        'velocities': _bracketed(len(timesteps), [0, 0], [10, 10]),
        'timesteps': timesteps,
        'velocity_threshold': 1,
        'minimum_duration': 1,
    }


def _microsaccades_kwargs(timesteps):
    return {
        'velocities': _bracketed(len(timesteps), [10, 10], [0, 0]),
        'timesteps': timesteps,
        'threshold': (1, 1),
        'minimum_duration': 1,
    }


def _blink_kwargs(timesteps):
    return {
        'pupil': _bracketed(len(timesteps), np.nan, 1.0),
        'timesteps': timesteps,
        'minimum_duration': 1,
        'maximum_duration': None,
        'minimum_gap': 0,
    }


def _fill_kwargs(timesteps):
    # The existing events cover the first and the last sample, fill reads their offsets as
    # exclusive.
    return {
        'events': Events(
            onsets=[timesteps[0], timesteps[-1]],
            offsets=[timesteps[1], timesteps[-1] + 1],
        ),
        'timesteps': timesteps,
        'minimum_duration': 1,
    }


def _out_of_screen_kwargs(timesteps):
    return {
        'pixels': _bracketed(len(timesteps), [-1, 0], [5, 5]),
        'timesteps': timesteps,
        'x_max': 10,
        'y_max': 10,
    }


def _ihmm_kwargs(timesteps):
    return {
        'velocities': _bracketed(len(timesteps), [0, 0], [10, 10]),
        'timesteps': timesteps,
        'mu': [0, 10],
        'sigma': [1, 1],
        'init_state': [0.5, 0.5],
        'transition_probabilities': [[0.5, 0.5], [0.5, 0.5]],
        'minimum_duration': 1,
    }


def _idt_kwargs(timesteps):
    # A constant position: every sample belongs to the one fixation.
    return {
        'positions': [[0, 0]] * len(timesteps),
        'timesteps': timesteps,
        'dispersion_threshold': 1,
        'minimum_duration': 1,
    }


# Each detector with a builder of a minimal input over the given timesteps and the slice of
# the samples that form its single event.
DETECTORS = [
    pytest.param(ivt, _ivt_kwargs, slice(1, -1), id='ivt'),
    pytest.param(microsaccades, _microsaccades_kwargs, slice(1, -1), id='microsaccades'),
    pytest.param(blink, _blink_kwargs, slice(1, -1), id='blink'),
    pytest.param(fill, _fill_kwargs, slice(1, -1), id='fill'),
    pytest.param(out_of_screen, _out_of_screen_kwargs, slice(1, -1), id='out_of_screen'),
    pytest.param(ihmm, _ihmm_kwargs, slice(1, -1), id='ihmm'),
    pytest.param(idt, _idt_kwargs, slice(None), id='idt'),
]
# I-DT needs two samples for a fixation, every other detector forms a single-sample event.
SINGLE_SAMPLE_DETECTORS = [param for param in DETECTORS if param.id != 'idt']
# out_of_screen has no duration threshold.
THRESHOLD_DETECTORS = [param for param in DETECTORS if param.id != 'out_of_screen']


def _entries(sampling_rate):
    return {'offset': {'inclusive': True}, 'sampling_rate': sampling_rate}


@pytest.mark.parametrize(('detector', 'build_kwargs', 'event_samples'), DETECTORS)
def test_detector_requires_sampling_rate(detector, build_kwargs, event_samples):
    del event_samples
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'sampling_rate'",
    ):
        detector(**build_kwargs(np.arange(3)))


@pytest.mark.parametrize(('detector', 'build_kwargs', 'event_samples'), DETECTORS)
@pytest.mark.parametrize('sampling_rate', [0, -1])
def test_detector_raises_on_non_positive_sampling_rate(
        detector, build_kwargs, event_samples, sampling_rate,
):
    del event_samples
    with pytest.raises(ValueError, match='sampling_rate must be positive'):
        detector(**build_kwargs(np.arange(3)), sampling_rate=sampling_rate)


@pytest.mark.parametrize(('detector', 'build_kwargs', 'event_samples'), SINGLE_SAMPLE_DETECTORS)
@pytest.mark.parametrize('sampling_rate', [1000, 600])
def test_single_sample_event_lasts_one_nominal_interval(
        detector, build_kwargs, event_samples, sampling_rate,
):
    del event_samples
    events = detector(**build_kwargs(np.arange(3)), sampling_rate=sampling_rate)

    nominal_interval = timedelta(microseconds=round(1_000_000 / sampling_rate))
    assert events.frame['onset'].to_list() == [timedelta(milliseconds=1)]
    assert events.frame['duration'].to_list() == [nominal_interval]
    assert events.frame['offset'].to_list() == [timedelta(milliseconds=1)]
    assert events.metadata == _entries(sampling_rate)


@pytest.mark.parametrize(('detector', 'build_kwargs', 'event_samples'), DETECTORS)
def test_event_duration_reaches_one_nominal_interval_past_the_last_sample(
        detector, build_kwargs, event_samples,
):
    # Seven samples spaced 2 ms at 500 Hz.
    timesteps = np.arange(0, 14, 2)
    events = detector(**build_kwargs(timesteps), sampling_rate=500)

    event_timesteps = timesteps[event_samples]
    first, last = int(event_timesteps[0]), int(event_timesteps[-1])
    assert events.frame['onset'].to_list() == [timedelta(milliseconds=first)]
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=last - first + 2)]
    assert events.frame['offset'].to_list() == [timedelta(milliseconds=last)]
    assert events.metadata == _entries(500)


@pytest.mark.parametrize(('detector', 'build_kwargs', 'event_samples'), THRESHOLD_DETECTORS)
def test_minimum_duration_counts_the_nominal_interval(detector, build_kwargs, event_samples):
    timesteps = np.arange(10)
    n_event_samples = len(timesteps[event_samples])
    kwargs = build_kwargs(timesteps)

    passing = detector(**{**kwargs, 'minimum_duration': n_event_samples}, sampling_rate=1000)
    failing = detector(
        **{**kwargs, 'minimum_duration': n_event_samples + 1}, sampling_rate=1000,
    )

    assert passing.frame.height == 1
    assert failing.frame.height == 0


def test_blink_maximum_duration_counts_the_nominal_interval():
    # Three candidate samples.
    kwargs = _blink_kwargs(np.arange(5))

    failing = blink(**{**kwargs, 'maximum_duration': 2}, sampling_rate=1000)
    passing = blink(**{**kwargs, 'maximum_duration': 3}, sampling_rate=1000)

    assert failing.frame.height == 0
    assert passing.frame.height == 1


def test_idt_duplicate_last_timestamp_ends_on_the_nominal_interval():
    timesteps = np.append(np.arange(10), 9)
    kwargs = _idt_kwargs(timesteps)

    events = idt(**{**kwargs, 'minimum_duration': 10}, sampling_rate=1000)

    assert events.frame['onset'].to_list() == [timedelta(0)]
    assert events.frame['duration'].to_list() == [timedelta(milliseconds=10)]


@pytest.mark.parametrize(
    ('detector', 'kwargs'),
    [
        pytest.param(
            ivt, {'velocities': [[10, 10]] * 3, 'velocity_threshold': 1}, id='ivt',
        ),
        pytest.param(
            microsaccades, {'velocities': [[0, 0]] * 3, 'threshold': (1, 1)}, id='microsaccades',
        ),
        pytest.param(blink, {'pupil': [1.0, 1.0, 1.0]}, id='blink'),
        pytest.param(
            fill,
            {'events': Events(onsets=[0], offsets=[3]), 'timesteps': np.arange(3)},
            id='fill',
        ),
        pytest.param(
            out_of_screen, {'pixels': [[5, 5]] * 3, 'x_max': 10, 'y_max': 10}, id='out_of_screen',
        ),
        pytest.param(ihmm, {'velocities': [[np.nan, np.nan]] * 3}, id='ihmm'),
        pytest.param(idt, {'positions': [[0, 0]]}, id='idt'),
    ],
)
def test_empty_result_carries_the_entries(detector, kwargs):
    events = detector(**kwargs, sampling_rate=1000)

    assert events.frame.height == 0
    assert events.metadata == _entries(1000)
