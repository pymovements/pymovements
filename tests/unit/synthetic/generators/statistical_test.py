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
"""Test the statistical gaze generator."""
import dataclasses

import polars as pl
import pytest

from pymovements import Events
from pymovements.synthetic import ConditioningMode
from pymovements.synthetic import GeneratorConfig
from pymovements.synthetic import InputContractError
from pymovements.synthetic import SignalRepresentation
from pymovements.synthetic.generators.statistical import StatisticalConfig
from pymovements.synthetic.generators.statistical import StatisticalGenerator
from pymovements.synthetic.generators.statistical import VELOCITY_MAGNITUDE_COLUMN

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


def _fixations():
    """Return a short fixation sequence to condition on."""
    return Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560], time_unit='ms')


def _mixed_events():
    """Return a sequence covering all three event types the generator produces."""
    return Events(
        name=['fixation', 'saccade', 'smooth_pursuit'],
        onsets=[0, 200, 240],
        offsets=[200, 240, 640],
        time_unit='ms',
    )


@pytest.mark.parametrize(
    ('config', 'expected_samples'),
    [
        pytest.param(
            StatisticalConfig(sampling_rate=1000.0, duration_s=1.0, seed=42),
            1000,
            id='one_second_at_1000hz',
        ),
        pytest.param(
            StatisticalConfig(sampling_rate=500.0, duration_s=2.0, seed=42),
            1000,
            id='two_seconds_at_500hz',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0, duration_s=0.5, seed=7, noise_std_dps=2.0,
            ),
            500,
            id='with_measurement_noise',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
                seed=3,
                event_sequence=('fixation', 'saccade'),
            ),
            1000,
            id='with_prescribed_event_sequence',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
                seed=3,
                smooth_pursuit_profile='growing',
                event_sequence=('smooth_pursuit',),
            ),
            1000,
            id='with_growing_pursuit',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
                seed=3,
                smooth_pursuit_profile='decreasing',
                fixation_duration_distribution='normal',
                event_sequence=('smooth_pursuit', 'fixation'),
            ),
            1000,
            id='with_decreasing_pursuit_and_normal_durations',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
                seed=5,
                fixation_consistency_min_dps=0.0,
                fixation_consistency_max_dps=0.0,
                saccade_consistency_min_dps=0.0,
                saccade_consistency_max_dps=0.0,
                smooth_pursuit_consistency_min_dps=0.0,
                smooth_pursuit_consistency_max_dps=0.0,
            ),
            1000,
            id='without_any_jitter',
        ),
        pytest.param(
            StatisticalConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
                seed=5,
                saccade_skewness_min=1.99,
                saccade_skewness_max=1.999,
                saccade_duration_min_ms=0.05,
                saccade_duration_max_ms=0.05,
                event_sequence=('saccade',),
            ),
            1000,
            id='with_extreme_skewness_and_single_sample_saccades',
        ),
    ],
)
def test_statistical_returns(config, expected_samples):
    gaze = StatisticalGenerator().generate(config)

    assert gaze.samples.height == expected_samples
    assert VELOCITY_MAGNITUDE_COLUMN in gaze.samples.columns
    assert gaze.samples[VELOCITY_MAGNITUDE_COLUMN].min() >= 0.0
    assert len(gaze.events) > 0


@pytest.mark.parametrize(
    'sampling_rate',
    [
        pytest.param(250.0, id='250hz'),
        pytest.param(500.0, id='500hz'),
        pytest.param(1000.0, id='1000hz'),
        pytest.param(2000.0, id='2000hz'),
        pytest.param(333.0, id='333hz_not_a_divisor'),
    ],
)
def test_arbitrary_sampling_rate(sampling_rate):
    """A generator declaring arbitrary rates must match the rate it was asked for."""
    config = StatisticalConfig(sampling_rate=sampling_rate, duration_s=2.0, seed=42)

    gaze = StatisticalGenerator().generate(config)

    assert gaze.samples.height == round(sampling_rate * 2.0)


def test_reproducible_under_seed():
    config = StatisticalConfig(sampling_rate=1000.0, duration_s=1.0, seed=42)

    first = StatisticalGenerator().generate(config)
    second = StatisticalGenerator().generate(config)
    other = StatisticalGenerator().generate(dataclasses.replace(config, seed=43))

    assert first.samples.equals(second.samples)
    assert not first.samples.equals(other.samples)


def test_output_representation_is_velocity_magnitude():
    """The output is a directionless magnitude, and the metadata records it as one."""
    config = StatisticalConfig(sampling_rate=1000.0, duration_s=1.0, seed=42)

    gaze = StatisticalGenerator().generate(config)

    assert gaze.metadata['synthetic']['signal_representations'] == [
        SignalRepresentation.VELOCITY_MAGNITUDE.value,
    ]
    for component_column in ('position', 'pixel', 'velocity', 'acceleration'):
        assert component_column not in gaze.samples.columns


def test_event_conditioned_generation_follows_supplied_events():
    config = StatisticalConfig(sampling_rate=1000.0, seed=42)

    gaze = StatisticalGenerator().generate(config, events=_mixed_events())

    assert gaze.events.frame['name'].to_list() == ['fixation', 'saccade', 'smooth_pursuit']
    assert gaze.samples.height == 640


def test_event_conditioned_generation_fills_gaps_with_saccades():
    """A fixation sequence has gaps where its saccades were; the output must not drop them."""
    config = StatisticalConfig(sampling_rate=1000.0, seed=42)

    gaze = StatisticalGenerator().generate(config, events=_fixations())

    assert gaze.events.frame['name'].to_list() == ['fixation', 'saccade', 'fixation']
    assert gaze.samples.height == 560


def test_event_conditioned_generation_keeps_the_input_timeline():
    """The fixations come back at the onsets they were supplied at, not shifted by the gaps."""
    events = _fixations()
    config = StatisticalConfig(sampling_rate=1000.0, seed=42)

    gaze = StatisticalGenerator().generate(config, events=events)

    fixations = gaze.events.frame.filter(pl.col('name') == 'fixation')
    assert fixations['onset'].to_list() == events.frame['onset'].to_list()
    assert fixations['offset'].to_list() == events.frame['offset'].to_list()


def test_event_conditioned_generation_starts_at_the_first_onset():
    events = Events(name=['fixation'], onsets=[500], offsets=[700], time_unit='ms')
    config = StatisticalConfig(sampling_rate=1000.0, seed=42)

    gaze = StatisticalGenerator().generate(config, events=events)

    assert gaze.samples['time'].dt.total_milliseconds().to_list()[0] == 500
    assert gaze.samples.height == 200


def test_declared_capabilities_match_the_method():
    capabilities = StatisticalGenerator.capabilities

    assert capabilities.conditioning_modes == frozenset({
        ConditioningMode.UNCONDITIONAL, ConditioningMode.EVENTS,
    })
    assert capabilities.signal_representations == frozenset({
        SignalRepresentation.VELOCITY_MAGNITUDE,
    })
    assert capabilities.sampling_rates is None
    assert not capabilities.requires_artifacts


def test_plain_generator_config_is_rejected():
    """The generator's own parameters are not optional, so a base configuration is not enough."""
    with pytest.raises(ValueError) as excinfo:
        StatisticalGenerator().generate(
            GeneratorConfig(sampling_rate=1000.0, duration_s=1.0, seed=42),
        )

    msg, = excinfo.value.args
    assert 'StatisticalConfig' in msg


def test_config_round_trips_through_dict():
    config = StatisticalConfig(sampling_rate=1000.0, duration_s=1.0, seed=42, noise_std_dps=1.5)

    restored = StatisticalConfig.from_dict(config.to_dict())

    assert restored == config


@pytest.mark.parametrize(
    ('kwargs', 'msg_substrings'),
    [
        pytest.param(
            {'fixation_duration_distribution': 'gaussian'},
            ('fixation_duration_distribution', 'normal', 'uniform'),
            id='unknown_duration_distribution',
        ),
        pytest.param(
            {'saccade_consistency_distribution': 'poisson'},
            ('saccade_consistency_distribution', 'uniform'),
            id='unknown_consistency_distribution',
        ),
        pytest.param(
            {'smooth_pursuit_profile': 'exponential'},
            ('smooth_pursuit_profile', 'constant', 'growing', 'decreasing'),
            id='unknown_pursuit_profile',
        ),
        pytest.param(
            {'fixation_duration_std_ms': -1.0},
            ('fixation_duration_std_ms', 'negative'),
            id='negative_standard_deviation',
        ),
        pytest.param(
            {'noise_std_dps': -0.5},
            ('noise_std_dps', 'negative'),
            id='negative_noise',
        ),
        pytest.param(
            {'fixation_duration_min_ms': 500.0, 'fixation_duration_max_ms': 100.0},
            ('fixation_duration_min_ms', 'fixation_duration_max_ms'),
            id='inverted_duration_range',
        ),
        pytest.param(
            {'saccade_peak_velocity_min_dps': 900.0},
            ('saccade_peak_velocity_min_dps', 'saccade_peak_velocity_max_dps'),
            id='inverted_velocity_range',
        ),
        pytest.param(
            {'saccade_skewness_min': 0.0},
            ('saccade_skewness_min', 'greater than zero'),
            id='non_positive_skewness',
        ),
        pytest.param(
            {'saccade_skewness_max': 2.0},
            ('saccade_skewness_max', 'below two', 'onset'),
            id='skewness_without_interior_peak',
        ),
        pytest.param(
            {'profile_sampling_rate': 100.0},
            ('profile_sampling_rate', 'sampling_rate'),
            id='profile_rate_below_target_rate',
        ),
        pytest.param(
            {'event_sequence': ()},
            ('event_sequence', 'empty'),
            id='empty_event_sequence',
        ),
        pytest.param(
            {'event_sequence': ('blink',)},
            ('event_sequence', 'blink', 'fixation'),
            id='unproducible_event_type',
        ),
    ],
)
def test_statistical_raises_error(kwargs, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        StatisticalConfig(sampling_rate=1000.0, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('config', 'msg_substrings'),
    [
        pytest.param(
            StatisticalConfig(sampling_rate=1000.0),
            ('statistical', 'duration_s'),
            id='unconditional_without_duration',
        ),
        pytest.param(
            None,
            ('statistical', 'StatisticalConfig'),
            id='no_configuration_at_all',
        ),
    ],
)
def test_statistical_raises_error_on_generate(config, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        StatisticalGenerator().generate(config)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_duration_alongside_events_is_the_callers_error():
    config = StatisticalConfig(sampling_rate=1000.0, duration_s=2.0, seed=42)

    with pytest.raises(InputContractError) as excinfo:
        StatisticalGenerator().generate(config, events=_fixations())

    msg, = excinfo.value.args
    assert 'duration_s' in msg
