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
"""Test the statistical sample generator."""
import json
import math

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements import Events
from pymovements.synthetic import Range
from pymovements.synthetic import SampleGeneratorRegistry
from pymovements.synthetic import StatisticalSampleGenerator

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)

#: Hyperparameters without any jitter, so a profile can be read off the samples exactly.
WITHOUT_JITTER = {
    'fixation_consistency': 0.0,
    'saccade_consistency': 0.0,
    'smooth_pursuit_consistency': 0.0,
}


def _events(*segments):
    """Return a gap-free sequence from (name, duration in ms) pairs, starting at zero."""
    onsets, offsets = [], []
    elapsed = 0.0
    for _, duration in segments:
        onsets.append(elapsed)
        elapsed += duration
        offsets.append(elapsed)

    return Events(
        name=[name for name, _ in segments],
        onsets=onsets,
        offsets=offsets,
        time_unit='ms',
    )


def _magnitudes(gaze, *, start=-math.inf, end=math.inf):
    """Return the velocity magnitudes of the samples in [start, end) ms."""
    time = pl.col('time').dt.total_microseconds() / 1000.0
    return gaze.samples.filter(time.is_between(start, end, closed='left'))['velocity_magnitude']


def _saccade_peak(duration, *, seed=42, **hyperparameters):
    """Return the peak velocity of one saccade, built and sampled at the same rate."""
    generator = StatisticalSampleGenerator(
        sampling_rate=1000.0, profile_sampling_rate=1000.0, **WITHOUT_JITTER, **hyperparameters,
    )
    gaze = generator.generate(events=_events(('saccade', duration)), seed=seed)
    return _magnitudes(gaze).max()


def _pursuit(*, duration=500.0, **hyperparameters):
    """Return the magnitudes of one smooth pursuit, built and sampled at the same rate."""
    generator = StatisticalSampleGenerator(
        sampling_rate=1000.0, profile_sampling_rate=1000.0, **WITHOUT_JITTER, **hyperparameters,
    )
    return generator.generate(events=_events(('smooth_pursuit', duration)), seed=42)


@pytest.mark.parametrize(
    'hyperparameters',
    [
        pytest.param({}, id='defaults'),
        pytest.param(WITHOUT_JITTER, id='without_any_jitter'),
        pytest.param(
            {'noise_percentage': 10.0, 'noise_velocity': (5.0, 20.0)},
            id='with_measurement_noise',
        ),
        pytest.param({'smooth_pursuit_profile': 'growing'}, id='growing_pursuit'),
        pytest.param({'smooth_pursuit_profile': 'decreasing'}, id='decreasing_pursuit'),
        pytest.param(
            {
                'saccade_peak_velocity': Range(300.0, 500.0, 'normal', std=50.0),
                'smooth_pursuit_velocity': Range(10.0, 30.0, 'normal'),
                'smooth_pursuit_onset_duration': Range(40.0, 120.0, 'normal', std=20.0),
                'saccade_consistency': Range(0.0, 5.0, 'normal', std=1.0),
            },
            id='normal_distributions',
        ),
        pytest.param({'saccade_skewness': (1.99, 1.999)}, id='extreme_skewness'),
        pytest.param({'saccade_skewness': (0.01, 0.02)}, id='almost_symmetric_saccade'),
    ],
)
def test_statistical_returns(contiguous_events, hyperparameters):
    gaze = StatisticalSampleGenerator(
        sampling_rate=1000.0, **hyperparameters,
    ).generate(events=contiguous_events, seed=42)

    assert gaze.samples.columns == ['time', 'velocity_magnitude']
    assert gaze.samples.height == 640
    assert _magnitudes(gaze).min() >= 0.0
    assert gaze.experiment.sampling_rate == 1000.0


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
    """The output has as many samples as the requested rate gives over the events' span."""
    events = _events(('fixation', 1200.0), ('saccade', 50.0), ('fixation', 750.0))

    gaze = StatisticalSampleGenerator(sampling_rate=sampling_rate).generate(events=events, seed=42)

    assert gaze.samples.height == round(sampling_rate * 2.0)


def test_output_is_a_velocity_magnitude(contiguous_events):
    """A magnitude has no x and y, so none of the Gaze component columns is filled."""
    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(
        events=contiguous_events, seed=42,
    )

    for component_column in ('position', 'pixel', 'velocity', 'acceleration'):
        assert component_column not in gaze.samples.columns


def test_attaches_a_copy_of_the_events(contiguous_events):
    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(
        events=contiguous_events, seed=42,
    )

    assert_frame_equal(gaze.events.frame, contiguous_events.frame)


def test_event_shorter_than_one_sample_gets_one_sample():
    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(
        events=_events(('saccade', 0.25)), seed=42,
    )

    assert gaze.samples.height == 1


def test_samples_start_at_the_first_onset():
    events = Events(
        name=['fixation', 'saccade'], onsets=[500, 700], offsets=[700, 740], time_unit='ms',
    )

    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(events=events, seed=42)

    assert gaze.samples['time'].dt.total_milliseconds().to_list()[0] == 500
    assert gaze.samples.height == 240


def test_keeps_sub_millisecond_times():
    """Event times between whole milliseconds, as at 333 or 2000 Hz, are not truncated."""
    events = Events(
        name=['fixation', 'saccade'], onsets=[0.5, 250.25], offsets=[250.25, 300.75],
        time_unit='ms',
    )

    gaze = StatisticalSampleGenerator(sampling_rate=2000.0).generate(events=events, seed=42)

    assert gaze.samples['time'].dt.total_microseconds().to_list()[:2] == [500, 1000]
    assert gaze.samples.height == round(300.25 * 2.0)


def test_orders_unsorted_events(contiguous_events):
    shuffled = Events(contiguous_events.frame.reverse())
    generator = StatisticalSampleGenerator(sampling_rate=1000.0)

    unsorted = generator.generate(events=shuffled, seed=42)
    ordered = generator.generate(events=contiguous_events, seed=42)

    assert_frame_equal(unsorted.samples, ordered.samples)


def test_sample_at_an_onset_belongs_to_that_event():
    """Each sample averages the profile up to the next sample, so the ground truth does not lag.

    The fixation has no jitter, so every sample stamped inside it must be exactly zero, the one at
    its onset included. A window reaching back to the previous sample would mix in the saccade.
    """
    events = _events(('saccade', 50.0), ('fixation', 50.0))

    gaze = StatisticalSampleGenerator(
        sampling_rate=1000.0, **WITHOUT_JITTER,
    ).generate(events=events, seed=42)

    assert (_magnitudes(gaze, start=50.0) == 0.0).all()
    assert _magnitudes(gaze, start=49.0, end=50.0).item() > 0.0


def test_sample_averages_the_profile_until_the_next_sample():
    """Fuhl et al. §3.5: averaging a dense profile flattens the saccade's peak a little."""
    events = _events(('saccade', 40.0))
    hyperparameters = {**WITHOUT_JITTER, 'saccade_peak_velocity': 400.0}

    exact = StatisticalSampleGenerator(
        sampling_rate=1000.0, profile_sampling_rate=1000.0, **hyperparameters,
    ).generate(events=events, seed=42)
    averaged = StatisticalSampleGenerator(
        sampling_rate=1000.0, **hyperparameters,
    ).generate(events=events, seed=42)

    assert _magnitudes(exact).max() == pytest.approx(400.0)
    assert _magnitudes(averaged).max() < 400.0


@pytest.mark.parametrize(
    ('duration', 'hyperparameters', 'expected'),
    [
        pytest.param(
            20.0,
            {'saccade_duration': (20.0, 80.0), 'saccade_peak_velocity': (300.0, 500.0)},
            300.0,
            id='shortest_saccade_is_limited_to_the_lowest_peak',
        ),
        pytest.param(
            50.0,
            {'saccade_peak_velocity': 400.0},
            400.0,
            id='single_peak_velocity_is_reached_at_any_length',
        ),
        pytest.param(
            80.0,
            {
                'saccade_duration': (20.0, 80.0),
                'saccade_peak_velocity': Range(300.0, 500.0, 'normal', std=0.1),
            },
            400.0,
            id='longest_saccade_reaches_the_drawn_peak',
        ),
        pytest.param(
            200.0,
            {
                'saccade_duration': (20.0, 80.0),
                'saccade_peak_velocity': Range(300.0, 500.0, 'normal', std=0.1),
            },
            400.0,
            id='saccade_beyond_the_longest_counts_as_longest',
        ),
        pytest.param(
            50.0,
            {
                'saccade_duration': 50.0,
                'saccade_peak_velocity': Range(300.0, 500.0, 'normal', std=0.1),
            },
            400.0,
            id='single_duration_does_not_limit_the_peak',
        ),
    ],
)
def test_saccade_peak_is_limited_by_its_length(duration, hyperparameters, expected):
    """Fuhl et al. §3.3: the length's and the peak velocity's fraction of their range multiply."""
    assert _saccade_peak(duration, **hyperparameters) == pytest.approx(expected, abs=1.0)


def test_saccade_peak_velocity_stays_within_its_range():
    peaks = [
        _saccade_peak(80.0, seed=seed, saccade_peak_velocity=(300.0, 500.0))
        for seed in range(20)
    ]

    assert min(peaks) >= 300.0
    assert max(peaks) <= 500.0
    assert len(set(peaks)) > 1


@pytest.mark.parametrize(
    ('hyperparameters', 'expected_first', 'expected_last'),
    [
        pytest.param(
            {'smooth_pursuit_profile': 'constant', 'smooth_pursuit_velocity': 20.0},
            20.0,
            20.0,
            id='constant',
        ),
        pytest.param(
            {'smooth_pursuit_profile': 'growing', 'smooth_pursuit_velocity': (10.0, 30.0)},
            10.0,
            30.0,
            id='growing',
        ),
        pytest.param(
            {'smooth_pursuit_profile': 'decreasing', 'smooth_pursuit_velocity': (10.0, 30.0)},
            30.0,
            10.0,
            id='decreasing',
        ),
    ],
)
def test_smooth_pursuit_follows_its_profile(hyperparameters, expected_first, expected_last):
    """Fuhl et al. §3.4: a constant, linearly growing or linearly decreasing pursuit."""
    gaze = _pursuit(smooth_pursuit_onset_duration=0.0, **hyperparameters)

    magnitudes = _magnitudes(gaze)
    assert magnitudes[1] == pytest.approx(expected_first, abs=0.1)
    assert magnitudes[-1] == pytest.approx(expected_last)


def test_smooth_pursuit_velocity_takes_its_distribution():
    gaze = _pursuit(
        smooth_pursuit_onset_duration=0.0,
        smooth_pursuit_velocity=Range(10.0, 30.0, 'normal', std=0.1),
    )

    assert _magnitudes(gaze, start=1.0).to_list() == pytest.approx([20.0] * 499, abs=0.5)


def test_smooth_pursuit_onset_rises_to_the_pursuit_velocity():
    """The onset is a sigmoid that reaches the pursuit velocity at its end and stays there."""
    gaze = _pursuit(smooth_pursuit_velocity=20.0, smooth_pursuit_onset_duration=80.0)

    magnitudes = _magnitudes(gaze)
    assert magnitudes[0] < 1.0
    assert (magnitudes.head(80).diff().drop_nulls() >= 0.0).all()
    assert _magnitudes(gaze, start=80.0).to_list() == pytest.approx([20.0] * 420)


def test_smooth_pursuit_onset_duration_takes_its_distribution():
    """A normal onset duration centred on 80 ms is over well before the uniform maximum."""
    gaze = _pursuit(
        smooth_pursuit_velocity=20.0,
        smooth_pursuit_onset_duration=Range(40.0, 120.0, 'normal', std=1.0),
    )

    assert _magnitudes(gaze, start=70.0, end=71.0).item() < 20.0
    assert _magnitudes(gaze, start=90.0).to_list() == pytest.approx([20.0] * 410)


def test_fixation_without_jitter_is_still():
    gaze = StatisticalSampleGenerator(sampling_rate=1000.0, fixation_consistency=0.0).generate(
        events=_events(('fixation', 1000.0)), seed=42,
    )

    assert (_magnitudes(gaze) == 0.0).all()


def test_fixation_consistency_adds_jitter():
    gaze = StatisticalSampleGenerator(
        sampling_rate=1000.0, fixation_consistency=(1.0, 1.0),
    ).generate(events=_events(('fixation', 1000.0)), seed=42)

    assert (_magnitudes(gaze) > 0.0).any()


def _still_fixation(**noise):
    """Generate a still fixation, so every sample that is not zero is noise."""
    generator = StatisticalSampleGenerator(sampling_rate=1000.0, fixation_consistency=0.0, **noise)
    gaze = generator.generate(events=_events(('fixation', 1000.0)), seed=42)
    return _magnitudes(gaze)


def test_noise_is_off_by_default():
    assert (_still_fixation() == 0.0).all()


@pytest.mark.parametrize(
    'noise_velocity',
    [
        pytest.param(Range(50.0, 60.0), id='uniform'),
        pytest.param(Range(50.0, 60.0, 'normal', std=5.0), id='normal'),
    ],
)
def test_noise_changes_the_requested_share_of_samples_within_its_range(noise_velocity):
    """Fuhl et al. §3.6: a percentage of the samples, each changed within a minimum and maximum."""
    magnitudes = _still_fixation(noise_percentage=10.0, noise_velocity=noise_velocity)

    noisy = magnitudes.filter(magnitudes > 0.0)
    assert noisy.len() == 100
    assert noisy.min() >= 50.0
    assert noisy.max() <= 60.0


def test_reproducible_under_seed(contiguous_events):
    generator = StatisticalSampleGenerator(sampling_rate=1000.0)

    first = generator.generate(events=contiguous_events, seed=42)
    second = generator.generate(events=contiguous_events, seed=42)
    other = generator.generate(events=contiguous_events, seed=43)

    assert_frame_equal(first.samples, second.samples)
    assert not first.samples.equals(other.samples)


@pytest.mark.parametrize(
    ('trials', 'trial_columns'),
    [
        pytest.param({'trial': [1, 1, 2]}, ['trial'], id='one_trial_column'),
        pytest.param(
            {'subject': ['a', 'a', 'b'], 'trial': [1, 1, 1]},
            ['subject', 'trial'],
            id='two_trial_columns',
        ),
    ],
)
def test_generates_each_trial_separately(trials, trial_columns):
    """Each trial restarts at zero, so each is its own continuous signal."""
    events = Events(
        pl.DataFrame({
            **trials,
            'name': ['fixation', 'saccade', 'fixation'],
            'onset': [0, 250, 0],
            'offset': [250, 300, 200],
        }),
        trial_columns=trial_columns,
    )

    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(events=events, seed=42)

    assert gaze.trial_columns == trial_columns
    first, second = gaze.samples.partition_by(trial_columns, maintain_order=True)
    assert first.height == 300
    assert second.height == 200
    assert second['time'].dt.total_milliseconds().to_list()[0] == 0


def test_does_not_fill_the_time_between_trials():
    events = Events(
        name=['fixation'] * 2, onsets=[0, 5000], offsets=[250, 5300], trials=[1, 2],
        time_unit='ms',
    )

    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(events=events, seed=42)

    assert gaze.samples.height == 550


def test_generates_from_bridged_fixations(fixations):
    """A detected fixation sequence has gaps; bridging them first makes it renderable."""
    bridged = pm.synthetic.bridge_gaps(fixations)

    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(events=bridged, seed=42)

    assert gaze.samples.height == 560
    assert gaze.events.frame['name'].to_list() == ['fixation', 'saccade', 'fixation']


def test_is_registered_by_name(contiguous_events):
    assert SampleGeneratorRegistry.get('statistical') is StatisticalSampleGenerator

    gaze = pm.synthetic.generate_samples(
        'statistical', events=contiguous_events, sampling_rate=1000.0, seed=42,
        saccade_peak_velocity=(300.0, 450.0),
    )
    explicit = StatisticalSampleGenerator(
        sampling_rate=1000.0, saccade_peak_velocity=(300.0, 450.0),
    ).generate(events=contiguous_events, seed=42)

    assert_frame_equal(gaze.samples, explicit.samples)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param((300.0, 450.0), Range(300.0, 450.0), id='pair'),
        pytest.param(400.0, Range(400.0, 400.0), id='single_value'),
        pytest.param(
            Range(300.0, 450.0, 'normal', std=20.0),
            Range(300.0, 450.0, 'normal', std=20.0),
            id='range',
        ),
    ],
)
def test_reads_a_range_from_each_accepted_form(value, expected):
    generator = StatisticalSampleGenerator(sampling_rate=1000.0, saccade_peak_velocity=value)

    assert generator.saccade_peak_velocity == expected


def test_records_its_hyperparameters(contiguous_events):
    gaze = StatisticalSampleGenerator(
        sampling_rate=1000.0, saccade_peak_velocity=(300.0, 450.0),
    ).generate(events=contiguous_events, seed=42)

    provenance = gaze.metadata['synthetic']
    assert provenance['generator'] == 'statistical'
    assert provenance['hyperparameters']['saccade_peak_velocity'] == {
        'minimum': 300.0, 'maximum': 450.0, 'distribution': 'uniform', 'std': None,
    }
    assert provenance['hyperparameters']['profile_sampling_rate'] == 10_000.0


def test_output_can_be_saved_and_written_to_json(contiguous_events, tmp_path):
    """The provenance holds the ranges as plain values, so it comes back equal."""
    gaze = StatisticalSampleGenerator(sampling_rate=1000.0).generate(
        events=contiguous_events, seed=42,
    )

    assert json.loads(json.dumps(gaze.metadata)) == gaze.metadata

    gaze.save(tmp_path, verbose=0)

    with open(tmp_path / 'metadata.yaml', encoding='utf-8') as file:
        assert yaml.safe_load(file) == gaze.metadata


def test_run_is_replayed_from_its_record(contiguous_events):
    gaze = StatisticalSampleGenerator(
        sampling_rate=1000.0, saccade_peak_velocity=Range(300.0, 450.0, 'normal', std=30.0),
    ).generate(events=contiguous_events, seed=42)
    record = gaze.metadata['synthetic']

    replayed = pm.synthetic.generate_samples(
        record['generator'],
        events=contiguous_events,
        seed=record['seed'],
        **record['hyperparameters'],
    )

    assert_frame_equal(replayed.samples, gaze.samples)


@pytest.mark.parametrize(
    ('hyperparameters', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'smooth_pursuit_profile': 'exponential'},
            ValueError,
            ('smooth_pursuit_profile', 'constant', 'growing', 'decreasing'),
            id='unknown_pursuit_profile',
        ),
        pytest.param(
            {'saccade_skewness': (0.0, 1.0)},
            ValueError,
            ('saccade_skewness', 'greater than zero'),
            id='non_positive_skewness',
        ),
        pytest.param(
            {'saccade_skewness': (0.5, 2.0)},
            ValueError,
            ('saccade_skewness', 'below two', 'onset'),
            id='skewness_without_a_rising_onset',
        ),
        pytest.param(
            {'saccade_peak_velocity': (-10.0, 500.0)},
            ValueError,
            ('saccade_peak_velocity', 'negative'),
            id='negative_peak_velocity',
        ),
        pytest.param(
            {'saccade_duration': (-1.0, 80.0)},
            ValueError,
            ('saccade_duration', 'negative'),
            id='negative_saccade_duration',
        ),
        pytest.param(
            {'smooth_pursuit_velocity': (-5.0, 30.0)},
            ValueError,
            ('smooth_pursuit_velocity', 'negative'),
            id='negative_pursuit_velocity',
        ),
        pytest.param(
            {'smooth_pursuit_onset_duration': (-5.0, 30.0)},
            ValueError,
            ('smooth_pursuit_onset_duration', 'negative'),
            id='negative_onset_duration',
        ),
        pytest.param(
            {'fixation_consistency': (-1.0, 1.0)},
            ValueError,
            ('fixation_consistency', 'negative'),
            id='negative_consistency',
        ),
        pytest.param(
            {'noise_percentage': -1.0},
            ValueError,
            ('noise_percentage', '0', '100'),
            id='negative_noise_percentage',
        ),
        pytest.param(
            {'noise_percentage': 100.5},
            ValueError,
            ('noise_percentage', '0', '100'),
            id='noise_percentage_above_100',
        ),
        pytest.param(
            {'profile_sampling_rate': 0.0},
            ValueError,
            ('profile_sampling_rate', 'greater than zero'),
            id='non_positive_profile_sampling_rate',
        ),
        pytest.param(
            {'saccade_peak_velocity': 'fast'},
            TypeError,
            ('range', 'str', 'fast'),
            id='range_of_wrong_type',
        ),
        pytest.param(
            {'noise_percentage': '10'},
            TypeError,
            ('noise_percentage', 'number', 'str'),
            id='noise_percentage_of_wrong_type',
        ),
        pytest.param(
            {'profile_sampling_rate': 500.0},
            ValueError,
            ('profile_sampling_rate', '500.0', 'sampling_rate', '1000.0'),
            id='profile_rate_below_sampling_rate',
        ),
        pytest.param(
            {'profile_sampling_rate': True},
            TypeError,
            ('profile_sampling_rate', 'number', 'bool'),
            id='profile_sampling_rate_of_wrong_type',
        ),
        pytest.param(
            {'saccade_duration': Range(20.0, 80.0, 'normal')},
            ValueError,
            ('saccade_duration', 'bounds', "'normal'", 'no effect'),
            id='saccade_duration_with_a_distribution',
        ),
        pytest.param(
            {
                'smooth_pursuit_profile': 'growing',
                'smooth_pursuit_velocity': Range(10.0, 30.0, 'normal', std=2.0),
            },
            ValueError,
            ('smooth_pursuit_velocity', "'growing'", "'normal'", 'no effect'),
            id='distribution_of_a_growing_pursuit',
        ),
        pytest.param(
            {
                'smooth_pursuit_profile': 'decreasing',
                'smooth_pursuit_velocity': Range(10.0, 30.0, 'normal'),
            },
            ValueError,
            ('smooth_pursuit_velocity', "'decreasing'", "'normal'", 'no effect'),
            id='distribution_of_a_decreasing_pursuit',
        ),
        pytest.param(
            {'noise_velocity': (5.0, 20.0)},
            ValueError,
            ('noise_velocity', 'noise_percentage', 'no effect'),
            id='noise_velocity_without_noise_percentage',
        ),
        pytest.param(
            {'noise_percentage': 10.0},
            ValueError,
            ('noise_percentage', 'noise_velocity', 'no effect'),
            id='noise_percentage_without_noise_velocity',
        ),
    ],
)
def test_statistical_raises_error(hyperparameters, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        StatisticalSampleGenerator(sampling_rate=1000.0, **hyperparameters)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('events', 'hyperparameters', 'msg_substrings'),
    [
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560], time_unit='ms'),
            {},
            ('statistical', 'gap', '250.0', '300.0', 'bridge_gaps'),
            id='gap_between_events',
        ),
        pytest.param(
            Events(
                name=['fixation', 'saccade'], onsets=[0.0, 250.75], offsets=[250.5, 300.0],
                time_unit='ms',
            ),
            {},
            ('statistical', 'gap', '250.5 ms', '250.75 ms', 'saccade'),
            id='sub_millisecond_gap',
        ),
        pytest.param(
            Events(name=['fixation'] * 2, onsets=[0, 200], offsets=[250, 500], time_unit='ms'),
            {},
            ('statistical', 'overlap'),
            id='overlapping_events',
        ),
        pytest.param(
            _events(('fixation', 200.0), ('blink', 100.0)),
            {},
            ('statistical', 'blink', 'fixation', 'saccade', 'smooth_pursuit'),
            id='event_type_without_a_profile',
        ),
        pytest.param(
            Events(
                pl.DataFrame({'name': ['fixation'] * 2, 'onset': [0, 300], 'duration': [250, 260]}),
            ),
            {},
            ('gap', '250.0 ms', '300.0 ms', 'bridge_gaps'),
            id='gap_is_read_from_the_duration',
        ),
        pytest.param(
            Events(name=[], onsets=[], offsets=[], time_unit='ms'),
            {},
            ('statistical', 'empty'),
            id='empty_events',
        ),
    ],
)
def test_statistical_raises_error_on_generate(events, hyperparameters, msg_substrings):
    generator = StatisticalSampleGenerator(sampling_rate=1000.0, **hyperparameters)

    with pytest.raises(ValueError) as excinfo:
        generator.generate(events=events, seed=42)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('run_fixtures', 'msg_substrings'),
    [
        pytest.param({}, ('missing', "'events'"), id='without_events'),
        pytest.param(
            {'events': 'contiguous_events', 'gaze': 'contiguous_events'},
            ('unknown', "'gaze'", "'events'"),
            id='gaze_is_no_run_parameter',
        ),
    ],
)
def test_statistical_raises_type_error_on_run_parameters(request, run_fixtures, msg_substrings):
    """The generator declares ``events`` as its only run parameter."""
    run = {key: request.getfixturevalue(fixture) for key, fixture in run_fixtures.items()}

    with pytest.raises(TypeError) as excinfo:
        StatisticalSampleGenerator(sampling_rate=1000.0).generate(seed=42, **run)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_statistical_raises_type_error_on_events_of_wrong_type():
    with pytest.raises(TypeError) as excinfo:
        StatisticalSampleGenerator(sampling_rate=1000.0).generate(events=None, seed=42)

    msg, = excinfo.value.args
    for msg_substring in ('statistical', 'events', 'Events', 'NoneType'):
        assert msg_substring in msg
