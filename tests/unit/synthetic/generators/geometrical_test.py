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
"""Test the geometrical sample generator."""
import json
import math

import numpy as np
import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements import Events
from pymovements.synthetic import GeometricalSampleGenerator
from pymovements.synthetic import Range
from pymovements.synthetic import SampleGeneratorRegistry

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)

#: The locations of the three fixations ``_fixations`` returns by default, in degrees.
LOCATIONS = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0))

#: A location that shows up in the output if an event of zero duration is rendered after all.
IGNORED = (99.0, 99.0)


def _fixations(
        onsets=(0, 300, 620),
        offsets=(250, 560, 900),
        locations=LOCATIONS,
        names=None,
        **trials,
):
    """Return events with locations, by default three fixations with two gaps between them.

    A location of ``None`` leaves it missing, as ``bridge_gaps`` does for its saccades.
    """
    return Events(
        pl.DataFrame(
            {
                **trials,
                'name': names or ['fixation'] * len(onsets),
                'onset': list(onsets),
                'offset': list(offsets),
                'location': list(locations),
            },
            schema_overrides={'location': pl.List(pl.Float64)},
        ),
        trial_columns=list(trials) or None,
    )


def _generate(events=None, *, sampling_rate=1000.0, seed=42, **hyperparameters):
    """Return the output of the generator for the events, by default the three fixations."""
    generator = GeometricalSampleGenerator(sampling_rate=sampling_rate, **hyperparameters)
    return generator.generate(events=_fixations() if events is None else events, seed=seed)


def _positions(gaze, column='position'):
    """Return the generated positions as an array of shape (n_samples, 2)."""
    return np.array(gaze.samples[column].to_list())


def _times(gaze):
    """Return the sample times in fractional milliseconds."""
    return gaze.samples['time'].dt.total_microseconds().to_numpy() / 1000.0


@pytest.mark.parametrize(
    ('hyperparameters', 'column'),
    [
        pytest.param({}, 'position', id='defaults'),
        pytest.param({'interpolation': 'linear'}, 'position', id='linear'),
        pytest.param({'location_column_group': 'pixel'}, 'pixel', id='pixel'),
        pytest.param(
            {'fixation_deviation': Range(0.1, 0.3, 'normal', std=0.05)},
            'position',
            id='normal_deviation',
        ),
        pytest.param({'fixation_deviation': np.float64(0.2)}, 'position', id='numpy_deviation'),
    ],
)
def test_geometrical_returns(hyperparameters, column):
    gaze = _generate(**hyperparameters)

    assert gaze.samples.columns == ['time', column]
    assert gaze.samples.height == 900
    assert gaze.samples[column].null_count() == 0
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
    gaze = _generate(sampling_rate=sampling_rate)

    assert gaze.samples.height == round(sampling_rate * 0.9)


def test_default_interpolation_is_a_cubic_spline():
    assert GeometricalSampleGenerator(sampling_rate=1000.0).interpolation == 'cubic_spline'


@pytest.mark.parametrize(
    'interpolation',
    [
        pytest.param('linear', id='linear'),
        pytest.param('cubic_spline', id='cubic_spline'),
    ],
)
def test_fixation_samples_hold_their_location(interpolation):
    """Fuhl & Kasneci §3.7: the fixation centres are the positions the gaze moves between."""
    gaze = _generate(interpolation=interpolation)

    positions = _positions(gaze)
    times = _times(gaze)
    for onset, offset, location in zip((0, 300, 620), (250, 560, 900), LOCATIONS):
        inside = (times >= onset) & (times < offset)
        np.testing.assert_array_equal(positions[inside], np.tile(location, (inside.sum(), 1)))


def test_linear_gap_is_a_straight_line_at_constant_velocity():
    """Gibaldi & Sabatini: the straight motion from one fixation to the next."""
    gaze = _generate(interpolation='linear')

    # From the last sample of the first fixation to the first sample of the second.
    first_gap = _positions(gaze)[249:301]
    np.testing.assert_allclose(np.diff(first_gap[:, 0]), 10.0 / 51)
    np.testing.assert_array_equal(first_gap[:, 1], 0.0)


def test_cubic_gap_connects_the_two_locations():
    """A specific case only, not a general claim that a cubic spline never overshoots."""
    gaze = _generate()

    first_gap = _positions(gaze)[250:300, 0]
    assert first_gap.min() > 0.0
    assert first_gap.max() < 10.0
    assert (np.diff(first_gap) > 0.0).all()


def test_cubic_gap_peaks_at_its_middle():
    """The disclosed limitation: the velocity of a cubic gap peaks around its middle."""
    gaze = _generate()

    # The 51 steps from the last sample of the first fixation to the first of the second.
    speeds = np.abs(np.diff(_positions(gaze)[249:301, 0]))
    assert abs(int(np.argmax(speeds)) - 25) <= 1


def test_cubic_gap_can_overshoot_around_a_short_fixation():
    """The disclosed limitation: the spline is fitted over the trial, not per gap."""
    events = _fixations(
        onsets=(0, 300, 600),
        offsets=(100, 310, 700),
        locations=((0.0, 0.0), (10.0, 0.0), (10.0, 0.0)),
    )

    cubic = _positions(_generate(events, sampling_rate=100.0))
    linear = _positions(_generate(events, sampling_rate=100.0, interpolation='linear'))

    assert cubic[:, 0].max() == pytest.approx(11.86, abs=0.01)
    assert linear[:, 0].max() == 10.0


def test_attaches_a_copy_of_the_events():
    """The gaps are filled in the samples only; no saccade is added to the events."""
    events = _fixations()

    gaze = _generate(events)

    assert_frame_equal(gaze.events.frame, events.frame)
    assert gaze.events is not events


@pytest.mark.parametrize(
    ('onsets', 'offsets', 'locations', 'names', 'expected_times'),
    [
        pytest.param(
            (0, 100, 300), (0, 200, 500), (IGNORED, (0.0, 0.0), (10.0, 0.0)), None, (100.0, 499.0),
            id='fixation_first',
        ),
        pytest.param(
            (0, 250, 300), (200, 250, 500), ((0.0, 0.0), IGNORED, (10.0, 0.0)), None, (0.0, 499.0),
            id='fixation_in_the_middle',
        ),
        pytest.param(
            (0, 300, 700), (200, 500, 700), ((0.0, 0.0), (10.0, 0.0), IGNORED), None, (0.0, 499.0),
            id='fixation_last',
        ),
        pytest.param(
            (0, 200, 200), (200, 200, 400), ((0.0, 0.0), None, (10.0, 0.0)),
            ['fixation', 'saccade', 'fixation'], (0.0, 399.0),
            id='saccade_in_the_middle',
        ),
        pytest.param(
            (0, 200, 200), (200, 400, 200), ((0.0, 0.0), (10.0, 0.0), None),
            ['fixation', 'fixation', 'saccade'], (0.0, 399.0),
            id='saccade_sorted_after_the_fixation_at_its_onset',
        ),
        pytest.param(
            (0, 200, 300), (200, 200, 300), ((0.0, 0.0), None, None),
            ['fixation', 'saccade', 'saccade'], (0.0, 199.0),
            id='saccade_last',
        ),
    ],
)
def test_event_of_zero_duration_gets_no_samples(onsets, offsets, locations, names, expected_times):
    """An impulse neither moves the gaze nor stretches the signal, and stays in the ground truth."""
    events = _fixations(onsets=onsets, offsets=offsets, locations=locations, names=names)

    gaze = _generate(events)

    assert _times(gaze)[[0, -1]].tolist() == list(expected_times)
    assert _positions(gaze)[:, 1].max() == 0.0
    assert_frame_equal(gaze.events.frame, events.frame)


def test_trial_of_only_zero_durations_gets_no_samples():
    gaze = _generate(
        _fixations(onsets=(0, 0), offsets=(200, 0), locations=LOCATIONS[:2], trial=[1, 2]),
    )

    assert gaze.samples['trial'].unique().to_list() == [1]
    assert gaze.samples.height == 200


def test_reads_the_end_of_a_fixation_from_its_duration():
    """PMEP 1: an event ends at onset + duration, so it needs no offset."""
    events = Events(
        pl.DataFrame(
            {
                'name': ['fixation'] * 3,
                'onset': [0, 300, 620],
                'duration': [250, 260, 280],
                'location': [list(location) for location in LOCATIONS],
            },
            schema_overrides={'location': pl.List(pl.Float64)},
        ),
    )

    assert_frame_equal(_generate(events).samples, _generate().samples)


def test_interpolates_the_saccades_of_bridged_fixations():
    """Bridged saccades are interpolated like gaps, so the samples do not change."""
    events = _fixations()
    bridged = pm.synthetic.bridge_gaps(events)

    from_bridged = _generate(bridged)

    assert_frame_equal(from_bridged.samples, _generate(events).samples)
    assert from_bridged.events.frame['name'].to_list() == [
        'fixation', 'saccade', 'fixation', 'saccade', 'fixation',
    ]


def test_ignores_the_location_of_a_saccade():
    """A saccade moves between fixations, so a location it carries is not a position to hold."""
    located = _fixations(
        onsets=(0, 250, 300),
        offsets=(250, 300, 560),
        locations=((0.0, 0.0), (50.0, 50.0), (10.0, 0.0)),
        names=['fixation', 'saccade', 'fixation'],
    )
    unlocated = _fixations(
        onsets=(0, 300), offsets=(250, 560), locations=((0.0, 0.0), (10.0, 0.0)),
    )

    assert_frame_equal(_generate(located).samples, _generate(unlocated).samples)


@pytest.mark.parametrize(
    'interpolation',
    [
        pytest.param('linear', id='linear'),
        pytest.param('cubic_spline', id='cubic_spline'),
    ],
)
def test_holds_the_nearest_location_at_the_trial_ends(interpolation):
    """Before the first and after the last fixation there is nothing to interpolate towards."""
    events = _fixations(
        onsets=(0, 50, 300),
        offsets=(50, 300, 350),
        locations=(None, (1.0, 2.0), None),
        names=['saccade', 'fixation', 'saccade'],
    )

    gaze = _generate(events, interpolation=interpolation)

    np.testing.assert_array_equal(_positions(gaze), np.tile((1.0, 2.0), (350, 1)))


def test_fixation_shorter_than_one_sample_gets_one_sample():
    events = _fixations(onsets=(0.0,), offsets=(0.25,), locations=((1.0, 2.0),))

    gaze = _generate(events)

    assert gaze.samples['position'].to_list() == [[1.0, 2.0]]


def test_fixation_between_samples_keeps_its_location_if_one_sample_falls_inside():
    events = _fixations(onsets=(0, 195, 300), offsets=(100, 205, 400), locations=LOCATIONS)

    gaze = _generate(events, sampling_rate=10.0, interpolation='linear')

    assert _positions(gaze)[2].tolist() == [10.0, 0.0]


@pytest.mark.parametrize(
    ('sampling_rate', 'onset'),
    [
        pytest.param(12.0, 16250, id='12hz'),
        pytest.param(19.0, 1000, id='19hz'),
        pytest.param(76.0, 250, id='76hz'),
    ],
)
def test_sample_on_an_onset_belongs_to_that_fixation(sampling_rate, onset):
    """These rates round the time of the sample just below the onset it falls on."""
    events = _fixations(
        onsets=(0, onset), offsets=(onset, onset + 500), locations=((0.0, 0.0), (10.0, 0.0)),
    )

    positions = _positions(_generate(events, sampling_rate=sampling_rate, interpolation='linear'))

    index = round(onset * sampling_rate / 1000.0)
    assert positions[index - 1].tolist() == [0.0, 0.0]
    assert positions[index].tolist() == [10.0, 0.0]


@pytest.mark.parametrize(
    ('sampling_rate', 'end'),
    [
        pytest.param(12.0, 16250, id='12hz'),
        pytest.param(19.0, 1000, id='19hz'),
        pytest.param(76.0, 250, id='76hz'),
    ],
)
def test_sample_on_an_end_is_outside_that_fixation(sampling_rate, end):
    """An event spans ``[onset, onset + duration)``, so the sample moves into the gap."""
    events = _fixations(
        onsets=(0, end + 500),
        offsets=(end, end + 1000),
        locations=((0.0, 0.0), (10.0, 0.0)),
    )

    positions = _positions(_generate(events, sampling_rate=sampling_rate, interpolation='linear'))

    index = round(end * sampling_rate / 1000.0)
    assert positions[index - 1].tolist() == [0.0, 0.0]
    assert positions[index][0] > 0.0


def test_samples_start_at_the_first_onset():
    events = _fixations(onsets=(500,), offsets=(740,), locations=((1.0, 2.0),))

    gaze = _generate(events)

    assert _times(gaze)[0] == 500.0
    assert gaze.samples.height == 240


def test_keeps_sub_millisecond_times():
    """Event times between whole milliseconds, as at 333 or 2000 Hz, are not truncated."""
    events = _fixations(
        onsets=(0.5, 150.75),
        offsets=(100.25, 300.75),
        locations=((0.0, 0.0), (5.0, 5.0)),
    )

    gaze = _generate(events, sampling_rate=2000.0, interpolation='linear')

    positions = dict(zip(_times(gaze), _positions(gaze)[:, 0]))
    assert _times(gaze)[0] == 0.5
    assert gaze.samples.height == round(300.25 * 2.0)
    assert positions[100.0] == 0.0
    assert 0.0 < positions[100.5] < 5.0
    assert 0.0 < positions[150.5] < 5.0
    assert positions[151.0] == 5.0


def test_orders_unsorted_events():
    events = _fixations()
    shuffled = Events(events.frame.reverse())

    assert_frame_equal(_generate(shuffled).samples, _generate(events).samples)


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
    """Each trial restarts at zero, so no gap between two trials is interpolated."""
    events = _fixations(
        onsets=(0, 300, 0),
        offsets=(250, 560, 200),
        locations=((0.0, 0.0), (10.0, 0.0), (5.0, 5.0)),
        **trials,
    )

    gaze = _generate(events)

    assert gaze.trial_columns == trial_columns
    first, second = gaze.samples.partition_by(trial_columns, maintain_order=True)
    assert first.height == 560
    assert second.height == 200
    assert second['time'].dt.total_milliseconds().to_list()[0] == 0
    np.testing.assert_array_equal(second['position'].to_list(), np.tile((5.0, 5.0), (200, 1)))


@pytest.mark.parametrize(
    ('location_column_group', 'trial_column'),
    [
        pytest.param('position', 'x', id='x_with_position'),
        pytest.param('position', 'y', id='y_with_position'),
        pytest.param('pixel', 'x', id='x_with_pixel'),
        pytest.param('pixel', 'y', id='y_with_pixel'),
    ],
)
def test_trial_column_may_be_named_like_an_axis(location_column_group, trial_column):
    events = _fixations(
        onsets=(0,), offsets=(2,), locations=((1.0, 2.0),), **{trial_column: [7]},
    )

    gaze = _generate(events, location_column_group=location_column_group)

    assert gaze.samples.columns == [trial_column, 'time', location_column_group]
    assert gaze.samples[trial_column].to_list() == [7, 7]
    assert gaze.samples[location_column_group].to_list() == [[1.0, 2.0], [1.0, 2.0]]


def test_does_not_fill_the_time_between_trials():
    events = _fixations(
        onsets=(0, 5000),
        offsets=(250, 5300),
        locations=((0.0, 0.0), (10.0, 0.0)),
        trial=[1, 2],
    )

    gaze = _generate(events)

    assert gaze.samples.height == 550


def test_is_noise_free_by_default():
    """Without a deviation, the seed changes nothing."""
    assert_frame_equal(_generate(seed=1).samples, _generate(seed=2).samples)


@pytest.mark.parametrize(
    'fixation_deviation',
    [
        pytest.param((0.1, 0.2), id='uniform'),
        pytest.param(Range(0.1, 0.2, 'normal', std=0.05), id='normal'),
    ],
)
def test_deviation_scatters_only_the_fixation_samples(fixation_deviation):
    """The gaps are interpolated from the true locations, before the scatter is added."""
    still = _generate()
    scattered = _generate(fixation_deviation=fixation_deviation)

    times = _times(still)
    in_gap = ((times >= 250) & (times < 300)) | ((times >= 560) & (times < 620))
    np.testing.assert_array_equal(_positions(scattered)[in_gap], _positions(still)[in_gap])
    assert not np.isin(_positions(scattered)[~in_gap], (0.0, 5.0, 10.0)).any()


def test_deviation_is_the_standard_deviation_of_the_scatter():
    """Fuhl & Kasneci §3.7 name the deviation; drawn per fixation, it scatters each sample."""
    events = _fixations(onsets=(0,), offsets=(10_000,), locations=((1.0, 2.0),))

    gaze = _generate(events, fixation_deviation=0.5)

    scatter = _positions(gaze) - (1.0, 2.0)
    np.testing.assert_allclose(scatter.std(axis=0), 0.5, rtol=0.05)
    np.testing.assert_allclose(scatter.mean(axis=0), 0.0, atol=0.05)


def test_reproducible_under_seed():
    first = _generate(seed=42, fixation_deviation=(0.1, 0.2))
    second = _generate(seed=42, fixation_deviation=(0.1, 0.2))
    other = _generate(seed=43, fixation_deviation=(0.1, 0.2))

    assert_frame_equal(first.samples, second.samples)
    assert not first.samples.equals(other.samples)


@pytest.mark.parametrize(
    ('location_column_group', 'locations', 'experiment', 'transforms'),
    [
        pytest.param(
            'position',
            LOCATIONS,
            pm.Experiment(sampling_rate=1000.0),
            ['pos2vel'],
            id='degrees_without_a_screen',
        ),
        pytest.param(
            'pixel',
            ((200.0, 300.0), (600.0, 300.0), (600.0, 500.0)),
            pm.Experiment(
                screen_width_px=1280,
                screen_height_px=1024,
                screen_width_cm=38,
                screen_height_cm=30,
                distance_cm=68,
                origin='upper left',
                sampling_rate=1000.0,
            ),
            ['pix2deg', 'pos2vel'],
            id='pixels_with_a_screen',
        ),
    ],
)
def test_output_runs_through_the_pipeline(
        location_column_group, locations, experiment, transforms,
):
    """The positions go through the transforms and event detection without any conversion.

    ``detect`` appends to the events the output already carries, so the detected ones are the
    rows after the input's. The generator attaches an experiment with the sampling rate only, so
    a screen is set on the output.
    """
    gaze = _generate(
        _fixations(locations=locations), location_column_group=location_column_group,
    )
    gaze.experiment = experiment

    for transform in transforms:
        getattr(gaze, transform)()
    gaze.detect('ivt')

    detected = gaze.events.frame.slice(3)
    assert detected['name'].to_list() == ['fixation'] * 3


def test_is_registered_by_name():
    assert SampleGeneratorRegistry.get('geometrical') is GeometricalSampleGenerator

    gaze = pm.synthetic.generate_samples(
        'geometrical', events=_fixations(), sampling_rate=1000.0, seed=42,
        interpolation='linear',
    )

    assert_frame_equal(gaze.samples, _generate(interpolation='linear').samples)


def test_records_its_hyperparameters():
    gaze = _generate(fixation_deviation=(0.1, 0.2))

    provenance = gaze.metadata['synthetic']
    assert provenance['generator'] == 'geometrical'
    assert provenance['hyperparameters'] == {
        'sampling_rate': 1000.0,
        'interpolation': 'cubic_spline',
        'location_column_group': 'position',
        'fixation_deviation': {
            'minimum': 0.1, 'maximum': 0.2, 'distribution': 'uniform', 'std': None,
        },
    }
    assert provenance['run'] == {'n_events': 3}


def test_output_can_be_saved_and_written_to_json(tmp_path):
    """The provenance holds the deviation as plain values, so it comes back equal."""
    gaze = _generate(fixation_deviation=Range(0.1, 0.2, 'normal', std=0.05))

    assert json.loads(json.dumps(gaze.metadata)) == gaze.metadata

    gaze.save(tmp_path, verbose=0)

    with open(tmp_path / 'metadata.yaml', encoding='utf-8') as file:
        assert yaml.safe_load(file) == gaze.metadata


def test_run_is_replayed_from_its_record():
    gaze = _generate(fixation_deviation=Range(0.1, 0.2, 'normal', std=0.05))
    record = gaze.metadata['synthetic']

    replayed = pm.synthetic.generate_samples(
        record['generator'],
        events=_fixations(),
        seed=record['seed'],
        **record['hyperparameters'],
    )

    assert_frame_equal(replayed.samples, gaze.samples)


@pytest.mark.parametrize(
    ('hyperparameters', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'interpolation': 'quadratic'},
            ValueError,
            ('interpolation', 'linear', 'cubic_spline', 'quadratic'),
            id='unknown_interpolation',
        ),
        pytest.param(
            {'location_column_group': 'velocity'},
            ValueError,
            ('location_column_group', 'position', 'pixel', 'velocity'),
            id='unknown_location_column_group',
        ),
        pytest.param(
            {'fixation_deviation': (-0.1, 0.2)},
            ValueError,
            ('fixation_deviation', 'negative', '-0.1'),
            id='negative_deviation',
        ),
        pytest.param(
            {'fixation_deviation': (0.2, 0.1)},
            ValueError,
            ('fixation_deviation', 'minimum', 'maximum'),
            id='inverted_deviation',
        ),
        pytest.param(
            {'fixation_deviation': math.nan},
            ValueError,
            ('fixation_deviation', 'minimum', 'finite', 'nan'),
            id='nan_deviation',
        ),
        pytest.param(
            {'fixation_deviation': (0.0, math.inf)},
            ValueError,
            ('fixation_deviation', 'maximum', 'finite', 'inf'),
            id='infinite_deviation',
        ),
        pytest.param(
            {'fixation_deviation': 'large'},
            TypeError,
            ('range', 'str', 'large'),
            id='deviation_of_wrong_type',
        ),
    ],
)
def test_geometrical_raises_error(hyperparameters, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        GeometricalSampleGenerator(sampling_rate=1000.0, **hyperparameters)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('events', 'sampling_rate', 'msg_substrings'),
    [
        pytest.param(
            Events(name=['fixation'], onsets=[0], offsets=[250], time_unit='ms'),
            1000.0,
            ('geometrical', "'location'", 'columns'),
            id='missing_location_column',
        ),
        pytest.param(
            _fixations(onsets=(0,), offsets=(250,), locations=((1.0, 2.0, 3.0),)),
            1000.0,
            ('geometrical', 'location', 'two', '[1.0, 2.0, 3.0]'),
            id='location_with_three_components',
        ),
        pytest.param(
            _fixations(onsets=(0,), offsets=(250,), locations=(None,)),
            1000.0,
            ('geometrical', 'location', 'two', 'None'),
            id='missing_location',
        ),
        pytest.param(
            _fixations(onsets=(0,), offsets=(250,), locations=((1.0, None),)),
            1000.0,
            ('geometrical', 'location', 'finite number', '[1.0, None]'),
            id='location_with_a_missing_component',
        ),
        pytest.param(
            Events(
                pl.DataFrame({
                    'name': ['fixation'], 'onset': [0], 'offset': [250], 'location': [['a', 'b']],
                }),
            ),
            1000.0,
            ('geometrical', 'location', 'finite number', "['a', 'b']"),
            id='location_of_text',
        ),
        pytest.param(
            Events(
                pl.DataFrame({
                    'name': ['fixation'],
                    'onset': [0],
                    'offset': [250],
                    'location': [[True, False]],
                }),
            ),
            1000.0,
            ('geometrical', 'location', 'finite number', '[True, False]'),
            id='location_of_booleans',
        ),
        pytest.param(
            _fixations(onsets=(0, 300), offsets=(None, 500), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'fixation at 0.0 ms', 'no known duration', 'drop it'),
            id='null_duration_of_the_first_fixation',
        ),
        pytest.param(
            _fixations(offsets=(250, None, 900)),
            1000.0,
            ('geometrical', 'fixation at 300.0 ms', 'no known duration'),
            id='null_duration_of_a_fixation_in_the_middle',
        ),
        pytest.param(
            _fixations(onsets=(0, 300), offsets=(250, None), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'fixation at 300.0 ms', 'no known duration'),
            id='null_duration_of_the_last_fixation',
        ),
        pytest.param(
            _fixations(onsets=(0,), offsets=(None,), locations=LOCATIONS[:1]),
            1000.0,
            ('geometrical', 'fixation at 0.0 ms', 'no known duration'),
            id='null_duration_of_the_only_fixation',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 250), offsets=(250, None), locations=(LOCATIONS[0], None),
                names=['fixation', 'saccade'],
            ),
            1000.0,
            ('geometrical', 'saccade at 250.0 ms', 'no known duration'),
            id='null_duration_of_a_saccade',
        ),
        pytest.param(
            _fixations(onsets=(0, 300), offsets=(250, 100), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'fixation at 300.0 ms', '-200.0 ms', 'negative', 'drop it'),
            id='negative_duration_of_a_fixation',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 250), offsets=(250, 240), locations=(LOCATIONS[0], None),
                names=['fixation', 'saccade'],
            ),
            1000.0,
            ('geometrical', 'saccade at 250.0 ms', '-10.0 ms', 'negative'),
            id='negative_duration_of_a_saccade',
        ),
        pytest.param(
            _fixations(onsets=(0, 300), offsets=(0, 300), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'all events', 'duration of zero', 'nothing to generate'),
            id='all_durations_of_zero',
        ),
        pytest.param(
            _fixations(onsets=(0, 200), offsets=(250, 500), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'overlap'),
            id='overlapping_fixations',
        ),
        pytest.param(
            _fixations(onsets=(0,), offsets=(250,), locations=LOCATIONS[:1], time=[1]),
            1000.0,
            ('geometrical', "['time']", 'generated', 'rename'),
            id='trial_column_named_time',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 250), offsets=(250, 650), locations=(LOCATIONS[0], None),
                names=['fixation', 'smooth_pursuit'],
            ),
            1000.0,
            ('geometrical', 'smooth_pursuit', 'fixation', 'saccade'),
            id='event_type_it_cannot_render',
        ),
        pytest.param(
            _fixations(names=[None, 'fixation', 'fixation']),
            1000.0,
            ('geometrical', '[None]', 'fixation', 'saccade'),
            id='null_event_name',
        ),
        pytest.param(
            _fixations(names=[None, 'blink', 'fixation']),
            1000.0,
            ('geometrical', "[None, 'blink']", 'fixation', 'saccade'),
            id='null_event_name_beside_an_unknown_type',
        ),
        pytest.param(
            Events(name=[], onsets=[], offsets=[], time_unit='ms'),
            1000.0,
            ('geometrical', 'empty'),
            id='empty_events',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 0), offsets=(250, 50), locations=(LOCATIONS[0], None),
                names=['fixation', 'saccade'], trial=[1, 2],
            ),
            1000.0,
            ('geometrical', 'starting at 0.0 ms', 'no fixation', 'no location'),
            id='trial_without_a_fixation',
        ),
        pytest.param(
            _fixations(
                onsets=(500,), offsets=(550,), locations=(None,), names=['saccade'],
            ),
            1000.0,
            ('geometrical', 'starting at 500.0 ms', 'no fixation', 'no location'),
            id='only_a_saccade',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 100), offsets=(0, 150), locations=(LOCATIONS[0], None),
                names=['fixation', 'saccade'],
            ),
            1000.0,
            ('geometrical', 'starting at 100.0 ms', 'no fixation', 'no location'),
            id='only_a_fixation_of_zero_duration',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 3), offsets=(3, 4), locations=(None, (1.0, 2.0)),
                names=['saccade', 'fixation'],
            ),
            250.0,
            ('geometrical', 'no sample', 'fixation at 3.0 ms', '250.0 Hz', 'sampling rate'),
            id='fixation_shorter_than_one_sample',
        ),
        pytest.param(
            _fixations(onsets=(0, 105, 200), offsets=(100, 108, 300), locations=LOCATIONS),
            10.0,
            ('geometrical', 'no sample', 'fixation at 105.0 ms', '10.0 Hz', 'sampling rate'),
            id='fixation_between_two_samples',
        ),
    ],
)
def test_geometrical_raises_error_on_generate(events, sampling_rate, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        GeometricalSampleGenerator(sampling_rate=sampling_rate).generate(events=events, seed=42)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    'location_column_group',
    [
        pytest.param('position', id='position_output'),
        pytest.param('pixel', id='pixel_output'),
    ],
)
@pytest.mark.parametrize(
    'trial_column',
    [
        pytest.param('position', id='position'),
        pytest.param('pixel', id='pixel'),
        pytest.param('velocity', id='velocity'),
        pytest.param('acceleration', id='acceleration'),
        pytest.param('distance', id='distance'),
    ],
)
def test_trial_column_named_like_a_gaze_column_raises_error(location_column_group, trial_column):
    """The one matching the output would be overwritten, which loses the trials silently."""
    events = _fixations(
        onsets=(0,), offsets=(250,), locations=LOCATIONS[:1], **{trial_column: [1]},
    )

    with pytest.raises(ValueError) as excinfo:
        _generate(events, location_column_group=location_column_group)

    msg, = excinfo.value.args
    assert 'geometrical' in msg
    assert f"['{trial_column}']" in msg
    assert 'Gaze' in msg
    assert 'rename' in msg


@pytest.mark.parametrize(
    'interpolation',
    [
        pytest.param('linear', id='linear'),
        pytest.param('cubic_spline', id='cubic_spline'),
    ],
)
@pytest.mark.parametrize(
    'component',
    [
        pytest.param(math.nan, id='nan'),
        pytest.param(math.inf, id='infinity'),
        pytest.param(-math.inf, id='negative_infinity'),
    ],
)
def test_location_that_is_not_finite_raises_error(interpolation, component):
    """Linear interpolation would carry a NaN into the samples without any error."""
    events = _fixations(
        onsets=(0, 300), offsets=(200, 500), locations=((0.0, 0.0), (component, 1.0)),
    )

    with pytest.raises(ValueError) as excinfo:
        _generate(events, interpolation=interpolation)

    msg, = excinfo.value.args
    assert 'finite number' in msg
    assert str(component) in msg


@pytest.mark.parametrize(
    ('run', 'msg_substrings'),
    [
        pytest.param({}, ('missing', "'events'"), id='without_events'),
        pytest.param(
            {'events': _fixations(), 'gaze': None},
            ('unknown', "'gaze'", "'events'"),
            id='gaze_is_no_run_parameter',
        ),
        pytest.param(
            {'events': None},
            ('geometrical', 'events', 'Events', 'NoneType'),
            id='events_of_wrong_type',
        ),
    ],
)
def test_geometrical_raises_type_error_on_run_parameters(run, msg_substrings):
    """The generator declares ``events`` as its only run parameter."""
    with pytest.raises(TypeError) as excinfo:
        GeometricalSampleGenerator(sampling_rate=1000.0).generate(seed=42, **run)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg
