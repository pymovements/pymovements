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

import numpy as np
import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements import Events
from pymovements.synthetic import GeometricalSampleGenerator
from pymovements.synthetic import Range
from pymovements.synthetic import SampleGeneratorLibrary

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)

#: The locations of the three fixations ``_fixations`` returns by default, in degrees.
LOCATIONS = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0))


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
    """The disclosed limitation: the velocity of a cubic gap is symmetric, peaking mid-gap."""
    gaze = _generate()

    # The 51 steps from the last sample of the first fixation to the first of the second.
    speeds = np.abs(np.diff(_positions(gaze)[249:301, 0]))
    assert abs(int(np.argmax(speeds)) - 25) <= 1


def test_attaches_a_copy_of_the_events():
    """The gaps are filled in the samples only; no saccade is added to the events."""
    events = _fixations()

    gaze = _generate(events)

    assert_frame_equal(gaze.events.frame, events.frame)
    assert gaze.events is not events


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
    assert SampleGeneratorLibrary.get('geometrical') is GeometricalSampleGenerator

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
            ('minimum', 'maximum'),
            id='inverted_deviation',
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
            ('geometrical', 'location', 'number', '[1.0, None]'),
            id='location_with_a_missing_component',
        ),
        pytest.param(
            _fixations(onsets=(0, 200), offsets=(250, 500), locations=LOCATIONS[:2]),
            1000.0,
            ('geometrical', 'overlap'),
            id='overlapping_fixations',
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
            ('geometrical', 'no sample', 'fixation', '1000.0 Hz'),
            id='trial_without_a_fixation',
        ),
        pytest.param(
            _fixations(
                onsets=(0, 3), offsets=(3, 4), locations=(None, (1.0, 2.0)),
                names=['saccade', 'fixation'],
            ),
            250.0,
            ('geometrical', 'no sample', 'fixation', '250.0 Hz'),
            id='fixation_shorter_than_one_sample',
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
