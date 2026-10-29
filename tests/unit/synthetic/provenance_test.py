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
"""Test the provenance record written into generated data, through the public API."""
import json

import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements import __version__
from pymovements.synthetic import register_generator

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


@pytest.mark.parametrize(
    ('run', 'expected_run'),
    [
        pytest.param({}, {'sampling_rate': 500.0}, id='unconditioned'),
        pytest.param(
            {'events': 'fixations'},
            {'sampling_rate': 500.0, 'n_events': 2},
            id='event_conditioned',
        ),
    ],
)
def test_sample_generator_records_provenance(request, noise_sample_generator, run, expected_run):
    run = {key: request.getfixturevalue(value) for key, value in run.items()}

    gaze = noise_sample_generator(scale=2.0).generate(sampling_rate=500.0, seed=42, **run)

    assert gaze.metadata['synthetic'] == {
        'generator': 'test_noise',
        'level': 'samples',
        'hyperparameters': {'scale': 2.0},
        'run': expected_run,
        'seed': 42,
        'pymovements_version': __version__,
    }


def test_generator_can_override_how_its_run_is_recorded(noise_sample_generator, fixations):
    class SummarisingGenerator(noise_sample_generator):
        def _record_run(self, run):
            return {'event_names': sorted(set(run['events'].frame['name']))}

    gaze = SummarisingGenerator().generate(events=fixations, sampling_rate=500.0, seed=42)

    assert gaze.metadata['synthetic']['run'] == {
        'sampling_rate': 500.0, 'event_names': ['fixation'],
    }


def test_event_generator_records_provenance(uniform_event_generator):
    events = uniform_event_generator(max_duration=200.0).generate(duration=1.5, seed=42)

    assert events.metadata['synthetic'] == {
        'generator': 'test_uniform',
        'level': 'events',
        'hyperparameters': {'min_duration': 100.0, 'max_duration': 200.0},
        'run': {'duration': 1.5},
        'seed': 42,
        'pymovements_version': __version__,
    }


def test_drawn_seed_is_recorded(noise_sample_generator):
    """The drawn seed is exact as a float too, so JSON readers and Int64 columns keep it."""
    gaze = noise_sample_generator().generate(sampling_rate=500.0)

    seed = gaze.metadata['synthetic']['seed']
    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 53


def test_provenance_records_a_range_as_plain_values(ranged_noise_sample_generator, tmp_path):
    """A record of plain values can be written to JSON and saved with the gaze."""
    gaze = ranged_noise_sample_generator().generate(sampling_rate=500.0, seed=42)

    hyperparameters = gaze.metadata['synthetic']['hyperparameters']
    assert hyperparameters['offset'] == {
        'minimum': 0.0, 'maximum': 1.0, 'distribution': 'normal', 'std': 0.5,
    }
    json.dumps(gaze.metadata)
    gaze.save(tmp_path, verbose=0)


@pytest.mark.usefixtures('isolated_registry')
def test_run_is_replayed_from_a_record_with_a_range(ranged_noise_sample_generator):
    generator_class = register_generator(ranged_noise_sample_generator)
    gaze = generator_class(offset=(2.0, 3.0)).generate(sampling_rate=500.0, seed=42)
    record = gaze.metadata['synthetic']

    replayed = pm.synthetic.generate_samples(
        record['generator'],
        sampling_rate=record['run']['sampling_rate'],
        seed=record['seed'],
        **record['hyperparameters'],
    )

    assert_frame_equal(replayed.samples, gaze.samples)
    assert replayed.metadata == gaze.metadata
