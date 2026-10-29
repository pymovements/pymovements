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
"""Test the entry points that generate synthetic data by generator name."""
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements.synthetic import register_generator

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
    ),
    pytest.mark.usefixtures('isolated_registry'),
]


@pytest.fixture(name='registered_sample_generator')
def fixture_registered_sample_generator(noise_sample_generator):
    """Register the toy sample generator for the test."""
    return register_generator(noise_sample_generator)


@pytest.fixture(name='registered_event_generator')
def fixture_registered_event_generator(uniform_event_generator):
    """Register the toy event generator for the test."""
    return register_generator(uniform_event_generator)


def test_generate_samples_matches_the_explicit_call(registered_sample_generator, fixations):
    gaze = pm.synthetic.generate_samples(
        'test_noise', events=fixations, sampling_rate=1000.0, seed=42, scale=2.0,
    )
    explicit = registered_sample_generator(scale=2.0).generate(
        events=fixations, sampling_rate=1000.0, seed=42,
    )

    assert_frame_equal(gaze.samples, explicit.samples)
    assert gaze.metadata == explicit.metadata


@pytest.mark.usefixtures('registered_sample_generator')
def test_generate_samples_passes_hyperparameters_to_the_constructor():
    gaze = pm.synthetic.generate_samples('test_noise', sampling_rate=1000.0, seed=42, scale=3.0)

    assert gaze.metadata['synthetic']['hyperparameters'] == {'scale': 3.0}


@pytest.mark.usefixtures('registered_sample_generator')
def test_generate_samples_passes_the_experiment():
    experiment = pm.Experiment(sampling_rate=250.0)

    gaze = pm.synthetic.generate_samples(
        'test_noise', sampling_rate=250.0, seed=42, experiment=experiment,
    )

    assert gaze.experiment is experiment


@pytest.mark.usefixtures('registered_sample_generator')
def test_generate_samples_passes_the_other_keywords_to_generate(fixations):
    """Keywords that are not a field of the generator are run parameters."""
    gaze = pm.synthetic.generate_samples(
        'test_noise', events=fixations, sampling_rate=500.0, seed=42, scale=2.0,
    )

    assert gaze.metadata['synthetic']['hyperparameters'] == {'scale': 2.0}
    assert gaze.metadata['synthetic']['run'] == {'sampling_rate': 500.0, 'n_events': 2}


def test_generate_events_matches_the_explicit_call(registered_event_generator):
    events = pm.synthetic.generate_events(
        'test_uniform', duration=2.0, seed=42, max_duration=150.0,
    )
    explicit = registered_event_generator(max_duration=150.0).generate(duration=2.0, seed=42)

    assert_frame_equal(events.frame, explicit.frame)
    assert events.metadata == explicit.metadata


@pytest.mark.parametrize(
    ('entry_point', 'kwargs'),
    [
        pytest.param(
            pm.synthetic.generate_samples, {'sampling_rate': 1000.0}, id='generate_samples',
        ),
        pytest.param(pm.synthetic.generate_events, {'duration': 1.0}, id='generate_events'),
    ],
)
@pytest.mark.usefixtures('registered_sample_generator', 'registered_event_generator')
def test_misspelled_hyperparameter_raises_type_error(entry_point, kwargs):
    name = 'test_noise' if entry_point is pm.synthetic.generate_samples else 'test_uniform'

    with pytest.raises(TypeError) as excinfo:
        entry_point(name, scael=2.0, **kwargs)

    msg, = excinfo.value.args
    assert 'scael' in msg


@pytest.mark.parametrize(
    ('entry_point', 'kwargs', 'library_name'),
    [
        pytest.param(
            pm.synthetic.generate_samples, {'sampling_rate': 1000.0}, 'SampleGeneratorLibrary',
            id='generate_samples',
        ),
        pytest.param(
            pm.synthetic.generate_events, {'duration': 1.0}, 'EventGeneratorLibrary',
            id='generate_events',
        ),
    ],
)
def test_unknown_name_raises_key_error(entry_point, kwargs, library_name):
    with pytest.raises(KeyError) as excinfo:
        entry_point('does_not_exist', **kwargs)

    msg, = excinfo.value.args
    for msg_substring in ('does_not_exist', library_name):
        assert msg_substring in msg


@pytest.mark.usefixtures('registered_sample_generator')
def test_sample_generator_is_not_found_among_event_generators():
    with pytest.raises(KeyError):
        pm.synthetic.generate_events('test_noise', duration=1.0)
