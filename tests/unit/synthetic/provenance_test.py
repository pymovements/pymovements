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
"""Test the provenance record written into generated data."""
import pytest

from pymovements import __version__
from pymovements.synthetic.provenance import generation_provenance
from pymovements.synthetic.provenance import PROVENANCE_KEY

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


def test_generation_provenance_returns():
    provenance = generation_provenance(
        generator='example',
        level='samples',
        hyperparameters={'scale': 2.0},
        run={'sampling_rate': 1000.0},
        seed=42,
    )

    assert provenance == {
        PROVENANCE_KEY: {
            'generator': 'example',
            'level': 'samples',
            'hyperparameters': {'scale': 2.0},
            'run': {'sampling_rate': 1000.0},
            'seed': 42,
            'pymovements_version': __version__,
        },
    }


def test_generation_provenance_copies_its_inputs():
    hyperparameters = {'scale': 2.0}

    provenance = generation_provenance(
        generator='example', level='samples', hyperparameters=hyperparameters, run={}, seed=42,
    )
    hyperparameters['scale'] = 3.0

    assert provenance[PROVENANCE_KEY]['hyperparameters'] == {'scale': 2.0}


@pytest.mark.parametrize(
    ('events', 'expected_run'),
    [
        pytest.param(
            False,
            {'sampling_rate': 500.0, 'n_events': None, 'n_gaze_samples': None},
            id='unconditioned',
        ),
        pytest.param(
            True,
            {'sampling_rate': 500.0, 'n_events': 2, 'n_gaze_samples': None},
            id='event_conditioned',
        ),
    ],
)
def test_sample_generator_records_provenance(
        noise_sample_generator, fixations, events, expected_run,
):
    gaze = noise_sample_generator(scale=2.0).generate(
        fixations if events else None, sampling_rate=500.0, seed=42,
    )

    assert gaze.metadata[PROVENANCE_KEY] == {
        'generator': 'test_noise',
        'level': 'samples',
        'hyperparameters': {'scale': 2.0},
        'run': expected_run,
        'seed': 42,
        'pymovements_version': __version__,
    }


def test_sample_generator_records_gaze_conditioning(noise_sample_generator):
    conditioning = noise_sample_generator().generate(sampling_rate=500.0, seed=1)

    gaze = noise_sample_generator().generate(gaze=conditioning, sampling_rate=500.0, seed=42)

    assert gaze.metadata[PROVENANCE_KEY]['run']['n_gaze_samples'] == 10


def test_event_generator_records_provenance(uniform_event_generator):
    events = uniform_event_generator(max_duration=200.0).generate(duration=1.5, seed=42)

    assert events.metadata[PROVENANCE_KEY] == {
        'generator': 'test_uniform',
        'level': 'events',
        'hyperparameters': {'min_duration': 100.0, 'max_duration': 200.0},
        'run': {'duration': 1.5},
        'seed': 42,
        'pymovements_version': __version__,
    }


def test_drawn_seed_is_recorded(noise_sample_generator):
    gaze = noise_sample_generator().generate(sampling_rate=500.0)

    seed = gaze.metadata[PROVENANCE_KEY]['seed']
    assert isinstance(seed, int)
    assert seed >= 0
