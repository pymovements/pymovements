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
"""Test the public generation API.

This is the framework's contract suite. ``CONTRACT_GENERATORS`` lists every generator that must
be reachable through the single ``generate`` signature, and each one is exercised by the same
test bodies, so that reaching every generation method through one signature is demonstrated
rather than asserted. The two
two generators defined here are structurally as unlike each other as the framework allows:
unconditional, fixed-rate, stochastic, position-emitting against event-conditioned,
arbitrary-rate, deterministic, velocity-emitting. Real generators join them as they land -- the
statistical generator appears twice, once per conditioning mode it declares.
"""
import numpy as np
import polars as pl
import pytest

from pymovements import Events
from pymovements import Gaze
from pymovements.synthetic import ConditioningMode
from pymovements.synthetic import GazeGenerator
from pymovements.synthetic import generate
from pymovements.synthetic import GeneratorCapabilities
from pymovements.synthetic import GeneratorConfig
from pymovements.synthetic import GeneratorLibrary
from pymovements.synthetic import InputContractError
from pymovements.synthetic import register_generator
from pymovements.synthetic import SignalRepresentation
from pymovements.warnings import ExperimentalWarning

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


class NoiseGenerator(GazeGenerator):
    """Unconditional, fixed-rate, stochastic, emits positions."""

    name = 'contract_noise'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
        signal_representations=frozenset({SignalRepresentation.POSITION}),
        sampling_rates=frozenset({1000.0}),
        sequence_duration_s=0.5,
    )

    def _generate(self, config, rng, **kwargs):
        n_samples = round(config.sampling_rate * self.capabilities.sequence_duration_s)
        positions = rng.normal(loc=500.0, scale=5.0, size=(n_samples, 2))
        return Gaze(
            samples=pl.DataFrame({'x': positions[:, 0], 'y': positions[:, 1]}),
            position_columns=['x', 'y'],
        )


class RampGenerator(GazeGenerator):
    """Event-conditioned, arbitrary-rate, deterministic, emits velocities and carries events."""

    name = 'contract_ramp'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.EVENTS}),
        signal_representations=frozenset({SignalRepresentation.VELOCITY_2D}),
        event_types=frozenset({'fixation'}),
    )

    def _generate(self, config, rng, events=None, **kwargs):
        n_samples = 10 * len(events)
        ramp = np.linspace(0.0, 1.0, n_samples)
        return Gaze(
            samples=pl.DataFrame({'vx': ramp, 'vy': -ramp}),
            events=events,
            velocity_columns=['vx', 'vy'],
        )


def _fixations():
    """Return the fixation sequence the event-conditioned generators are called with."""
    return Events(name=['fixation'] * 2, onsets=[0, 300], offsets=[250, 560], time_unit='ms')


CONTRACT_GENERATORS = [
    pytest.param(
        'contract_noise',
        GeneratorConfig(sampling_rate=1000.0, duration_s=0.5, seed=42),
        {},
        id='contract_noise',
    ),
    pytest.param(
        'contract_ramp',
        GeneratorConfig(sampling_rate=500.0, seed=42),
        {'events': _fixations()},
        id='contract_ramp',
    ),
]


@pytest.fixture(name='registered_generators', autouse=True)
def fixture_registered_generators():
    """Register the contract generators, and leave the registry as it was found."""
    original = GeneratorLibrary.generators.copy()
    register_generator(NoiseGenerator)
    register_generator(RampGenerator)
    yield
    GeneratorLibrary.generators.clear()
    GeneratorLibrary.generators.update(original)


@pytest.mark.parametrize(('name', 'config', 'inputs'), CONTRACT_GENERATORS)
def test_generate_dispatches_to_registered_generator(name, config, inputs):
    gaze = generate(name, config, **inputs)

    assert isinstance(gaze, Gaze)
    assert not gaze.samples.is_empty()


@pytest.mark.parametrize(('name', 'config', 'inputs'), CONTRACT_GENERATORS)
def test_generate_records_provenance(name, config, inputs):
    gaze = generate(name, config, **inputs)

    assert gaze.metadata['synthetic']['generator'] == name


@pytest.mark.parametrize(('name', 'config', 'inputs'), CONTRACT_GENERATORS)
def test_generate_is_reproducible_under_seed(name, config, inputs):
    first = generate(name, config, **inputs)
    second = generate(name, config, **inputs)

    assert first.samples.equals(second.samples)


@pytest.mark.parametrize(('name', 'config', 'inputs'), CONTRACT_GENERATORS)
def test_generate_matches_explicit_resolution(name, config, inputs):
    through_api = generate(name, config, **inputs)
    explicit = GeneratorLibrary.get(name)().generate(config, **inputs)

    assert through_api.samples.equals(explicit.samples)
    assert through_api.metadata['synthetic'] == explicit.metadata['synthetic']


@pytest.mark.parametrize(('name', 'config', 'inputs'), CONTRACT_GENERATORS)
def test_generate_emits_experimental_warning(name, config, inputs):
    with pytest.warns(ExperimentalWarning, match='Synthetic gaze generation is experimental'):
        generate(name, config, **inputs)


def test_generate_unknown_name_raises_key_error():
    with pytest.raises(KeyError) as excinfo:
        generate('does_not_exist', GeneratorConfig(sampling_rate=1000.0))

    msg, = excinfo.value.args
    for msg_substring in ('does_not_exist', 'not found', 'contract_noise'):
        assert msg_substring in msg


def test_generate_validates_input_like_the_explicit_path():
    with pytest.raises(InputContractError) as excinfo:
        generate('contract_noise', GeneratorConfig(sampling_rate=1000.0), events=_fixations())

    msg, = excinfo.value.args
    assert 'contract_noise' in msg


def test_generate_forwards_generator_specific_kwargs():
    received = {}

    @register_generator
    class EchoGenerator(GazeGenerator):
        name = 'contract_echo'

        def _generate(self, config, rng, **kwargs):
            received.update(kwargs)
            return Gaze(
                samples=pl.DataFrame({'x': [0.0], 'y': [0.0]}),
                position_columns=['x', 'y'],
            )

    generate('contract_echo', GeneratorConfig(sampling_rate=1000.0), amplitude=3.0)

    assert GeneratorLibrary.get('contract_echo') is EchoGenerator
    assert received == {'amplitude': 3.0}
