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
"""Test the provenance metadata written by the synthetic gaze generation framework.

The generators here are not registered, so this module leaves the registry alone.
"""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Gaze
from pymovements.synthetic import ConditioningMode
from pymovements.synthetic import GazeGenerator
from pymovements.synthetic import GeneratorCapabilities
from pymovements.synthetic import GeneratorConfig
from pymovements.synthetic import SignalRepresentation
from pymovements.synthetic.provenance import generation_provenance
from pymovements.synthetic.provenance import PROVENANCE_KEY


class NoiseGenerator(GazeGenerator):
    """A stochastic generator drawing its samples from the seeded generator."""

    name = 'test_noise'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
        signal_representations=frozenset({SignalRepresentation.POSITION}),
    )

    def _generate(self, config, rng, **kwargs):
        return Gaze(
            samples=pl.DataFrame({'x': rng.normal(size=8), 'y': rng.normal(size=8)}),
            position_columns=['x', 'y'],
        )


class UndeclaredGenerator(GazeGenerator):
    """A generator that declares no capabilities."""

    name = 'test_undeclared'

    def _generate(self, config, rng, **kwargs):
        return Gaze(samples=pl.DataFrame({'x': [0.0], 'y': [0.0]}), position_columns=['x', 'y'])


class ExistingMetadataGenerator(GazeGenerator):
    """A generator whose output already carries metadata of its own."""

    name = 'test_existing_metadata'

    def _generate(self, config, rng, **kwargs):
        return Gaze(
            samples=pl.DataFrame({'x': [0.0], 'y': [0.0]}),
            position_columns=['x', 'y'],
            metadata={'sampling_rate': 1000.0},
        )


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {'generator': 'test_noise', 'config': None, 'seed': 42},
            {
                'generator': 'test_noise',
                'config': None,
                'seed': 42,
                'signal_representations': None,
                'artifacts': {},
            },
            id='without_config_or_capabilities',
        ),

        pytest.param(
            {
                'generator': 'test_noise',
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=2.5, seed=42),
                'seed': 42,
            },
            {
                'generator': 'test_noise',
                'config': {'sampling_rate': 1000.0, 'duration_s': 2.5, 'seed': 42},
                'seed': 42,
                'signal_representations': None,
                'artifacts': {},
            },
            id='config_is_recorded_as_dictionary',
        ),

        pytest.param(
            {
                'generator': 'test_noise',
                'config': None,
                'seed': 7,
                'capabilities': GeneratorCapabilities(
                    conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
                    signal_representations=frozenset({
                        SignalRepresentation.VELOCITY_2D,
                        SignalRepresentation.POSITION,
                    }),
                ),
            },
            {
                'generator': 'test_noise',
                'config': None,
                'seed': 7,
                'signal_representations': ['position', 'velocity_2d'],
                'artifacts': {},
            },
            id='signal_representations_are_sorted_values',
        ),

        pytest.param(
            {
                'generator': 'test_noise',
                'config': None,
                'seed': 7,
                'artifacts': {'weights': 'sha256:abc'},
            },
            {
                'generator': 'test_noise',
                'config': None,
                'seed': 7,
                'signal_representations': None,
                'artifacts': {'weights': 'sha256:abc'},
            },
            id='artifacts_are_recorded',
        ),
    ],
)
def test_generation_provenance_returns(kwargs, expected):
    provenance = generation_provenance(**kwargs)[PROVENANCE_KEY]

    expected = {**expected, 'pymovements_version': __version__}
    assert provenance == expected


def test_generation_provenance_records_pymovements_version():
    provenance = generation_provenance(generator='test_noise', config=None, seed=42)

    assert provenance[PROVENANCE_KEY]['pymovements_version'] == __version__


def test_metadata_records_generator_config_and_seed():
    config = GeneratorConfig(sampling_rate=1000.0, duration_s=2.5, seed=42)

    gaze = NoiseGenerator().generate(config=config)

    assert gaze.metadata[PROVENANCE_KEY] == {
        'generator': 'test_noise',
        'config': {'sampling_rate': 1000.0, 'duration_s': 2.5, 'seed': 42},
        'seed': 42,
        'signal_representations': ['position'],
        'artifacts': {},
        'pymovements_version': __version__,
    }


def test_metadata_records_no_signal_representations_when_undeclared():
    gaze = UndeclaredGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0))

    assert gaze.metadata[PROVENANCE_KEY]['signal_representations'] is None


def test_metadata_records_config_of_none_when_none_supplied():
    gaze = UndeclaredGenerator().generate()

    assert gaze.metadata[PROVENANCE_KEY]['config'] is None


def test_provenance_preserves_existing_metadata():
    gaze = ExistingMetadataGenerator().generate()

    assert gaze.metadata['sampling_rate'] == 1000.0
    assert PROVENANCE_KEY in gaze.metadata


def test_same_seed_produces_identical_output():
    config = GeneratorConfig(sampling_rate=1000.0, seed=42)

    first = NoiseGenerator().generate(config=config)
    second = NoiseGenerator().generate(config=config)

    assert_frame_equal(first.samples, second.samples)


def test_different_seed_produces_different_output():
    first = NoiseGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0, seed=42))
    second = NoiseGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0, seed=7))

    assert not first.samples.equals(second.samples)


def test_seed_is_recorded_when_none_supplied():
    gaze = NoiseGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0))

    seed = gaze.metadata[PROVENANCE_KEY]['seed']
    assert isinstance(seed, int)
    assert gaze.metadata[PROVENANCE_KEY]['config']['seed'] == seed


def test_recorded_seed_reproduces_output():
    generator = NoiseGenerator()
    gaze = generator.generate(config=GeneratorConfig(sampling_rate=1000.0))

    recorded_config = GeneratorConfig.from_dict(gaze.metadata[PROVENANCE_KEY]['config'])
    repeated = generator.generate(config=recorded_config)

    assert_frame_equal(gaze.samples, repeated.samples)


def test_drawn_seeds_differ_between_runs():
    first = NoiseGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0))
    second = NoiseGenerator().generate(config=GeneratorConfig(sampling_rate=1000.0))

    assert (
        first.metadata[PROVENANCE_KEY]['seed'] != second.metadata[PROVENANCE_KEY]['seed']
    )
