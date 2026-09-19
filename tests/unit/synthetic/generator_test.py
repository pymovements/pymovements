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
"""Test the synthetic gaze generator abstraction.

The generators here are not registered, so this module leaves the registry alone.
Registration is covered in ``library_test.py``.
"""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import Gaze
from pymovements.synthetic import GazeGenerator
from pymovements.synthetic import GeneratorConfig

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


class MinimalGenerator(GazeGenerator):
    """A minimal generator returning a fixed two-sample gaze sequence."""

    name = 'minimal'

    def _generate(self, config, rng, **kwargs):
        return Gaze(
            samples=pl.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
            position_columns=['x', 'y'],
        )


class EchoKwargsGenerator(GazeGenerator):
    """A generator recording the arguments it was called with."""

    name = 'echo_kwargs'

    def __init__(self):
        self.received = None
        self.received_config = None
        self.received_rng = None

    def _generate(self, config, rng, **kwargs):
        self.received = kwargs
        self.received_config = config
        self.received_rng = rng
        return Gaze(samples=pl.DataFrame({'x': [0.0], 'y': [0.0]}), position_columns=['x', 'y'])


def test_generator_is_abstract():
    with pytest.raises(TypeError) as excinfo:
        GazeGenerator()  # pylint: disable=abstract-class-instantiated

    msg, = excinfo.value.args
    for msg_substring in ('abstract', 'GazeGenerator', '_generate'):
        assert msg_substring in msg


def test_subclass_without_generate_is_abstract():
    class IncompleteGenerator(GazeGenerator):
        name = 'incomplete'

    with pytest.raises(TypeError) as excinfo:
        IncompleteGenerator()  # pylint: disable=abstract-class-instantiated

    msg, = excinfo.value.args
    for msg_substring in ('abstract', 'IncompleteGenerator', '_generate'):
        assert msg_substring in msg


def test_generate_returns_gaze():
    gaze = MinimalGenerator().generate()

    assert isinstance(gaze, Gaze)
    assert_frame_equal(
        gaze.samples,
        pl.DataFrame({'position': [[0.0, 0.0], [1.0, 1.0]]}),
    )


def test_generate_forwards_kwargs_to_private_generate():
    generator = EchoKwargsGenerator()

    generator.generate(centers=[(0.0, 0.0)], foo='bar')

    assert generator.received == {'centers': [(0.0, 0.0)], 'foo': 'bar'}


def test_generate_passes_random_generator_to_private_generate():
    generator = EchoKwargsGenerator()

    generator.generate()

    assert isinstance(generator.received_rng, np.random.Generator)


def test_generate_passes_config_to_private_generate():
    generator = EchoKwargsGenerator()
    config = GeneratorConfig(sampling_rate=1000.0, seed=42)

    generator.generate(config=config)

    assert generator.received_config == config


def test_generate_passes_no_config_when_none_supplied():
    generator = EchoKwargsGenerator()

    generator.generate()

    assert generator.received_config is None


def test_generate_passes_drawn_seed_in_config_to_private_generate():
    generator = EchoKwargsGenerator()

    generator.generate(config=GeneratorConfig(sampling_rate=1000.0))

    assert isinstance(generator.received_config.seed, int)


def test_generate_does_not_mutate_the_supplied_config():
    config = GeneratorConfig(sampling_rate=1000.0)

    EchoKwargsGenerator().generate(config=config)

    assert config.seed is None


@pytest.mark.parametrize(
    ('first_seed', 'second_seed', 'expected_equal'),
    [
        pytest.param(42, 42, True, id='same_seed_draws_identically'),
        pytest.param(42, 7, False, id='different_seed_draws_differently'),
    ],
)
def test_random_generator_is_seeded_from_config(first_seed, second_seed, expected_equal):
    generator = EchoKwargsGenerator()

    generator.generate(config=GeneratorConfig(sampling_rate=1000.0, seed=first_seed))
    first = generator.received_rng.normal(size=4)

    generator.generate(config=GeneratorConfig(sampling_rate=1000.0, seed=second_seed))
    second = generator.received_rng.normal(size=4)

    assert np.array_equal(first, second) == expected_equal, (
        f'first = {first}, second = {second}'
    )


@pytest.mark.parametrize(
    ('generator_class', 'expected_name'),
    [
        pytest.param(MinimalGenerator, 'minimal', id='minimal'),
        pytest.param(EchoKwargsGenerator, 'echo_kwargs', id='echo_kwargs'),
    ],
)
def test_subclass_declares_name(generator_class, expected_name):
    assert generator_class.name == expected_name
