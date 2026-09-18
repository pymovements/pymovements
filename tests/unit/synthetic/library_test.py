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
"""Test the GeneratorLibrary."""
import polars as pl
import pytest

from pymovements import Gaze
from pymovements.synthetic import GazeGenerator
from pymovements.synthetic import GeneratorLibrary
from pymovements.synthetic import register_generator


@pytest.fixture(name='isolated_registry', autouse=True)
def fixture_isolated_registry():
    """Restore the registry after each test, so registrations do not leak between tests."""
    original = GeneratorLibrary.generators.copy()
    yield
    GeneratorLibrary.generators.clear()
    GeneratorLibrary.generators.update(original)


def _make_generator_class(name):
    """Create a GazeGenerator subclass with the given registered name."""
    class _Generator(GazeGenerator):
        def _generate(self, config, rng, **kwargs):
            return Gaze(
                samples=pl.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
                position_columns=['x', 'y'],
            )

    _Generator.name = name
    _Generator.__name__ = f'{name}_generator'
    return _Generator


def test_generator_registered():
    generator_class = register_generator(_make_generator_class('statistical'))

    assert 'statistical' in GeneratorLibrary.generators
    assert GeneratorLibrary.get('statistical') is generator_class
    assert 'statistical' in GeneratorLibrary.names()


def test_register_generator_returns_class_unchanged():
    generator_class = _make_generator_class('geometrical')

    assert register_generator(generator_class) is generator_class


def test_generator_defined_outside_framework_package_registers():
    @register_generator
    class ThirdPartyGenerator(GazeGenerator):
        name = 'third_party'

        def _generate(self, config, rng, **kwargs):
            return Gaze(
                samples=pl.DataFrame({'x': [0.0], 'y': [0.0]}),
                position_columns=['x', 'y'],
            )

    assert GeneratorLibrary.get('third_party') is ThirdPartyGenerator


def test_get_unknown_generator_raises_key_error():
    register_generator(_make_generator_class('known'))

    with pytest.raises(KeyError) as excinfo:
        GeneratorLibrary.get('does_not_exist')

    msg, = excinfo.value.args
    for msg_substring in ('does_not_exist', 'not found', 'GeneratorLibrary', 'known'):
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('name', 'msg_substrings'),
    [
        pytest.param(
            None,
            ('must define', 'name', 'GeneratorLibrary'),
            id='missing_name_raises_value_error',
        ),
        pytest.param(
            '',
            ('must define', 'name', 'GeneratorLibrary'),
            id='empty_name_raises_value_error',
        ),
        pytest.param(
            123,
            ('must define', 'name', 'GeneratorLibrary'),
            id='non_string_name_raises_value_error',
        ),
    ],
)
def test_add_generator_without_valid_name_raises_value_error(name, msg_substrings):
    class NamelessGenerator(GazeGenerator):
        def _generate(self, config, rng, **kwargs):
            return Gaze()

    if name is not None:
        NamelessGenerator.name = name

    with pytest.raises(ValueError) as excinfo:
        GeneratorLibrary.add(NamelessGenerator)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_names_returns_sorted_names():
    for name in ('gan', 'diffusion', 'statistical'):
        register_generator(_make_generator_class(name))

    names = GeneratorLibrary.names()

    assert names == sorted(names)
    for name in ('diffusion', 'gan', 'statistical'):
        assert name in names


def test_registering_same_name_twice_overwrites():
    first = register_generator(_make_generator_class('duplicate'))
    second = register_generator(_make_generator_class('duplicate'))

    assert first is not second
    assert GeneratorLibrary.get('duplicate') is second


def test_resolved_generator_generates_gaze():
    register_generator(_make_generator_class('resolvable'))

    gaze = GeneratorLibrary.get('resolvable')().generate()

    assert isinstance(gaze, Gaze)
    assert gaze.samples.height == 2
