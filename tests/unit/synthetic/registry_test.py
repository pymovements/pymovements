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
"""Test the event and sample generator registries."""
import pytest
from polars.testing import assert_frame_equal

from pymovements.synthetic import EventGeneratorRegistry
from pymovements.synthetic import register_generator
from pymovements.synthetic import SampleGeneratorRegistry

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
    ),
    pytest.mark.usefixtures('isolated_registry'),
]


def _named(generator_class, name):
    """Return a subclass of a generator class registered under another name."""
    return type(f'{name}_generator', (generator_class,), {'name': name})


@pytest.mark.parametrize(
    ('generator', 'registry', 'other_registry'),
    [
        pytest.param(
            'noise_sample_generator', SampleGeneratorRegistry, EventGeneratorRegistry,
            id='sample_generator',
        ),
        pytest.param(
            'uniform_event_generator', EventGeneratorRegistry, SampleGeneratorRegistry,
            id='event_generator',
        ),
    ],
)
def test_generator_is_registered_in_the_registry_of_its_level(
        request, generator, registry, other_registry,
):
    generator_class = _named(request.getfixturevalue(generator), 'registered')

    assert register_generator(generator_class) is generator_class
    assert registry.get('registered') is generator_class
    assert 'registered' in registry.names()
    assert 'registered' not in other_registry.names()


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        pytest.param('registered', True, id='registered_name'),
        pytest.param('does_not_exist', False, id='unknown_name'),
    ],
)
def test_registry_contains_registered_names(noise_sample_generator, name, expected):
    """Mirrors ``name in EventDetectionLibrary()`` and ``name in SampleMeasureLibrary()``."""
    register_generator(_named(noise_sample_generator, 'registered'))

    assert (name in SampleGeneratorRegistry()) is expected


def test_both_levels_can_use_the_same_name(noise_sample_generator, uniform_event_generator):
    sample_class = register_generator(_named(noise_sample_generator, 'statistical'))
    event_class = register_generator(_named(uniform_event_generator, 'statistical'))

    assert SampleGeneratorRegistry.get('statistical') is sample_class
    assert EventGeneratorRegistry.get('statistical') is event_class


def test_register_generator_rejects_a_class_of_neither_level():
    class NotAGenerator:
        name = 'not_a_generator'

    with pytest.raises(TypeError) as excinfo:
        register_generator(NotAGenerator)

    msg, = excinfo.value.args
    for msg_substring in ('NotAGenerator', 'EventGenerator', 'SampleGenerator'):
        assert msg_substring in msg


@pytest.mark.parametrize(
    'registry',
    [
        pytest.param(EventGeneratorRegistry, id='event_registry'),
        pytest.param(SampleGeneratorRegistry, id='sample_registry'),
    ],
)
def test_get_unknown_generator_raises_key_error(registry):
    with pytest.raises(KeyError) as excinfo:
        registry.get('does_not_exist')

    msg, = excinfo.value.args
    for msg_substring in ('does_not_exist', 'not found', registry.__name__):
        assert msg_substring in msg


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('', id='empty_name'),
        pytest.param(123, id='non_string_name'),
    ],
)
def test_add_generator_without_valid_name_raises_value_error(noise_sample_generator, name):
    generator_class = _named(noise_sample_generator, 'placeholder')
    generator_class.name = name

    with pytest.raises(ValueError) as excinfo:
        SampleGeneratorRegistry.add(generator_class)

    msg, = excinfo.value.args
    for msg_substring in ('must define', 'name', 'SampleGeneratorRegistry'):
        assert msg_substring in msg


def test_names_returns_sorted_names(noise_sample_generator):
    for name in ('gan', 'diffusion', 'statistical'):
        register_generator(_named(noise_sample_generator, name))

    names = SampleGeneratorRegistry.names()

    assert names == sorted(names)
    assert {'diffusion', 'gan', 'statistical'} <= set(names)


def test_registering_same_name_twice_overwrites(noise_sample_generator):
    first = register_generator(_named(noise_sample_generator, 'duplicate'))
    second = register_generator(_named(noise_sample_generator, 'duplicate'))

    assert first is not second
    assert SampleGeneratorRegistry.get('duplicate') is second


@pytest.mark.parametrize(
    ('generator', 'registry', 'parameters'),
    [
        pytest.param(
            'noise_sample_generator', SampleGeneratorRegistry,
            {'sampling_rate': 500.0, 'scale': 2.0},
            id='sample_generator',
        ),
        pytest.param(
            'uniform_event_generator', EventGeneratorRegistry, {'max_duration': 150.0},
            id='event_generator',
        ),
    ],
)
def test_create_returns_the_generator_the_constructor_returns(
        request, generator, registry, parameters,
):
    generator_class = register_generator(request.getfixturevalue(generator))

    created = registry.create(generator_class.name, **parameters)

    assert created == generator_class(**parameters)


def test_created_sample_generator_generates_what_the_constructed_one_does(
        noise_sample_generator, fixations,
):
    register_generator(noise_sample_generator)

    created = SampleGeneratorRegistry.create('test_noise', sampling_rate=1000.0, scale=2.0)
    constructed = noise_sample_generator(sampling_rate=1000.0, scale=2.0)
    gaze = created.generate(events=fixations, seed=42)
    expected = constructed.generate(events=fixations, seed=42)

    assert_frame_equal(gaze.samples, expected.samples)
    assert gaze.metadata == expected.metadata


def test_created_event_generator_generates_what_the_constructed_one_does(
        uniform_event_generator,
):
    register_generator(uniform_event_generator)

    created = EventGeneratorRegistry.create('test_uniform', max_duration=150.0)
    constructed = uniform_event_generator(max_duration=150.0)
    events = created.generate(duration=2.0, seed=42)
    expected = constructed.generate(duration=2.0, seed=42)

    assert_frame_equal(events.frame, expected.frame)
    assert events.metadata == expected.metadata


@pytest.mark.parametrize(
    ('registry', 'name'),
    [
        pytest.param(SampleGeneratorRegistry, 'does_not_exist', id='sample_registry'),
        pytest.param(EventGeneratorRegistry, 'does_not_exist', id='event_registry'),
        pytest.param(
            EventGeneratorRegistry, 'test_noise',
            id='sample_generator_is_not_an_event_generator',
        ),
    ],
)
def test_create_unknown_generator_raises_key_error(noise_sample_generator, registry, name):
    register_generator(noise_sample_generator)

    with pytest.raises(KeyError) as excinfo:
        registry.create(name)

    msg, = excinfo.value.args
    for msg_substring in (name, 'not found', registry.__name__):
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('generator', 'registry', 'parameters'),
    [
        pytest.param(
            'noise_sample_generator', SampleGeneratorRegistry, {'sampling_rate': 1000.0},
            id='sample_generator',
        ),
        pytest.param('uniform_event_generator', EventGeneratorRegistry, {}, id='event_generator'),
    ],
)
def test_create_with_unknown_parameter_raises_type_error(
        request, generator, registry, parameters,
):
    generator_class = register_generator(request.getfixturevalue(generator))

    with pytest.raises(TypeError) as excinfo:
        registry.create(generator_class.name, scael=2.0, **parameters)

    msg, = excinfo.value.args
    assert 'scael' in msg
