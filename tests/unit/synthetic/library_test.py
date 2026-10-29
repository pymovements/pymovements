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
"""Test the event and sample generator libraries."""
import pytest

from pymovements.synthetic import EventGeneratorLibrary
from pymovements.synthetic import register_generator
from pymovements.synthetic import SampleGeneratorLibrary

pytestmark = pytest.mark.usefixtures('isolated_registry')


def _named(generator_class, name):
    """Return a subclass of a generator class registered under another name."""
    return type(f'{name}_generator', (generator_class,), {'name': name})


@pytest.mark.parametrize(
    ('generator', 'library', 'other_library'),
    [
        pytest.param(
            'noise_sample_generator', SampleGeneratorLibrary, EventGeneratorLibrary,
            id='sample_generator',
        ),
        pytest.param(
            'uniform_event_generator', EventGeneratorLibrary, SampleGeneratorLibrary,
            id='event_generator',
        ),
    ],
)
def test_generator_is_registered_in_the_library_of_its_level(
        request, generator, library, other_library,
):
    generator_class = _named(request.getfixturevalue(generator), 'registered')

    assert register_generator(generator_class) is generator_class
    assert library.get('registered') is generator_class
    assert 'registered' in library.names()
    assert 'registered' not in other_library.names()


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        pytest.param('registered', True, id='registered_name'),
        pytest.param('does_not_exist', False, id='unknown_name'),
    ],
)
def test_library_contains_registered_names(noise_sample_generator, name, expected):
    """Mirrors ``name in EventDetectionLibrary()`` and ``name in SampleMeasureLibrary()``."""
    register_generator(_named(noise_sample_generator, 'registered'))

    assert (name in SampleGeneratorLibrary()) is expected


def test_both_levels_can_use_the_same_name(noise_sample_generator, uniform_event_generator):
    sample_class = register_generator(_named(noise_sample_generator, 'statistical'))
    event_class = register_generator(_named(uniform_event_generator, 'statistical'))

    assert SampleGeneratorLibrary.get('statistical') is sample_class
    assert EventGeneratorLibrary.get('statistical') is event_class


def test_register_generator_rejects_a_class_of_neither_level():
    class NotAGenerator:
        name = 'not_a_generator'

    with pytest.raises(TypeError) as excinfo:
        register_generator(NotAGenerator)

    msg, = excinfo.value.args
    for msg_substring in ('NotAGenerator', 'EventGenerator', 'SampleGenerator'):
        assert msg_substring in msg


@pytest.mark.parametrize(
    'library',
    [
        pytest.param(EventGeneratorLibrary, id='event_library'),
        pytest.param(SampleGeneratorLibrary, id='sample_library'),
    ],
)
def test_get_unknown_generator_raises_key_error(library):
    with pytest.raises(KeyError) as excinfo:
        library.get('does_not_exist')

    msg, = excinfo.value.args
    for msg_substring in ('does_not_exist', 'not found', library.__name__):
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
        SampleGeneratorLibrary.add(generator_class)

    msg, = excinfo.value.args
    for msg_substring in ('must define', 'name', 'SampleGeneratorLibrary'):
        assert msg_substring in msg


def test_names_returns_sorted_names(noise_sample_generator):
    for name in ('gan', 'diffusion', 'statistical'):
        register_generator(_named(noise_sample_generator, name))

    names = SampleGeneratorLibrary.names()

    assert names == sorted(names)
    assert {'diffusion', 'gan', 'statistical'} <= set(names)


def test_registering_same_name_twice_overwrites(noise_sample_generator):
    first = register_generator(_named(noise_sample_generator, 'duplicate'))
    second = register_generator(_named(noise_sample_generator, 'duplicate'))

    assert first is not second
    assert SampleGeneratorLibrary.get('duplicate') is second
