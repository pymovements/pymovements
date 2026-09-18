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
"""Test generator capability declarations."""
import dataclasses
import json

import pytest

from pymovements.synthetic import ConditioningMode
from pymovements.synthetic import GeneratorCapabilities
from pymovements.synthetic import SignalRepresentation


def test_conditioning_modes_are_exactly_the_supported_three():
    """Stimulus conditioning is an extensibility target and must stay unimplemented."""
    assert {mode.value for mode in ConditioningMode} == {'unconditional', 'events', 'gaze'}


def test_signal_representations_cover_the_observed_output_quantities():
    assert {representation.value for representation in SignalRepresentation} == {
        'position', 'velocity_2d', 'velocity_magnitude',
    }


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.UNCONDITIONAL},
                'signal_representations': {SignalRepresentation.VELOCITY_MAGNITUDE},
                'event_types': {'fixation', 'saccade', 'smooth_pursuit'},
                'sampling_rates': None,
                'sequence_duration_s': None,
                'requires_artifacts': False,
            },
            {
                'conditioning_modes': frozenset({ConditioningMode.UNCONDITIONAL}),
                'signal_representations': frozenset({SignalRepresentation.VELOCITY_MAGNITUDE}),
                'event_types': frozenset({'fixation', 'saccade', 'smooth_pursuit'}),
                'sampling_rates': None,
                'sequence_duration_s': None,
                'requires_artifacts': False,
            },
            id='fuhl_et_al_2018_statistical',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {
                    SignalRepresentation.VELOCITY_2D, SignalRepresentation.POSITION,
                },
                'event_types': {'fixation', 'saccade'},
                'sampling_rates': {1000.0},
                'sequence_duration_s': None,
                'requires_artifacts': True,
            },
            {
                'conditioning_modes': frozenset({ConditioningMode.EVENTS}),
                'signal_representations': frozenset({
                    SignalRepresentation.VELOCITY_2D, SignalRepresentation.POSITION,
                }),
                'event_types': frozenset({'fixation', 'saccade'}),
                'sampling_rates': frozenset({1000.0}),
                'sequence_duration_s': None,
                'requires_artifacts': True,
            },
            id='sp_eyegan_prasse_et_al_2023',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.UNCONDITIONAL},
                'signal_representations': {SignalRepresentation.VELOCITY_MAGNITUDE},
                'event_types': set(),
                'sampling_rates': {1000.0},
                'sequence_duration_s': 0.2,
                'requires_artifacts': True,
            },
            {
                'conditioning_modes': frozenset({ConditioningMode.UNCONDITIONAL}),
                'signal_representations': frozenset({SignalRepresentation.VELOCITY_MAGNITUDE}),
                'event_types': frozenset(),
                'sampling_rates': frozenset({1000.0}),
                'sequence_duration_s': 0.2,
                'requires_artifacts': True,
            },
            id='bhandari_et_al_2025_gan',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.GAZE},
                'signal_representations': {SignalRepresentation.VELOCITY_2D},
                'event_types': set(),
                'sampling_rates': {1000.0},
                'sequence_duration_s': 5.0,
                'requires_artifacts': True,
            },
            {
                'conditioning_modes': frozenset({ConditioningMode.GAZE}),
                'signal_representations': frozenset({SignalRepresentation.VELOCITY_2D}),
                'event_types': frozenset(),
                'sampling_rates': frozenset({1000.0}),
                'sequence_duration_s': 5.0,
                'requires_artifacts': True,
            },
            id='diffeyesyn_jiao_et_al_2025',
        ),
    ],
)
def test_capabilities_of_the_four_reference_methods(kwargs, expected):
    """The four reference methods must be expressible before any generator exists.

    Mirrors the heterogeneity evidence table; a method that cannot be written down here
    means the type is wrong.
    """
    capabilities = GeneratorCapabilities(**kwargs)

    assert dataclasses.asdict(capabilities) == expected


def test_diffeyesyn_fixed_window_is_expressible():
    """The 5-second/1,000 Hz constraint of DiffEyeSyn must be visible to a caller."""
    capabilities = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.GAZE},
        signal_representations={SignalRepresentation.VELOCITY_2D},
        sampling_rates={1000.0},
        sequence_duration_s=5.0,
        requires_artifacts=True,
    )

    assert capabilities.sampling_rates == frozenset({1000.0})
    assert capabilities.sequence_duration_s == 5.0
    assert capabilities.requires_artifacts


def test_directionless_velocity_is_expressible():
    """Velocities without directions must not imply positions."""
    capabilities = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.UNCONDITIONAL},
        signal_representations={SignalRepresentation.VELOCITY_MAGNITUDE},
    )

    assert SignalRepresentation.POSITION not in capabilities.signal_representations
    assert SignalRepresentation.VELOCITY_2D not in capabilities.signal_representations


def test_arbitrary_sampling_rate_is_distinct_from_a_fixed_rate():
    """``None`` means arbitrary and must not be confused with an empty declaration."""
    arbitrary = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.UNCONDITIONAL},
        signal_representations={SignalRepresentation.VELOCITY_MAGNITUDE},
    )

    assert arbitrary.sampling_rates is None
    assert arbitrary.sequence_duration_s is None


@pytest.mark.parametrize(
    'field_name',
    [
        pytest.param('conditioning_modes', id='conditioning_modes'),
        pytest.param('signal_representations', id='signal_representations'),
        pytest.param('event_types', id='event_types'),
        pytest.param('sampling_rates', id='sampling_rates'),
        pytest.param('sequence_duration_s', id='sequence_duration_s'),
        pytest.param('requires_artifacts', id='requires_artifacts'),
    ],
)
def test_capabilities_is_immutable(field_name):
    capabilities = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.EVENTS},
        signal_representations={SignalRepresentation.POSITION},
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(capabilities, field_name, None)


@pytest.mark.parametrize(
    'iterable_type',
    [
        pytest.param(list, id='list'),
        pytest.param(tuple, id='tuple'),
        pytest.param(set, id='set'),
        pytest.param(frozenset, id='frozenset'),
    ],
)
def test_collections_are_coerced_to_frozensets(iterable_type):
    capabilities = GeneratorCapabilities(
        conditioning_modes=iterable_type([ConditioningMode.EVENTS]),
        signal_representations=iterable_type([SignalRepresentation.POSITION]),
        event_types=iterable_type(['fixation']),
        sampling_rates=iterable_type([1000.0]),
    )

    assert isinstance(capabilities.conditioning_modes, frozenset)
    assert isinstance(capabilities.signal_representations, frozenset)
    assert isinstance(capabilities.event_types, frozenset)
    assert isinstance(capabilities.sampling_rates, frozenset)


def test_integer_sampling_rates_are_coerced_to_float():
    capabilities = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.EVENTS},
        signal_representations={SignalRepresentation.POSITION},
        sampling_rates={1000},
    )

    assert all(isinstance(rate, float) for rate in capabilities.sampling_rates)


def test_equal_declarations_are_equal_and_hashable():
    kwargs = {
        'conditioning_modes': [ConditioningMode.EVENTS],
        'signal_representations': [SignalRepresentation.POSITION],
        'sampling_rates': [1000.0],
    }

    assert GeneratorCapabilities(**kwargs) == GeneratorCapabilities(**kwargs)
    assert len({GeneratorCapabilities(**kwargs), GeneratorCapabilities(**kwargs)}) == 1


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'conditioning_modes': set(),
                'signal_representations': {SignalRepresentation.POSITION},
            },
            ValueError,
            ('conditioning_modes', 'empty', 'at least one'),
            id='empty_conditioning_modes_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': set(),
            },
            ValueError,
            ('signal_representations', 'empty', 'at least one'),
            id='empty_signal_representations_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sampling_rates': set(),
            },
            ValueError,
            ('sampling_rates', 'empty', 'none', 'arbitrary'),
            id='empty_sampling_rates_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sampling_rates': {0.0},
            },
            ValueError,
            ('sampling_rates', 'greater than zero', '0.0'),
            id='zero_sampling_rate_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sampling_rates': {-1000.0},
            },
            ValueError,
            ('sampling_rates', 'greater than zero', '-1000.0'),
            id='negative_sampling_rate_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sequence_duration_s': 0.0,
            },
            ValueError,
            ('sequence_duration_s', 'greater than zero', '0.0'),
            id='zero_sequence_duration_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sequence_duration_s': -5.0,
            },
            ValueError,
            ('sequence_duration_s', 'greater than zero', '-5.0'),
            id='negative_sequence_duration_raises_value_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {'events'},
                'signal_representations': {SignalRepresentation.POSITION},
            },
            TypeError,
            ('conditioning_modes', 'ConditioningMode', 'events'),
            id='string_conditioning_mode_raises_type_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {'position'},
            },
            TypeError,
            ('signal_representations', 'SignalRepresentation', 'position'),
            id='string_signal_representation_raises_type_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'event_types': {1},
            },
            TypeError,
            ('event_types', 'str', '1'),
            id='non_string_event_type_raises_type_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sampling_rates': {'1000'},
            },
            TypeError,
            ('sampling_rates', 'number', '1000'),
            id='non_numeric_sampling_rate_raises_type_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'sequence_duration_s': '5.0',
            },
            TypeError,
            ('sequence_duration_s', 'number', '5.0'),
            id='non_numeric_sequence_duration_raises_type_error',
        ),
        pytest.param(
            {
                'conditioning_modes': {ConditioningMode.EVENTS},
                'signal_representations': {SignalRepresentation.POSITION},
                'requires_artifacts': 'yes',
            },
            TypeError,
            ('requires_artifacts', 'bool', 'yes'),
            id='non_bool_requires_artifacts_raises_type_error',
        ),
    ],
)
def test_capabilities_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        GeneratorCapabilities(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    'capabilities',
    [
        pytest.param(
            GeneratorCapabilities(
                conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
                signal_representations=frozenset({SignalRepresentation.VELOCITY_MAGNITUDE}),
                event_types=frozenset({'fixation', 'saccade', 'smooth_pursuit'}),
            ),
            id='fuhl_et_al_2018_statistical',
        ),
        pytest.param(
            GeneratorCapabilities(
                conditioning_modes=frozenset({ConditioningMode.EVENTS}),
                signal_representations=frozenset({
                    SignalRepresentation.VELOCITY_2D, SignalRepresentation.POSITION,
                }),
                event_types=frozenset({'fixation', 'saccade'}),
                sampling_rates=frozenset({1000.0}),
                requires_artifacts=True,
            ),
            id='sp_eyegan_prasse_et_al_2023',
        ),
        pytest.param(
            GeneratorCapabilities(
                conditioning_modes=frozenset({ConditioningMode.GAZE}),
                signal_representations=frozenset({SignalRepresentation.VELOCITY_2D}),
                sampling_rates=frozenset({1000.0}),
                sequence_duration_s=5.0,
                requires_artifacts=True,
            ),
            id='diffeyesyn_jiao_et_al_2025',
        ),
    ],
)
def test_capabilities_are_serialisable(capabilities):
    """A declaration must survive a round trip through a plain dict."""
    as_dict = capabilities.to_dict()

    assert json.loads(json.dumps(as_dict)) == as_dict
    assert GeneratorCapabilities.from_dict(as_dict) == capabilities


def test_to_dict_is_deterministically_ordered():
    """Set iteration order must not leak into the dictionary."""
    capabilities = GeneratorCapabilities(
        conditioning_modes={ConditioningMode.EVENTS, ConditioningMode.GAZE},
        signal_representations={
            SignalRepresentation.VELOCITY_2D, SignalRepresentation.POSITION,
        },
        event_types={'saccade', 'fixation'},
        sampling_rates={2000.0, 1000.0},
    )

    assert capabilities.to_dict() == {
        'conditioning_modes': ['events', 'gaze'],
        'signal_representations': ['position', 'velocity_2d'],
        'event_types': ['fixation', 'saccade'],
        'sampling_rates': [1000.0, 2000.0],
        'sequence_duration_s': None,
        'requires_artifacts': False,
    }


def test_from_dict_accepts_omitted_optional_keys():
    capabilities = GeneratorCapabilities.from_dict({
        'conditioning_modes': ['unconditional'],
        'signal_representations': ['velocity_magnitude'],
    })

    assert capabilities == GeneratorCapabilities(
        conditioning_modes={ConditioningMode.UNCONDITIONAL},
        signal_representations={SignalRepresentation.VELOCITY_MAGNITUDE},
    )


def test_from_dict_accepts_enum_members():
    """A hand-written declaration may hold members rather than values."""
    capabilities = GeneratorCapabilities.from_dict({
        'conditioning_modes': [ConditioningMode.UNCONDITIONAL],
        'signal_representations': [SignalRepresentation.VELOCITY_MAGNITUDE],
    })

    assert capabilities == GeneratorCapabilities.from_dict({
        'conditioning_modes': ['unconditional'],
        'signal_representations': ['velocity_magnitude'],
    })


@pytest.mark.parametrize(
    ('as_dict', 'msg_substrings'),
    [
        pytest.param(
            {
                'conditioning_modes': ['stimulus'],
                'signal_representations': ['position'],
            },
            ('conditioning_modes', 'stimulus', 'unconditional', 'events', 'gaze'),
            id='unknown_conditioning_mode',
        ),
        pytest.param(
            {
                'conditioning_modes': ['events'],
                'signal_representations': ['acceleration'],
            },
            ('signal_representations', 'acceleration', 'position', 'velocity_2d'),
            id='unknown_signal_representation',
        ),
    ],
)
def test_from_dict_raises_error_for_unknown_member(as_dict, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        GeneratorCapabilities.from_dict(as_dict)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


def test_from_dict_raises_error_for_unknown_key():
    with pytest.raises(ValueError) as excinfo:
        GeneratorCapabilities.from_dict({
            'conditioning_modes': ['events'],
            'signal_representations': ['position'],
            'supports_stimulus': True,
        })

    msg, = excinfo.value.args
    for msg_substring in ('supports_stimulus', 'unknown', 'conditioning_modes'):
        assert msg_substring.lower() in msg.lower()
