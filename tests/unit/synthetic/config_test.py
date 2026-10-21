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
"""Test the configuration objects of the synthetic gaze generation framework."""
import dataclasses

import pytest

from pymovements.synthetic import GeneratorConfig


@dataclasses.dataclass
class NoiseConfig(GeneratorConfig):
    """A subclass adding a generator-specific parameter, as concrete generators will."""

    noise_px: float = 3.0


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {'sampling_rate': 1000.0},
            {'sampling_rate': 1000.0, 'duration_s': None, 'seed': None},
            id='sampling_rate_only_defaults_duration_and_seed',
        ),

        pytest.param(
            {'sampling_rate': 1000},
            {'sampling_rate': 1000.0},
            id='integer_sampling_rate_is_cast_to_float',
        ),

        pytest.param(
            {'sampling_rate': 500.0, 'duration_s': 5},
            {'duration_s': 5.0},
            id='integer_duration_is_cast_to_float',
        ),

        pytest.param(
            {'sampling_rate': 250.0, 'duration_s': 2.5, 'seed': 42},
            {'sampling_rate': 250.0, 'duration_s': 2.5, 'seed': 42},
            id='all_parameters_explicit',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'seed': 0},
            {'seed': 0},
            id='seed_zero_is_kept',
        ),
    ],
)
def test_config_returns(kwargs, expected):
    config = GeneratorConfig(**kwargs)

    for name, expected_value in expected.items():
        value = getattr(config, name)
        assert value == expected_value, f'{name} = {value}, expected = {expected_value}'
        assert type(value) is type(expected_value), (
            f'{name} is of type {type(value).__name__}, '
            f'expected {type(expected_value).__name__}'
        )


@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        pytest.param(
            GeneratorConfig(sampling_rate=1000.0),
            {'sampling_rate': 1000.0, 'duration_s': None, 'seed': None},
            id='defaults',
        ),

        pytest.param(
            GeneratorConfig(sampling_rate=250.0, duration_s=2.5, seed=42),
            {'sampling_rate': 250.0, 'duration_s': 2.5, 'seed': 42},
            id='all_parameters_explicit',
        ),

        pytest.param(
            NoiseConfig(sampling_rate=1000.0, seed=7, noise_px=4.0),
            {'sampling_rate': 1000.0, 'duration_s': None, 'seed': 7, 'noise_px': 4.0},
            id='subclass_includes_own_parameters',
        ),
    ],
)
def test_config_to_dict_returns(config, expected):
    assert config.to_dict() == expected


@pytest.mark.parametrize(
    'config',
    [
        pytest.param(GeneratorConfig(sampling_rate=1000.0), id='defaults'),

        pytest.param(GeneratorConfig(sampling_rate=1000.0, seed=0), id='seed_zero'),

        pytest.param(
            GeneratorConfig(sampling_rate=250.0, duration_s=2.5, seed=42),
            id='all_parameters_explicit',
        ),

        pytest.param(
            NoiseConfig(sampling_rate=1000.0, duration_s=5.0, seed=7, noise_px=4.0),
            id='subclass',
        ),
    ],
)
def test_config_roundtrip(config):
    assert type(config).from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ('config_class', 'declaration', 'expected'),
    [
        pytest.param(
            GeneratorConfig,
            {'sampling_rate': 1000.0},
            GeneratorConfig(sampling_rate=1000.0),
            id='omitted_keys_take_their_default',
        ),

        pytest.param(
            NoiseConfig,
            {'sampling_rate': 1000.0, 'noise_px': 4.0},
            NoiseConfig(sampling_rate=1000.0, noise_px=4.0),
            id='subclass_parameter',
        ),
    ],
)
def test_config_from_dict_returns(config_class, declaration, expected):
    assert config_class.from_dict(declaration) == expected


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {},
            TypeError,
            ('sampling_rate', 'missing'),
            id='no_sampling_rate_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': '1000'},
            TypeError,
            ('sampling_rate', 'number', 'str', '1000'),
            id='string_sampling_rate_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': True},
            TypeError,
            ('sampling_rate', 'number', 'bool'),
            id='bool_sampling_rate_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': None},
            TypeError,
            ('sampling_rate', 'number', 'nonetype'),
            id='none_sampling_rate_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': 0.0},
            ValueError,
            ('sampling_rate', 'greater than zero', '0.0'),
            id='zero_sampling_rate_raises_value_error',
        ),

        pytest.param(
            {'sampling_rate': -1000.0},
            ValueError,
            ('sampling_rate', 'greater than zero', '-1000.0'),
            id='negative_sampling_rate_raises_value_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'duration_s': '5'},
            TypeError,
            ('duration_s', 'number', 'str', '5'),
            id='string_duration_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'duration_s': 0.0},
            ValueError,
            ('duration_s', 'greater than zero', '0.0'),
            id='zero_duration_raises_value_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'duration_s': -2.5},
            ValueError,
            ('duration_s', 'greater than zero', '-2.5'),
            id='negative_duration_raises_value_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'seed': 1.5},
            TypeError,
            ('seed', 'integer', 'float', '1.5'),
            id='float_seed_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'seed': True},
            TypeError,
            ('seed', 'integer', 'bool'),
            id='bool_seed_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'seed': 'foo'},
            TypeError,
            ('seed', 'integer', 'str', 'foo'),
            id='string_seed_raises_type_error',
        ),

        pytest.param(
            {'sampling_rate': 1000.0, 'seed': -1},
            ValueError,
            ('seed', 'negative', '-1'),
            id='negative_seed_raises_value_error',
        ),
    ],
)
def test_config_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        GeneratorConfig(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('config_class', 'declaration', 'msg_substrings'),
    [
        pytest.param(
            GeneratorConfig,
            {'sampling_rate': 1000.0, 'noise_px': 4.0},
            ('unknown', 'noise_px', 'sampling_rate', 'duration_s', 'seed'),
            id='unknown_key_raises_value_error',
        ),

        pytest.param(
            NoiseConfig,
            {'sampling_rate': 1000.0, 'foo': 1.0},
            ('unknown', 'foo', 'noise_px'),
            id='unknown_key_on_subclass_lists_subclass_keys',
        ),
    ],
)
def test_config_from_dict_raises_error(config_class, declaration, msg_substrings):
    with pytest.raises(ValueError) as excinfo:
        config_class.from_dict(declaration)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()
