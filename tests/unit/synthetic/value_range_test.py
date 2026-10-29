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
"""Test the range type that generator hyperparameters are drawn from."""
import dataclasses

import numpy as np
import pytest

from pymovements.synthetic import Range


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {'minimum': 1, 'maximum': 2},
            Range(1.0, 2.0, 'uniform', None),
            id='ints_are_cast_to_float',
        ),
        pytest.param(
            {'minimum': 3.0, 'maximum': 3.0},
            Range(3.0, 3.0),
            id='single_value_range',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': 10.0, 'distribution': 'normal', 'std': 2},
            Range(0.0, 10.0, 'normal', 2.0),
            id='normal_with_std',
        ),
        pytest.param(
            {'minimum': np.int64(1), 'maximum': np.float64(2.5)},
            Range(1.0, 2.5),
            id='numpy_numbers',
        ),
    ],
)
def test_range_returns(kwargs, expected):
    value_range = Range(**kwargs)

    assert value_range == expected
    assert isinstance(value_range.minimum, float)
    assert isinstance(value_range.maximum, float)


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'minimum': 5.0, 'maximum': 1.0},
            ValueError,
            ('minimum (5.0)', 'maximum (1.0)'),
            id='inverted_range_raises_value_error',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': 1.0, 'distribution': 'gamma'},
            ValueError,
            ('distribution', 'normal', 'uniform', "'gamma'"),
            id='unknown_distribution_raises_value_error',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': 1.0, 'distribution': 'normal', 'std': -1.0},
            ValueError,
            ('std', 'negative'),
            id='negative_std_raises_value_error',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': 1.0, 'std': 0.2},
            ValueError,
            ('std', 'normal', "'uniform'", '0.2'),
            id='std_of_a_uniform_range_raises_value_error',
        ),
        pytest.param(
            {'minimum': '0', 'maximum': 1.0},
            TypeError,
            ('minimum', 'number', 'str'),
            id='string_minimum_raises_type_error',
        ),
        pytest.param(
            {'minimum': None, 'maximum': 1.0},
            TypeError,
            ('minimum', 'number', 'NoneType'),
            id='none_minimum_raises_type_error',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': True},
            TypeError,
            ('maximum', 'number', 'bool'),
            id='bool_maximum_raises_type_error',
        ),
        pytest.param(
            {'minimum': 0.0, 'maximum': 1.0, 'std': 'wide'},
            TypeError,
            ('std', 'number', 'str'),
            id='string_std_raises_type_error',
        ),
    ],
)
def test_range_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        Range(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_range_is_immutable():
    value_range = Range(0.0, 1.0)

    with pytest.raises(dataclasses.FrozenInstanceError) as excinfo:
        value_range.minimum = 0.5

    msg, = excinfo.value.args
    assert "'minimum'" in msg


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param(Range(1.0, 2.0, 'normal'), Range(1.0, 2.0, 'normal'), id='range'),
        pytest.param((300, 500), Range(300.0, 500.0), id='tuple'),
        pytest.param([300, 500], Range(300.0, 500.0), id='list'),
        pytest.param(5, Range(5.0, 5.0), id='int'),
        pytest.param(2.5, Range(2.5, 2.5), id='float'),
    ],
)
def test_from_value_returns(value, expected):
    assert Range.from_value(value) == expected


@pytest.mark.parametrize(
    'value',
    [
        pytest.param((1.0, 2.0, 3.0), id='triple'),
        pytest.param('1-2', id='string'),
        pytest.param(None, id='none'),
        pytest.param(True, id='bool'),
    ],
)
def test_from_value_raises_type_error(value):
    with pytest.raises(TypeError) as excinfo:
        Range.from_value(value)

    msg, = excinfo.value.args
    for msg_substring in ('Range', '(minimum, maximum)', 'single number'):
        assert msg_substring in msg


@pytest.mark.parametrize(
    'value_range',
    [
        pytest.param(Range(300.0, 500.0), id='uniform'),
        pytest.param(Range(300.0, 500.0, 'normal', std=50.0), id='normal'),
        pytest.param(Range(300.0, 500.0, 'normal'), id='normal_with_range_width_as_std'),
    ],
)
def test_draw_stays_within_the_range(value_range):
    rng = np.random.default_rng(42)

    values = [value_range.draw(rng) for _ in range(1000)]

    assert min(values) >= 300.0
    assert max(values) <= 500.0


@pytest.mark.parametrize(
    'value_range',
    [
        pytest.param(Range(300.0, 500.0), id='uniform'),
        pytest.param(Range(300.0, 500.0, 'normal', std=50.0), id='normal'),
    ],
)
def test_draw_is_reproducible_under_seed(value_range):
    first_rng, second_rng = np.random.default_rng(7), np.random.default_rng(7)

    first = [value_range.draw(first_rng) for _ in range(3)]
    second = [value_range.draw(second_rng) for _ in range(3)]

    assert first == second


def test_normal_draw_is_centred_on_the_range():
    rng = np.random.default_rng(42)

    values = [Range(0.0, 100.0, 'normal', std=5.0).draw(rng) for _ in range(2000)]

    assert abs(np.mean(values) - 50.0) < 1.0


@pytest.mark.parametrize(
    'distribution',
    [
        pytest.param('uniform', id='uniform'),
        pytest.param('normal', id='normal'),
    ],
)
def test_single_value_range_always_draws_that_value(distribution):
    rng = np.random.default_rng(42)

    assert Range(3.0, 3.0, distribution).draw(rng) == 3.0
