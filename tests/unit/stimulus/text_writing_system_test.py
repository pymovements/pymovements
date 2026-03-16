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
"""Test writing_system parameter in TextStimulus class."""

import pytest

from pymovements.stimulus.text import WritingSystem


@pytest.mark.parametrize(
    ('posargs', 'kwargs', 'expected'),
    [
        pytest.param(
            [],
            {},
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='default',
        ),
        pytest.param(
            ['left-to-right'],
            {},
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='ltr_posarg',
        ),
        pytest.param(
            ['right-to-left'],
            {},
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='rtl_posarg',
        ),
        pytest.param(
            [],
            {'directionality': 'right-to-left'},
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='rtl_kwarg',
        ),
    ],
)
def test_writing_system_init(posargs, kwargs, expected):
    """Test that init arguments lead to correct fields."""
    writing_system = WritingSystem(*posargs, **kwargs)
    assert writing_system.directionality == expected['directionality']
    assert writing_system.axis == expected['axis']
    assert writing_system.lining == expected['lining']


@pytest.mark.parametrize(
    ('descriptor', 'expected'),
    [
        pytest.param(
            'left-to-right',
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='left-to-right',
        ),
        pytest.param(
            'LEFT-TO-RIGHT',
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='LEFT-TO-RIGHT',
        ),
        pytest.param(
            'ltr',
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='ltr',
        ),
        pytest.param(
            'LTR',
            {'directionality': 'left-to-right', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='LTR',
        ),
        pytest.param(
            'right-to-left',
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='right-to-left',
        ),
        pytest.param(
            'RIGHT-TO-LEFT',
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='RIGHT-TO-LEFT',
        ),
        pytest.param(
            'rtl',
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='rtl',
        ),
        pytest.param(
            'RTL',
            {'directionality': 'right-to-left', 'axis': 'horizontal', 'lining': 'top-to-bottom'},
            id='RTL',
        ),
    ],
)
def test_writing_system_from_descriptor(descriptor, expected):
    """Test that writing_system descriptor strings lead to correct fields."""
    writing_system = WritingSystem.from_descriptor(descriptor)
    assert writing_system.directionality == expected['directionality']
    assert writing_system.axis == expected['axis']
    assert writing_system.lining == expected['lining']


def test_writing_system_from_descriptor_unknown_raises_exception():
    with pytest.raises(ValueError, match="Unknown descriptor 'test'"):
        WritingSystem.from_descriptor('test')
