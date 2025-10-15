# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test read from IPC/feather."""
import re

import pytest

from pymovements import __version__
from pymovements.gaze import from_ipc


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'shape'),
    [
        pytest.param(
            'monocular_example.feather',
            {},
            (10, 2),
            id='feather_mono_shape',
        ),
        pytest.param(
            'binocular_example.feather',
            {},
            (10, 3),
            id='feather_bino_shape',
        ),
        pytest.param(
            'monocular_example.feather',
            {
                'read_ipc_kwargs': {'columns': ['time']},
            },
            (10, 1),
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='read_ipc_kwargs',
        ),
        pytest.param(
            'monocular_example.feather',
            {
                'columns': ['time'],
            },
            (10, 1),
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
                'ignore:.*kwargs.*:DeprecationWarning',
            ),
            id='**kwargs',
        ),
        pytest.param(
            'monocular_example.feather',
            {
                'column_map': {'pixel': 'pixel_coordinates'},
            },
            (10, 2),
            marks=pytest.mark.filterwarnings(
                'ignore:Gaze contains samples but no.*:UserWarning',
            ),
            id='feather_mono_shape_column_map',
        ),
    ],
)
def test_shapes(filename, kwargs, shape, make_example_file):
    filepath = make_example_file(filename)
    gaze = from_ipc(file=filepath, **kwargs)

    assert gaze.samples.shape == shape


@pytest.mark.parametrize(
    ('filename', 'kwargs'),
    [
        pytest.param(
            'monocular_example.feather',
            {
                'n_rows': 1,
            },
            id='**kwargs',
        ),
    ],
)
def test_from_ipc_parameter_is_deprecated(filename, kwargs, make_example_file):
    filepath = make_example_file(filename)

    with pytest.raises(DeprecationWarning) as info:
        from_ipc(filepath, **kwargs)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    argument_name = list(kwargs.keys())[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'keyword argument {argument_name} was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )
