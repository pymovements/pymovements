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
"""Test pymovements plotting utils."""
import matplotlib.pyplot
import numpy as np
import pytest

from pymovements import __version__
from pymovements.utils.plotting import draw_image_stimulus
from pymovements.utils.plotting import draw_line_data
from pymovements.utils.plotting import setup_matplotlib


@pytest.fixture(
    name='axes',
    params=[
        (10, 15),
        (15, 15),
    ],
)
def axes_fixture(request):
    fig = matplotlib.pyplot.figure(figsize=request.param)
    yield fig.gca()


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {
                'x_signal': np.array([0.0, 1.0]),
                'y_signal': np.array([0.0, 2.0]),
                'figsize': (10, 15),
            },
            id='both_signals_figsize',
        ),
    ],
)
def test_setup_matplotlib(kwargs):
    setup_matplotlib(**kwargs)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_draw_image_stimulus(axes, make_example_file):
    filepath = make_example_file('pexels-zoorg-1000498.jpg')
    draw_image_stimulus(image_stimulus=filepath, ax=axes)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_draw_line_data(axes):
    draw_line_data(x_signal=np.array([0.0, 0.0]), y_signal=np.array([0.0, 0.0]), ax=axes)


@pytest.mark.parametrize(
    ('plotting_function', 'kwargs'),
    [
        pytest.param(
            setup_matplotlib,
            {
                'x_signal': np.array([0.0, 0.0]),
                'y_signal': np.array([0.0, 0.0]),
                'figsize': (10, 15),
            },
            id='_setup_matplotlib',
        ),

        pytest.param(
            draw_image_stimulus,
            {},
            id='draw_image_stimulus',
        ),

        pytest.param(
            draw_line_data,
            {},
            id='_draw_line_data',
        ),
    ],
)
def test_plotting_function_removed(plotting_function, kwargs, assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        plotting_function(**kwargs)
    assert_deprecation_is_removed('utils/filters.py', info.value.args[0], __version__)
