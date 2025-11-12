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
from pathlib import Path
from unittest.mock import Mock

import pytest
from matplotlib import pyplot

from pymovements.stimulus.image import _draw_image_stimulus


@pytest.mark.parametrize(
    ('image_stimulus'),
    (
        pytest.param(
            './tests/files/pexels-zoorg-1000498.jpg',
            id='image_stimulus_str',
        ),
        pytest.param(
            Path('./tests/files/pexels-zoorg-1000498.jpg'),
            id='image_stimulus_Path',
        ),
    ),
)
@pytest.mark.parametrize(
    ('origin'),
    (
        pytest.param(
            'upper',
            id='origin_upper',
        ),
        pytest.param(
            'lower',
            id='origin_lower',
        ),
    ),
)
def test_show_image_stimulus(image_stimulus, origin, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)
    _draw_image_stimulus(image_stimulus, origin=origin, show=True)
    pyplot.close()
    mock.assert_called_once()


@pytest.mark.parametrize(
    ('image_stimulus'),
    (
        pytest.param(
            './tests/files/pexels-zoorg-1000498.jpg',
            id='image_stimulus_str',
        ),
        pytest.param(
            Path('./tests/files/pexels-zoorg-1000498.jpg'),
            id='image_stimulus_Path',
        ),
    ),
)
@pytest.mark.parametrize(
    ('origin'),
    (
        pytest.param(
            'upper',
            id='origin_upper',
        ),
        pytest.param(
            'lower',
            id='origin_lower',
        ),
    ),
)
def test_no_show_image_stimulus(image_stimulus, origin, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)
    _draw_image_stimulus(image_stimulus, origin=origin, show=False)
    pyplot.close()
    mock.assert_not_called()


@pytest.mark.parametrize(
    ('image_stimulus'),
    (
        pytest.param(
            './tests/files/pexels-zoorg-1000498.jpg',
            id='image_stimulus_str',
        ),
        pytest.param(
            Path('./tests/files/pexels-zoorg-1000498.jpg'),
            id='image_stimulus_Path',
        ),
    ),
)
@pytest.mark.parametrize(
    ('origin'),
    (
        pytest.param(
            'upper',
            id='origin_upper',
        ),
        pytest.param(
            'lower',
            id='origin_lower',
        ),
    ),
)
def test_image_stimulus_fig_not_None(image_stimulus, origin, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)
    fig, ax = pyplot.subplots(figsize=(15, 10))
    _draw_image_stimulus(
        image_stimulus,
        origin=origin,
        fig=fig,
        ax=ax,
        show=True,
    )
    pyplot.close()
    mock.assert_called_once()
