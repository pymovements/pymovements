# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Test Image stimulus class."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from matplotlib import pyplot

import pymovements as pm


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_image_stimulus_from_file(image_path):
    image_stimulus = pm.stimulus.image.from_file(image_path)
    assert image_stimulus.images[0].as_posix() == 'tests/files/pexels-zoorg-1000498.jpg'


@pytest.mark.parametrize(
    ('path'),
    (
        pytest.param('tests/files/', id='image_path_str'),
        pytest.param(Path('tests/files/'), id='image_path_Path'),
    ),
)
def test_image_stimulus_from_files(path):
    image_stimulus = pm.stimulus.image.from_files(path, r'{book_name}-{page_num}-{line_num}.jpg')
    assert image_stimulus.images[0].as_posix() == 'tests/files/pexels-zoorg-1000498.jpg'


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
@pytest.mark.parametrize(
    ('stimulus_id'),
    (
        pytest.param(0, id='stimulus_id_0'),
    ),
)
@pytest.mark.parametrize(
    ('origin'),
    (
        pytest.param('upper', id='origin_upper'),
        pytest.param('lower', id='origin_lower'),
    ),
)
def test_not_showing_image_stimulus_from_file(image_path, stimulus_id, origin, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)
    image_stimulus = pm.stimulus.image.from_file(image_path)
    assert image_stimulus.images[0].as_posix() == 'tests/files/pexels-zoorg-1000498.jpg'
    image_stimulus.show(stimulus_id, origin)
    pyplot.close()
    mock.assert_called_once()
