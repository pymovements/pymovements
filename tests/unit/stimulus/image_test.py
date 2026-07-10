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

import matplotlib.pyplot as plt
import pytest
from matplotlib import pyplot

from pymovements.stimulus.image import from_file
from pymovements.stimulus.image import from_files
from pymovements.stimulus.image import ImageStimulus


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_image_stimulus_from_file(image_path):
    image_stimulus = from_file(image_path)
    assert image_stimulus.images[0].as_posix() == 'tests/files/stimuli/pexels-zoorg-1000498.jpg'


@pytest.mark.parametrize(
    ('path'),
    (
        pytest.param('tests/files/', id='image_path_str'),
        pytest.param(Path('tests/files/'), id='image_path_Path'),
    ),
)
def test_image_stimulus_from_files(path):
    image_stimulus = from_files(path, r'{book_name}-{page_num}-{line_num}.jpg')
    assert image_stimulus.images[0].as_posix() == 'tests/files/stimuli/pexels-zoorg-1000498.jpg'


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
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
    image_stimulus = from_file(image_path)
    assert image_stimulus.images[0].as_posix() == 'tests/files/stimuli/pexels-zoorg-1000498.jpg'
    with pytest.warns(DeprecationWarning, match='This method is deprecated'):
        image_stimulus.show(stimulus_id, origin)
    pyplot.close()
    mock.assert_called_once()


# NEW TESTS BELOW

@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
@pytest.mark.parametrize(
    ('stimulus_id'),
    (
        pytest.param(0, id='stimulus_id_0'),
    ),
)
def test_plot_image_stimulus(image_path, stimulus_id, monkeypatch):
    """Test the new plot method."""
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)

    image_stimulus = from_file(image_path)
    assert image_stimulus.images[0].as_posix() == 'tests/files/stimuli/pexels-zoorg-1000498.jpg'

    fig, ax = image_stimulus.plot(stimulus_id)

    assert fig is not None
    assert ax is not None
    mock.assert_not_called()


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
@pytest.mark.parametrize(
    ('stimulus_id'),
    (
        pytest.param(0, id='stimulus_id_0'),
    ),
)
def test_plot_image_stimulus_with_custom_axes(image_path, stimulus_id):
    """Test plotting on custom axes."""

    fig, ax = pyplot.subplots(figsize=(10, 8))
    image_stimulus = from_file(image_path)

    returned_fig, returned_ax = image_stimulus.plot(stimulus_id, ax=ax)

    assert returned_fig is fig
    assert returned_ax is ax
    pyplot.close(fig)


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_show_method_deprecation(image_path, monkeypatch):
    """Test that the show method raises a deprecation warning."""
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)

    image_stimulus = from_file(image_path)

    with pytest.warns(DeprecationWarning, match='This method is deprecated'):
        image_stimulus.show(0, 'upper')

    mock.assert_called_once()


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_show_method_updates_origin(image_path, monkeypatch):
    """Test that show still updates the origin attribute."""
    mock = Mock()
    monkeypatch.setattr(pyplot, 'show', mock)

    image_stimulus = from_file(image_path)
    # image_stimulus.origin
    

    with pytest.warns(DeprecationWarning, match='This method is deprecated'):
        image_stimulus.show(0, 'lower')

    assert image_stimulus.origin == 'lower'
    mock.assert_called_once()


def test_plot_with_invalid_stimulus_id_raises_error():
    """Test that plot raises IndexError with invalid stimulus_id."""
    image_stimulus = from_file('tests/files/stimuli/pexels-zoorg-1000498.jpg')

    with pytest.raises(IndexError):
        image_stimulus.plot(999)


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_plot_returns_figure_and_axes(image_path):
    """Test that plot returns figure and axes."""
    image_stimulus = from_file(image_path)
    fig, ax = image_stimulus.plot(0)

    assert fig is not None
    assert ax is not None
<<<<<<< HEAD
    
    assert isinstance(fig, pyplot.Figure)
    assert isinstance(ax, pyplot.Axes)
    pyplot.close(fig)
=======

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
>>>>>>> ae958d8115e062a60909881240b21c93488752ca


@pytest.mark.parametrize(
    ('image_path'),
    (
        pytest.param('tests/files/stimuli/pexels-zoorg-1000498.jpg', id='image_path_str'),
        pytest.param(Path('tests/files/stimuli/pexels-zoorg-1000498.jpg'), id='image_path_Path'),
    ),
)
def test_multiple_stimuli(image_path):
    """Test ImageStimulus with multiple images."""

    images = [Path(image_path), Path(image_path)]
    image_stimulus = ImageStimulus(images=images)

    assert len(image_stimulus.images) == 2
    fig1, _ = image_stimulus.plot(0)
    assert fig1 is not None
<<<<<<< HEAD
    
    pyplot.close(fig1)
=======

    plt.close(fig1)
>>>>>>> ae958d8115e062a60909881240b21c93488752ca

    fig2, _ = image_stimulus.plot(1)
    assert fig2 is not None
    pyplot.close(fig2)


def test_from_file_returns_image_stimulus():
    """Test from_file returns ImageStimulus instance."""

    result = from_file('tests/files/stimuli/pexels-zoorg-1000498.jpg')
    assert isinstance(result, ImageStimulus)
    assert len(result.images) == 1


def test_from_files_returns_image_stimulus():
    """Test from_files returns ImageStimulus instance."""

    result = from_files('tests/files/', r'{book_name}-{page_num}-{line_num}.jpg')
    assert isinstance(result, ImageStimulus)
    assert len(result.images) >= 1
