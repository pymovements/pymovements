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
"""Module for the ImageDataFrame."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot
import PIL.Image

from pymovements._utils._html import repr_html
from pymovements._utils._paths import get_filepaths
from pymovements._utils._strings import curly_to_regex


@repr_html()
class ImageStimulus:
    """A DataFrame for image stimulus.

    Parameters
    ----------
    images: list[Path]
        Image stimulus list.
    """

    def __init__(self, images: list[Path]) -> None:
        self.images = images

    def show(self, stimulus_id: int, origin: str = 'upper') -> None:
        """Show image stimulus.

        Parameters
        ----------
        stimulus_id: int
            Number of stimulus to be shown.
        origin: str
            Origin of the stimulus to be shown.
        """
        _draw_image_stimulus(self.images[stimulus_id], origin=origin, show=True)


def from_file(image_path: str | Path) -> ImageStimulus:
    """Load image stimulus from file.

    Parameters
    ----------
    image_path:  str | Path
        Path to file to be read.

    Returns
    -------
    ImageStimulus
        Returns the image stimulus file.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)

    return ImageStimulus(images=[image_path])


def from_files(path: str | Path, filename_format: str) -> ImageStimulus:
    """Load image stimulus from file.

    Parameters
    ----------
    path:  str | Path
        Path to directory with image stimulus files.
    filename_format:  str
        Format of the image stimulus file names.

    Returns
    -------
    ImageStimulus
        Returns the image stimulus file.
    """
    filenames = get_filepaths(
        path,
        regex=curly_to_regex(filename_format),
    )
    image_stimuli = []
    for filename in filenames:
        image_stimuli.append(filename)

    return ImageStimulus(image_stimuli)


def _draw_image_stimulus(
        image_stimulus: str | Path,
        origin: str = 'upper',
        show: bool = False,
        figsize: tuple[float, float] = (15, 10),
        extent: list[float] | None = None,
        fig: matplotlib.pyplot.figure | None = None,
        ax: matplotlib.pyplot.Axes | None = None,
) -> tuple[matplotlib.pyplot.figure, matplotlib.pyplot.Axes]:
    """Draw stimulus.

    Parameters
    ----------
    image_stimulus: str | Path
        Path to image stimulus.
    origin: str
        Origin how to draw the image.
    show: bool
        Boolean whether to show the image. (default: False)
    figsize: tuple[float, float]
        Size of the figure. (default: (15, 10))
    extent: list[float] | None
        Extent of image. (default: None)
    fig: matplotlib.pyplot.figure | None
        Matplotlib canvas. (default: None)
    ax: matplotlib.pyplot.Axes | None
        Matplotlib axes. (default: None)

    Returns
    -------
    fig: matplotlib.pyplot.figure
    ax: matplotlib.pyplot.Axes
    """
    img = PIL.Image.open(image_stimulus)
    if not fig:
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
    assert ax
    ax.imshow(img, origin=origin, extent=extent)
    if show:
        matplotlib.pyplot.show()
    return fig, ax
