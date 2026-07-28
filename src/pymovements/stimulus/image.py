# Copyright (c) 2023-2026 The pymovements Project Authors
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
from typing import Any

import matplotlib.pyplot as plt
import PIL.Image
from deprecated.sphinx import deprecated

from pymovements._utils._html import repr_html
from pymovements._utils._paths import get_filepaths
from pymovements._utils._strings import curly_to_regex


@repr_html(['images', 'metadata'])
class ImageStimulus:
    """A DataFrame for image stimulus.

    Parameters
    ----------
    images: list[Path]
        Image stimulus list.
    origin : str
        Image origin position for plotting.
        (default: 'upper')
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata.
        (default: None)
    """

    def __init__(
            self,
            images: list[Path],
            origin: str = 'upper',
            metadata: dict[str, Any] | None = None,
    ) -> None:
        self.images = images
        self.origin = origin
        self.metadata = metadata if metadata is not None else {}

    @deprecated(
        reason='Please use ImageStimulus.plot() instead. '
               'This method will be removed in v0.33.0.',
        version='v0.28.0',
    )
    def show(self, stimulus_id: int, origin: str = 'upper') -> None:
        """Display an image stimulus.

        .. deprecated:: v0.28.0
           Please use :py:meth:`~pymovements.stimulus.ImageStimulus.plot` instead.
           This method will be removed in v0.33.0.

        Parameters
        ----------
        stimulus_id : int
            Index of the stimulus to display.
        origin : str
            Image origin position for plotting.
            (default: 'upper')
        """
        self.origin = origin

        self.plot(stimulus_id)

        plt.show()

    def plot(
        self,
        stimulus_id: int,
        *,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot an image stimulus.

        Parameters
        ----------
        stimulus_id : int
            Index of the stimulus to plot.
        ax : plt.Axes | None
            Axes to draw the image on.
            (default: None)

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Figure and axes containing the plot.
        """
        if ax is not None:
            fig = ax.figure
        else:
            fig = None

        return _draw_image_stimulus(self.images[stimulus_id], fig=fig, ax=ax, origin=self.origin)

    @staticmethod
    def from_file(path: str | Path, metadata: dict[str, Any] | None = None) -> ImageStimulus:
        """Load image stimulus from file.

        Parameters
        ----------
        path:  str | Path
            Path to image file to be read.
        metadata: dict[str, Any] | None
            Dictionary containing additional metadata. (default: None)

        Returns
        -------
        ImageStimulus
            Returns an ImageStimulus initialized with the image stimulus file.
        """
        return ImageStimulus(images=[Path(path)], metadata=metadata)


def from_file(image_path: str | Path, metadata: dict[str, Any] | None = None) -> ImageStimulus:
    """Load image stimulus from file.

    Parameters
    ----------
    image_path:  str | Path
        Path to file to be read.
    metadata: dict[str, Any] | None
        Dictionary containing additional metadata. (default: None)

    Returns
    -------
    ImageStimulus
        Returns the image stimulus file.
    """
    return ImageStimulus.from_file(path=image_path, metadata=metadata)


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
    filenames = get_filepaths(path, regex=curly_to_regex(filename_format))
    return ImageStimulus(list(filenames))


def _draw_image_stimulus(
        image_stimulus: str | Path,
        origin: str = 'upper',
        show: bool = False,
        figsize: tuple[float, float] = (15, 10),
        extent: list[float] | None = None,
        fig: plt.figure | None = None,
        ax: plt.Axes | None = None,
) -> tuple[plt.figure, plt.Axes]:
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
    fig: plt.figure | None
        Matplotlib canvas. (default: None)
    ax: plt.Axes | None
        Matplotlib axes. (default: None)

    Returns
    -------
    fig: plt.figure
    ax: plt.Axes
    """
    try:
        img = PIL.Image.open(image_stimulus)
    except PIL.UnidentifiedImageError as exception:
        raise ValueError(
            f"Unsupported image file '{image_stimulus}'. "
            "Use 'PIL.features.pilinfo()' to get an overview of supported types.",
        ) from exception

    if not fig:
        fig, ax = plt.subplots(figsize=figsize)
    assert ax
    ax.imshow(img, origin=origin, extent=extent)
    if show:
        plt.show()
    return fig, ax
