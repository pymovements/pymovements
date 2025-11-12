# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Provides the Experiment class."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pymovements._utils import _checks
from pymovements._utils._html import repr_html
from pymovements.gaze import transforms_numpy
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.screen import Screen


@repr_html()
class Experiment:
    """Experiment class for holding experiment properties.

    Parameters
    ----------
    screen_width_px: int | None
        Screen width in pixels. (default: None)
    screen_height_px: int | None
        Screen height in pixels. (default: None)
    screen_width_cm: float | None
        Screen width in centimeters. (default: None)
    screen_height_cm: float | None
        Screen height in centimeters. (default: None)
    distance_cm: float | None
        Eye-to-screen distance in centimeters. If None, a `distance_column` must be provided
        in the `DatasetDefinition` or `Gaze`, which contains the eye-to-screen
        distance for each sample in millimeters. (default: None)
    origin: str | None
        Specifies the screen location of the origin of the pixel coordinate system.
        (default: None)
    sampling_rate: float | None
        Sampling rate in Hz. (default: None)
    screen: Screen | None
        Scree object for experiment. Mutually exclusive with explicit screen arguments.
        (default: None)
    eyetracker: EyeTracker | None
        EyeTracker object for experiment. Mutually exclusive with sampling_rate. (default: None)

    Examples
    --------
    >>> experiment = Experiment(
    ...     screen_width_px=1280,
    ...     screen_height_px=1024,
    ...     screen_width_cm=38.0,
    ...     screen_height_cm=30.0,
    ...     distance_cm=68.0,
    ...     origin='upper left',
    ...     sampling_rate=1000.0,
    ... )
    >>> print(experiment)
    Experiment(screen=Screen(width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0,
     distance_cm=68.0, origin='upper left'), eyetracker=EyeTracker(sampling_rate=1000.0, left=None,
      right=None, model=None, version=None, vendor=None, mount=None))

    We can also access the screen boundaries in degrees of visual angle via the
    :py:attr:`~pymovements.gaze.Screen` object. This only works if the
    `distance_cm` attribute is specified.

    >>> experiment.screen.x_min_dva# doctest:+ELLIPSIS
    -15.59...
    >>> experiment.screen.x_max_dva# doctest:+ELLIPSIS
    15.59...
    >>> experiment.screen.y_min_dva# doctest:+ELLIPSIS
    -12.42...
    >>> experiment.screen.y_max_dva# doctest:+ELLIPSIS
    12.42...
    """

    def __init__(
            self,
            screen_width_px: int | None = None,
            screen_height_px: int | None = None,
            screen_width_cm: float | None = None,
            screen_height_cm: float | None = None,
            distance_cm: float | None = None,
            origin: str | None = None,
            sampling_rate: float | None = None,
            *,
            screen: Screen | None = None,
            eyetracker: EyeTracker | None = None,
    ):
        _checks.check_is_mutual_exclusive(screen_width_px=screen_width_px, screen=screen)
        _checks.check_is_mutual_exclusive(screen_height_px=screen_height_px, screen=screen)
        _checks.check_is_mutual_exclusive(screen_width_cm=screen_width_cm, screen=screen)
        _checks.check_is_mutual_exclusive(screen_height_cm=screen_height_cm, screen=screen)
        _checks.check_is_mutual_exclusive(distance_cm=distance_cm, screen=screen)
        _checks.check_is_mutual_exclusive(origin=origin, screen=screen)
        _checks.check_is_mutual_exclusive(sampling_rate=sampling_rate, eyetracker=eyetracker)

        if screen is None:
            screen = Screen(
                width_px=screen_width_px,
                height_px=screen_height_px,
                width_cm=screen_width_cm,
                height_cm=screen_height_cm,
                distance_cm=distance_cm,
                origin=origin,
            )
        self.screen = screen

        if eyetracker is None:
            eyetracker = EyeTracker(sampling_rate=sampling_rate)
        self.eyetracker = eyetracker

        if self.sampling_rate is not None:
            _checks.check_is_greater_than_zero(sampling_rate=self.sampling_rate)

    @staticmethod
    def from_dict(dictionary: dict[str, Any]) -> Experiment:
        """Create an Experiment instance from a dictionary.

        Parameters
        ----------
        dictionary : dict[str, Any]
            A dictionary containing Experiment parameters.

        Notes
        -----
        The dictionary may contain nested dictionaries for 'screen' and 'eyetracker'.
        These will be automatically converted into Screen and EyeTracker instances.

        Examples
        --------
        Passing a flat dictionary:

        >>> experiment = Experiment.from_dict({
        ...     "screen_width_px": 1280,
        ...     "screen_height_px": 1024,
        ...     "screen_width_cm": 38.0,
        ...     "screen_height_cm": 30.0,
        ...     "distance_cm": 68.0,
        ...     "sampling_rate": 1000.0,
        ... })
        >>> print(experiment)
        Experiment(screen=Screen(width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0,
                                 distance_cm=68.0, origin=None),
                   eyetracker=EyeTracker(sampling_rate=1000.0, left=None, right=None,
                                        model=None, version=None, vendor=None, mount=None))

        The same result using nested dictionaries for `screen` and `eyetracker`:

        >>> experiment = Experiment.from_dict({
        ...     "screen": {
        ...         "width_px": 1280,
        ...         "height_px": 1024,
        ...         "width_cm": 38.0,
        ...         "height_cm": 30.0,
        ...         "distance_cm": 68.0,
        ...         "origin": "upper left"
        ...     },
        ...     "eyetracker": {
        ...         "sampling_rate": 1000.0
        ...     }
        ... })
        >>> print(experiment)
        Experiment(screen=Screen(width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0,
                                 distance_cm=68.0, origin='upper left'),
                   eyetracker=EyeTracker(sampling_rate=1000.0, left=None, right=None,
                                        model=None, version=None, vendor=None, mount=None))

        Returns
        -------
        Experiment
            An initialized Experiment instance.
        """
        dictionary = deepcopy(dictionary)
        screen = None
        eyetracker = None

        if 'screen' in dictionary:
            screen = Screen(**dictionary.pop('screen'))

        if 'eyetracker' in dictionary:
            eyetracker = EyeTracker(**dictionary.pop('eyetracker'))

        return Experiment(
            **dictionary,
            screen=screen,
            eyetracker=eyetracker,
        )

    @property
    def sampling_rate(self) -> float | None:
        """Get sampling rate of experiment."""
        return self.eyetracker.sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate: float | None = None) -> None:
        """Set sampling rate of experiment."""
        self.eyetracker.sampling_rate = sampling_rate

    def pos2vel(
            self,
            arr: list[float] | list[list[float]] | np.ndarray,
            method: str = 'smooth',
            **kwargs: int | float | str,
    ) -> np.ndarray:
        """Compute velocity time series from 2-dimensional position time series.

        Methods 'smooth', 'neighbors' and 'preceding' are adapted from
            Engbert et al.: Microsaccade Toolbox 0.9.

        Parameters
        ----------
        arr: list[float] | list[list[float]] | np.ndarray
            Continuous 2D position time series.
        method: str
            Computation method. See :func:`~transforms.pos2vel` for details. (default: 'smooth')
        **kwargs: int | float | str
            Additional keyword arguments used for savitzky golay method.

        Returns
        -------
        np.ndarray
            Velocity time series in input_unit / sec

        Raises
        ------
        ValueError
            If selected method is invalid, input array is too short for the
            selected method or the sampling rate is below zero

        Examples
        --------
        >>> experiment = Experiment(
        ...     screen_width_px=1280,
        ...     screen_height_px=1024,
        ...     screen_width_cm=38,
        ...     screen_height_cm=30,
        ...     distance_cm=68,
        ...     origin='upper left',
        ...     sampling_rate=1000.0,
        ... )
        >>> arr = [[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]
        >>> experiment.pos2vel(
        ...    arr=arr,
        ...    method="smooth",
        ... )
        array([[ 500.,  500.],
               [1000., 1000.],
               [1000., 1000.],
               [1000., 1000.],
               [1000., 1000.],
               [ 500.,  500.]])
        """
        assert self.sampling_rate is not None
        return transforms_numpy.pos2vel(
            arr=arr, sampling_rate=self.sampling_rate, method=method, **kwargs,
        )

    def __eq__(self: Experiment, other: Experiment) -> bool:
        """Compare equality to other Experiment."""
        return self.screen == other.screen and self.eyetracker == other.eyetracker

    def to_dict(
        self, *, exclude_none: bool = True,
    ) -> dict[str, Any | dict[str, str | float | None]]:
        """Convert the experiment instance into a dictionary.

        Parameters
        ----------
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.

        Returns
        -------
        dict[str, Any | dict[str, str | float | None]]
            Experiment as dictionary.
        """
        data: dict[str, dict[str, str | float | None]] = {}

        if self.eyetracker or not exclude_none:
            data['eyetracker'] = self.eyetracker.to_dict(exclude_none=exclude_none)
        if self.screen or not exclude_none:
            data['screen'] = self.screen.to_dict(exclude_none=exclude_none)

        return data

    def __str__(self: Experiment) -> str:
        """Return Experiment string."""
        return (
            f"{type(self).__name__}(screen={self.screen}, "
            f"eyetracker={self.eyetracker})"
        )

    def __bool__(self) -> bool:
        """Return True if the experiment has data defined, else False."""
        return not all(not value for value in self.__dict__.values())

    def to_yaml(
        self,
        path: str | Path,
        *,
        exclude_none: bool = True,
    ) -> None:
        """Save an experiment to a YAML file.

        Parameters
        ----------
        path: str | Path
            Path where to save the YAML file to.
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.
        """
        data = self.to_dict(exclude_none=exclude_none)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
