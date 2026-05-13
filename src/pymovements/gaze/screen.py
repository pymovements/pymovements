# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Provides the Screen class."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np

from pymovements._utils import _checks
from pymovements._utils._html import repr_html
from pymovements.gaze import transforms_numpy


@repr_html()
@dataclass
class Screen:
    """Screen class for holding screen properties.

    Attributes
    ----------
    distance_cm: float | None
        Eye-to-screen distance in centimeters. If None, a `distance_column` must be provided
        in the `DatasetDefinition` or `Gaze`, which contains the eye-to-screen
        distance for each sample in millimeters.
        (default: None)
    origin: str | None
        Specifies the screen location of the origin of the pixel
        coordinate system.
        (default: None)

    Parameters
    ----------
    width_px: int | None
        Screen width in pixels.
        (default: None)
    height_px: int | None
        Screen height in pixels.
        (default: None)
    width_cm: float | None
        Screen width in centimeters.
        (default: None)
    height_cm: float | None
        Screen height in centimeters.
        (default: None)
    distance_cm: float | None
        Eye-to-screen distance in centimeters. If None, a `distance_column` must be provided
        in the `DatasetDefinition` or `Gaze`, which contains the eye-to-screen
        distance for each sample in millimeters.
        (default: None)
    origin: str | None
        Specifies the screen location of the origin of the pixel
        coordinate system.
        (default: None)
    resolution: tuple[int | None, int | None] | None
        Screen resolution in pixels as tuple of width and height.
        (default: None)
    size: tuple[float | None, float | None] | None
        Screen size in centimeters as tuple of width and height.
        (default: None)

    Examples
    --------
    >>> screen = Screen(
    ...     width_px=1280,
    ...     height_px=1024,
    ...     width_cm=38.0,
    ...     height_cm=30.0,
    ...     distance_cm=68.0,
    ...     origin='upper left',
    ... )
    >>> print(screen)
    Screen(width_px=1280, height_px=1024, width_cm=38.0,
    height_cm=30.0, distance_cm=68.0, origin='upper left')

    We can also access the screen boundaries in degrees of visual angle. This only works if the
    `distance_cm` attribute is specified.

    >>> screen.x_min_dva# doctest:+ELLIPSIS
    -15.59...
    >>> screen.x_max_dva# doctest:+ELLIPSIS
    15.59...
    >>> screen.y_min_dva# doctest:+ELLIPSIS
    -12.42...
    >>> screen.y_max_dva# doctest:+ELLIPSIS
    12.42...

    """

    distance_cm: float | None = None
    origin: str | None = None

    _width_px: int | None = None
    _height_px: int | None = None
    _width_cm: float | None = None
    _height_cm: float | None = None

    def __init__(
        self,
        width_px: int | None = None,
        height_px: int | None = None,
        width_cm: float | None = None,
        height_cm: float | None = None,
        distance_cm: float | None = None,
        origin: str | None = None,
        *,
        resolution: tuple[int | None, int | None] | None = None,
        size: tuple[float | None, float | None] | None = None,
    ):
        # Check mutual exclusivity.
        _checks.check_is_mutual_exclusive(width_px=width_px, resolution=resolution)
        _checks.check_is_mutual_exclusive(height_px=height_px, resolution=resolution)
        _checks.check_is_mutual_exclusive(width_cm=width_cm, size=size)
        _checks.check_is_mutual_exclusive(height_cm=height_cm, size=size)

        # Build the tuple from the individual px/cm arguments if not provided.
        self.resolution = resolution if resolution is not None else (width_px, height_px)
        self.size = size if size is not None else (width_cm, height_cm)
        self.distance_cm = distance_cm
        self.origin = origin

    def __post_init__(self) -> None:
        """Check fields for validity."""
        if self.width_px is not None:
            _checks.check_is_greater_than_zero(width_px=self.width_px)

        if self.height_px is not None:
            _checks.check_is_greater_than_zero(height_px=self.height_px)

        if self.width_cm is not None:
            _checks.check_is_greater_than_zero(width_cm=self.width_cm)

        if self.height_cm is not None:
            _checks.check_is_greater_than_zero(height_cm=self.height_cm)

        if self.distance_cm is not None:
            _checks.check_is_greater_than_zero(distance_cm=self.distance_cm)

    @property
    def width_px(self) -> int | None:
        """Width of screen in pixels."""
        return self._width_px

    @width_px.setter
    def width_px(self, value: int | None = None) -> None:
        if value is not None and not isinstance(value, Number):
            raise TypeError(f"'width_px' must be a number but is of type {type(value).__name__}")
        if value is not None and value <= 0:
            raise ValueError(f"'width_px' must be a positive number but is: {value}")
        self._width_px = value

    @property
    def height_px(self) -> int | None:
        """Height of screen in pixels."""
        return self._height_px

    @height_px.setter
    def height_px(self, value: int | None = None) -> None:
        if value is not None and not isinstance(value, Number):
            raise TypeError(f"'height_px' must be a number but is of type {type(value).__name__}")
        if value is not None and value <= 0:
            raise ValueError(f"'height_px' must be a positive number but is: {value}")
        self._height_px = value

    @property
    def resolution(self) -> tuple[int | None, int | None] | None:
        """Resolution of screen in pixels (width, height)."""
        if self.width_px is None and self.height_px is None:
            return None
        return (self.width_px, self.height_px)

    @resolution.setter
    def resolution(self, values: Sequence[int | None] | None) -> None:
        if values is None:
            self.width_px = None
            self.height_px = None
            return

        if not isinstance(values, Sequence):
            raise TypeError(
                f"'resolution' must be a sequence but is of type {type(values).__name__}",
            )

        if len(values) != 2:
            raise ValueError(f"'resolution' must be of length 2 but its length is {len(values)}")

        self.width_px = values[0]
        self.height_px = values[1]

    @property
    def width_cm(self) -> float | None:
        """Width of screen in cm."""
        return self._width_cm

    @width_cm.setter
    def width_cm(self, value: float | None = None) -> None:
        if value is not None and not isinstance(value, Number):
            raise TypeError(f"'width_cm' must be a number but is of type {type(value).__name__}")
        if value is not None and value <= 0:
            raise ValueError(f"'width_cm' must be a positive number but is: {value}")
        self._width_cm = value

    @property
    def height_cm(self) -> float | None:
        """Height of screen in cm."""
        return self._height_cm

    @height_cm.setter
    def height_cm(self, value: float | None = None) -> None:
        if value is not None and not isinstance(value, Number):
            raise TypeError(f"'height_cm' must be a number but is of type {type(value).__name__}")
        if value is not None and value <= 0:
            raise ValueError(f"'height_cm' must be a positive number but is: {value}")
        self._height_cm = value

    @property
    def size(self) -> tuple[float | None, float | None] | None:
        """Size of screen in centimeters (width, height)."""
        if self.width_cm is None and self.height_cm is None:
            return None
        return (self.width_cm, self.height_cm)

    @size.setter
    def size(self, values: Sequence[float | None] | None) -> None:
        if values is None:
            self.width_cm = None
            self.height_cm = None
            return

        if not isinstance(values, Sequence):
            raise TypeError(f"'size' must be a sequence but is of type {type(values).__name__}")

        if len(values) != 2:
            raise ValueError(f"'size' must be of length 2 but its length is {len(values)}")

        self.width_cm = values[0]
        self.height_cm = values[1]

    @property
    def x_max_dva(self) -> float:
        """Maximum screen x-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        _checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                self.width_px - 1,
                screen_px=self.width_px,
                screen_cm=self.width_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def y_max_dva(self) -> float:
        """Maximum screen y-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        _checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                self.height_px - 1,
                screen_px=self.height_px,
                screen_cm=self.height_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def x_min_dva(self) -> float:
        """Minimum screen x-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        _checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                0,
                screen_px=self.width_px,
                screen_cm=self.width_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def y_min_dva(self) -> float:
        """Minimum screen y-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        _checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                0,
                screen_px=self.height_px,
                screen_cm=self.height_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    def pix2deg(
            self,
            arr: float | list[float] | list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """Convert pixel screen coordinates to degrees of visual angle.

        Parameters
        ----------
        arr: float | list[float] | list[list[float]] | np.ndarray
            Pixel coordinates to transform into degrees of visual angle

        Returns
        -------
        np.ndarray
            Coordinates in degrees of visual angle

        Raises
        ------
        ValueError
            If positions aren't two-dimensional.

        Examples
        --------
        >>> arr = [(123.0, 865.0)]
        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ...     origin='upper left',
        ... )
        >>> screen.pix2deg(arr=arr)
        array([[-12.70732231, 8.65963972]])

        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ...     origin='center',
        ... )
        >>> screen.pix2deg(arr=arr)
        array([[ 3.07379946, 20.43909054]])
        """
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        _checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return transforms_numpy.pix2deg(
            arr=arr,
            screen_px=(self.width_px, self.height_px),
            screen_cm=(self.width_cm, self.height_cm),
            distance_cm=self.distance_cm,
            origin=self.origin,
        )

    def _check_numerical_attribute(self, key: str) -> None:
        """Check if numerical attribute is not None and greater than zero."""
        value = getattr(self, key, None)
        _checks.check_is_not_none(**{key: value})
        assert isinstance(value, (int, float))
        _checks.check_is_greater_than_zero(**{key: value})

    def to_dict(
            self,
            *,
            exclude_none: bool = True,
            exclude_private: bool = True,
            prefer_resolution: bool = False,
            prefer_size: bool = False,
    ) -> dict[str, Any]:
        """Convert the Screen instance into a dictionary.

        Parameters
        ----------
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.
            (default: True)
        exclude_private: bool
            Exclude attributes that start with ``_``.
            (default: True)
        prefer_resolution: bool
            If ``True`` include ``resolution`` instead of ``width_px`` and ``height_px`` in output
            dictionary.
            (default: False)
        prefer_size: bool
            If ``True`` include ``size`` instead of ``width_cm`` and ``height_cm`` in output
            dictionary.
            (default: False)

        Returns
        -------
        dict[str, Any]
            Screen as dictionary.
        """
        data = asdict(self)  # this does not include properties, just explicit attributes.

        # Include properties in dictionary.
        if prefer_resolution:
            data['resolution'] = self.resolution
        else:
            data['width_px'] = self.width_px
            data['height_px'] = self.height_px
        if prefer_size:
            data['size'] = self.size
        else:
            data['width_cm'] = self.width_cm
            data['height_cm'] = self.height_cm

        # Delete private fields from dictionary.
        if exclude_private:
            # we need a separate list of keys here or else we get a
            # RuntimeError: dictionary changed size during iteration
            for key in list(data.keys()):
                if key.startswith('_'):
                    del data[key]

        # Delete fields that evaluate to False (False, None, [], {})
        if exclude_none:
            for key, value in list(data.items()):
                if not isinstance(value, (bool, int, float)) and not value:
                    del data[key]

        return data

    def __bool__(self) -> bool:
        """Return True if the screen has data defined, else False."""
        return not all(not value for value in self.__dict__.values())

    def __str__(self) -> str:
        """Return Screen string."""
        # Make sure a string is enclosed by ' characters.
        origin_str = f"'{self.origin}'" if isinstance(self.origin, str) else str(self.origin)

        return (
            f'{type(self).__name__}('
            f'width_px={self.width_px}, '
            f'height_px={self.height_px}, '
            f'width_cm={self.width_cm}, '
            f'height_cm={self.height_cm}, '
            f'distance_cm={self.distance_cm}, '
            f"origin={origin_str})"
        )
