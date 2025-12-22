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
"""Transforms module."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from pymovements._utils import _checks


def pix2deg(
        arr: float | list[float] | list[list[float]] | np.ndarray,
        screen_px: float | list[float] | tuple[float, float] | np.ndarray,
        screen_cm: float | list[float] | tuple[float, float] | np.ndarray,
        distance_cm: float,
        origin: str,
) -> np.ndarray:
    """Convert pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    arr: float | list[float] | list[list[float]] | np.ndarray
        Pixel coordinates to transform into degrees of visual angle
    screen_px: float | list[float] | tuple[float, float] | np.ndarray
        Screen dimension in pixels
    screen_cm: float | list[float] | tuple[float, float] | np.ndarray
        Screen dimension in centimeters
    distance_cm: float
        Eye-to-screen distance in centimeters
    origin : str
        Specifies the screen location of the origin of the pixel coordinate system. Valid values
        are: center, upper left.

    Returns
    -------
    np.ndarray
        Coordinates in degrees of visual angle

    Raises
    ------
    TypeError
        If arr is None.
    ValueError
        If dimension screen_px or screen_cm don't match dimension of arr.
        If screen_px or screen_cm or one of its elements is zero.
        If distance_cm is zero.
        If origin value is not supported.

    Examples
    --------
    >>> pix2deg(
    ...    arr=[(123.0, 865.0)],
    ...    screen_px=(1280, 1024),
    ...    screen_cm=(38.0, 30.0),
    ...    distance_cm=68.0,
    ...    origin='upper left',
    ... )
    array([[-12.70732231, 8.65963972]])

    >>> pix2deg(
    ...    arr=[(123.0, 865.0)],
    ...    screen_px=(1280, 1024),
    ...    screen_cm=(38.0, 30.0),
    ...    distance_cm=68.0,
    ...    origin='center',
    ... )
    array([[ 3.07379946, 20.43909054]])

    If the eye-movement of both eyes was recorded, the input array must contain
    first the x- and y-coordinates of one eye and then the x- and y-coordinates
    of the other eye as shown in the example below.

    >>> x_left, y_left = 123.0, 865.0
    >>> x_right, y_right = 183.0, 865.0
    >>> arr = [(x_left, y_left, x_right, y_right)]
    >>> pix2deg(
    ...    arr=arr,
    ...    screen_px=(1280, 1024),
    ...    screen_cm=(38.0, 30.0),
    ...    distance_cm=68.0,
    ...    origin='center',
    ... )
    array([[ 3.07379946, 20.43909054,  4.56790364, 20.43909054]])
    """
    if arr is None:
        raise TypeError('arr must not be None')

    _checks.check_no_zeros(screen_px, 'screen_px')
    _checks.check_no_zeros(screen_cm, 'screen_px')
    _checks.check_no_zeros(distance_cm, 'distance_cm')

    arr = np.array(arr)
    screen_px = np.array(screen_px)
    screen_cm = np.array(screen_cm)

    # Check basic arr dimensions.
    if arr.ndim not in [0, 1, 2]:
        raise ValueError(
            'Number of dimensions of arr must be either 0, 1 or 2'
            f' (arr.ndim: {arr.ndim})',
        )
    if arr.ndim == 2 and arr.shape[-1] not in [1, 2, 4, 6]:
        raise ValueError(
            'Last coord dimension must have length 1, 2, 4 or 6.'
            f' (arr.shape: {arr.shape})',
        )

    # check if arr dimensions match screen_px and screen_cm dimensions
    if arr.ndim in {0, 1}:
        if screen_px.ndim != 0 and screen_px.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_px is not')
        if screen_cm.ndim != 0 and screen_cm.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 2:
        if screen_px.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_px is not')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 4:
        if screen_px.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_px is not 2-dimensional')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_cm is not 2-dimensional')

        # We have binocular data. Double tile screen parameters.
        screen_px = np.tile(screen_px, 2)
        screen_cm = np.tile(screen_cm, 2)

    if arr.ndim != 0 and arr.shape[-1] == 6:
        if screen_px.shape != (2,):
            raise ValueError('arr is 6-dimensional, but screen_px is not 2-dimensional')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 6-dimensional, but screen_cm is not 2-dimensional')

        # We have binocular and cyclopian data. Triple tile screen parameters.
        screen_px = np.tile(screen_px, 3)
        screen_cm = np.tile(screen_cm, 3)

    # Compute eye-to-screen-distance in pixels.
    distance_px = distance_cm * (screen_px / screen_cm)

    # If pixel coordinate system is not centered, shift pixel coordinate to the center.
    if origin == 'upper left':
        arr = arr - (screen_px - 1) / 2
    elif origin != 'center':
        raise ValueError(f'origin {origin} is not supported.')

    # 180 / pi transforms arc measure to degrees.
    return np.arctan2(arr, distance_px) * 180 / np.pi


def pos2acc(
        arr: list[float] | list[list[float]] | np.ndarray,
        sampling_rate: float,
        window_length: int = 7,
        degree: int = 2,
        mode: str = 'interp',
        cval: float = 0.0,
) -> np.ndarray:
    """Compute velocity time series from 2-dimensional position time series.

    Parameters
    ----------
    arr: list[float] | list[list[float]] | np.ndarray
        Continuous 2D position time series
    sampling_rate: float
        Sampling rate of input time series.
    window_length: int
        The window size to use. (default: 7)
    degree: int
        The degree of the polynomial to use. (default: 2)
    mode: str
        The padding mode to use. (default: 'interp')
    cval: float
        A constant value for padding. (default: 0.0)

    Returns
    -------
    np.ndarray
        Velocity time series in input_unit / sec

    Examples
    --------
    >>> arr = [(0., 0.), (1., 1.), (4., 4.), (9., 9.), (16., 16.), (25., 25.)]
    >>> pos2acc(
    ...    arr=arr,
    ...    sampling_rate=1000,
    ...    window_length=5,
    ... )
    array([[2000., 2000.],
           [2000., 2000.],
           [2000., 2000.],
           [2000., 2000.],
           [2000., 2000.],
           [2000., 2000.]])
    """
    if sampling_rate <= 0:
        raise ValueError('sampling_rate needs to be above zero')

    # make sure that we're operating on a numpy array
    arr = np.array(arr)

    if arr.ndim not in [1, 2]:
        raise ValueError(
            'arr needs to have 1 or 2 dimensions (are: {arr.ndim = })',
        )

    acc = np.zeros(arr.shape)

    # transform to velocities
    if arr.ndim == 1:
        acc = savgol_filter(
            x=arr,
            deriv=2,
            window_length=window_length,
            polyorder=degree,
            mode=mode,
            cval=cval,
        )
    else:  # we already checked for error cases

        for channel_id in range(arr.shape[1]):
            acc[:, channel_id] = savgol_filter(
                x=arr[:, channel_id],
                deriv=2,
                window_length=window_length,
                polyorder=degree,
                mode=mode,
                cval=cval,
            )
    acc *= sampling_rate

    return acc


def pos2vel(
        arr: list[float] | list[list[float]] | np.ndarray,
        sampling_rate: float = 1000,
        method: str = 'smooth',
        **kwargs: int | float | str,
) -> np.ndarray:
    """Compute velocity time series from 2-dimensional position time series.

    Methods 'smooth', 'neighbors' and 'preceding' are adapted from
        Engbert et al.: Microsaccade Toolbox 0.9.

    Parameters
    ----------
    arr: list[float] | list[list[float]] | np.ndarray
        Continuous 2D position time series
    sampling_rate: float
        Sampling rate of input time series. (default: 1000)
    method: str
        Following methods are available:
        * *smooth*: velocity is calculated from the difference of the mean values
        of the subsequent two samples and the preceding two samples
        * *neighbors*: velocity is calculated from difference of the subsequent
        sample and the preceding sample
        * *preceding*: velocity is calculated from the difference of the current
        sample to the preceding sample
        (default: 'smooth')
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
    >>> arr = [(0., 0.), (1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.)]
    >>> pos2vel(
    ...    arr=arr,
    ...    sampling_rate=1000,
    ...    method="smooth",
    ... )
    array([[ 500.,  500.],
           [1000., 1000.],
           [1000., 1000.],
           [1000., 1000.],
           [1000., 1000.],
           [ 500.,  500.]])
    """
    if sampling_rate <= 0:
        raise ValueError('sampling_rate needs to be above zero')

    # make sure that we're operating on a numpy array
    arr = np.array(arr)

    if arr.ndim not in [1, 2]:
        raise ValueError(
            'arr needs to have 1 or 2 dimensions (are: {arr.ndim = })',
        )
    if method == 'smooth' and arr.shape[0] < 6:
        raise ValueError(
            'arr has to have at least 6 elements for method "smooth"',
        )
    if method == 'neighbors' and arr.shape[0] < 3:
        raise ValueError(
            'arr has to have at least 3 elements for method "neighbors"',
        )
    if method == 'preceding' and arr.shape[0] < 2:
        raise ValueError(
            'arr has to have at least 2 elements for method "preceding"',
        )
    if method != 'savitzky_golay' and kwargs:
        raise ValueError(
            'selected method doesn\'t support any additional kwargs',
        )

    N = arr.shape[0]
    v = np.zeros(arr.shape)

    valid_methods = ['smooth', 'neighbors', 'preceding', 'savitzky_golay']
    if method == 'smooth':
        # center is N - 2
        moving_avg = arr[4:N] + arr[3:N - 1] - arr[1:N - 3] - arr[0:N - 4]
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        v[2:N - 2] = moving_avg * sampling_rate / 6

        # for second and second last sample:
        # calculate vocity from preceding and subsequent sample
        v[1] = (arr[2] - arr[0]) * sampling_rate / 2
        v[N - 2] = (arr[N - 1] - arr[N - 3]) * sampling_rate / 2

        # for first and second sample:
        # calculate velocity from current and neighboring sample
        v[0] = (arr[1] - arr[0]) * sampling_rate / 2
        v[N - 1] = (arr[N - 1] - arr[N - 2]) * sampling_rate / 2

    elif method == 'neighbors':
        # window size is two, so we need to divide by two
        v[1:N - 1] = (arr[2:N] - arr[0:N - 2]) * sampling_rate / 2

    elif method == 'preceding':
        v[1:N] = (arr[1:N] - arr[0:N - 1]) * sampling_rate

    elif method == 'savitzky_golay':
        # transform to velocities
        if arr.ndim == 1:
            v = savgol_filter(x=arr, deriv=1, **kwargs)
        else:  # we already checked for error cases

            for channel_id in range(arr.shape[1]):
                v[:, channel_id] = savgol_filter(
                    x=arr[:, channel_id], deriv=1, **kwargs,
                )
        v *= sampling_rate

    else:
        raise ValueError(
            f'Method needs to be in {valid_methods}'
            f' (is: {method})',
        )

    return v


def norm(arr: np.ndarray, axis: int | None = None) -> np.ndarray | Any:
    r"""Take the norm of an array.

    The norm is defined by :math:`\sqrt{x^2 + y^2}` with :math:`x` being the yaw component and
    :math:`y` being the pitch component of a coordinate.

    Parameters
    ----------
    arr: np.ndarray
        Input velocity sequence.
    axis: int | None
        Set axis to take norm. If None it is inferred from arr.shape. (default: None)

    Returns
    -------
    np.ndarray | Any
        Norm of input array.

    Raises
    ------
    ValueError
        If no axis is given but the array dimensions are lager than 3.
        In that case the axis cannot be inferred.

    Examples
    --------
    >>> arr = np.array([[1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.]])
    >>> norm(arr=arr)
    array([1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356,
           1.41421356])
    """
    if axis is None:
        # for single vector and array of vectors the axis is 0
        # shape is assumed to be either (2, ) or (2, sequence_length)
        if arr.ndim in {1, 2}:
            axis = 0

        # for batched array of vectors, the
        # shape is assumed to be (n_batches, 2, sequence_length)
        elif arr.ndim == 3:
            axis = 1

        else:
            raise ValueError(
                f'Axis can not be inferred in case of more than 3 '
                f'input array dimensions (arr.shape={arr.shape}).'
                f' Either reduce the number of input array '
                f'dimensions or specify `axis` explicitly.',
            )

    return np.linalg.norm(arr, axis=axis)


def split(
        arr: np.ndarray,
        window_size: int,
        keep_padded: bool = True,
) -> np.ndarray:
    """Split sequence into subsequences of equal length.

    Parameters
    ----------
    arr: np.ndarray
        Input sequence of shape (N, L, C).
    window_size: int
        size of subsequences
    keep_padded: bool
        If True, last subsequence (if not of length `window_size`) is kept in the output array and
        padded with np.nan. (default: True)

    Returns
    -------
    np.ndarray
        Array of split sequences with new window length of shape (N', L', C)

    Examples
    --------
    We first create an array with `2` channels and a sequence length of `13`.

    >>> arr = np.ones((1, 13, 2))
    >>> arr.shape
    (1, 13, 2)

    We now split the original array into three subsequences of length `5`.

    >>> arr_split = split(
    ...    arr=arr,
    ...    window_size=5,
    ...    keep_padded=True,
    ... )
    >>> arr_split.shape
    (3, 5, 2)

    The last subsequence is padded with `nan` to have a length of `5`.

    >>> arr_split[-1]
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [nan, nan],
           [nan, nan]])

    We can also drop any remaining sequences that would need padding by passing
    ``keep_padded=False``.

    >>> arr_split = split(
    ...    arr=arr,
    ...    window_size=5,
    ...    keep_padded=False,
    ... )
    >>> arr_split.shape
    (2, 5, 2)

    """
    n, rest = np.divmod(arr.shape[1], window_size)

    if rest > 0 and keep_padded:
        n_rows = arr.shape[0] * (n + 1)
    else:
        n_rows = arr.shape[0] * n

    arr_split = np.nan * np.ones((n_rows, window_size, arr.shape[2]))

    idx = 0
    for arr_instance in arr:
        # Create an array of indicies where sequence will be split.
        split_indices = np.arange(start=window_size, stop=(n + 1) * window_size, step=window_size)

        # The length of the last element in list will be equal to rest.
        arr_split_list = np.split(arr_instance, split_indices)

        # Put the first n elements of split list into output array.
        arr_split[idx:idx + n] = arr_split_list[:-1]
        idx += n

        if rest > 0 and keep_padded:
            # Insert last split, remaining padded samples are already np.nan.
            arr_split[idx, :rest] = arr_split_list[-1]
            idx += 1

    return arr_split


def downsample(
        arr: np.ndarray,
        factor: int,
) -> np.ndarray:
    """Downsamples array by integer factor.

    Parameters
    ----------
    arr: np.ndarray
        sequence to be downsampled
    factor: int
        factor to be downsampled with

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> arr = np.array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5.])
    >>> downsample(arr=arr, factor=2)
    array([0., 1., 2., 3., 4., 5.])

    >>> arr2 = np.array([(0., 0.), (1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.)])
    >>> downsample(arr=arr2, factor=2)
    array([[0., 0.],
           [2., 2.],
           [4., 4.]])
    """
    sequence_length = arr.shape[0]
    select = [i % factor == 0 for i in range(sequence_length)]

    return arr[select].copy()


def consecutive(arr: np.ndarray) -> list[np.ndarray]:
    """Split array into groups of consecutive numbers.

    Parameters
    ----------
    arr : np.ndarray
        Array to be split into groups of consecutive numbers.

    Returns
    -------
    list[np.ndarray]
        List of arrays with consecutive numbers.

    Example
    -------
    >>> arr = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    >>> consecutive(arr)
    [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
