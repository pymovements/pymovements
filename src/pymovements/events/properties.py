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
"""Provides functions for calculating event properties."""
from __future__ import annotations

from collections.abc import Callable

import polars as pl

EVENT_PROPERTIES: dict[str, Callable] = {}


def register_event_property(function: Callable) -> Callable:
    """Register a function as a valid property.

    Parameters
    ----------
    function: Callable
        Function to be registered as a valid property.

    Returns
    -------
    Callable
        The function that was passed as an argument.
    """
    EVENT_PROPERTIES[function.__name__] = function
    return function


@register_event_property
def amplitude(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Amplitude of an event.

    The amplitude is calculated as:

    .. math::
        \text{Amplitude} = \sqrt{(x_{\text{max}} - x_{\text{min}})^2 +
        (y_{\text{max}} - y_{\text{min}})^2}

    where :math:`(x_{\text{min}},\; x_{\text{max}})` and
    :math:`(y_{\text{min}},\; y_{\text{max}})` are the minimum and maximum values of the
    :math:`x` and :math:`y` components of the gaze positions during an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The amplitude of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    return (
        (x_position.max() - x_position.min()).pow(2)
        + (y_position.max() - y_position.min()).pow(2)
    ).sqrt()


@register_event_property
def dispersion(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Dispersion of an event.

    The dispersion is calculated as:

    .. math::
        \text{Dispersion} = x_{\text{max}} - x_{\text{min}} + y_{\text{max}} - y_{\text{min}}

    where :math:`(x_{\text{min}},\; x_{\text{max}})` and
    :math:`(y_{\text{min}},\; y_{\text{max}})` are the minimum and maximum values of the
    :math:`x` and :math:`y` components of the gaze positions during an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The dispersion of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    return x_position.max() - x_position.min() + y_position.max() - y_position.min()


@register_event_property
def disposition(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Disposition of an event.

    The disposition is calculated as:

    .. math::
        \text{Disposition} = \sqrt{(x_0 - x_n)^2 + (y_0 - y_n)^2}

    where :math:`x_0` and :math:`y_0` are the coordinates of the starting position and
    :math:`x_n` and :math:`y_n` are the coordinates of the ending position of an event.

    Parameters
    ----------
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The disposition of the event.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).list.get(0)
    y_position = pl.col(position_column).list.get(1)

    return (
        (x_position.head(n=1) - x_position.reverse().head(n=1)).pow(2)
        + (y_position.head(n=1) - y_position.reverse().head(n=1)).pow(2)
    ).sqrt()


@register_event_property
def duration() -> pl.Expr:
    """Duration of an event.

    The duration is defined as the difference between offset time and onset time.

    Returns
    -------
    pl.Expr
        The duration of the event.
    """
    return pl.col('offset') - pl.col('onset')


@register_event_property
def location(
        method: str = 'mean',
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    r"""Location of an event.

    For method ``mean`` the location is calculated as:

    .. math::
        \text{Location} = \frac{1}{n} \sum_{i=1}^n \text{position}_i

    For method ``median`` the location is calculated as:

    .. math::
        \text{Location} = \text{median} \left(\text{position}_1, \ldots,
         \text{position}_n \right)


    Parameters
    ----------
    method: str
        The centroid method to be used for calculation. Supported methods are ``mean``, ``median``.
        (default: 'mean')
    position_column: str
        The column name of the position tuples. (default: 'position')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The location of the event.

    Raises
    ------
    ValueError
        If method is not one of the supported methods.
    """
    if method not in {'mean', 'median'}:
        raise ValueError(
            f"Method '{method}' not supported. "
            f"Please choose one of the following: ['mean', 'median'].",
        )

    component_expressions = []
    for component in range(n_components):
        position_component = (
            pl.col(position_column)
            .list.slice(0, None)
            .list.get(component)
        )

        if method == 'mean':
            expression_component = position_component.mean()
        else:  # by exclusion this must be median
            expression_component = position_component.median()

        component_expressions.append(expression_component)

    # Not sure why first() is needed here, but an outer list is being created somehow.
    return pl.concat_list(component_expressions).first()


@register_event_property
def peak_velocity(
        *,
        velocity_column: str = 'velocity',
        n_components: int = 2,
) -> pl.Expr:
    r"""Peak velocity of an event.

    The peak velocity is calculated as:

    .. math::
        \text{Peak Velocity} = \max \left(\sqrt{v_x^2 + v_y^2} \right)

    where :math:`v_x` and :math:`v_y` are the velocity components in :math:`x` and :math:`y`
    direction, respectively.

    Parameters
    ----------
    velocity_column: str
        The column name of the velocity tuples. (default: 'velocity')
    n_components: int
        Number of positional components. Usually these are the two components yaw and pitch.
        (default: 2)

    Returns
    -------
    pl.Expr
        The peak velocity of the event.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_velocity = pl.col(velocity_column).list.get(0)
    y_velocity = pl.col(velocity_column).list.get(1)

    return (x_velocity.pow(2) + y_velocity.pow(2)).sqrt().max()


def _check_has_two_componenents(n_components: int) -> None:
    """Check that number of componenents is two.

    Parameters
    ----------
    n_components: int
        Number of components.
    """
    if n_components != 2:
        raise ValueError('data must have exactly two components')
