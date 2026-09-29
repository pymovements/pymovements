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
"""Registry for correctors that do not fit the drift-algorithm shape.

The algorithms in :py:data:`~pymovements.events.correction.ALL_DRIFT_ALGORITHMS` return a polars
expression, which carries no state and never sees the trial. A corrector that has to hold
something -- precomputed geometry, a fitted parameter, loaded weights -- needs a different shape
and is registered here, sharing the ``algorithm=`` namespace of
:py:func:`~pymovements.events.correction.correct_fixations`.

The registry holds **factories, not instances**: registering a name constructs nothing and
imports none of the corrector's dependencies. It never calls what a factory returns.
"""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

import polars as pl

#: A corrector handed one trial at a time, called as
#: ``corrector(fixations, aois, location_column=...)``. It returns one ``[x, y]`` list per
#: fixation, or ``None`` to decline the trial, which skips it with a warning as a drift
#: algorithm would. Anything callable qualifies; a stateful corrector is an object whose
#: ``__call__`` has this shape, built once per
#: :py:func:`~pymovements.events.correction.correct_fixations` call rather than per trial.
TrialCorrector = Callable[..., pl.Series | None]

_CORRECTORS: dict[str, Callable[..., TrialCorrector]] = {}
_RESERVED_NAMES: set[str] = set()


def reserve_names(names: Iterable[str]) -> None:
    """Mark names as taken by drift algorithms, so a registered corrector cannot shadow them.

    Called by :py:mod:`pymovements.events.correction.fixation_correction` once its algorithm
    table is built. The dependency runs that way round on purpose: the registry must not
    import the module that imports it.

    Parameters
    ----------
    names: Iterable[str]
        Names that registered correctors must not use.
    """
    _RESERVED_NAMES.update(names)


def register_corrector(
        factory: Callable[..., TrialCorrector],
) -> Callable[..., TrialCorrector]:
    """Register a factory that builds a trial corrector, under the factory's own name.

    Meant as a decorator. The factory is called when a correction runs, not at import time.

    Parameters
    ----------
    factory: Callable[..., TrialCorrector]
        Callable returning a :py:data:`TrialCorrector`. Its ``__name__`` becomes the name
        passed as ``algorithm=``.

    Returns
    -------
    Callable[..., TrialCorrector]
        The factory that was passed in, so this works as a decorator.

    Raises
    ------
    ValueError
        If the name is already taken, here or by a drift algorithm.
    """
    name = factory.__name__
    if name in _CORRECTORS:
        raise ValueError(f'a corrector named {name!r} is already registered')
    if name in _RESERVED_NAMES:
        raise ValueError(
            f'{name!r} is already a drift algorithm; registered correctors and drift '
            'algorithms share one namespace so that algorithm= takes either kind',
        )
    _CORRECTORS[name] = factory
    return factory


def is_registered_corrector(name: str) -> bool:
    """Return whether a name selects a registered corrector rather than a drift algorithm.

    Parameters
    ----------
    name: str
        The name passed as ``algorithm=``.

    Returns
    -------
    bool
        True if a corrector is registered under that name.
    """
    return name in _CORRECTORS


def all_registered_correctors() -> list[str]:
    """Return the names of all registered correctors.

    Returns
    -------
    list[str]
        Registered names, in registration order.
    """
    return list(_CORRECTORS)


def build_corrector(name: str, **kwargs: Any) -> TrialCorrector:
    """Build the corrector registered under a name.

    Parameters
    ----------
    name: str
        Registered name.
    **kwargs: Any
        Passed to the factory.

    Returns
    -------
    TrialCorrector
        The corrector the factory returns.

    Raises
    ------
    ValueError
        If no corrector is registered under that name.
    """
    if name not in _CORRECTORS:
        raise ValueError(
            f'no corrector named {name!r}; registered correctors are '
            f'{all_registered_correctors()}',
        )
    return _CORRECTORS[name](**kwargs)
