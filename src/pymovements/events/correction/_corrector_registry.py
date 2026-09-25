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

The eleven algorithms in :py:data:`~pymovements.events.correction.ALL_DRIFT_ALGORITHMS` are
functions returning a polars expression over a column. That shape carries no state: each call
builds an expression, polars evaluates it, nothing survives. A corrector that has to *hold*
something -- weights it loaded, geometry it precomputed, a fitted per-reader parameter --
cannot be written that way. Its cost would also be hidden inside an expression, where neither
the caller nor the ensemble vote can see it.

Correctors of that kind therefore live in a registry of their own, while sharing the
``algorithm=`` namespace of :py:func:`~pymovements.events.correction.correct_fixations`, so
that selecting one does not look different from selecting ``'warp'``.

The registry holds **factories, not instances**. Registering a name must not construct
anything and must not import the corrector's dependencies: ``import pymovements`` stays cheap,
and an optional extra is only touched when someone actually asks for the corrector.

.. note::

   What a registered corrector *looks like* is deliberately not fixed here. This module knows
   names and factories; it never calls what a factory returns. The call shape is the open
   design question, and it is settled in the caller, not in the registry.
"""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

_CORRECTORS: dict[str, Callable[..., Any]] = {}
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


def register_corrector(name: str, factory: Callable[..., Any]) -> None:
    """Register a factory that builds a corrector.

    Parameters
    ----------
    name: str
        Name under which the corrector is selected via ``algorithm=``.
    factory: Callable[..., Any]
        Callable returning the corrector. It is called at correction time, not at import time,
        so that registering costs nothing.

    Raises
    ------
    ValueError
        If the name is already taken, here or by a drift algorithm.
    """
    if name in _CORRECTORS:
        raise ValueError(f'a corrector named {name!r} is already registered')
    if name in _RESERVED_NAMES:
        raise ValueError(
            f'{name!r} is already a drift algorithm; registered correctors and drift '
            'algorithms share one namespace so that algorithm= takes either kind',
        )
    _CORRECTORS[name] = factory


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


def build_corrector(name: str, **kwargs: Any) -> Any:
    """Build the corrector registered under a name.

    Parameters
    ----------
    name: str
        Registered name.
    **kwargs: Any
        Passed to the factory.

    Returns
    -------
    Any
        Whatever the factory returns. This module does not constrain that; see the note in the
        module docstring.

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
