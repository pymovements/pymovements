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
"""Test the registry for correctors that do not fit the drift-algorithm shape."""
from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest

import pymovements as pm
from pymovements.events.correction import _corrector_registry
from pymovements.events.correction.fixation_correction import _resolve_algorithms
from pymovements.events.correction.fixation_correction import ALL_DRIFT_ALGORITHMS


class Stub:
    """A stand-in for whatever a factory returns; the registry never calls it."""


@pytest.fixture(name='temporary_registration')
def fixture_temporary_registration() -> Iterator[str]:
    """Register a stub corrector and remove it again afterwards.

    Yields
    ------
    str
        The registered name.

    """
    name = 'stub_corrector'
    _corrector_registry.register_corrector(name, lambda **kwargs: Stub())
    yield name
    _corrector_registry._CORRECTORS.pop(name, None)


def test_drift_algorithms_are_not_registered_correctors():
    assert not any(
        _corrector_registry.is_registered_corrector(name) for name in ALL_DRIFT_ALGORITHMS
    )


def test_registering_a_duplicate_is_refused(temporary_registration):
    with pytest.raises(ValueError, match='already registered'):
        _corrector_registry.register_corrector(
            temporary_registration, lambda **kwargs: Stub(),
        )


def test_registering_over_a_drift_algorithm_is_refused():
    with pytest.raises(ValueError, match='already a drift algorithm'):
        _corrector_registry.register_corrector('warp', lambda **kwargs: Stub())


def test_building_an_unknown_corrector_names_the_known_ones():
    with pytest.raises(ValueError, match='no corrector named'):
        _corrector_registry.build_corrector('nonexistent')


def test_registry_holds_a_factory_not_an_instance():
    """Registration must not construct anything, or a name would cost its dependencies."""
    built = []

    def factory(**kwargs: object) -> Stub:
        built.append(kwargs)
        return Stub()

    _corrector_registry.register_corrector('counting', factory)
    try:
        assert not built
        _corrector_registry.build_corrector('counting', answer=42)
        assert built == [{'answer': 42}]
    finally:
        _corrector_registry._CORRECTORS.pop('counting', None)


def test_a_registered_name_in_an_ensemble_is_refused(temporary_registration):
    """A registered corrector's vote has no obvious weight against eleven others; say so."""
    events = pl.DataFrame({
        'name': ['fixation'] * 3,
        'onset': [0, 1, 2], 'offset': [1, 2, 3],
        'location_x': [1.0, 2.0, 3.0], 'location_y': [1.0, 1.0, 1.0],
    })
    aois = pl.DataFrame({
        'char': ['a', 'b'], 'start_x': [0.0, 10.0], 'start_y': [0.0, 20.0],
        'end_x': [10.0, 20.0], 'end_y': [10.0, 30.0],
    })

    with pytest.raises(ValueError, match='cannot take part in an ensemble'):
        pm.events.correction.correct_fixations(
            events, aois, algorithm=['attach', temporary_registration],
        )


def test_unknown_algorithm_lists_the_registered_correctors_too(temporary_registration):
    events = pl.DataFrame({
        'name': ['fixation'], 'onset': [0], 'offset': [1],
        'location_x': [1.0], 'location_y': [1.0],
    })
    aois = pl.DataFrame({
        'char': ['a'], 'start_x': [0.0], 'start_y': [0.0], 'end_x': [10.0], 'end_y': [10.0],
    })

    with pytest.raises(ValueError, match="Unknown drift algorithm 'nope'") as error:
        pm.events.correction.correct_fixations(events, aois, algorithm='nope')

    assert temporary_registration in str(error.value)


def test_a_registered_name_resolves_to_itself_and_not_to_an_ensemble(temporary_registration):
    """Selecting a registered corrector by name must bypass the drift-algorithm table.

    This is the resolution step only. What happens with the resolved name afterwards depends
    on the corrector's call shape, which is still open; see the note in _corrector_registry.
    """
    resolved, ensemble = _resolve_algorithms(
        temporary_registration, has_word_coords=True, right_to_left=False,
    )

    assert resolved == [temporary_registration]
    assert ensemble is False
