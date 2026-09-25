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

    def __call__(
            self,
            fixations: pl.DataFrame,
            aois: pl.DataFrame,
            *,
            location_column: str,
    ) -> None:
        """Decline every trial.

        Parameters
        ----------
        fixations: pl.DataFrame
            The trial's fixations.
        aois: pl.DataFrame
            The trial's areas of interest.
        location_column: str
            Name of the location column.

        """
        del fixations, aois, location_column


def make_events(n: int = 3) -> pl.DataFrame:
    """Return a minimal fixation events frame.

    Parameters
    ----------
    n: int
        Number of fixations.

    Returns
    -------
    pl.DataFrame
        Fixation events with component location columns.

    """
    return pl.DataFrame({
        'name': ['fixation'] * n,
        'onset': list(range(n)),
        'offset': list(range(1, n + 1)),
        'location_x': [1.0 * i for i in range(n)],
        'location_y': [1.0] * n,
    })


def make_aois() -> pl.DataFrame:
    """Return a minimal two-line character AOI frame.

    Returns
    -------
    pl.DataFrame
        Areas of interest.

    """
    return pl.DataFrame({
        'char': ['a', 'b'],
        'start_x': [0.0, 10.0], 'start_y': [0.0, 20.0],
        'end_x': [10.0, 20.0], 'end_y': [10.0, 30.0],
    })


def shift_down(
        fixations: pl.DataFrame,
        aois: pl.DataFrame,
        *,
        location_column: str,
) -> pl.Series:
    """Move every fixation onto the second line; a trial corrector in its simplest form.

    Parameters
    ----------
    fixations: pl.DataFrame
        The trial's fixations.
    aois: pl.DataFrame
        The trial's areas of interest.
    location_column: str
        Name of the location column.

    Returns
    -------
    pl.Series
        One [x, y] list per fixation.

    """
    del aois, location_column
    return fixations.select(
        pl.concat_list([pl.col('location_x'), pl.lit(25.0)]).alias('location'),
    ).to_series()


def decline(
        fixations: pl.DataFrame,
        aois: pl.DataFrame,
        *,
        location_column: str,
) -> None:
    """Decline every trial, the way a corrector reports it cannot serve one.

    Parameters
    ----------
    fixations: pl.DataFrame
        The trial's fixations.
    aois: pl.DataFrame
        The trial's areas of interest.
    location_column: str
        Name of the location column.

    """
    del fixations, aois, location_column


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


def test_a_callable_corrects_without_being_registered():
    """Daniel's second point: pass the corrector itself, no registration step."""
    corrected = pm.events.correction.correct_fixations(
        make_events(), make_aois(), algorithm=shift_down,
    )

    assert corrected['correction_algorithm'].unique().to_list() == ['shift_down']
    assert corrected['location_y'].to_list() == [25.0, 25.0, 25.0]
    assert corrected['location_y_original'].to_list() == [1.0, 1.0, 1.0]


def test_a_declining_corrector_skips_the_trial_and_says_so():
    """Returning None must behave like a drift algorithm skipping a trial, not like an error."""
    events = make_events()

    with pytest.warns(UserWarning, match='stay uncorrected'):
        corrected = pm.events.correction.correct_fixations(
            events, make_aois(), algorithm=decline,
        )

    assert corrected['location_y'].to_list() == events['location_y'].to_list()
    assert 'location_y_original' not in corrected.columns
    assert 'correction_algorithm' not in corrected.columns


def test_a_stateful_corrector_is_built_once_not_once_per_trial():
    """The whole reason for the second form: what it holds is prepared once."""
    builds: list[int] = []

    class Stateful:
        def __init__(self) -> None:
            builds.append(1)

        def __call__(self, fixations, aois, *, location_column):
            del aois, location_column
            return fixations.select(
                pl.concat_list([pl.col('location_x'), pl.lit(25.0)]).alias('location'),
            ).to_series()

    _corrector_registry.register_corrector('stateful', lambda **kwargs: Stateful())
    try:
        events = pl.concat([
            make_events().with_columns(pl.lit(subject).alias('subject_id'))
            for subject in (1, 2, 3)
        ])
        pm.events.correction.correct_fixations(
            events, make_aois(), algorithm='stateful', trial_columns=['subject_id'],
        )
    finally:
        _corrector_registry._CORRECTORS.pop('stateful', None)

    assert len(builds) == 1


def test_algorithm_kwargs_are_refused_for_a_callable():
    """There is no factory to take them, and per-trial binding would undo building once."""
    with pytest.raises(ValueError, match='functools.partial'):
        pm.events.correction.correct_fixations(
            make_events(), make_aois(), algorithm=shift_down,
            algorithm_kwargs={'answer': 42},
        )


def test_algorithm_kwargs_reach_a_registered_factory():
    seen: list[dict] = []

    def factory(**kwargs):
        seen.append(kwargs)
        return shift_down

    _corrector_registry.register_corrector('configurable', factory)
    try:
        pm.events.correction.correct_fixations(
            make_events(), make_aois(), algorithm='configurable',
            algorithm_kwargs={'answer': 42},
        )
    finally:
        _corrector_registry._CORRECTORS.pop('configurable', None)

    assert seen == [{'answer': 42}]


def test_a_callable_in_an_ensemble_is_refused():
    with pytest.raises(ValueError, match='cannot take part in an ensemble'):
        pm.events.correction.correct_fixations(
            make_events(), make_aois(), algorithm=['attach', shift_down],
        )


def test_a_stateful_corrector_is_labelled_by_its_class():
    """A callable object has no __name__; the column must still get a usable name."""
    class NamedByClass:
        def __call__(self, fixations, aois, *, location_column):
            del aois, location_column
            return fixations.select(
                pl.concat_list([pl.col('location_x'), pl.lit(25.0)]).alias('location'),
            ).to_series()

    corrected = pm.events.correction.correct_fixations(
        make_events(), make_aois(), algorithm=NamedByClass(),
    )

    assert corrected['correction_algorithm'].unique().to_list() == ['NamedByClass']


def test_a_non_callable_is_still_a_type_error():
    """Widening algorithm= must not swallow nonsense: only a callable is a trial corrector.

    Without this, an int is taken for a corrector named 'int' and fails much later with a
    KeyError, which is what happened while this was being written.
    """
    with pytest.raises(TypeError, match='algorithm must be a string or a list of strings'):
        pm.events.correction.correct_fixations(
            make_events(), make_aois(), algorithm=123,
        )
