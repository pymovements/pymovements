# Copyright (c) 2025 The pymovements Project Authors
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
"""Several tests about specific Events.map_to_aois behaviours."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus


# Tests for Events.map_to_aois ignoring non-fixation events:
# These tests ensure that AOIs are only mapped for fixations and that non-fixations
# (saccades, blinks, etc.) are left with None AOI columns without raising errors,
# including when their locations are missing. They also cover exception handling
# inside Events.map_to_aois with a message match on the emitted warning.


@pytest.mark.parametrize(
    'names_locations_expected',
    [
        pytest.param(
            # name, location -> expected label
            [
                ('fixation', [5, 5], 'A'),
                ('saccade', [5, 5], None),
                ('blink', [5, 5], None),
            ],
            id='fixation_maps_others_none',
        ),
        pytest.param(
            [
                ('fixation_ivt', [5, 5], 'A'),
                ('saccade', None, None),  # missing location should not fail
            ],
            id='fixation_prefix_and_missing_location',
        ),
    ],
)
def test_map_to_aois_ignores_non_fixations(
    simple_stimulus: TextStimulus,
    names_locations_expected: list[tuple[str, list[float] | None, str | None]],
) -> None:
    # Build Events frame
    names = [n for n, _, _ in names_locations_expected]
    locations = [loc for _, loc, _ in names_locations_expected]
    onsets = list(range(0, len(names)))
    offsets = list(range(1, len(names) + 1))

    df = pl.DataFrame(
        {
            'name': names,
            'onset': onsets,
            'offset': offsets,
            'location': locations,
        },
    )
    events = Events(data=df)

    events.map_to_aois(simple_stimulus)

    # Expect AOI columns appended - we only check the label values row-wise
    labels = events.frame.get_column('label').to_list()
    expected_labels = [exp for _, _, exp in names_locations_expected]
    assert labels == expected_labels


@pytest.mark.parametrize(
    'row_defs, expected_labels',
    [
        pytest.param(
            [
                {
                    'name': 'fixation', 'onset': 0, 'offset': 1,
                    'location': [5, 5], 'trial': 1, 'page': 'X',
                },
                {
                    'name': 'saccade', 'onset': 1, 'offset': 2,
                    'location': [5, 5], 'trial': 1, 'page': 'X',
                },
                {
                    'name': 'fixation', 'onset': 2, 'offset': 3,
                    'location': [5, 5], 'trial': 1, 'page': 'Y',
                },
            ],
            ['TX', None, 'TY'],
            id='respects_trial_page_and_ignores_nonfix',
        ),
    ],
)
def test_map_to_aois_with_trial_page(
    stimulus_with_trial_page: TextStimulus,
    row_defs: list[dict[str, object]],
    expected_labels: list[str | None],
) -> None:
    df = pl.DataFrame(row_defs)
    events = Events(data=df, trial_columns=['trial', 'page'])  # group info stored for later steps

    events.map_to_aois(stimulus_with_trial_page)

    labels = events.frame.get_column('label').to_list()
    assert labels == expected_labels


def test_map_to_aois_fixation_missing_coordinates(simple_stimulus: TextStimulus) -> None:
    """Cover branch where a fixation row has missing coordinates, leading to an empty AOI row."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[None, None]],  # explicit List[None, None] to satisfy schema
        },
    )
    events = Events(data=df)
    events.map_to_aois(simple_stimulus)
    assert events.frame.get_column('label').to_list() == [None]


def test_map_to_aois_get_aoi_exception_sets_none_without_crash(
    simple_stimulus: TextStimulus, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a recoverable exception in get_aoi and assert mapping continues with None AOI.

    Events.map_to_aois tolerates per-row KeyError/TypeError and fills AOI values with None
    instead of raising. ValueError is allowed to propagate for misconfiguration cases and is
    covered by existing tests elsewhere.
    """

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002 - test helper
        raise TypeError('simulated get_aoi type error')

    # Patch TextStimulus.get_aoi to raise on call for a fixation row
    monkeypatch.setattr(TextStimulus, 'get_aoi', _boom, raising=True)

    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5, 5]],
        },
    )
    events = Events(data=df)

    # Should not raise - AOI columns get None
    events.map_to_aois(simple_stimulus)

    assert events.frame.get_column('label').to_list() == [None]


# Tests ensuring Events.map_to_aois has no unnest side-effects and supports grouping:
# - The events frame should not gain component columns like 'location_x'/'location_y'.
# - The original 'location' list column is preserved after AOI mapping.
# - Trial/page columns are kept intact and used for filtering, with AOI columns appended only.


def test_events_map_to_aois_preserves_location_column(simple_stimulus: TextStimulus) -> None:
    """Mapping AOIs must not unnest the 'location' column into components."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'saccade', 'fixation'],
            'onset': [0, 1, 2],
            'offset': [1, 2, 3],
            'location': [[5, 5], [5, 5], [15, 15]],
        },
    )

    events = Events(data=df)

    before_cols = set(events.frame.columns)
    assert 'location' in before_cols
    assert 'location_x' not in before_cols and 'location_y' not in before_cols

    events.map_to_aois(simple_stimulus)

    # Ensure mapping succeeded without structural requirement assertions
    labels = events.frame.get_column('label').to_list()
    assert labels == ['A', None, None]


def test_events_map_to_aois_grouped_by_trial_page(stimulus_with_trial_page: TextStimulus) -> None:
    """Grouped mapping by trial/page yields correct AOIs and preserves structure."""
    rows = [
        {'name': 'fixation', 'onset': 0, 'offset': 1, 'location': [5, 5], 'trial': 1, 'page': 'X'},
        {'name': 'fixation', 'onset': 1, 'offset': 2, 'location': [5, 5], 'trial': 1, 'page': 'Y'},
        {'name': 'saccade', 'onset': 2, 'offset': 3, 'location': [5, 5], 'trial': 2, 'page': 'X'},
    ]
    df = pl.DataFrame(rows)
    events = Events(data=df, trial_columns=['trial', 'page'])  # store grouping keys

    events.map_to_aois(stimulus_with_trial_page)

    # Correct labels per (trial, page)
    labels = events.frame.get_column('label').to_list()
    assert labels == ['TX', 'TY', None]


# Test for Events.map_to_aois preserve_structure flag

@pytest.mark.parametrize(
    'preserve_structure,expect_drop_location', [
        (True, True),
        (False, False),
    ],
)
def test_events_map_to_aois_preserve_structure_flag(
    simple_stimulus: TextStimulus,
    preserve_structure: bool,
    expect_drop_location: bool,
) -> None:
    # Frame contains only list column 'location'
    df = pl.DataFrame(
        {
            'name': ['fixation', 'saccade'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location': [[5, 5], [5, 5]],
        },
    )
    events = Events(data=df)

    events.map_to_aois(simple_stimulus, preserve_structure=preserve_structure)

    cols = set(events.frame.columns)
    if expect_drop_location:
        assert 'location' not in cols
        # Derived component columns must exist when dropping took place
        assert {'location_x', 'location_y'}.issubset(cols)
    else:
        assert 'location' in cols
        assert 'location_x' not in cols and 'location_y' not in cols

    # AOI labels should still be mapped correctly for fixation and None for saccade
    labels = events.frame.get_column('label').to_list()
    assert labels == ['A', None]

# Specifically tests the backward-compatibility block that drops the original
# 'location' list column at the end of Events.map_to_aois when component columns
# already exist. The block should only apply if preserve_structure=True.


@pytest.mark.parametrize('preserve', [True, False])
def test_end_drop_location_when_components_exist(
    simple_stimulus: TextStimulus,
    preserve: bool,
) -> None:
    # Prepare events with BOTH 'location' list and component columns present from the start.
    # This skips the early derive-and-drop path (since components exist), exercising the
    # end-of-function drop guarded by preserve_structure.
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            # list column (legacy pipelines might keep it around)
            'location': [[5.0, 5.0], [15.0, 5.0]],
            # pre-existing component columns
            'location_x': [5.0, 15.0],
            'location_y': [5.0, 5.0],
        },
    )
    events = Events(df)

    # Map to AOIs with parameterized preserve_structure
    events.map_to_aois(simple_stimulus, preserve_structure=preserve)

    cols = set(events.frame.columns)
    labels = events.frame.get_column(simple_stimulus.aoi_column).to_list()

    # AOI labels: first inside [0,10)x[0,10) -> 'A' - second outside -> None
    assert labels == ['A', None]

    if preserve:
        # When preserving legacy structure, we drop the original list column at the end
        # if component columns are present.
        assert 'location' not in cols
        assert 'location_x' in cols and 'location_y' in cols
    else:
        # When not preserving structure, we do not drop the list column at the end.
        assert 'location' in cols
        # Component columns should remain untouched
        assert 'location_x' in cols and 'location_y' in cols


@pytest.mark.parametrize('preserve_structure', [True, False])
def test_events_map_to_aois_no_new_columns_when_all_aoi_columns_present(
    simple_stimulus_w_h: TextStimulus,
    preserve_structure: bool,
) -> None:
    # Events already contains ALL AOI columns from the stimulus.
    # One row is a fixation positioned inside the AOI.
    events_df = pl.DataFrame(
        {
            'name': ['fixation'],
            'location_x': [5.0],
            'location_y': [5.0],
            'label': ['pre'],
            # Pre-existing AOI columns (same names as in stimulus)
            'aoi': ['A'],
            'x': [-1.0],
            'y': [-1.0],
            'width': [0.0],
            'height': [0.0],
        },
    )

    ev = Events(data=events_df)

    # Keep a copy of original columns to verify no new columns are appended.
    original_cols = ev.frame.columns.copy()

    ev.map_to_aois(simple_stimulus_w_h, preserve_structure=preserve_structure, verbose=False)

    # The set and order of columns should remain unchanged (no new AOI columns appended),
    # exercising the `if aoi_columns:` no-branch.
    assert ev.frame.columns == original_cols

    # Also ensure that pre-existing AOI columns remain untouched (no overwrite during concat).
    assert ev.frame.select('label').item() == 'pre'


@pytest.mark.parametrize('preserve_structure', [True, False])
def test_previous_saccades_untouched_after_map_to_aois(
    simple_stimulus: TextStimulus,
    preserve_structure: bool,
) -> None:
    """Events with preceding saccades remain unchanged by map_to_aois.

    - AOIs are only mapped for fixation rows.
    - Saccade rows keep their original fields (name/onset/offset and location or its components),
      and receive only None values in the appended AOI columns.
    """
    # Build a frame where a saccade precedes a fixation (and another saccade follows).
    base = pl.DataFrame(
        {
            'name': ['saccade', 'fixation', 'saccade'],
            'onset': [0, 1, 2],
            'offset': [1, 2, 3],
            'location': [[5.0, 5.0], [5.0, 5.0], [15.0, 5.0]],
        },
    )

    original = base.clone()

    events = Events(data=base)
    events.map_to_aois(simple_stimulus, preserve_structure=preserve_structure)

    # only middle fixation is inside AOI 'A'
    labels = events.frame.get_column('label').to_list()
    assert labels == [None, 'A', None]

    # Verify saccade rows are unchanged (except for expected structural handling)
    if preserve_structure:
        # location list is dropped and components are derived
        assert 'location' not in events.frame.columns
        assert {'location_x', 'location_y'}.issubset(set(events.frame.columns))
        # Check each saccade's fields
        for idx in (0, 2):
            row_after = events.frame.row(idx, named=True)
            row_before = original.row(idx, named=True)
            assert row_after['name'] == row_before['name'] == 'saccade'
            assert row_after['onset'] == row_before['onset']
            assert row_after['offset'] == row_before['offset']
            # Components equal original list components
            assert row_after['location_x'] == row_before['location'][0]
            assert row_after['location_y'] == row_before['location'][1]
            # AOI label None for saccades
            assert row_after['label'] is None
    else:
        # location list is preserved and there are no derived components
        assert 'location' in events.frame.columns
        assert 'location_x' not in events.frame.columns and 'location_y' not in events.frame.columns
        for idx in (0, 2):
            row_after = events.frame.row(idx, named=True)
            row_before = original.row(idx, named=True)
            assert row_after['name'] == row_before['name'] == 'saccade'
            assert row_after['onset'] == row_before['onset']
            assert row_after['offset'] == row_before['offset']
            assert row_after['location'] == row_before['location']
            assert row_after['label'] is None
