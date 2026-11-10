# Copyright (c) 2025-2026 The pymovements Project Authors
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

import math
from typing import Any

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


def test_map_to_aois_null_key_uses_per_row_lookup(stimulus_with_trials: TextStimulus) -> None:
    """A null key routes the row through the per-row lookup and still matches by the other key."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5.0, 5.0]],
            'trial': [None],
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus_with_trials)

    assert events.frame.get_column('label').to_list() == ['A']


def test_map_to_aois_uses_first_overlapping_aoi_without_misalignment() -> None:
    """Map only the first overlapping AOI row and keep event/AOI rows aligned."""
    stimulus = TextStimulus(
        pl.DataFrame(
            {
                'content': ['AOI1', 'AOI2'],
                'left': [0, 100],
                'right': [200, 300],
                'top': [0, 100],
                'bottom': [200, 300],
            },
        ),
        aoi_column='content',
        start_x_column='left',
        end_x_column='right',
        start_y_column='top',
        end_y_column='bottom',
    )
    events = Events(
        pl.DataFrame(
            {
                'name': ['fixation', 'fixation', 'fixation', 'fixation'],
                'location': [(50, 50), (150, 150), (250, 250), (350, 350)],
                'onset': [0, 1000, 2000, 3000],
                'offset': [500, 1500, 2500, 3500],
            },
        ),
    )

    with pytest.warns(UserWarning, match='Multiple AOIs matched this point'):
        events.map_to_aois(stimulus, verbose=False)

    assert events.frame.height == 4
    assert events.frame.get_column('content').to_list() == ['AOI1', 'AOI1', 'AOI2', None]


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
    with pytest.warns(UserWarning, match='Invalid eye coordinates'):
        events.map_to_aois(simple_stimulus)
    assert events.frame.get_column('label').to_list() == [None]


def test_map_to_aois_get_aoi_exception_sets_none_without_crash(
    stimulus_with_trial_page: TextStimulus, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a recoverable exception in get_aoi and assert mapping continues with None AOI.

    Events.map_to_aois tolerates per-row KeyError/TypeError and fills AOI values with None
    instead of raising. ValueError is allowed to propagate for misconfiguration cases and is
    covered by existing tests elsewhere.
    """

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002 - test helper
        raise TypeError('simulated get_aoi type error')

    # Patch TextStimulus.get_aoi to raise. It is only invoked for rows with a null
    # trial/page key; the exception must be tolerated and the AOI values set to None.
    monkeypatch.setattr(TextStimulus, 'get_aoi', _boom, raising=True)

    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5, 5]],
            'trial': [1],
            'page': [None],
        },
    )
    events = Events(data=df, trial_columns=['trial', 'page'])

    # Should not raise - AOI columns get None
    events.map_to_aois(stimulus_with_trial_page)

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


def test_events_map_to_aois_preserve_structure_true(
    simple_stimulus: TextStimulus,
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

    events.map_to_aois(simple_stimulus, preserve_structure=True)

    cols = set(events.frame.columns)
    assert 'location' not in cols
    # Derived component columns must exist when dropping took place
    assert {'location_x', 'location_y'}.issubset(cols)

    # AOI labels should still be mapped correctly for fixation and None for saccade
    labels = events.frame.get_column('label').to_list()
    assert labels == ['A', None]


def test_events_map_to_aois_preserve_structure_false(
    simple_stimulus: TextStimulus,
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

    events.map_to_aois(simple_stimulus, preserve_structure=False)

    cols = set(events.frame.columns)
    assert 'location' in cols
    assert 'location_x' not in cols and 'location_y' not in cols

    # AOI labels should still be mapped correctly for fixation and None for saccade
    labels = events.frame.get_column('label').to_list()
    assert labels == ['A', None]


# Specifically tests the backward-compatibility block that drops the original
# 'location' list column at the end of Events.map_to_aois when component columns
# already exist. The block should only apply if preserve_structure=True.


def test_end_drop_location_when_components_exist_preserve_true(
    simple_stimulus: TextStimulus,
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

    # Map to AOIs with preserve_structure=True
    events.map_to_aois(simple_stimulus, preserve_structure=True)

    cols = set(events.frame.columns)
    labels = events.frame.get_column(simple_stimulus.aoi_column).to_list()

    # AOI labels: first inside [0,10)x[0,10) -> 'A' - second outside -> None
    assert labels == ['A', None]

    # When preserving legacy structure, we drop the original list column at the end
    # if component columns are present.
    assert 'location' not in cols
    assert 'location_x' in cols and 'location_y' in cols


def test_end_drop_location_when_components_exist_preserve_false(
    simple_stimulus: TextStimulus,
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

    # Map to AOIs with preserve_structure=False
    events.map_to_aois(simple_stimulus, preserve_structure=False)

    cols = set(events.frame.columns)
    labels = events.frame.get_column(simple_stimulus.aoi_column).to_list()

    # AOI labels: first inside [0,10)x[0,10) -> 'A' - second outside -> None
    assert labels == ['A', None]

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


def test_map_to_aois_preserves_saccades_with_structure(
    simple_stimulus: TextStimulus,
) -> None:
    """Events with preceding saccades remain unchanged by map_to_aois with preserve_structure=True.

    - AOIs are only mapped for fixation rows.
    - Saccade rows keep their original fields (name/onset/offset and location or its components),
      and receive only None values in the appended AOI columns.
    - location list is dropped and components are derived.
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

    events = Events(data=base)
    original = events.frame.clone()
    events.map_to_aois(simple_stimulus, preserve_structure=True)
    labels = events.frame.get_column('label').to_list()
    assert labels == [None, 'A', None]

    # Verify saccade rows are unchanged (except for expected structural handling)
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


def test_map_to_aois_preserves_saccades_without_structure(
    simple_stimulus: TextStimulus,
) -> None:
    """Events with preceding saccades remain unchanged by map_to_aois with preserve_structure=False.

    - AOIs are only mapped for fixation rows.
    - Saccade rows keep their original fields (name/onset/offset and location or its components),
      and receive only None values in the appended AOI columns.
    - location list is preserved and there are no derived components.
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

    events = Events(data=base)
    original = events.frame.clone()
    events.map_to_aois(simple_stimulus, preserve_structure=False)
    labels = events.frame.get_column('label').to_list()
    assert labels == [None, 'A', None]

    # Verify saccade rows are unchanged (except for expected structural handling)
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


@pytest.mark.parametrize(
    'trial_value',
    ['x', 1.5],
    ids=['non-numeric-string', 'non-integral-float'],
)
def test_map_to_aois_key_that_changes_on_cast_gets_no_aoi(
    stimulus_with_trials: TextStimulus,
    trial_value: object,
) -> None:
    """A non-null key that cannot be cast to the AOI dtype unchanged matches no AOI."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5.0, 5.0]],
            'trial': [trial_value],
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus_with_trials, verbose=False)

    assert events.frame.get_column('label').to_list() == [None]


@pytest.mark.parametrize(
    ('trial_value', 'location', 'expected_label'),
    [
        pytest.param('1', [5.0, 5.0], 'A', id='plain'),
        pytest.param('02', [25.0, 25.0], 'B', id='leading-zero'),
    ],
)
def test_map_to_aois_numeric_string_key_matches_integer_key(
    stimulus_with_trials: TextStimulus,
    trial_value: str,
    location: list[float],
    expected_label: str,
) -> None:
    """A string key that parses to the integer AOI key matches that trial's AOIs."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [location],
            'trial': [trial_value],
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus_with_trials, verbose=False)

    assert events.frame.get_column('label').to_list() == [expected_label]


def test_map_to_aois_integer_key_matches_string_key() -> None:
    """An integer key matches a string AOI key with the same textual value."""
    stimulus = TextStimulus(
        pl.DataFrame(
            {
                'label': ['A', 'B'],
                'sx': [0, 20],
                'sy': [0, 20],
                'ex': [10, 30],
                'ey': [10, 30],
                'trial': ['1', '02'],
            },
        ),
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        trial_column='trial',
    )
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location': [[5.0, 5.0], [25.0, 25.0]],
            'trial': [1, 2],
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A', None]


def test_map_to_aois_valid_key_still_matches(stimulus_with_trials: TextStimulus) -> None:
    """A key that casts to the AOI dtype unchanged still matches its AOIs."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5.0, 5.0]],
            'trial': [1],
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus_with_trials, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A']


def test_map_to_aois_non_numeric_coordinates_get_no_match_and_warn(
    simple_stimulus: TextStimulus,
) -> None:
    """String coordinates are not numeric, so they get no match and are reported once."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location_x': ['5', '5'],
            'location_y': ['5', '5'],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 2 of 2 fixations'):
        events.map_to_aois(simple_stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == [None, None]


def test_map_to_aois_nan_coordinates_warn_once(simple_stimulus: TextStimulus) -> None:
    """A NaN coordinate counts as invalid, matches nothing, and is reported once."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location_x': [5.0, math.nan],
            'location_y': [5.0, 5.0],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 1 of 2 fixations'):
        events.map_to_aois(simple_stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A', None]


def test_map_to_aois_no_matches_returns_all_none(simple_stimulus: TextStimulus) -> None:
    """A frame without any spatial match fills every AOI column with nulls."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location': [[50.0, 50.0], [60.0, 60.0]],
        },
    )
    events = Events(data=df)

    events.map_to_aois(simple_stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == [None, None]


def test_map_to_aois_overlap_with_keyed_null_key_and_unmatched_rows() -> None:
    """Overlapping AOIs keep first-match order for keyed and null-key rows, others get none."""
    stimulus = TextStimulus(
        pl.DataFrame(
            {
                'content': ['L1', 'L2', 'Y'],
                'left': [0, 5, 100],
                'right': [10, 15, 110],
                'top': [0, 5, 0],
                'bottom': [10, 15, 10],
                'trial': [1, 1, 1],
                'page': ['X', 'X', 'Y'],
            },
        ),
        aoi_column='content',
        start_x_column='left',
        end_x_column='right',
        start_y_column='top',
        end_y_column='bottom',
        trial_column='trial',
        page_column='page',
    )
    df = pl.DataFrame(
        [
            {
                'name': 'fixation', 'onset': 0, 'offset': 1,
                'location': [7.0, 7.0], 'trial': 1, 'page': 'X',
            },
            {
                'name': 'fixation', 'onset': 1, 'offset': 2,
                'location': [7.0, 7.0], 'trial': 1, 'page': None,
            },
            {
                'name': 'fixation', 'onset': 2, 'offset': 3,
                'location': [50.0, 50.0], 'trial': 1, 'page': 'X',
            },
        ],
    )
    events = Events(data=df, trial_columns=['trial', 'page'])

    with pytest.warns(UserWarning, match='Multiple AOIs matched this point'):
        events.map_to_aois(stimulus, verbose=False)

    assert events.frame.get_column('content').to_list() == ['L1', 'L1', None]


@pytest.mark.parametrize(
    'bounds_kwargs',
    [
        pytest.param({'width_column': 'w', 'height_column': None}, id='width-without-height'),
        pytest.param({'end_x_column': 'ex', 'end_y_column': None}, id='end-x-without-end-y'),
    ],
)
def test_map_to_aois_raises_for_mismatched_bounds(
    bounds_kwargs: dict[str, Any],
) -> None:
    """A bounds pair that is only half defined raises the mutual-none check."""
    columns: dict[str, list[object]] = {'content': ['A'], 'left': [0], 'top': [0]}
    if 'width_column' in bounds_kwargs:
        columns['w'] = [10]
    else:
        columns['ex'] = [10]
    stimulus = TextStimulus(
        pl.DataFrame(columns),
        aoi_column='content',
        start_x_column='left',
        start_y_column='top',
        **bounds_kwargs,
    )
    events = Events(
        pl.DataFrame(
            {
                'name': ['fixation'],
                'onset': [0],
                'offset': [1],
                'location': [[5, 5]],
            },
        ),
    )

    with pytest.raises(ValueError, match='must be either both None or both not None'):
        events.map_to_aois(stimulus, verbose=False)


def test_map_to_aois_numeric_string_list_coordinates_get_no_match(
    simple_stimulus: TextStimulus,
) -> None:
    """Numeric strings in a location list are not numeric coordinates and match nothing."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [['5', '5']],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 1 of 1 fixations'):
        events.map_to_aois(simple_stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == [None]


def test_map_to_aois_categorical_key_matches_string_key() -> None:
    """String-like keys with different dtypes are still castable and match."""
    stimulus = TextStimulus(
        pl.DataFrame(
            {
                'label': ['A'],
                'sx': [0],
                'sy': [0],
                'ex': [10],
                'ey': [10],
                'trial': ['1'],
            },
        ),
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        trial_column='trial',
    )
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5.0, 5.0]],
            'trial': pl.Series(['1'], dtype=pl.Categorical),
        },
    )
    events = Events(data=df, trial_columns=['trial'])

    events.map_to_aois(stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A']


def test_map_to_aois_non_numeric_list_without_structure_gets_no_match(
    simple_stimulus: TextStimulus,
) -> None:
    """A non-numeric location list is rejected when it is read directly."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [['5', '5']],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 1 of 1 fixations'):
        events.map_to_aois(simple_stimulus, preserve_structure=False, verbose=False)

    assert events.frame.get_column('label').to_list() == [None]


@pytest.mark.parametrize('preserve_structure', [True, False])
def test_map_to_aois_null_component_falls_back_to_location_list(
    simple_stimulus: TextStimulus,
    preserve_structure: bool,
) -> None:
    """A row with a null component coordinate takes both coordinates from the list."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation', 'fixation'],
            'onset': [0, 1, 2],
            'offset': [1, 2, 3],
            'location_x': [5.0, None, 5.0],
            'location_y': [5.0, 5.0, None],
            'location': [[50.0, 50.0], [5.0, 5.0], [5.0, 5.0]],
        },
    )
    events = Events(data=df)

    events.map_to_aois(simple_stimulus, preserve_structure=preserve_structure, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A', 'A', 'A']


@pytest.mark.parametrize('preserve_structure', [True, False])
def test_map_to_aois_short_location_list_gets_no_match(
    simple_stimulus: TextStimulus,
    preserve_structure: bool,
) -> None:
    """A null component with a list of fewer than two entries gets no match and warns."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation', 'fixation'],
            'onset': [0, 1, 2],
            'offset': [1, 2, 3],
            'location_x': [None, None, 5.0],
            'location_y': [5.0, 5.0, None],
            'location': [[5.0], [], [5.0, 5.0]],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 2 of 3 fixations'):
        events.map_to_aois(simple_stimulus, preserve_structure=preserve_structure, verbose=False)

    assert events.frame.get_column('label').to_list() == [None, None, 'A']


@pytest.mark.parametrize('preserve_structure', [True, False])
def test_map_to_aois_short_location_list_without_components_gets_no_match(
    simple_stimulus: TextStimulus,
    preserve_structure: bool,
) -> None:
    """A list of fewer than two entries gets no match when it is the only coordinate source."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation', 'fixation'],
            'onset': [0, 1, 2],
            'offset': [1, 2, 3],
            'location': [[5.0], [], [5.0, 5.0]],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 2 of 3 fixations'):
        events.map_to_aois(simple_stimulus, preserve_structure=preserve_structure, verbose=False)

    assert events.frame.get_column('label').to_list() == [None, None, 'A']


def test_map_to_aois_single_component_column_reads_location_list(
    simple_stimulus: TextStimulus,
) -> None:
    """With only one component column present, every row reads the list."""
    df = pl.DataFrame(
        {
            'name': ['fixation', 'fixation'],
            'onset': [0, 1],
            'offset': [1, 2],
            'location_x': [50.0, 5.0],
            'location': [[5.0, 5.0], [50.0, 50.0]],
        },
    )
    events = Events(data=df)

    events.map_to_aois(simple_stimulus, preserve_structure=False, verbose=False)

    assert events.frame.get_column('label').to_list() == ['A', None]


def test_map_to_aois_missing_coordinate_columns_get_no_match(
    simple_stimulus: TextStimulus,
) -> None:
    """A fixation frame without any coordinate column gets no match and warns."""
    df = pl.DataFrame(
        {
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
        },
    )
    events = Events(data=df)

    with pytest.warns(UserWarning, match='Invalid eye coordinates in 1 of 1 fixations'):
        events.map_to_aois(simple_stimulus, verbose=False)

    assert events.frame.get_column('label').to_list() == [None]
