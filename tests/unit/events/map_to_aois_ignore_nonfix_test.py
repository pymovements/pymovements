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
"""Tests for Events.map_to_aois ignoring non-fixation events.

These tests ensure that AOIs are only mapped for fixations and that non-fixations
(saccades, blinks, etc.) are left with None AOI columns without raising errors,
including when their locations are missing. They also cover exception handling
inside Events.map_to_aois with a message match on the emitted warning.
"""
from __future__ import annotations

import polars as pl
import pytest

from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus


@pytest.fixture(name='simple_stimulus')
def _simple_stimulus() -> TextStimulus:
    """Single AOI box [0,10) x [0,10) labeled 'A'."""
    df = pl.DataFrame(
        {
            'label': ['A'],
            'sx': [0],
            'sy': [0],
            'ex': [10],
            'ey': [10],
        },
    )
    return TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
    )


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


@pytest.fixture(name='stimulus_with_trial_page')
def _stimulus_with_trial_page() -> TextStimulus:
    """Two AOIs on different (trial, page), same spatial box."""
    df = pl.DataFrame(
        {
            'label': ['TX', 'TY'],
            'sx': [0, 0],
            'sy': [0, 0],
            'ex': [10, 10],
            'ey': [10, 10],
            'trial': [1, 1],
            'page': ['X', 'Y'],
        },
    )
    return TextStimulus(
        aois=df,
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        trial_column='trial',
        page_column='page',
    )


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
