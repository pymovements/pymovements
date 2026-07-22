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
"""Tests for fixation drift correction helper routines."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import pymovements as pm
from pymovements.events.correction.fixation_correction import _get_lines_of_text_from_aois
from pymovements.events.correction.fixation_correction import add_corrected_fixations
from pymovements.events.correction.fixation_correction import create_corrected_fixations_locations


@pytest.fixture
def sample_events_and_aois():
    """Return sample events DataFrame and AOIs DataFrame for testing."""
    events_df = pl.DataFrame({
        'trial': ['TRIAL1'] * 6,
        'name': ['fixation'] * 6,
        'onset': [0, 100, 200, 300, 400, 500],
        'location': [
            [100.0, 105.0], [200.0, 102.0], [300.0, 198.0],
            [400.0, 201.0], [100.0, 305.0], [200.0, 301.0],
        ],
    })

    aois_df = pl.DataFrame({
        'trial': ['TRIAL1'] * 6,
        'word': ['Word1', 'Word2', 'Word3', 'Word4', 'Word5', 'Word6'],
        'start_x': [50.0, 250.0, 50.0, 250.0, 50.0, 250.0],
        'start_y': [80.0, 80.0, 180.0, 180.0, 280.0, 280.0],
        'end_x': [200.0, 400.0, 200.0, 400.0, 200.0, 400.0],
        'end_y': [120.0, 120.0, 220.0, 220.0, 320.0, 320.0],
        'width': [150.0, 150.0, 150.0, 150.0, 150.0, 150.0],
        'height': [40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
    })

    return events_df, aois_df


def test_get_lines_of_text_from_aois(sample_events_and_aois):
    _, aois_df = sample_events_and_aois
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert line_Y == [100.0, 200.0, 300.0]


def test_create_corrected_fixations_locations_default_woc(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = create_corrected_fixations_locations(events_df, aois_df)
    assert locs.shape == (6, 2)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])


def test_create_corrected_fixations_locations_specific_algos(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    algos = (
        'attach', 'chain', 'cluster', 'compare', 'merge',
        'regress', 'segment', 'slice', 'split', 'stretch', 'warp',
    )
    for algo in algos:
        locs = create_corrected_fixations_locations(events_df, aois_df, algorithm=algo)
        assert locs.shape == (6, 2)


def test_create_corrected_fixations_locations_invalid_location_raises():
    events_df = pl.DataFrame({'name': ['fixation'], 'trial': ['TRIAL1']})
    aois_df = pl.DataFrame({'start_y': [80.0], 'height': [40.0]})
    with pytest.raises(ValueError, match='No valid location coordinates found'):
        create_corrected_fixations_locations(events_df, aois_df)


def test_add_corrected_fixations_default_woc(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    res_df = add_corrected_fixations(events_df, aois_df, trial_id='TRIAL1')
    assert res_df.height == 12
    corrected_rows = res_df.filter(pl.col('name') == 'fixation_corrected_wisdom_of_the_crowd')
    assert corrected_rows.height == 6


def test_add_corrected_fixations_events_object(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events_obj = pm.Events(events_df)
    res_df = add_corrected_fixations(events_obj, aois_df, algorithm='chain')
    assert res_df.height == 12
    assert res_df.filter(pl.col('name') == 'fixation_corrected_chain').height == 6


def test_add_corrected_fixations_empty_fixations(sample_events_and_aois):
    _, aois_df = sample_events_and_aois
    empty_events_df = pl.DataFrame({'name': ['saccade'], 'trial': ['TRIAL1']})
    res_df = add_corrected_fixations(empty_events_df, aois_df)
    assert res_df.height == 1


def test_get_lines_of_text_from_aois_top_left_y():
    aois_df = pl.DataFrame({
        'top_left_y': [80.0, 180.0],
        'height': [40.0, 40.0],
    })
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert line_Y == [100.0, 200.0]


def test_get_lines_of_text_from_aois_varying_heights():
    aois_df = pl.DataFrame({
        'start_y': [80.0, 80.0, 180.0],
        'height': [40.0, 60.0, 40.0],
    })
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert line_Y == [105.0, 200.0]


def test_get_lines_of_text_from_aois_with_line_idx():
    aois_df = pl.DataFrame({
        'line_idx': [0, 0, 1],
        'top_left_y': [80.0, 80.0, 180.0],
        'height': [40.0, 50.0, 40.0],
    })
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert line_Y == [102.5, 200.0]


def test_create_corrected_fixations_locations_split_columns():
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'location_x': [100.0, 200.0],
        'location_y': [105.0, 198.0],
    })
    aois_df = pl.DataFrame({
        'start_y': [80.0, 180.0],
        'height': [40.0, 40.0],
    })
    locs = create_corrected_fixations_locations(events_df, aois_df, algorithm='attach')
    assert locs.shape == (2, 2)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 200.0])


def test_add_corrected_fixations_1d_and_split_columns(monkeypatch):
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'location_x': [100.0, 200.0],
        'location_y': [105.0, 198.0],
    })
    aois_df = pl.DataFrame({
        'start_y': [80.0, 180.0],
        'height': [40.0, 40.0],
    })

    monkeypatch.setattr(
        'pymovements.events.correction.fixation_correction.create_corrected_fixations_locations',
        lambda *args, **kwargs: np.array([100.0, 200.0]),
    )

    res_df = add_corrected_fixations(events_df, aois_df, algorithm='attach')
    assert res_df.height == 4
    corrected_rows = res_df.filter(pl.col('name') == 'fixation_corrected_attach')
    assert corrected_rows.height == 2
    assert 'location_x' in corrected_rows.columns
    assert 'location_y' in corrected_rows.columns
    np.testing.assert_array_equal(corrected_rows['location_x'].to_list(), [100.0, 200.0])
    np.testing.assert_array_equal(corrected_rows['location_y'].to_list(), [100.0, 200.0])


def test_add_corrected_fixations_1d_with_location_list(monkeypatch):
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'location': [[100.0, 105.0], [200.0, 198.0]],
    })
    aois_df = pl.DataFrame({
        'start_y': [80.0, 180.0],
        'height': [40.0, 40.0],
    })

    monkeypatch.setattr(
        'pymovements.events.correction.fixation_correction.create_corrected_fixations_locations',
        lambda *args, **kwargs: np.array([100.0, 200.0]),
    )

    res_df = add_corrected_fixations(events_df, aois_df, algorithm='attach')
    assert res_df.height == 4
    corrected_rows = res_df.filter(pl.col('name') == 'fixation_corrected_attach')
    assert corrected_rows['location'].to_list() == [[100.0, 100.0], [200.0, 200.0]]
