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

from pymovements.events.correction.fixation_correction import _get_lines_of_text_from_aois
from pymovements.events.correction.fixation_correction import _get_word_xy_from_aois
from pymovements.events.correction.fixation_correction import _has_word_x_coords
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


def test_has_word_x_coords(sample_events_and_aois):
    _, aois_df = sample_events_and_aois
    aois_no_x = aois_df.drop(['start_x', 'end_x'])

    assert _has_word_x_coords(aois_no_x, {'word_XY': np.array([[5, 5]])}) is True
    assert _has_word_x_coords(aois_df, {}) is True
    assert _has_word_x_coords(aois_no_x, {}) is False


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


def test_create_corrected_fixations_locations_woc_custom_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm=['attach', 'chain', 'cluster'],
    )
    assert locs.shape == (6, 2)


def test_create_corrected_fixations_locations_single_element_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm=['attach'],
    )
    assert locs.shape == (6, 2)


def test_create_corrected_fixations_locations_empty_list_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(ValueError, match='At least one algorithm must be provided'):
        create_corrected_fixations_locations(events_df, aois_df, algorithm=[])


def test_create_corrected_fixations_locations_woc_string(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm='wisdom_of_the_crowd',
    )
    assert locs.shape == (6, 2)


def test_create_corrected_fixations_locations_invalid_type_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(TypeError, match='algorithm must be a string or a list of strings'):
        create_corrected_fixations_locations(
            events_df, aois_df, algorithm=123,  # type: ignore[arg-type]
        )


def test_create_corrected_fixations_locations_missing_word_x_coords_warns(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    aois_no_x = aois_df.drop(['start_x', 'end_x'])
    with pytest.warns(
        UserWarning, match=r"Word X coordinates \('start_x', 'end_x'\) are missing",
    ):
        locs = create_corrected_fixations_locations(events_df, aois_no_x)
        assert locs.shape == (6, 2)

    with pytest.warns(
        UserWarning, match=r"Word X coordinates \('start_x', 'end_x'\) are missing",
    ):
        locs2 = create_corrected_fixations_locations(
            events_df, aois_no_x, algorithm=['attach', 'compare'],
        )
        assert locs2.shape == (6, 2)


def test_create_corrected_fixations_locations_single_compare_missing_x_coords_raises(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    aois_no_x = aois_df.drop(['start_x', 'end_x'])
    with pytest.raises(ValueError, match="Algorithm 'compare' requires word X coordinates"):
        create_corrected_fixations_locations(events_df, aois_no_x, algorithm='compare')


def test_create_corrected_fixations_locations_explicit_word_xy(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    word_XY = np.array([[100.0, 100.0], [200.0, 200.0]])
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm='compare', word_XY=word_XY, n_nearest_lines=2,
    )
    assert locs.shape == (6, 2)


def test_create_corrected_fixations_locations_default_woc_two_line_text():
    """Default ensemble must handle texts with fewer lines than compare's n_nearest_lines."""
    events_df = pl.DataFrame({
        'name': ['fixation'] * 4,
        'location': [
            [100.0, 105.0], [800.0, 102.0], [100.0, 198.0], [800.0, 201.0],
        ],
    })
    aois_df = pl.DataFrame({
        'start_x': [50.0, 700.0, 50.0, 700.0],
        'end_x': [150.0, 900.0, 150.0, 900.0],
        'start_y': [80.0, 80.0, 180.0, 180.0],
        'end_y': [120.0, 120.0, 220.0, 220.0],
        'height': [40.0] * 4,
    })
    locs = create_corrected_fixations_locations(events_df, aois_df)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0])


def test_create_corrected_fixations_locations_woc_votes_on_line_indices(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    # Explicit word_XY with y-values offset from the AOI line centers: index-based voting
    # must still map all ensemble votes onto the AOI line centers.
    word_XY = np.array([
        [125.0, 95.0], [325.0, 95.0],
        [125.0, 195.0], [325.0, 195.0],
        [125.0, 295.0], [325.0, 295.0],
    ])
    locs = create_corrected_fixations_locations(events_df, aois_df, word_XY=word_XY)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])


def test_create_corrected_fixations_locations_woc_word_xy_without_aoi_coordinates():
    events_df = pl.DataFrame({
        'name': ['fixation'] * 6,
        'location': [
            [100.0, 105.0], [200.0, 102.0], [300.0, 198.0],
            [400.0, 201.0], [100.0, 305.0], [200.0, 301.0],
        ],
    })
    aois_df = pl.DataFrame({'word': ['Word1', 'Word2', 'Word3']})
    word_XY = np.array([
        [125.0, 100.0], [325.0, 100.0],
        [125.0, 200.0], [325.0, 200.0],
        [125.0, 300.0], [325.0, 300.0],
    ])
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm=['compare', 'warp'], word_XY=word_XY,
    )
    assert set(locs[:, 1]).issubset({100.0, 200.0, 300.0})


def test_create_corrected_fixations_locations_invalid_location_raises():
    events_df = pl.DataFrame({'name': ['fixation'], 'trial': ['TRIAL1']})
    aois_df = pl.DataFrame({'start_y': [80.0], 'height': [40.0]})
    with pytest.raises(ValueError, match='No valid location coordinates found'):
        create_corrected_fixations_locations(events_df, aois_df)


def test_add_corrected_fixations_default_woc(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    res_df = add_corrected_fixations(events_df, aois_df, trial_columns='trial')
    assert res_df.height == 12
    corrected_rows = res_df.filter(pl.col('name') == 'fixation_corrected_wisdom_of_the_crowd')
    assert corrected_rows.height == 6


def test_add_corrected_fixations_algorithm_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    res_df = add_corrected_fixations(
        events_df, aois_df, algorithm=['attach', 'chain'],
    )
    assert res_df.height == 12
    assert res_df.filter(pl.col('name') == 'fixation_corrected_wisdom_of_the_crowd').height == 6

    res_single = add_corrected_fixations(
        events_df, aois_df, algorithm=['attach'],
    )
    assert res_single.filter(pl.col('name') == 'fixation_corrected_attach').height == 6


def test_add_corrected_fixations_multiple_trials_corrected_independently(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events_two_trials = pl.concat([
        events_df,
        events_df.with_columns(pl.lit('TRIAL2').alias('trial')),
    ])
    aois_two_trials = pl.concat([
        aois_df,
        aois_df.with_columns(pl.lit('TRIAL2').alias('trial')),
    ])

    single_trial_result = add_corrected_fixations(events_df, aois_df, algorithm='segment')
    expected_locations = single_trial_result.filter(
        pl.col('name') == 'fixation_corrected_segment',
    )['location'].to_list()

    res_df = add_corrected_fixations(
        events_two_trials, aois_two_trials, algorithm='segment', trial_columns='trial',
    )
    corrected_rows = res_df.filter(pl.col('name') == 'fixation_corrected_segment')
    assert corrected_rows.height == 12

    # Identical trials must receive identical corrections, each matching the single-trial result.
    for trial in ('TRIAL1', 'TRIAL2'):
        trial_locations = corrected_rows.filter(pl.col('trial') == trial)['location'].to_list()
        assert trial_locations == expected_locations


def test_add_corrected_fixations_missing_trial_columns_raises():
    events_df = pl.DataFrame({
        'name': ['fixation'],
        'location': [[100.0, 105.0]],
    })
    aois_df = pl.DataFrame({'start_y': [80.0], 'height': [40.0]})
    with pytest.raises(ValueError, match=r"trial columns \['trial'\] are missing"):
        add_corrected_fixations(events_df, aois_df, trial_columns='trial')


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


def test_get_word_xy_from_aois_uses_line_center_y():
    # Word bounding box centers differ from line centers due to varying AOI heights.
    aois_df = pl.DataFrame({
        'line_idx': [0, 0, 1],
        'start_x': [50.0, 250.0, 50.0],
        'end_x': [200.0, 400.0, 200.0],
        'start_y': [80.0, 80.0, 180.0],
        'end_y': [120.0, 130.0, 220.0],
        'height': [40.0, 50.0, 40.0],
    })
    word_XY = _get_word_xy_from_aois(aois_df)
    np.testing.assert_array_equal(word_XY[:, 0], [125.0, 325.0, 125.0])
    # Word y-coordinates are the line centers, identical to _get_lines_of_text_from_aois.
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert sorted(set(word_XY[:, 1])) == line_Y


def test_create_corrected_fixations_locations_warp_returns_line_centers():
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'location': [[100.0, 105.0], [200.0, 198.0]],
    })
    # end_y offsets make word bounding box centers deviate from line centers.
    aois_df = pl.DataFrame({
        'start_x': [50.0, 250.0, 50.0, 250.0],
        'end_x': [200.0, 400.0, 200.0, 400.0],
        'start_y': [80.0, 80.0, 180.0, 180.0],
        'end_y': [121.0, 121.0, 221.0, 221.0],
        'height': [40.0, 40.0, 40.0, 40.0],
    })
    line_Y = _get_lines_of_text_from_aois(aois_df)
    locs = create_corrected_fixations_locations(events_df, aois_df, algorithm='warp')
    assert set(locs[:, 1]).issubset(set(line_Y))


def test_create_corrected_fixations_locations_compare_varying_word_centers():
    events_df = pl.DataFrame({
        'name': ['fixation'] * 4,
        'location': [
            [100.0, 105.0], [300.0, 102.0], [100.0, 198.0], [300.0, 201.0],
        ],
    })
    # Varying AOI heights within a line must not create spurious extra lines for compare.
    aois_df = pl.DataFrame({
        'line_idx': [0, 0, 1, 1],
        'start_x': [50.0, 250.0, 50.0, 250.0],
        'end_x': [200.0, 400.0, 200.0, 400.0],
        'start_y': [80.0, 75.0, 180.0, 175.0],
        'end_y': [120.0, 125.0, 220.0, 225.0],
        'height': [40.0, 50.0, 40.0, 50.0],
    })
    line_Y = _get_lines_of_text_from_aois(aois_df)
    assert len(line_Y) == 2
    locs = create_corrected_fixations_locations(
        events_df, aois_df, algorithm='compare', n_nearest_lines=2,
    )
    assert set(locs[:, 1]).issubset(set(line_Y))


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
