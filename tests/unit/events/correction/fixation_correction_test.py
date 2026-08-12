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
from pymovements.events.correction.fixation_correction import _get_word_xy_from_aois
from pymovements.events.correction.fixation_correction import _has_word_x_coords
from pymovements.events.correction.fixation_correction import correct_fixation_locations
from pymovements.events.correction.fixation_correction import correct_fixations


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

    assert _has_word_x_coords(aois_df) is True
    assert _has_word_x_coords(aois_no_x) is False


def test_correct_fixation_locations_default_woc(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = correct_fixation_locations(events_df, aois_df)
    assert locs.shape == (6, 2)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])


def test_correct_fixation_locations_specific_algos(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    algos = (
        'attach', 'chain', 'cluster', 'compare', 'merge',
        'regress', 'segment', 'slice', 'split', 'stretch', 'warp',
    )
    for algo in algos:
        locs = correct_fixation_locations(events_df, aois_df, algorithm=algo)
        assert locs.shape == (6, 2)


def test_correct_fixation_locations_woc_custom_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm=['attach', 'chain', 'cluster'],
    )
    assert locs.shape == (6, 2)


def test_correct_fixation_locations_single_element_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm=['attach'],
    )
    assert locs.shape == (6, 2)


def test_correct_fixation_locations_empty_list_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(ValueError, match='At least one algorithm must be provided'):
        correct_fixation_locations(events_df, aois_df, algorithm=[])


def test_correct_fixation_locations_woc_string(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm='wisdom_of_the_crowd',
    )
    assert locs.shape == (6, 2)


def test_correct_fixation_locations_woc_routes_algorithm_specific_kwargs(
    sample_events_and_aois,
):
    """Algorithm-specific kwargs must not break ensemble algorithms that do not accept them."""
    events_df, aois_df = sample_events_and_aois
    # x_thresh is only accepted by chain, compare and slice.
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm_kwargs={'x_thresh': 250.0},
    )
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])


def test_correct_fixation_locations_woc_right_to_left():
    """Right-to-Left reading support must work with the default ensemble."""
    events_df = pl.DataFrame({
        'name': ['fixation'] * 4,
        'location': [
            [800.0, 105.0], [100.0, 102.0], [800.0, 198.0], [100.0, 201.0],
        ],
    })
    aois_df = pl.DataFrame({
        'start_x': [700.0, 50.0, 700.0, 50.0],
        'end_x': [900.0, 150.0, 900.0, 150.0],
        'start_y': [80.0, 80.0, 180.0, 180.0],
        'end_y': [120.0, 120.0, 220.0, 220.0],
        'height': [40.0] * 4,
    })
    with pytest.warns(
        UserWarning, match="'compare' does not support right-to-left reading",
    ):
        locs = correct_fixation_locations(events_df, aois_df, text_right_to_left=True)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0])


def test_correct_fixation_locations_single_compare_right_to_left_raises(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(
        ValueError, match="'compare' does not support right-to-left reading",
    ):
        correct_fixation_locations(
            events_df, aois_df, algorithm='compare', text_right_to_left=True,
        )


def test_correct_fixation_locations_woc_unknown_kwarg_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(ValueError, match=r"\['bogus_thresh'\] are not accepted"):
        correct_fixation_locations(
            events_df, aois_df, algorithm_kwargs={'bogus_thresh': 1.0},
        )


def test_correct_fixation_locations_reserved_algorithm_kwargs_raise(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(ValueError, match="'text_right_to_left' must be passed as an explicit"):
        correct_fixation_locations(
            events_df, aois_df, algorithm_kwargs={'text_right_to_left': True},
        )


def test_correct_fixation_locations_unknown_algorithm_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(ValueError, match="Unknown drift algorithm 'atach'"):
        correct_fixation_locations(events_df, aois_df, algorithm='atach')

    with pytest.raises(ValueError, match=r"Unknown drift algorithms \['atach'\]"):
        correct_fixation_locations(
            events_df, aois_df, algorithm=['attach', 'atach'],
        )


def test_correct_fixation_locations_invalid_type_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    with pytest.raises(TypeError, match='algorithm must be a string or a list of strings'):
        correct_fixation_locations(
            events_df, aois_df, algorithm=123,  # type: ignore[arg-type]
        )


def test_correct_fixation_locations_missing_word_x_coords_warns(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    aois_no_x = aois_df.drop(['start_x', 'end_x'])
    with pytest.warns(
        UserWarning, match=r"Word X coordinates \('start_x', 'end_x'\) are missing",
    ):
        locs = correct_fixation_locations(events_df, aois_no_x)
        assert locs.shape == (6, 2)

    with pytest.warns(
        UserWarning, match=r"Word X coordinates \('start_x', 'end_x'\) are missing",
    ):
        locs2 = correct_fixation_locations(
            events_df, aois_no_x, algorithm=['attach', 'compare'],
        )
        assert locs2.shape == (6, 2)


def test_correct_fixation_locations_single_compare_missing_x_coords_raises(
    sample_events_and_aois,
):
    events_df, aois_df = sample_events_and_aois
    aois_no_x = aois_df.drop(['start_x', 'end_x'])
    with pytest.raises(ValueError, match="Algorithm 'compare' requires word X coordinates"):
        correct_fixation_locations(events_df, aois_no_x, algorithm='compare')


def test_correct_fixation_locations_explicit_word_xy(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    word_XY = np.array([[100.0, 100.0], [200.0, 200.0]])
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm='compare', word_XY=word_XY,
        algorithm_kwargs={'n_nearest_lines': 2},
    )
    assert locs.shape == (6, 2)


def test_correct_fixation_locations_default_woc_two_line_text():
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
    locs = correct_fixation_locations(events_df, aois_df)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0])


def test_correct_fixation_locations_woc_votes_on_line_indices(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    # Explicit word_XY with y-values offset from the AOI line centers: index-based voting
    # must still map all ensemble votes onto the AOI line centers.
    word_XY = np.array([
        [125.0, 95.0], [325.0, 95.0],
        [125.0, 195.0], [325.0, 195.0],
        [125.0, 295.0], [325.0, 295.0],
    ])
    locs = correct_fixation_locations(events_df, aois_df, word_XY=word_XY)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 100.0, 200.0, 200.0, 300.0, 300.0])


def test_correct_fixation_locations_woc_word_xy_without_aoi_coordinates():
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
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm=['compare', 'warp'], word_XY=word_XY,
    )
    assert set(locs[:, 1]).issubset({100.0, 200.0, 300.0})


def test_correct_fixation_locations_invalid_location_raises():
    events_df = pl.DataFrame({'name': ['fixation'], 'trial': ['TRIAL1']})
    aois_df = pl.DataFrame({'start_y': [80.0], 'height': [40.0]})
    with pytest.raises(ValueError, match='No valid location coordinates found'):
        correct_fixation_locations(events_df, aois_df)


def test_correct_fixations_default_woc(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    res_df = correct_fixations(events_df, aois_df, trial_columns='trial')
    assert res_df.height == 6
    corrected_rows = res_df.filter(pl.col('correction_algorithm') == 'wisdom_of_the_crowd')
    assert corrected_rows.height == 6
    assert corrected_rows['name'].to_list() == ['fixation'] * 6
    corrected_y = [location[1] for location in corrected_rows['location'].to_list()]
    assert corrected_y == [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]
    # Original locations are preserved.
    assert corrected_rows['location_original'].to_list() == events_df['location'].to_list()


def test_correct_fixations_algorithm_list(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    res_df = correct_fixations(
        events_df, aois_df, algorithm=['attach', 'chain'],
    )
    assert res_df.filter(pl.col('correction_algorithm') == 'wisdom_of_the_crowd').height == 6

    res_single = correct_fixations(
        events_df, aois_df, algorithm=['attach'],
    )
    assert res_single.filter(pl.col('correction_algorithm') == 'attach').height == 6


def test_correct_fixations_multiple_trials_corrected_independently(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events_two_trials = pl.concat([
        events_df,
        events_df.with_columns(pl.lit('TRIAL2').alias('trial')),
    ])
    aois_two_trials = pl.concat([
        aois_df,
        aois_df.with_columns(pl.lit('TRIAL2').alias('trial')),
    ])

    single_trial_result = correct_fixations(events_df, aois_df, algorithm='segment')
    expected_locations = single_trial_result['location'].to_list()

    res_df = correct_fixations(
        events_two_trials, aois_two_trials, algorithm='segment', trial_columns='trial',
    )
    assert res_df.height == 12
    assert res_df.filter(pl.col('correction_algorithm') == 'segment').height == 12

    # Identical trials must receive identical corrections, each matching the single-trial result.
    for trial in ('TRIAL1', 'TRIAL2'):
        trial_locations = res_df.filter(pl.col('trial') == trial)['location'].to_list()
        assert trial_locations == expected_locations


def test_correct_fixations_rerun_raises(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    once = correct_fixations(events_df, aois_df, algorithm='attach')
    with pytest.raises(ValueError, match="'fixation' events have already been corrected"):
        correct_fixations(once, aois_df, algorithm='chain')


def test_correct_fixations_custom_fixation_name(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events_named = events_df.with_columns(pl.lit('fixation_left').alias('name'))
    res_df = correct_fixations(
        events_named, aois_df, algorithm='attach', fixation_name='fixation_left',
    )
    corrected_rows = res_df.filter(pl.col('correction_algorithm') == 'attach')
    assert corrected_rows.height == 6
    assert corrected_rows['name'].to_list() == ['fixation_left'] * 6


def test_events_correct_fixations(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events = pm.Events(events_df, trial_columns='trial')
    result = events.correct_fixations(aois_df, algorithm='attach')
    assert result is None
    corrected_rows = events.frame.filter(pl.col('correction_algorithm') == 'attach')
    assert corrected_rows.height == 6


def test_events_correct_fixations_not_inplace(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events = pm.Events(events_df, trial_columns='trial')
    result = events.correct_fixations(aois_df, algorithm='attach', inplace=False)
    assert 'correction_algorithm' not in events.frame.columns  # original object unchanged
    assert result is not None
    assert result.trial_columns == ['trial']
    assert result.frame.filter(pl.col('correction_algorithm') == 'attach').height == 6


def test_events_correct_fixations_with_text_stimulus(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    stimulus = pm.stimulus.TextStimulus(
        aois=aois_df,
        aoi_column='word',
        start_x_column='start_x',
        start_y_column='start_y',
        end_x_column='end_x',
        end_y_column='end_y',
    )
    events = pm.Events(events_df, trial_columns='trial')
    events.correct_fixations(stimulus, algorithm='attach')
    corrected_rows = events.frame.filter(pl.col('correction_algorithm') == 'attach')
    assert corrected_rows.height == 6


def test_correct_fixations_missing_trial_columns_raises():
    events_df = pl.DataFrame({
        'name': ['fixation'],
        'location': [[100.0, 105.0]],
    })
    aois_df = pl.DataFrame({'start_y': [80.0], 'height': [40.0]})
    with pytest.raises(ValueError, match=r"trial columns \['trial'\] are missing"):
        correct_fixations(events_df, aois_df, trial_columns='trial')


def test_correct_fixations_empty_fixations(sample_events_and_aois):
    _, aois_df = sample_events_and_aois
    empty_events_df = pl.DataFrame({'name': ['saccade'], 'trial': ['TRIAL1']})
    res_df = correct_fixations(empty_events_df, aois_df)
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


def test_correct_fixation_locations_warp_returns_line_centers():
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
    locs = correct_fixation_locations(events_df, aois_df, algorithm='warp')
    assert set(locs[:, 1]).issubset(set(line_Y))


def test_correct_fixation_locations_compare_varying_word_centers():
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
    locs = correct_fixation_locations(
        events_df, aois_df, algorithm='compare', algorithm_kwargs={'n_nearest_lines': 2},
    )
    assert set(locs[:, 1]).issubset(set(line_Y))


def test_correct_fixation_locations_split_columns():
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'location_x': [100.0, 200.0],
        'location_y': [105.0, 198.0],
    })
    aois_df = pl.DataFrame({
        'start_y': [80.0, 180.0],
        'height': [40.0, 40.0],
    })
    locs = correct_fixation_locations(events_df, aois_df, algorithm='attach')
    assert locs.shape == (2, 2)
    np.testing.assert_array_equal(locs[:, 1], [100.0, 200.0])


def test_correct_fixations_split_columns(sample_events_and_aois):
    _, aois_df = sample_events_and_aois
    events_df = pl.DataFrame({
        'name': ['fixation', 'saccade', 'fixation'],
        'location_x': [100.0, 150.0, 200.0],
        'location_y': [105.0, 150.0, 198.0],
    })

    res_df = correct_fixations(events_df, aois_df.head(4), algorithm='attach')

    assert res_df.height == 3
    assert res_df['location_x'].to_list() == [100.0, 150.0, 200.0]
    assert res_df['location_y'].to_list() == [100.0, 150.0, 200.0]
    assert res_df['location_x_original'].to_list() == [100.0, None, 200.0]
    assert res_df['location_y_original'].to_list() == [105.0, None, 198.0]
    assert res_df['correction_algorithm'].to_list() == ['attach', None, 'attach']


def test_correct_fixations_preserves_non_fixation_rows(sample_events_and_aois):
    events_df, aois_df = sample_events_and_aois
    events_with_saccade = pl.concat([
        events_df,
        pl.DataFrame({
            'trial': ['TRIAL1'],
            'name': ['saccade'],
            'onset': [50],
            'location': [[150.0, 150.0]],
        }),
    ])

    res_df = correct_fixations(events_with_saccade, aois_df, algorithm='attach')

    saccade_row = res_df.filter(pl.col('name') == 'saccade')
    assert saccade_row['location'].to_list() == [[150.0, 150.0]]
    assert saccade_row['location_original'].to_list() == [None]
    assert saccade_row['correction_algorithm'].to_list() == [None]
    # Row order is unchanged.
    assert res_df['name'].to_list() == events_with_saccade['name'].to_list()
