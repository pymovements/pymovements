# Copyright (c) 2026 The pymovements Project Authors
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
"""Reading measure tests."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events import Events
from pymovements.measure.reading import ReadingMeasures
from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_fixation_count
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import first_reading_time
from pymovements.measure.reading.measures import landing_position
from pymovements.measure.reading.measures import non_aoi_fixation_count_ratio
from pymovements.measure.reading.measures import non_aoi_fixation_duration_ratio
from pymovements.measure.reading.measures import regression_count_in
from pymovements.measure.reading.measures import regression_count_out
from pymovements.measure.reading.measures import regression_path_duration_exclusive
from pymovements.measure.reading.measures import regression_path_duration_inclusive
from pymovements.measure.reading.measures import rereading_time
from pymovements.measure.reading.measures import right_bounded_reading_time
from pymovements.measure.reading.measures import saccade_length_in
from pymovements.measure.reading.measures import saccade_length_out
from pymovements.measure.reading.measures import total_fixation_count
from pymovements.measure.reading.processing import compute_reading_measures
from pymovements.stimulus.text import TextStimulus


# Synthetic char-level AOI table
# "The quick" — 2 words, each 3 chars, on page_1
CHAR_AOI_DF = pl.DataFrame({
    'char': ['T', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k'],
    'char_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'word_idx': [0, 0, 0, 0, 1, 1, 1, 1, 1],  # blank space is part of the previous word
    'word': ['The', 'The', 'The', 'The', 'quick', 'quick', 'quick', 'quick', 'quick'],
    'top_left_x': [10., 20., 30., 40., 50., 60., 70., 80., 90.],
    'top_left_y': [10., 10., 10., 10., 10., 10., 10., 10., 10.],
    'width': [10., 10., 10., 10., 10., 10., 10., 10., 10.],
    'height': [20., 20., 20., 20., 20., 20., 20., 20., 20.],
    'page': ['page_1'] * 9,
    'trial': ['trial_1'] * 9,
})


@pytest.fixture(name='stimulus', scope='function')
def fixture_stimulus():
    return TextStimulus(
        aois=CHAR_AOI_DF.clone(),
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        trial_column='trial',
    )


@pytest.fixture(name='mapped_events', scope='function')
def fixture_mapped_events(stimulus):
    events_df = pl.DataFrame({
        'name': ['fixation', 'fixation'],
        'onset': [0, 200],
        'offset': [200, 400],
        'duration': [200, 200],
        'location': [[15., 15.], [55., 15.]],
        'trial': ['trial_1', 'trial_1'],
        'page': ['page_1', 'page_1'],
    })
    events = Events(data=events_df)
    events.map_to_aois(stimulus)
    return events.frame


@pytest.fixture(name='annotated_events', scope='function')
def fixture_annotated_events(mapped_events):
    return annotate_fixations(mapped_events, group_columns=['trial', 'page'])


def test_fixture_mapped_events_has_word_and_char_columns(mapped_events):
    assert 'word_idx' in mapped_events.columns
    assert 'word' in mapped_events.columns
    assert 'char_idx' in mapped_events.columns


def test_annotate_fixations(annotated_events):
    assert 'is_first_pass' in annotated_events.columns
    assert 'run_id' in annotated_events.columns
    assert annotated_events.height == 2


def test_reading_measures_init_none():
    reading_measures = ReadingMeasures()
    assert isinstance(reading_measures.frame, pl.DataFrame)
    assert reading_measures.frame.is_empty()


def test_reading_measures_init_df():
    df = pl.DataFrame({'a': [1, 2, 3]})
    reading_measures = ReadingMeasures(df)
    assert isinstance(reading_measures.frame, pl.DataFrame)
    assert reading_measures.frame.shape == (3, 1)
    assert reading_measures.frame['a'].to_list() == [1, 2, 3]


def test_compute_reading_measures_preserves_zero_based_word_indices():
    aois = pl.DataFrame({
        'word_idx': [0, 0, 1, 1],
        'word': ['zero', 'zero', 'one', 'one'],
    })
    fixations = pl.DataFrame({
        'word_idx': [0, 1],
        'duration': [100, 200],
    })

    result = compute_reading_measures(fixations, aois)

    assert result['word_index'].to_list() == [0, 1]
    assert result['word'].to_list() == ['zero', 'one']


def test_compute_reading_measures_word_level(mapped_events):
    result = compute_reading_measures(mapped_events, CHAR_AOI_DF.clone())
    assert result.height == 2  # one row per word
    assert 'FFD' in result.columns
    assert 'TFT' in result.columns
    assert result.columns[:4] == ['trial', 'page', 'word_index', 'word']
    # both fixations land on the first character of their word
    assert result['LP'].to_list() == [0, 0]
    assert result.filter(pl.col('word') == 'The')['FFD'][0] == 200
    assert result.filter(pl.col('word') == 'quick')['FFD'][0] == 200


def _aggregate(annotated_events, expression):
    return annotated_events.group_by(['trial', 'page', 'word_idx']).agg(expression)


def test_compute_first_duration(annotated_events):
    result = _aggregate(annotated_events, first_duration())
    assert 'FD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FD'][0] == 200
    assert result.filter(pl.col('word_idx') == 1)['FD'][0] == 200


def test_compute_first_fixation_duration(annotated_events):
    result = _aggregate(annotated_events, first_fixation_duration())
    assert 'FFD' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FFD'][0] == 200


def test_compute_first_pass_reading_time(annotated_events):
    result = _aggregate(annotated_events, first_pass_reading_time())
    assert 'FPRT' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FPRT'][0] == 200


def test_compute_total_fixation_count(annotated_events):
    result = _aggregate(annotated_events, total_fixation_count())
    assert 'TFC' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['TFC'][0] == 1
    assert result.filter(pl.col('word_idx') == 1)['TFC'][0] == 1


def test_first_pass_fixation_count(annotated_events):
    result = _aggregate(annotated_events, first_pass_fixation_count())
    assert 'FPFC' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FPFC'][0] == 1


def test_first_reading_time(annotated_events):
    result = _aggregate(annotated_events, first_reading_time())
    assert 'FRT' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['FRT'][0] == 200


def test_rereading_time(annotated_events):
    result = _aggregate(annotated_events, rereading_time())
    assert 'RRT' in result.columns
    assert result['RRT'].to_list() == [0, 0]


def test_measures_aggregate_without_grouping(annotated_events):
    # Expression measures work with any grouping the caller chooses, including word-only.
    result = annotated_events.group_by('word_idx').agg(total_fixation_count())
    assert sorted(result['TFC'].to_list()) == [1, 1]


def test_regression_count_in(annotated_events):
    result = _aggregate(annotated_events, regression_count_in())
    assert 'TRC_in' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['TRC_in'][0] == 0


def test_regression_count_out(annotated_events):
    result = _aggregate(annotated_events, regression_count_out())
    assert 'TRC_out' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['TRC_out'][0] == 0


def test_landing_position(annotated_events):
    result = _aggregate(annotated_events, landing_position())
    assert 'LP' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['LP'][0] == 0


def test_saccade_length_in(annotated_events):
    result = _aggregate(annotated_events, saccade_length_in())
    assert 'SL_in' in result.columns
    assert result.filter(pl.col('word_idx') == 1)['SL_in'][0] == 1


def test_saccade_length_out(annotated_events):
    result = _aggregate(annotated_events, saccade_length_out())
    assert 'SL_out' in result.columns
    assert result.filter(pl.col('word_idx') == 0)['SL_out'][0] == 1


def test_regression_path_duration(annotated_events):
    result = annotated_events.group_by(['trial', 'page', 'regression_path_word']).agg([
        regression_path_duration_inclusive(),
        regression_path_duration_exclusive(),
        right_bounded_reading_time(),
    ])
    assert result.filter(pl.col('regression_path_word') == 0)['RPD_inc'][0] == 200
    assert result.filter(pl.col('regression_path_word') == 0)['RPD_exc'][0] == 0
    assert result.filter(pl.col('regression_path_word') == 0)['RBRT'][0] == 200


def test_regression_path_duration_with_regression():
    # Sequence 2 -> 1 -> 2: word 1 is entered from the right, so it never opens a
    # regression-path window; both regression fixations belong to word 2's window.
    events = pl.DataFrame({
        'name': ['fixation'] * 3,
        'word_idx': [2, 1, 2],
        'onset': [0, 100, 200],
        'duration': [100, 150, 120],
    })
    annotated = annotate_fixations(events)

    result = annotated.group_by('regression_path_word').agg([
        regression_path_duration_inclusive(),
        regression_path_duration_exclusive(),
        right_bounded_reading_time(),
    ])

    assert result['regression_path_word'].to_list() == [2]
    assert result['RPD_inc'][0] == 370
    assert result['RPD_exc'][0] == 150
    assert result['RBRT'][0] == 220


# ---------------------------
# non_aoi_fixation_count_ratio
# ---------------------------


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        # All fixations inside AOI -> NAFCR == 0.0
        (
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'page': ['1', '1', '1'],
                'word_idx': [0, 1, 2],
            }),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFCR': [0.0],
            }),
        ),
        # All fixations outside AOI -> NAFCR == 1.0
        (
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': ['1', '1'],
                'word_idx': [None, None],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFCR': [1.0],
            }),
        ),
        # Mix of inside / outside fixations -> NAFCR == 0.5
        (
            pl.DataFrame({
                'trial': ['1', '1', '1', '1'],
                'page': ['1', '1', '1', '1'],
                'word_idx': [0, None, 1, None],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFCR': [0.5],
            }),
        ),
        # Multiple trials get separate NAFCR values
        (
            pl.DataFrame({
                'trial': ['1', '1', '2', '2', '2'],
                'page': ['1', '1', '1', '1', '1'],
                'word_idx': [0, None, None, None, 0],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1', '2'],
                'page': ['1', '1'],
                'NAFCR': [0.5, 2 / 3],
            }),
        ),
        # Empty fixations table yields empty output DataFrame
        (
            pl.DataFrame(schema={'trial': pl.String, 'page': pl.String, 'word_idx': pl.Int64}),
            pl.DataFrame(schema={'trial': pl.String, 'page': pl.String, 'NAFCR': pl.Float64}),
        ),
    ],
)
def test_non_aoi_fixation_count_ratio(fixations, expected):
    result = fixations.group_by(['trial', 'page'], maintain_order=True).agg(
        non_aoi_fixation_count_ratio(),
    )
    assert_frame_equal(result, expected)


# ---------------------------
# non_aoi_fixation_duration_ratio
# ---------------------------


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        # All fixations inside AOI -> NAFDR == 0.0
        (
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': ['1', '1'],
                'word_idx': [0, 1],
                'duration': [100, 200],
            }),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFDR': [0.0],
            }),
        ),
        # All fixations outside AOI -> NAFDR == 1.0
        (
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': ['1', '1'],
                'word_idx': [None, None],
                'duration': [50, 150],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFDR': [1.0],
            }),
        ),
        # Mixed fixations -> NAFDR == 150 / 400
        (
            pl.DataFrame({
                'trial': ['1', '1', '1', '1'],
                'page': ['1', '1', '1', '1'],
                'word_idx': [0, None, 1, None],
                'duration': [100, 50, 150, 100],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1'],
                'page': ['1'],
                'NAFDR': [150 / 400],
            }),
        ),
        # Zero total duration yields None for NAFDR
        (
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': ['1', '1'],
                'word_idx': [0, 1],
                'duration': [0, 0],
            }),
            pl.DataFrame(
                {
                    'trial': ['1'],
                    'page': ['1'],
                    'NAFDR': [None],
                },
                schema={'trial': pl.String, 'page': pl.String, 'NAFDR': pl.Float64},
            ),
        ),
        # Multiple trials get separate NAFDR values
        (
            pl.DataFrame({
                'trial': ['1', '1', '2', '2'],
                'page': ['1', '1', '1', '1'],
                'word_idx': [0, None, None, 0],
                'duration': [100, 100, 300, 100],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1', '2'],
                'page': ['1', '1'],
                'NAFDR': [0.5, 0.75],
            }),
        ),
        # Multiple pages get separate NAFDR values
        (
            pl.DataFrame({
                'trial': ['1', '1', '1', '1'],
                'page': ['p1', 'p1', 'p2', 'p2'],
                'word_idx': [0, None, None, None],
                'duration': [100, 100, 200, 200],
            }).cast({'word_idx': pl.Int64}),
            pl.DataFrame({
                'trial': ['1', '1'],
                'page': ['p1', 'p2'],
                'NAFDR': [0.5, 1.0],
            }),
        ),
    ],
)
def test_non_aoi_fixation_duration_ratio(fixations, expected):
    result = fixations.group_by(['trial', 'page'], maintain_order=True).agg(
        non_aoi_fixation_duration_ratio(),
    )
    assert_frame_equal(result, expected, abs_tol=1e-5)
