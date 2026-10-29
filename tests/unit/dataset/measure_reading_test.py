# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Reading measure processing tests for Dataset."""
from pathlib import Path

import polars as pl
import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import Events
from pymovements import Gaze


@pytest.fixture(name='dummy_dataset')
def fixture_dummy_dataset(tmp_path):
    """Create a dummy dataset with some fixation events."""
    definition = DatasetDefinition(name='dummy')
    dataset = Dataset(definition, path=tmp_path)

    # We need 'subject_id' and 'text_id' in trial_columns for measure_reading to work
    fixation_data = pl.DataFrame({
        'name': ['fixation', 'fixation', 'fixation', 'fixation'],
        'onset': [0, 200, 400, 600],
        'offset': [100, 300, 500, 700],
        'duration': [100, 100, 100, 100],
        'location_x': [100, 140, 200, 10000],  # 100->AOI 1, 140->AOI 2, 200->AOI 3
        'location_y': [50, 50, 50, 50],
        'subject_id': [5, 5, 5, 5],
        'text_id': ['b0', 'b0', 'b0', 'b0'],
    })
    events = Events(
        fixation_data,
        trial_columns=['subject_id', 'text_id'],
        metadata={'Sources': ['raw/sub_5.csv']},
    )
    dataset.gaze = [Gaze(events=events)]

    return dataset


def test_compute_reading_measures(dummy_dataset, make_example_file):
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    reading_measures = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=False,
    )

    expected_columns = [
        'word_index', 'word', 'subject_id', 'text_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FPFC',
        'FRT', 'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'skipped', 'FPF', 'RR',
        'FPReg', 'TRC_out', 'TRC_in', 'SL_in', 'SL_out', 'LP', 'TFC',
    ]
    result_frame = reading_measures.frame

    assert set(result_frame.columns) == set(expected_columns)

    assert len(result_frame) > 0
    assert (result_frame['subject_id'] == 5).all()
    assert (result_frame['text_id'] == 'b0').all()

    # The sources union the event sources with the AOI stimulus file. The AOI file lies
    # below the dataset root and is therefore recorded relative to it.
    assert reading_measures.metadata['Sources'] == ['raw/sub_5.csv', aoi_path.name]

    # The mapping happens on a copy. The dataset's events keep their own metadata.
    assert dummy_dataset.gaze[0].events.metadata['Sources'] == ['raw/sub_5.csv']


def test_compute_reading_measures_save(dummy_dataset, tmp_path, make_example_file):
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    dummy_dataset.measure_reading(
        aoi_dict,
        save_path=tmp_path,
        word_index_column='aoi',
        word_column='character',
        annotate_events=False,
    )

    expected_columns = [
        'word_index', 'word', 'subject_id', 'text_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FPFC',
        'FRT', 'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'skipped', 'FPF', 'RR',
        'FPReg', 'TRC_out', 'TRC_in', 'SL_in', 'SL_out', 'LP', 'TFC',
    ]
    expected_file = Path(tmp_path) / '5-b0-reading_measures.csv'

    assert expected_file.is_file()
    saved_df = pl.read_csv(expected_file)
    assert set(saved_df.columns) == set(expected_columns)


def test_measure_reading_default_keeps_single_sequence(dummy_dataset, make_example_file):
    """A varying ``trial`` column must not silently split a subject-text under the default."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    events = dummy_dataset.gaze[0].events
    events.frame = events.frame.with_columns(pl.Series('trial', [1, 1, 2, 2]))

    reading_measures = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=False,
    )

    assert 'trial' not in reading_measures.frame.columns


def test_measure_reading_group_columns_partition_output(dummy_dataset, make_example_file):
    """An explicit ``group_columns`` computes measures per group and keeps the column."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    events = dummy_dataset.gaze[0].events
    events.frame = events.frame.with_columns(pl.Series('trial', [1, 1, 2, 2]))

    with pytest.warns(UserWarning, match='broadcast'):
        reading_measures = dummy_dataset.measure_reading(
            aoi_dict,
            group_columns=['trial'],
            word_index_column='aoi',
            word_column='character',
            annotate_events=False,
        )

    assert 'trial' in reading_measures.frame.columns


def test_measure_reading_event_name_selects_events(dummy_dataset, make_example_file):
    """``event_name`` selects the events the measures are computed from."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}
    expected = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=False,
    )

    # map_to_aois only maps event names starting with 'fixation'.
    events = dummy_dataset.gaze[0].events
    events.frame = events.frame.with_columns(pl.lit('fixation_ivt').alias('name'))

    reading_measures = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        event_name='fixation_ivt',
        annotate_events=False,
    )

    assert reading_measures.frame.equals(expected.frame)


def test_measure_reading_default_warns_and_annotates_events(
        dummy_dataset, make_example_file, assert_deprecation_is_removed,
):
    """The default annotates the dataset's events in place and warns about the deprecation."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}
    events = dummy_dataset.gaze[0].events

    with pytest.warns(DeprecationWarning, match='annotate_events') as info:
        reading_measures = dummy_dataset.measure_reading(
            aoi_dict,
            word_index_column='aoi',
            word_column='character',
        )

    assert_deprecation_is_removed(
        function_name='Dataset.measure_reading default annotate_events=None',
        warning_message=str(info[0].message),
        scheduled_version='0.34.0',
    )

    assert len(reading_measures.frame) > 0
    assert {'aoi', 'character'} <= set(events.frame.columns)
    # Annotating in place records the AOI file as the events read it, not relative to the root.
    assert events.metadata['Sources'] == ['raw/sub_5.csv', aoi_path.as_posix()]


def test_measure_reading_annotate_events_true_annotates_without_warning(
        dummy_dataset, make_example_file,
):
    """``annotate_events=True`` keeps the in-place annotation and stays silent."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}
    events = dummy_dataset.gaze[0].events

    reading_measures = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=True,
    )

    assert len(reading_measures.frame) > 0
    assert {'aoi', 'character'} <= set(events.frame.columns)
    # Annotating in place records the AOI file as the events read it, not relative to the root.
    assert events.metadata['Sources'] == ['raw/sub_5.csv', aoi_path.as_posix()]


def test_measure_reading_annotate_events_false_keeps_events(dummy_dataset, make_example_file):
    """``annotate_events=False`` leaves the events untouched and computes the same measures."""
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}
    events = dummy_dataset.gaze[0].events
    expected_frame = events.frame.clone()

    untouched = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=False,
    )

    assert events.frame.equals(expected_frame)
    assert events.metadata['Sources'] == ['raw/sub_5.csv']

    annotated = dummy_dataset.measure_reading(
        aoi_dict,
        word_index_column='aoi',
        word_column='character',
        annotate_events=True,
    )

    assert untouched.frame.equals(annotated.frame)
    assert untouched.metadata == annotated.metadata


def test_compute_reading_measures_empty_dataset(tmp_path):
    definition = DatasetDefinition(name='dummy')
    dataset = Dataset(definition, path=tmp_path)

    empty_events = Events(
        pl.DataFrame(
            schema={
                'name': pl.String,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
                'location_x': pl.Float64,
                'location_y': pl.Float64,
                'subject_id': pl.Int64,
                'text_id': pl.String,
            },
        ),
        trial_columns=['subject_id', 'text_id'],
    )

    dataset.gaze = [Gaze(events=empty_events)]

    reading_measures = dataset.measure_reading(aoi_dict={}, annotate_events=False)

    assert reading_measures.frame.is_empty()
    assert isinstance(reading_measures.frame, pl.DataFrame)
    assert reading_measures.metadata == {}
