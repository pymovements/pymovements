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
    definition = DatasetDefinition(name='dummy')
    dataset = Dataset(definition, path=tmp_path)

    # We need 'subject_id' and 'text_id' in trial_columns for compute_reading_measures to work
    fixation_data = pl.DataFrame({
        'name': ['fixation', 'fixation', 'fixation', 'fixation'],
        'onset': [0, 200, 400, 600],
        'offset': [100, 300, 500, 700],
        'duration': [100, 100, 100, 100],
        'location_x': [100, 140, 200, 10000],  # 100->AOI 1, 140->AOI 2, 200->AOI 3
        'location_y': [50, 50, 50, 50],  # y=50 is within y bounds (21-99)
        'subject_id': [5, 5, 5, 5],
        'text_id': ['b0', 'b0', 'b0', 'b0'],
    })
    events = Events(fixation_data, trial_columns=['subject_id', 'text_id'])

    empty_events = Events(
        pl.DataFrame(schema=fixation_data.schema),
        trial_columns=['subject_id', 'text_id'],
    )

    gaze1 = Gaze(events=events)
    gaze2 = Gaze(events=empty_events)
    dataset.gaze = [gaze1, gaze2]

    return dataset


def test_compute_reading_measures(dummy_dataset, make_example_file):
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    reading_measures = dummy_dataset.compute_reading_measures(aoi_dict)

    expected_columns = [
        'word_index', 'word', 'subject_id', 'text_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FRT',
        'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC',
    ]
    result_frame = reading_measures.frame

    assert set(result_frame.columns) == set(expected_columns)

    assert len(result_frame) > 0
    assert (result_frame['subject_id'] == 5).all()
    assert (result_frame['text_id'] == 'b0').all()


def test_compute_reading_measures_save(dummy_dataset, tmp_path, make_example_file):
    aoi_path = make_example_file('potec_word_aoi_b0.tsv')
    aoi_dict = {'b0': aoi_path}

    dummy_dataset.compute_reading_measures(aoi_dict, save_path=tmp_path)

    expected_columns = [
        'word_index', 'word', 'subject_id', 'text_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FRT',
        'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC',
    ]
    expected_file = Path(tmp_path) / '5-b0-reading_measures.csv'

    assert expected_file.is_file()
    saved_df = pl.read_csv(expected_file)
    assert set(saved_df.columns) == set(expected_columns)
