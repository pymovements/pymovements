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
"""Test that detected events reload from their saved files without a metadata file."""
import shutil
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import Experiment

SUBJECT_IDS = [1, 2, 3, 4, 5]


@pytest.fixture(name='dataset_path')
def fixture_dataset_path(tmp_path, testfiles_dirpath):
    raw_dirpath = tmp_path / 'raw'
    raw_dirpath.mkdir()
    for subject_id in SUBJECT_IDS:
        shutil.copy2(
            testfiles_dirpath / 'monocular_example.csv',
            raw_dirpath / f'{subject_id}.csv',
        )
    return tmp_path


def make_dataset(path: Path) -> Dataset:
    definition = DatasetDefinition(
        resources=[{
            'content': 'gaze',
            'filename_pattern': '{subject_id:d}.csv',
            'load_kwargs': {
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
            },
        }],
        experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
    )
    return Dataset(definition, path=path)


def test_detected_events_reload_without_metadata_file_or_keyword(dataset_path):
    dataset = make_dataset(dataset_path)
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect('ivt', minimum_duration=1)
    dataset.save_events()
    detected_frames = [events.frame for events in dataset.events]

    reloaded = make_dataset(dataset_path)
    reloaded.load()
    reloaded.load_event_files()

    # The events directory holds the five frames and no metadata file beside them.
    saved_filenames = sorted(path.name for path in reloaded.paths.events.iterdir())
    assert saved_filenames == [f'{subject_id}.feather' for subject_id in SUBJECT_IDS]
    assert len(reloaded.events) == len(SUBJECT_IDS)
    for detected_frame, events in zip(detected_frames, reloaded.events):
        # The ten 1 ms samples hold one fixation, trimmed by the velocity window at both ends.
        assert events.frame['onset'].dt.total_microseconds().to_list() == [2000]
        assert events.frame['duration'].dt.total_microseconds().to_list() == [6000]
        assert events.frame['name'].to_list() == ['fixation']
        assert 'offset' not in events.frame.columns
        assert events.metadata['sampling_rate'] == 1000
        assert_frame_equal(events.frame, detected_frame)
