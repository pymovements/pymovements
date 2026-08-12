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
"""Tests for Dataset.correct_fixations."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import polars as pl
import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import Events
from pymovements import Gaze


@pytest.fixture(name='aois_df')
def fixture_aois_df():
    """Return an AOIs DataFrame with three lines of text."""
    return pl.DataFrame({
        'trial': ['TRIAL1'] * 6,
        'word': ['Word1', 'Word2', 'Word3', 'Word4', 'Word5', 'Word6'],
        'start_x': [50.0, 250.0, 50.0, 250.0, 50.0, 250.0],
        'start_y': [80.0, 80.0, 180.0, 180.0, 280.0, 280.0],
        'end_x': [200.0, 400.0, 200.0, 400.0, 200.0, 400.0],
        'end_y': [120.0, 120.0, 220.0, 220.0, 320.0, 320.0],
        'height': [40.0] * 6,
    })


@pytest.fixture(name='dummy_dataset')
def fixture_dummy_dataset(tmp_path):
    """Create a dummy dataset with fixation events and one empty events object."""
    definition = DatasetDefinition(name='dummy')
    dataset = Dataset(definition, path=tmp_path)

    events_df = pl.DataFrame({
        'trial': ['TRIAL1'] * 6,
        'name': ['fixation'] * 6,
        'onset': [0, 100, 200, 300, 400, 500],
        'offset': [50, 150, 250, 350, 450, 550],
        'location': [
            [100.0, 105.0], [200.0, 102.0], [300.0, 198.0],
            [400.0, 201.0], [100.0, 305.0], [200.0, 301.0],
        ],
    })
    events = Events(events_df, trial_columns='trial')
    dataset.gaze = [Gaze(events=events), Gaze(events=Events())]

    return dataset


def test_dataset_correct_fixations(dummy_dataset, aois_df):
    result = dummy_dataset.correct_fixations(aois_df, algorithm='attach', verbose=False)

    assert result is dummy_dataset

    corrected_rows = dummy_dataset.events[0].frame.filter(
        pl.col('name') == 'fixation_corrected_attach',
    )
    assert corrected_rows.height == 6

    # The empty events object is skipped and remains empty.
    assert dummy_dataset.events[1].frame.height == 0


def test_dataset_correct_fixations_corrected_locations(dummy_dataset, aois_df):
    dummy_dataset.correct_fixations(aois_df, algorithm='attach', verbose=False)

    corrected_rows = dummy_dataset.events[0].frame.filter(
        pl.col('name') == 'fixation_corrected_attach',
    )
    corrected_y = [location[1] for location in corrected_rows['location'].to_list()]
    assert corrected_y == [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]
