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
"""Tests for Dataset.__str__ and Dataset.summary."""
import polars as pl

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import Events
from pymovements import Experiment
from pymovements import Gaze


def test_dataset_str_unloaded(tmp_path):
    """Ensure that the string representation of an unloaded dataset shows zero recordings."""
    dataset = Dataset(DatasetDefinition(name='example'), path=tmp_path)

    expected = (
        f"Dataset(name='example', path={str(dataset.path)!r})\n"
        '  gaze: 0 recordings\n'
        '  events: 0 recordings'
    )
    assert str(dataset) == expected


def test_dataset_str_includes_experiment(tmp_path):
    """Ensure that the experiment definition is included in the string representation."""
    definition = DatasetDefinition(
        name='example',
        experiment=Experiment(1280, 1024, 38, 30, 68, 'upper left', 1000),
    )
    dataset = Dataset(definition, path=tmp_path)

    assert f'\n  experiment: {definition.experiment}\n' in str(dataset)


def test_dataset_str_includes_resources(tmp_path):
    """Ensure that the number of defined resources is included in the string representation."""
    definition = DatasetDefinition(
        name='example',
        resources=[{'content': 'gaze', 'filename_pattern': 'a.csv'}],
    )
    dataset = Dataset(definition, path=tmp_path)

    assert '\n  resources: 1 defined\n' in str(dataset)


def _two_recording_dataset(tmp_path):
    """Return a dataset with two loaded gaze recordings holding 5 samples and 3 events."""
    dataset = Dataset(DatasetDefinition(name='example'), path=tmp_path)
    dataset.gaze = [
        Gaze(
            samples=pl.DataFrame({'time': [0, 1, 2], 'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]}),
            pixel_columns=['x', 'y'],
            events=Events(name=['fixation', 'saccade'], onsets=[0, 1], offsets=[1, 2]),
        ),
        Gaze(
            samples=pl.DataFrame({'time': [0, 1], 'x': [0.0, 1.0], 'y': [0.0, 1.0]}),
            pixel_columns=['x', 'y'],
            events=Events(name=['fixation'], onsets=[0], offsets=[1]),
        ),
    ]
    return dataset


def test_dataset_str_counts_loaded_recordings(tmp_path):
    """Ensure that the string representation counts loaded gaze and event recordings."""
    dataset = _two_recording_dataset(tmp_path)

    assert '\n  gaze: 2 recordings\n' in str(dataset)
    assert str(dataset).endswith('\n  events: 2 recordings')


def test_dataset_summary_prints_str_and_aggregates(tmp_path, capsys):
    """Ensure that summary prints the string representation plus aggregated totals."""
    dataset = _two_recording_dataset(tmp_path)

    dataset.summary()

    out = capsys.readouterr().out
    assert out.startswith(f'{dataset}\n')
    assert '\n  total samples: 5 (0.0 MB estimated)\n' in out
    assert out.endswith('\n  total events: 3\n')
