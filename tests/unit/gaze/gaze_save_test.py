# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test Gaze save functionality."""
from __future__ import annotations

import os

import polars as pl
import pytest

from pymovements import Gaze


def test_gaze_save_csv(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.save(
        dirpath=tmp_path,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_feather(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.save(
        dirpath=tmp_path,
        verbose=2,
        extension='feather',
    )
    assert os.path.exists(tmp_path / 'samples.feather')
    assert os.path.exists(tmp_path / 'events.feather')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_events(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.save(
        dirpath=tmp_path,
        save_events=False,
        verbose=2,
        extension='csv',
    )
    assert not os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_samples(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.save(
        dirpath=tmp_path,
        save_samples=False,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert not os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_without_experiment(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.save(
        dirpath=tmp_path,
        save_experiment=False,
        verbose=1,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_with_empty_events(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.events = None

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_events=True,
            verbose=2,
            extension='csv',
        )


def test_gaze_save_wrong_extension_events(tmp_path, gaze_all):
    gaze = gaze_all

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            verbose=0,
            extension='blabla',
        )


def test_gaze_save_wrong_extension_samples(tmp_path, gaze_all):
    gaze = gaze_all

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_events=False,
            verbose=1,
            extension='blabla',
        )


def test_gaze_save_empty_experiment(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.experiment = None

    gaze.save(
        dirpath=tmp_path,
        verbose=1,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_empty_experiment_true_save(tmp_path, gaze_all):
    gaze = gaze_all
    gaze.experiment = None

    with pytest.raises(ValueError):
        gaze.save(
            dirpath=tmp_path,
            save_experiment=True,
            verbose=1,
            extension='csv',
        )


def test_gaze_save_samples_csv_no_warning_without_nested_columns(tmp_path):
    samples = pl.DataFrame(
        {'x': [0, 1], 'y': [1, 0], 'trial': [1, 2]},
        schema={'x': pl.Float64, 'y': pl.Float64, 'trial': pl.Int64},
    )
    gaze = Gaze(samples=samples, pixel_columns=['x', 'y'], trial_columns='trial')

    gaze.save(
        dirpath=tmp_path,
        save_samples=True,
        save_events=False,
        save_experiment=False,
        verbose=1,
        extension='csv',
    )


@pytest.mark.parametrize(
    'gaze,expected_file',
    [
        pytest.param(
            Gaze(
                pl.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                pixel_columns=['x', 'y'],
                metadata={'key': 'value', 'nested': {'inner': 42}},
            ),
            'metadata.yaml',
            id='with_metadata',
        ),
    ],
)
def test_gaze_save_metadata(tmp_path, gaze, expected_file):
    gaze.save(tmp_path, save_metadata=True, verbose=2)
    assert os.path.exists(tmp_path / expected_file)


@pytest.mark.parametrize(
    'gaze,save_func,expected_file,raises_error,error_match',
    [
        pytest.param(
            Gaze(
                pl.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                pixel_columns=['x', 'y'],
                messages=pl.DataFrame({'time': [0], 'content': ['msg']}),
            ),
            lambda g, p: g.save_messages(p / 'messages.feather', verbose=2),
            'messages.feather',
            False,
            None,
            id='messages_with_data',
        ),
        pytest.param(
            Gaze(pl.DataFrame({'x': [1, 2], 'y': [3, 4]}), pixel_columns=['x', 'y']),
            lambda g, p: g.save_messages(p / 'messages.feather', verbose=2),
            None,
            True,
            'No messages',
            id='messages_none',
        ),
        pytest.param(
            Gaze(
                pl.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                pixel_columns=['x', 'y'],
                calibrations=pl.DataFrame({'timestamp': [0], 'num_points': [9]}),
            ),
            lambda g, p: g.save_calibrations(p / 'calibrations.feather', verbose=2),
            'calibrations.feather',
            False,
            None,
            id='calibrations_with_data',
        ),
        pytest.param(
            Gaze(pl.DataFrame({'x': [1, 2], 'y': [3, 4]}), pixel_columns=['x', 'y']),
            lambda g, p: g.save_calibrations(p / 'calibrations.feather', verbose=2),
            None,
            True,
            'No calibrations',
            id='calibrations_none',
        ),
        pytest.param(
            Gaze(
                pl.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                pixel_columns=['x', 'y'],
                validations=pl.DataFrame({'timestamp': [0], 'accuracy_avg': [0.5]}),
            ),
            lambda g, p: g.save_validations(p / 'validations.feather', verbose=2),
            'validations.feather',
            False,
            None,
            id='validations_with_data',
        ),
        pytest.param(
            Gaze(pl.DataFrame({'x': [1, 2], 'y': [3, 4]}), pixel_columns=['x', 'y']),
            lambda g, p: g.save_validations(p / 'validations.feather', verbose=2),
            None,
            True,
            'No validations',
            id='validations_none',
        ),
    ],
)
def test_gaze_save_dataframes(
    tmp_path,
    gaze,
    save_func,
    expected_file,
    raises_error,
    error_match,
):
    if raises_error:
        with pytest.raises(ValueError, match=error_match):
            save_func(gaze, tmp_path)
    else:
        save_func(gaze, tmp_path)
        assert os.path.exists(tmp_path / expected_file)


@pytest.mark.parametrize(
    'save_method',
    [
        pytest.param('save_messages', id='messages'),
        pytest.param('save_calibrations', id='calibrations'),
        pytest.param('save_validations', id='validations'),
    ],
)
def test_gaze_save_wrong_extension(tmp_path, save_method):
    gaze = Gaze(pl.DataFrame({'x': [1, 2], 'y': [3, 4]}), pixel_columns=['x', 'y'])
    if save_method == 'save_messages':
        gaze.messages = pl.DataFrame({'time': [0], 'content': ['msg']})
    elif save_method == 'save_calibrations':
        gaze.calibrations = pl.DataFrame({'timestamp': [0], 'num_points': [9]})
    else:
        gaze.validations = pl.DataFrame({'timestamp': [0], 'accuracy_avg': [0.5]})

    with pytest.raises(ValueError, match='unsupported file format'):
        getattr(gaze, save_method)(tmp_path / 'test.txt')
