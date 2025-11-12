# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test basic preprocessing on all registered public datasets."""
import shutil

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    'dataset_name',
    list(pm.dataset.DatasetLibrary.definitions.keys()),
)
def test_public_dataset_processing(dataset_name, tmp_path):
    # Initialize dataset.
    dataset_path = tmp_path / dataset_name
    dataset = pm.Dataset(dataset_name, path=dataset_path)

    # Download and load in dataset.
    dataset.download()
    dataset.load()

    # Do some basic transformations.
    if dataset.definition.resources.has_content('gaze'):
        if 'pixel' in dataset.gaze[0].columns:
            dataset.pix2deg()
        dataset.pos2vel()
        dataset.pos2acc()

        for gaze in dataset.gaze:
            assert 'position' in gaze.columns
            assert 'velocity' in gaze.columns
            assert 'acceleration' in gaze.columns
            assert len(set(gaze.trial_columns)) == len(gaze.trial_columns)

        shutil.rmtree(dataset_path, ignore_errors=True)
