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
"""Test DatasetPaths."""

from pathlib import Path

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_paths'),
    [
        pytest.param(
            {'root': '/data/set/path'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'precomputed_events': Path('/data/set/path/precomputed_events'),
                'precomputed_reading_measures': Path('/data/set/path/precomputed_reading_measures'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='only_root',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': '.'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'precomputed_events': Path('/data/set/path/precomputed_events'),
                'precomputed_reading_measures': Path('/data/set/path/precomputed_reading_measures'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='dataset_dot',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': 'dataset'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_dataset',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': 'dataset', 'events': 'custom_events'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/custom_events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_events',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': 'dataset', 'preprocessed': 'custom_preprocessed'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'preprocessed': Path('/data/set/path/dataset/custom_preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_preprocessed',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': 'dataset', 'raw': 'custom_raw'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/custom_raw'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_raw',
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset': 'dataset', 'downloads': 'custom_downloads'},
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/custom_downloads'),
            },
            id='explicit_downloads',
        ),
        pytest.param(
            {
                'root': '/data/set/path',
                'dataset': 'dataset',
                'precomputed_events': 'custom_precomputed_events',
            },
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'precomputed_events': Path('/data/set/path/dataset/custom_precomputed_events'),
                'precomputed_reading_measures': Path(
                    '/data/set/path/dataset/precomputed_reading_measures'
                ),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_precomputed_events',
        ),
        pytest.param(
            {
                'root': '/data/set/path',
                'dataset': 'dataset',
                'precomputed_reading_measures': 'custom_prm',
            },
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'precomputed_events': Path('/data/set/path/dataset/precomputed_events'),
                'precomputed_reading_measures': Path('/data/set/path/dataset/custom_prm'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_precomputed_reading_measures',
        ),
    ],
)
def test_dataset_paths(init_kwargs, expected_paths):
    paths = pm.DatasetPaths(**init_kwargs)

    assert paths.root == expected_paths['root']
    assert paths.dataset == expected_paths['dataset']
    assert paths.raw == expected_paths['raw']
    assert paths.precomputed_events == expected_paths['precomputed_events']
    assert paths.precomputed_reading_measures == expected_paths['precomputed_reading_measures']
    assert paths.preprocessed == expected_paths['preprocessed']
    assert paths.events == expected_paths['events']
    assert paths.downloads == expected_paths['downloads']
