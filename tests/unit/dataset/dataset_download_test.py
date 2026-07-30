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
"""Test all download and extract functionality of pymovements.Dataset."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetPaths
from pymovements import ResourceDefinition
from pymovements.dataset.dataset_download import download_dataset
from pymovements.dataset.websource import WebSource


@pytest.fixture(
    name='dataset_definition',
    params=[
        'CustomGazeAndPrecomputedSingleMirror',
        'CustomGazeAndPrecomputedNoMirror',
        'CustomGazeOnlySingleMirror',
        'CustomGazeOnlyTwoMirrors',
        'CustomGazeOnlyNoMirror',
        'CustomGazeImageStimuli',
        'CustomGazeTextStimuli',
        'CustomPrecomputedOnlySingleMirror',
        'CustomPrecomputedOnlyNoMirror',
        'CustomPrecomputedOnlyNoExtractSingleMirror',
        'CustomPrecomputedOnlyNoExtractNoMirror',
        'CustomPrecomputedRMOnlySingleMirror',
        'CustomPrecomputedRMOnlyNoMirror',
        'CustomImageStimuli',
        'CustomTextStimuli',
    ],
)
def dataset_definition_fixture(request):  # pylint: disable=too-many-return-statements
    if request.param == 'CustomGazeAndPrecomputedSingleMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'mirrors': ['https://another_example.com/test.gz.tar'],
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'mirrors': ['https://another_example.com/test_pc.gz.tar'],
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeAndPrecomputedNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeOnlySingleMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'mirrors': ['https://another_example.com/test.gz.tar'],
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeOnlyTwoMirrors':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'mirrors': [
                            'https://mirror1.example.com/test.gz.tar',
                            'https://mirror2.example.com/test.gz.tar',
                        ],
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeImageStimuli':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
                {
                    'content': 'imagestimulus',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'stimuli.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomGazeTextStimuli':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
                {
                    'content': 'textstimulus',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'stimuli.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlySingleMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'mirrors': ['https://another_example.com/test_pc.gz.tar'],
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoExtractSingleMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'mirrors': ['https://another_example.com/test_pc.gz.tar'],
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoExtractNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'https://example.com/test_pc.gz.tar',
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedRMOnlySingleMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_reading_measures',
                    'source': {
                        'url': 'https://example.com/test_rm.gz.tar',
                        'mirrors': ['https://another_example.com/test_rm.gz.tar'],
                        'filename': 'test_rm.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomPrecomputedRMOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_reading_measures',
                    'source': {
                        'url': 'https://example.com/test_rm.gz.tar',
                        'filename': 'test_rm.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomImageStimuli':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'imagestimulus',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'stimuli.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    if request.param == 'CustomTextStimuli':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'textstimulus',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': 'stimuli.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                },
            ],
        )

    assert False, f'unknown dataset_definition fixture {request.param}'


@pytest.mark.parametrize(
    ('init_path', 'expected_paths'),
    [
        pytest.param(
            '/data/set/path',
            {
                'root': Path('/data/set/path'),
                'dataset': Path('/data/set/path'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='no_paths',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/CustomPublicDataset'),
                'downloads': Path('/data/set/path/CustomPublicDataset/downloads'),
            },
            id='no_paths',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='dataset_dot',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_dataset_dirname',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', downloads='custom_downloads'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/CustomPublicDataset'),
                'downloads': Path('/data/set/path/CustomPublicDataset/custom_downloads'),
            },
            id='explicit_download_dirname',
        ),
    ],
)
def test_paths(init_path, expected_paths, dataset_definition):
    dataset = Dataset(dataset_definition, path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.paths.downloads == expected_paths['downloads']


@pytest.mark.filterwarnings('ignore:Downloading resource .* failed.*:UserWarning')
def test_dataset_download_no_sources_raises(tmp_path):
    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset_definition = DatasetDefinition(
        name='test',
        resources=[{'content': 'gaze', 'filename_pattern': 'test.csv'}],
    )
    dataset = Dataset(dataset_definition, path=paths)

    message = (
        'No downloadable resources found in DatasetDefinition. '
        'ResourceDefinition.source must be specified to download a dataset.'
    )
    with pytest.raises(AttributeError, match=message):
        dataset.download()


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeOnlySingleMirror',
        'CustomGazeOnlyTwoMirrors',
        'CustomGazeOnlyNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_gaze(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomPrecomputedRMOnlySingleMirror',
        'CustomPrecomputedRMOnlyNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_rm(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_rm.gz.tar',
            destination_path=tmp_path / 'precomputed_reading_measures',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeAndPrecomputedSingleMirror',
        'CustomGazeAndPrecomputedNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_both(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomPrecomputedOnlySingleMirror',
        'CustomPrecomputedOnlyNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_precomputed(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeImageStimuli',
        'CustomGazeTextStimuli',
        'CustomImageStimuli',
        'CustomTextStimuli',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_stimuli(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'stimuli.gz.tar',
            destination_path=tmp_path / 'stimuli',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeAndPrecomputedSingleMirror',
        'CustomGazeAndPrecomputedNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_both(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeOnlySingleMirror',
        'CustomGazeOnlyTwoMirrors',
        'CustomGazeOnlyNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_gaze(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomPrecomputedOnlySingleMirror',
        'CustomPrecomputedOnlyNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_precomputed(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeImageStimuli',
        'CustomGazeTextStimuli',
        'CustomImageStimuli',
        'CustomTextStimuli',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_stimuli(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'stimuli.gz.tar',
            destination_path=tmp_path / 'stimuli',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@pytest.mark.parametrize(
    ('dataset_definition', 'expected_exception', 'expected_msg'),
    [
        pytest.param(
            DatasetDefinition(name='CustomPublicDataset'),
            AttributeError,
            'No downloadable resources found in DatasetDefinition. '
            'ResourceDefinition.source must be specified to download a dataset.',
            id='no_resources',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'source': {
                        'url': None,
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                }],
            ),
            AttributeError,
            'WebSource.url must not be None',
            id='url_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'source': {
                        'url': 'https://example.com/test.gz.tar',
                        'filename': None,
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                }],
            ),
            AttributeError,
            'WebSource.filename must not be None',
            id='filename_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'source': {
                        'url': 'test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_http_resource_gaze',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'precomputed_events',
                    'source': {
                        'url': 'test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_http_resource_events',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'precomputed_reading_measures',
                    'source': {
                        'url': 'test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_http_resource_measures',
        ),
    ],
)
def test_dataset_download_raises_exception(
        dataset_definition, expected_exception, expected_msg, tmp_path,
):
    with pytest.raises(expected_exception, match=expected_msg):
        Dataset(dataset_definition, path=tmp_path).download()


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_exception', 'expected_msg'),
    [
        pytest.param(
            {
                'name': 'CustomPublicDataset',
                'resources': [
                    {
                        'content': 'gaze',
                        'source': {
                            'url': None,
                            'filename': 'test.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    },
                ],
            },
            AttributeError,
            'WebSource.url must not be None',
            id='url_none',
        ),
        pytest.param(
            {
                'name': 'CustomPublicDataset',
                'resources': [
                    {
                        'content': 'gaze',
                        'source': {
                            'url': 'https://example.com/test.gz.tar',
                            'filename': None,
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    },
                ],
            },
            AttributeError,
            'WebSource.filename must not be None',
            id='filename_none',
        ),
    ],
)
def test_dataset_download_websource_missing_attributes_raises_exception(
        init_kwargs, expected_exception, expected_msg, tmp_path,
):
    dataset_definition = DatasetDefinition(**init_kwargs)
    with pytest.raises(expected_exception, match=expected_msg):
        Dataset(dataset_definition, path=tmp_path).download()


def test_public_dataset_registered_correct_attributes(tmp_path, dataset_definition):
    dataset = Dataset(dataset_definition, path=tmp_path)

    assert dataset.definition.resources == dataset_definition.resources
    assert dataset.definition.experiment == dataset_definition.experiment


def test_extract_dataset_precomputed_move_single_file(tmp_path, testfiles_dirpath):
    definition = DatasetDefinition(
        name='CustomPublicDataset',
        resources=[
            {
                'content': 'precomputed_events',
                'source': {'filename': '18sat_fixfinal.csv'},
            },
        ],
    )

    # Create directory and copy test file.
    (tmp_path / 'downloads').mkdir(parents=True)
    shutil.copyfile(
        testfiles_dirpath / '18sat_fixfinal.csv',
        tmp_path / 'downloads' / '18sat_fixfinal.csv',
    )

    Dataset(definition, path=tmp_path).extract()


def test_extract_dataset_precomputed_rm_move_single_file(tmp_path, testfiles_dirpath):
    definition = DatasetDefinition(
        name='CustomPublicDataset',
        resources=[
            {
                'content': 'precomputed_reading_measures',
                'source': {'filename': 'copco_rm_dummy.csv'},
            },
        ],
    )

    # Create directory and copy test file.
    (tmp_path / 'downloads').mkdir(parents=True)

    shutil.copyfile(
        testfiles_dirpath / 'copco_rm_dummy.csv',
        tmp_path / 'downloads' / 'copco_rm_dummy.csv',
    )

    Dataset(definition, path=tmp_path).extract()


def test_download_dataset_deduplication():
    """Test that download_dataset calls download on each unique source only once."""
    resource1 = ResourceDefinition(
        content='gaze',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    resource2 = ResourceDefinition(
        content='precomputed_events',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    definition = DatasetDefinition(name='test', resources=[resource1, resource2])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with mock.patch('pymovements.dataset.websource.WebSource.download') as mock_download:
        download_dataset(definition, paths, extract=False)

    # Even though there are 2 resources, there should be only 1 download call
    # because they share the same source (deduplicated by url and filename).
    assert mock_download.call_count == 1


def test_download_dataset_conflicting_sources_raises():
    """Test that sources sharing (url, filename) but differing md5 raise ValueError."""
    resource1 = ResourceDefinition(
        content='gaze',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip', md5='abc'),
    )
    resource2 = ResourceDefinition(
        content='precomputed_events',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip', md5='def'),
    )
    definition = DatasetDefinition(name='test', resources=[resource1, resource2])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with pytest.raises(ValueError, match="md5 differs between resources \\('abc' != 'def'\\)"):
        download_dataset(definition, paths, extract=False)


def test_download_dataset_resolves_named_source_reference():
    """Test that a string source reference is resolved to the named source and downloaded."""
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    resource = ResourceDefinition(content='gaze', source='main')
    definition = DatasetDefinition(name='test', resources=[resource], sources={'main': source})

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with mock.patch('pymovements.dataset.websource.WebSource.download') as mock_download:
        download_dataset(definition, paths, extract=False)

    assert mock_download.call_count == 1


def test_download_dataset_skips_resource_without_source():
    """Test that resources without a source are skipped when collecting downloads."""
    resource_with = ResourceDefinition(
        content='gaze',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    resource_without = ResourceDefinition(content='precomputed_events')
    definition = DatasetDefinition(name='test', resources=[resource_with, resource_without])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with mock.patch('pymovements.dataset.websource.WebSource.download') as mock_download:
        download_dataset(definition, paths, extract=False)

    assert mock_download.call_count == 1


def test_download_dataset_dangling_reference_raises():
    """Test that an unresolvable source reference raises ValueError at download time."""
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    resource = ResourceDefinition(content='gaze', source='main')
    definition = DatasetDefinition(name='test', resources=[resource], sources={'main': source})
    # Bypass construction-time validation to reach the runtime guard.
    definition.resources[0].source = 'ghost'

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with pytest.raises(ValueError, match="Dangling source reference: 'ghost'"):
        download_dataset(definition, paths, extract=False)


def test_download_dataset_triggers_extract():
    """Test that download_dataset extracts by default after downloading."""
    resource = ResourceDefinition(
        content='gaze',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    definition = DatasetDefinition(name='test', resources=[resource])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with mock.patch('pymovements.dataset.websource.WebSource.download'), \
            mock.patch('pymovements.dataset.dataset_download.extract_dataset') as mock_extract:
        download_dataset(definition, paths)

    assert mock_extract.call_count == 1


def test_extract_dataset_skips_resource_without_source(tmp_path):
    """Test that a resource without a source is skipped during extraction."""
    definition = DatasetDefinition(
        name='test',
        resources=[ResourceDefinition(content='precomputed_events')],
    )
    (tmp_path / 'downloads').mkdir(parents=True)

    # Should not raise even though there is nothing to extract.
    Dataset(definition, path=tmp_path).extract()


def test_download_dataset_named_source_shared_by_multiple_resources_deduplicated():
    """Test that a named source shared by several resources is downloaded only once."""
    source = WebSource(url='http://example.com/file.zip', filename='file.zip')
    resources = [
        ResourceDefinition(content='gaze', source='main'),
        ResourceDefinition(content='precomputed_events', source='main'),
        ResourceDefinition(content='imagestimulus', source='main'),
    ]
    definition = DatasetDefinition(name='test', resources=resources, sources={'main': source})

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with mock.patch('pymovements.dataset.websource.WebSource.download') as mock_download:
        download_dataset(definition, paths, extract=False)

    assert mock_download.call_count == 1


def test_download_dataset_conflicting_sources_asymmetric_md5_raises():
    """Test that a shared (url, filename) with md5 set on only one resource raises ValueError."""
    resource1 = ResourceDefinition(
        content='gaze',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip', md5='abc'),
    )
    resource2 = ResourceDefinition(
        content='precomputed_events',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    definition = DatasetDefinition(name='test', resources=[resource1, resource2])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with pytest.raises(ValueError, match="md5 differs between resources \\('abc' != 'None'\\)"):
        download_dataset(definition, paths, extract=False)


def test_download_dataset_conflicting_sources_mirrors_raises():
    """Test that a shared (url, filename) with differing mirrors raises ValueError."""
    resource1 = ResourceDefinition(
        content='gaze',
        source=WebSource(
            url='http://example.com/file.zip', filename='file.zip',
            mirrors=['http://mirror.com/file.zip'],
        ),
    )
    resource2 = ResourceDefinition(
        content='precomputed_events',
        source=WebSource(url='http://example.com/file.zip', filename='file.zip'),
    )
    definition = DatasetDefinition(name='test', resources=[resource1, resource2])

    paths = mock.Mock()
    paths.downloads = '/tmp/downloads'

    with pytest.raises(ValueError, match='mirrors differ between resources'):
        download_dataset(definition, paths, extract=False)


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
def test_extract_dataset_resolves_named_source(mock_extract_archive, tmp_path):
    """Test that extraction resolves a named source reference to its archive file."""
    mock_extract_archive.return_value = 'path'
    definition = DatasetDefinition(
        name='test',
        resources=[ResourceDefinition(content='gaze', source='main')],
        sources={'main': WebSource(url='http://example.com/file.gz.tar', filename='file.gz.tar')},
    )
    paths = DatasetPaths(root=tmp_path, dataset='.')

    Dataset(definition, path=paths).extract()

    mock_extract_archive.assert_called_once_with(
        source_path=tmp_path / 'downloads' / 'file.gz.tar',
        destination_path=tmp_path / 'raw',
        recursive=True,
        remove_finished=False,
        remove_top_level=True,
        resume=True,
        verbose=1,
    )
