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
            'resources must be specified to download a dataset.',
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
