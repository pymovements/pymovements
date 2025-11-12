# Copyright (c) 2025 The pymovements Project Authors
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
"""Test downloading and scanning all items in DatasetLibrary."""
import pytest

from pymovements import Dataset
from pymovements import DatasetLibrary


@pytest.mark.parametrize('dataset_name', DatasetLibrary.names())
def test_download_dataset(dataset_name, tmp_path):
    # Initialize dataset.
    dataset = Dataset(dataset_name, path=tmp_path)

    # Download dataset. This checks for integrity by matching the md5 checksum.
    dataset.download(extract=False)

    # Check that all resources are downloaded.
    download_dir = dataset.paths.downloads
    for resource in dataset.definition.resources:
        assert download_dir / resource.filename in download_dir.iterdir()
