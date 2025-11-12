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
"""Test autowriting sphinx docpages for datasets."""
from pymovements import DatasetLibrary
from pymovements._scripts import write_dataset_docpages


def test_write_dataset_docpages(tmp_path):
    write_dataset_docpages.main(doc_dirpath=tmp_path)

    assert (tmp_path / 'datasets.yaml').is_file()

    for dataset_name in DatasetLibrary.names():
        assert (tmp_path / f'{dataset_name}.rst').is_file()
        print(list((tmp_path / 'meta').iterdir()))
        assert (tmp_path / 'meta' / f'{dataset_name}.yaml').is_file()
