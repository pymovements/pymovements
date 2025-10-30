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
"""Write documentation pages for datasets."""
from __future__ import annotations

from pathlib import Path

import yaml

import pymovements as pm


def write_docfiles_for_dataset(
        dataset_name: str,
        doc_dirpath: Path,
        doc_meta_dirname: str,
) -> None:
    """Write sphinx documentation files for given dataset.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset for which to write documentation files.
    doc_dirpath: Path
        Target dirpath for documentation files.
    doc_meta_dirname: str
        Name of the meta directory.
    """
    definition = pm.DatasetLibrary.get(dataset_name)

    meta_dirpath = doc_dirpath / doc_meta_dirname
    meta_dirpath.mkdir(exist_ok=True)
    definition.to_yaml(meta_dirpath / f'{definition.name}.yaml')

    rst_content = f'''.. datatemplate:yaml:: meta/{definition.name}.yaml
    :template: dataset.rst\n'''

    with open(doc_dirpath / f'{definition.name}.rst', 'w', encoding='utf-8') as rst_file:
        rst_file.write(rst_content)


def main(
        doc_dirpath: str | Path = 'docs/source/datasets',
        doc_yaml_filename: str = 'datasets.yaml',
        doc_meta_dirname: str = 'meta',
) -> int:
    """Write sphinx documentation files for all datasets.

    Parameters
    ----------
    doc_dirpath: str | Path
        Target dirpath for documentation files.
    doc_yaml_filename: str
        Filename of the target yaml file containing the list of dataset names.
    doc_meta_dirname: str
        Name of the meta directory.

    Returns
    -------
    int
        ``0`` if success.
    """
    doc_dirpath = Path(doc_dirpath)

    dataset_names = pm.DatasetLibrary.names()

    with open(doc_dirpath / doc_yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_names, f)

    for dataset_name in dataset_names:
        write_docfiles_for_dataset(
            dataset_name=dataset_name,
            doc_dirpath=doc_dirpath,
            doc_meta_dirname=doc_meta_dirname,
        )

    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
