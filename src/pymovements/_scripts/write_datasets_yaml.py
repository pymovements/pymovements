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
"""Write datasets.yaml for DatasetLibrary."""
from __future__ import annotations

from pathlib import Path

import yaml


def main(
        datasets_dirpath: str | Path = './src/pymovements/datasets',
        datasets_yaml_filename: str = 'datasets.yaml',
) -> int:
    """Write datasets yaml file for DatasetLibrary.

    Parameters
    ----------
    datasets_dirpath: str | Path
        The path to the directory containing dataset definition yaml files.
        (default: './src/pymovements/datasets')
    datasets_yaml_filename: str
        The filename of the datasets yaml file. (default: 'datasets.yaml')

    Returns
    -------
    int
        ``0`` if no changes needed, ``1`` otherwise.
    """
    datasets_dirpath = Path(datasets_dirpath)

    dataset_filename_stems = sorted(
        [
            filepath.stem for filepath in datasets_dirpath.glob('*.yaml')
            if filepath.name != datasets_yaml_filename  # Ignore datasets.yaml file.
        ],
    )

    try:
        with open(datasets_dirpath / datasets_yaml_filename, encoding='utf-8') as f:
            dataset_yaml_content = yaml.safe_load(f)
    except FileNotFoundError:
        dataset_yaml_content = None

    # File content matches. Exit successfully.
    if dataset_filename_stems == dataset_yaml_content:
        return 0

    # We have some updates in the datasets directory. Update the datasets yaml file.
    with open(datasets_dirpath / datasets_yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_filename_stems, f)

    return 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
