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
"""Test the correctness of all shipped dataset definitions."""
import pytest

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import DatasetLibrary


@pytest.fixture(name='definition')
def fixture_definition(name: str) -> DatasetDefinition:
    """Get the dataset definition for a given name.

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    DatasetDefinition
        The dataset definition.

    """
    return DatasetLibrary.get(name)


@pytest.mark.parametrize('name', DatasetLibrary.names())
def test_shipped_definitions_references_resolve(definition: DatasetDefinition) -> None:
    """Test that every string source reference resolves to a named source.

    Parameters
    ----------
    definition : DatasetDefinition
        The dataset definition to test.

    """
    for resource in definition.resources:
        if isinstance(resource.source, str):
            assert resource.source in definition.sources, \
                f"Dangling reference {resource.source} in {definition.name}"


@pytest.mark.parametrize('name', DatasetLibrary.names())
def test_shipped_definitions_sources_no_duplicate_url_filename(
    definition: DatasetDefinition,
) -> None:
    """Test that sources contains no duplicate (url, filename) pairs.

    Parameters
    ----------
    definition : DatasetDefinition
        The dataset definition to test.

    """
    urls_filenames = [(s.url, s.filename) for s in definition.sources.values()]
    assert len(urls_filenames) == len(set(urls_filenames)), \
        f"Duplicate (url, filename) pairs in sources of {definition.name}: {urls_filenames}"


@pytest.mark.parametrize('name', DatasetLibrary.names())
def test_shipped_definitions_filenames_are_unique(definition: DatasetDefinition) -> None:
    """Test that all filenames are unique (non-None).

    Parameters
    ----------
    definition : DatasetDefinition
        The dataset definition to test.

    """
    filenames = [s.filename for s in definition.sources.values() if s.filename is not None]
    assert len(filenames) == len(set(filenames)), \
        f"Duplicate filenames in sources of {definition.name}: {filenames}"
