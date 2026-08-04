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
"""Test sources metadata population in gaze I/O functions."""
import pytest

from pymovements.gaze.io import from_asc
from pymovements.gaze.io import from_begaze
from pymovements.gaze.io import from_csv
from pymovements.gaze.io import from_ipc


_CSV_KWARGS = {'pixel_columns': ['x_left_pix', 'y_left_pix']}

_LOAD_FUNCTION_PARAMS = [
    pytest.param(from_csv, 'monocular_example.csv', _CSV_KWARGS, id='from_csv'),
    pytest.param(from_ipc, 'monocular_example.feather', {}, id='from_ipc'),
    pytest.param(from_asc, 'eyelink_monocular_example.asc', {}, id='from_asc'),
    pytest.param(from_begaze, 'didec_example.txt', {}, id='from_begaze'),
]


@pytest.mark.parametrize(
    (
        'load_function', 'example_filename',
        'load_kwargs',
    ), _LOAD_FUNCTION_PARAMS,
)
def test_load_function_adds_absolute_source(
        load_function, example_filename, load_kwargs, make_example_file,
):
    filepath = make_example_file(example_filename)

    gaze = load_function(filepath, **load_kwargs)

    expected_sources = [filepath.resolve().as_posix()]
    assert gaze.metadata['sources'] == expected_sources
    # The source files are propagated to the events container.
    assert gaze.events.metadata['sources'] == expected_sources


@pytest.mark.parametrize(
    (
        'load_function', 'example_filename',
        'load_kwargs',
    ), _LOAD_FUNCTION_PARAMS,
)
def test_load_function_accepts_str_filepath(
        load_function, example_filename, load_kwargs, make_example_file,
):
    filepath = make_example_file(example_filename)

    gaze = load_function(str(filepath), **load_kwargs)

    assert gaze.metadata['sources'] == [filepath.resolve().as_posix()]


def test_from_csv_respects_user_provided_sources(make_example_file):
    filepath = make_example_file('monocular_example.csv')

    gaze = from_csv(filepath, metadata={'sources': ['my/custom/source.csv']}, **_CSV_KWARGS)

    assert gaze.metadata['sources'] == ['my/custom/source.csv']


def test_from_csv_does_not_mutate_passed_metadata(make_example_file):
    filepath = make_example_file('monocular_example.csv')
    metadata = {'subject_id': 42}

    gaze = from_csv(filepath, metadata=metadata, **_CSV_KWARGS)

    assert metadata == {'subject_id': 42}
    assert gaze.metadata == {
        'subject_id': 42,
        'sources': [filepath.resolve().as_posix()],
    }
