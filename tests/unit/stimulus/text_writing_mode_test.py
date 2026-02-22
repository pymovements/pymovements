# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Test writing_system parameter in TextStimulus class."""
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pymovements.stimulus import TextStimulus
from pymovements.stimulus.text import from_file
from pymovements.stimulus.text import WritingSystem


# examining typical configurations
HORIZONTAL_LR = WritingSystem('horizontal', 'top-to-bottom', 'left-to-right')
HORIZONTAL_RL = WritingSystem('horizontal', 'top-to-bottom', 'right-to-left')
VERTICAL_RL = WritingSystem('vertical', 'right-to-left', 'top-to-bottom')
VERTICAL_LR = WritingSystem('vertical', 'left-to-right', 'top-to-bottom')


@pytest.fixture(name='sample_aoi_dataframe')
def fixture_sample_aoi_dataframe():
    """Create a sample AOI dataframe for testing."""
    return pl.DataFrame({
        'aoi': ['word1', 'word2', 'word3'],
        'x_min': [0, 100, 200],
        'y_min': [0, 0, 0],
        'width': [100, 100, 100],
        'height': [50, 50, 50],
        'page': [1, 1, 2],
    })


@pytest.fixture(name='sample_schema')
def fixture_sample_schema():
    """Create a sample schema for testing."""
    return {
        'aoi': 'aoi',
        'x_min': 'start_x',
        'y_min': 'start_y',
        'width': 'width',
        'height': 'height',
    }


def test_writing_system_default(sample_aoi_dataframe, sample_schema):
    """Test that default writing_system is horizontal-lr."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
    )
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == HORIZONTAL_LR


@pytest.mark.parametrize(
    'writing_system',
    [
        HORIZONTAL_LR,
        HORIZONTAL_RL,
        VERTICAL_RL,
        VERTICAL_LR,
    ],
)
def test_writing_system_explicit(sample_aoi_dataframe, sample_schema, writing_system):
    """Test that writing_system can be set explicitly."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
        writing_system=writing_system,
    )
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == writing_system


def test_writing_system_explicit_writing_system_object(sample_aoi_dataframe, sample_schema):
    """Test that writing_system accepts WritingSystem objects."""
    writing_system = WritingSystem(
        axis='vertical',
        lining='right-to-left',
        directionality='top-to-bottom',
    )
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
        writing_system=writing_system,
    )

    assert stimulus.writing_system == writing_system
    assert stimulus.writing_system == VERTICAL_RL


def test_writing_system_str_object(sample_aoi_dataframe, sample_schema):
    """Test that writing_system can be set from string descriptor."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
        writing_system='left-to-right',
    )
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == HORIZONTAL_LR


@pytest.mark.parametrize(
    'writing_system',
    [
        HORIZONTAL_LR,
        HORIZONTAL_RL,
        VERTICAL_RL,
        VERTICAL_LR,
    ],
)
def test_writing_system_preserved_by_split(sample_aoi_dataframe, sample_schema, writing_system):
    """Test that split() preserves writing_system."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
        page_column='page',
        writing_system=writing_system,
    )

    # Split by page
    split_parts = stimulus.split(by='page')

    # Check that all split parts preserve the writing_system
    assert len(split_parts) == 2
    for part in split_parts:
        assert isinstance(part.writing_system, WritingSystem)
        assert part.writing_system == writing_system


@pytest.mark.parametrize(
    'writing_system',
    [
        HORIZONTAL_LR,
        HORIZONTAL_RL,
        VERTICAL_RL,
        VERTICAL_LR,
    ],
)
def test_writing_system_preserved_by_from_csv(sample_aoi_dataframe, writing_system):
    """Test that from_csv() accepts and preserves writing_system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_aoi.csv'
        sample_aoi_dataframe.write_csv(csv_path)

        stimulus = TextStimulus.from_csv(
            path=csv_path,
            aoi_column='aoi',
            start_x_column='x_min',
            start_y_column='y_min',
            width_column='width',
            height_column='height',
            writing_system=writing_system,
        )

        assert isinstance(stimulus.writing_system, WritingSystem)
        assert stimulus.writing_system == writing_system


@pytest.mark.parametrize(
    'writing_system',
    [
        HORIZONTAL_LR,
        HORIZONTAL_RL,
        VERTICAL_RL,
        VERTICAL_LR,
    ],
)
def test_writing_system_preserved_by_from_file(sample_aoi_dataframe, writing_system):
    """Test that from_file() accepts and preserves writing_system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_aoi.csv'
        sample_aoi_dataframe.write_csv(csv_path)

        stimulus = from_file(
            aoi_path=csv_path,
            aoi_column='aoi',
            start_x_column='x_min',
            start_y_column='y_min',
            width_column='width',
            height_column='height',
            writing_system=writing_system,
        )

        assert isinstance(stimulus.writing_system, WritingSystem)
        assert stimulus.writing_system == writing_system


def test_writing_system_from_csv_default(sample_aoi_dataframe):
    """Test that from_csv() uses default writing_system when not specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_aoi.csv'
        sample_aoi_dataframe.write_csv(csv_path)

        stimulus = TextStimulus.from_csv(
            path=csv_path,
            aoi_column='aoi',
            start_x_column='x_min',
            start_y_column='y_min',
            width_column='width',
            height_column='height',
        )

        assert isinstance(stimulus.writing_system, WritingSystem)
        assert stimulus.writing_system == HORIZONTAL_LR


def test_writing_system_from_file_default(sample_aoi_dataframe):
    """Test that from_file() uses default writing_system when not specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_aoi.csv'
        sample_aoi_dataframe.write_csv(csv_path)

        stimulus = from_file(
            aoi_path=csv_path,
            aoi_column='aoi',
            start_x_column='x_min',
            start_y_column='y_min',
            width_column='width',
            height_column='height',
        )

        assert isinstance(stimulus.writing_system, WritingSystem)
        assert stimulus.writing_system == HORIZONTAL_LR


def test_writing_system_attribute_access(sample_aoi_dataframe, sample_schema):
    """Test that writing_system can be accessed as an attribute."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column=sample_schema['aoi'],
        start_x_column=sample_schema['x_min'],
        start_y_column=sample_schema['y_min'],
        width_column=sample_schema['width'],
        height_column=sample_schema['height'],
        writing_system=HORIZONTAL_RL,
    )

    # Test attribute access
    assert hasattr(stimulus, 'writing_system')
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == HORIZONTAL_RL
