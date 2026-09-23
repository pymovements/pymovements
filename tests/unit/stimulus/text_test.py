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
"""Test Text stimulus class."""
from copy import deepcopy
from dataclasses import replace

import math
import matplotlib.pyplot as plt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.stimulus import text
from pymovements.stimulus import TextStimulus
from pymovements.stimulus import WritingSystem


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


HORIZONTAL_LR = WritingSystem('left-to-right', axis='horizontal', lining='top-to-bottom')
HORIZONTAL_RL = WritingSystem('right-to-left', axis='horizontal', lining='top-to-bottom')
VERTICAL_RL = WritingSystem('top-to-bottom', axis='vertical', lining='right-to-left')
VERTICAL_LR = WritingSystem('top-to-bottom', axis='vertical', lining='left-to-right')


@pytest.fixture(
    name='writing_system',
    params=[
        'hlr',  # horizontal left to right
        'hrl',  # horizontal right to left
        'vlr',  # vertical left to right
        'vrl',  # vertical right to left
    ],
    scope='function',
)
def writing_system_fixture(request):
    writing_systems = {
        'hlr': HORIZONTAL_LR,
        'hrl': HORIZONTAL_RL,
        'vlr': VERTICAL_RL,
        'vrl': VERTICAL_LR,
    }
    yield replace(writing_systems[request.param])  # create copy


EXPECTED_DF = pl.DataFrame(
    {
        'char': [
            'A',
            'B',
            'S',
            'T',
            'R',
            'A',
            'C',
            'T',
            'p',
            'y',
            'm',
            'o',
        ],
        'top_left_x': [
            400.0,
            415.0,
            430.0,
            445.0,
            460.0,
            475.0,
            490.0,
            505.0,
            400.0,
            414.972602739726,
            429.94520547945206,
            444.9178082191781,
        ],
        'top_left_y': [
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            214.85148514851485,
            214.85148514851485,
            214.85148514851485,
            214.85148514851485,
        ],
        'width': [
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            14.972602739726028,
            14.972602739726028,
            14.972602739726028,
            14.972602739726028,
        ],
        'height': [
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            23,
            23,
            23,
            23,
        ],
        'char_idx_in_line': [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
        ],
        'line_idx': [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
        'page': ['page_2' for _ in range(12)],
        'word': [
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'pymovements:',
            'pymovements:',
            'pymovements:',
            'pymovements:',
        ],
        'bottom_left_x': [
            415.0,
            430.0,
            445.0,
            460.0,
            475.0,
            490.0,
            505.0,
            520.0,
            414.972602739726,
            429.94520547945206,
            444.9178082191781,
            459.8904109589041,
        ],
        'bottom_left_y': [
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            237.85148514851485,
            237.85148514851485,
            237.85148514851485,
            237.85148514851485,
        ],
    },
)


@pytest.mark.parametrize(
    ('filename', 'custom_read_kwargs', 'expected'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            None,
            EXPECTED_DF,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            {'separator': ','},
            EXPECTED_DF,
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_has_correct_aois(filename, custom_read_kwargs, expected, make_example_file):
    aoi_path = make_example_file(filename)
    aois = text.from_file(
        aoi_path,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )
    head = aois.aois.head(12)

    assert_frame_equal(
        head,
        expected,
    )
    assert len(aois.aois.columns) == len(expected.columns)


def test_text_stimulus_from_file_has_correct_metadata_default(make_example_file):
    aoi_path = make_example_file('stimuli/toy_text_aoi.csv')
    stimulus = text.from_file(
        aoi_path,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    assert stimulus.metadata == {}


@pytest.mark.parametrize(
    'metadata',
    [
        pytest.param({}, id='empty'),
        pytest.param({'key': 'value'}, id='dict'),
    ],
)
def test_text_stimulus_has_correct_metadata(metadata, make_example_file):
    metadata_pre = deepcopy(metadata)
    aoi_path = make_example_file('stimuli/toy_text_aoi.csv')

    stimulus = text.from_file(
        aoi_path,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        metadata=metadata,
    )

    assert stimulus.metadata == metadata_pre
    assert stimulus.metadata is metadata


def test_text_stimulus_unsupported_format(make_example_file):
    image_filepath = make_example_file('stimuli/pexels-zoorg-1000498.jpg')

    message = 'Stimulus file is not a valid CSV file: .*pexels-zoorg-1000498.jpg'
    with pytest.raises(ValueError, match=message):
        text.from_file(
            image_filepath,
            aoi_column='char',
            start_x_column='top_left_x',
            start_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
        )


def test_text_stimulus_file_not_found_raises():
    message = 'Stimulus file not found.*nonexistingfile[.]csv'
    with pytest.raises(FileNotFoundError, match=message):
        text.from_file(
            'nonexistingfile.csv',
            aoi_column='char',
            start_x_column='top_left_x',
            start_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
        )


@pytest.mark.parametrize(
    ('filename', 'custom_read_kwargs'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            None,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            {'separator': ','},
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting(filename, custom_read_kwargs, make_example_file):
    aoi_file = make_example_file(filename)
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    assert len(aois_df) == 2


@pytest.mark.parametrize(
    ('filename', 'custom_read_kwargs'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            None,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            {'separator': ','},
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting_unique_within(filename, custom_read_kwargs, make_example_file):
    aoi_file = make_example_file(filename)
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    assert all(df.aois.n_unique(subset=['line_idx']) == 1 for df in aois_df)


@pytest.mark.parametrize(
    ('filename', 'custom_read_kwargs'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            None,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            {'separator': ','},
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting_different_between(filename, custom_read_kwargs, make_example_file):
    aoi_file = make_example_file(filename)
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    unique_values = []
    for df in aois_df:
        unique_value = df.aois.unique(subset=['line_idx'])['line_idx'].to_list()
        unique_values.extend(unique_value)

    assert len(unique_values) == len(set(unique_values))


def test_writing_system_default(sample_aoi_dataframe):
    """Test that default writing_system is horizontal-lr."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
    )
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == HORIZONTAL_LR


def test_writing_system_object(sample_aoi_dataframe, writing_system):
    """Test that writing_system can be set explicitly."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
        writing_system=writing_system,
    )
    assert stimulus.writing_system == writing_system


@pytest.mark.parametrize(
    ('descriptor', 'writing_system'),
    [
        pytest.param('left-to-right', 'hlr', id='left-to-right'),
        pytest.param('LEFT-TO-RIGHT', 'hlr', id='LEFT-TO-RIGHT'),
        pytest.param('ltr', 'hlr', id='ltr'),
        pytest.param('LTR', 'hlr', id='LTR'),
        pytest.param('right-to-left', 'hrl', id='right-to-left'),
        pytest.param('RIGHT-TO-LEFT', 'hrl', id='RIGHT-TO-LEFT'),
        pytest.param('rtl', 'hrl', id='rtl'),
        pytest.param('RTL', 'hrl', id='RTL'),
    ],
    indirect=['writing_system'],
)
def test_writing_system_descriptor(descriptor, writing_system, sample_aoi_dataframe):
    """Test that writing_system can be set from string descriptor."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
        writing_system=descriptor,
    )
    assert stimulus.writing_system == writing_system


def test_writing_system_preserved_by_from_csv(sample_aoi_dataframe, writing_system, make_csv_file):
    """Test that from_csv() accepts and preserves writing_system."""
    filepath = make_csv_file('test_aoi.csv', sample_aoi_dataframe)

    stimulus = TextStimulus.from_csv(
        path=filepath,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
        writing_system=writing_system,
    )

    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == writing_system


def test_writing_system_from_csv_default(sample_aoi_dataframe, make_csv_file):
    """Test that from_csv() uses default writing_system when not specified."""
    filepath = make_csv_file('test_aoi.csv', sample_aoi_dataframe)

    stimulus = TextStimulus.from_csv(
        path=filepath,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
    )

    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == HORIZONTAL_LR


def test_writing_system_attribute_access(sample_aoi_dataframe, writing_system):
    """Test that writing_system can be accessed as an attribute."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
        writing_system=writing_system,
    )

    # Test attribute access
    assert hasattr(stimulus, 'writing_system')
    assert isinstance(stimulus.writing_system, WritingSystem)
    assert stimulus.writing_system == writing_system


def test_writing_system_preserved_by_split_sample_df(sample_aoi_dataframe, writing_system):
    """Test that split() preserves writing_system."""
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=writing_system,
    )

    # Split by page
    splits = stimulus.split(by='page')

    # Check that all split parts preserve the writing_system
    assert len(splits) == 2
    assert all(stimulus.writing_system == writing_system for stimulus in splits)


@pytest.mark.parametrize(
    ('filename', 'writing_system', 'expected_n_lines'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            'hlr',
            2,
            id='ltr_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            'hrl',
            2,
            id='rtl_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            'vrl',
            3,
            id='vertical_rl_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            'vlr',
            3,
            id='vertical_lr_split',
        ),
    ],
    indirect=['writing_system'],
)
def test_writing_system_preserved_by_split_example_file(
    filename,
    writing_system,
    expected_n_lines,
    make_example_file,
):
    filepath = make_example_file(filename)
    stimulus = text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=writing_system,
    )

    split_stimuli = stimulus.split(by='line_idx')

    assert len(split_stimuli) == expected_n_lines
    assert all(stimulus.writing_system == writing_system for stimulus in split_stimuli)


@pytest.mark.parametrize(
    ('filename', 'writing_system', 'row', 'expected_aoi'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            'hlr',
            {'x': 400, 'y': 125},
            'A',
            id='ltr_inside',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            'hlr',
            {'x': 500, 'y': 300},
            None,
            id='ltr_outside',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            'hrl',
            {'x': 1161, 'y': 125},
            'T',
            id='rtl_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            'hrl',
            {'x': 1279, 'y': 125},
            'A',
            id='rtl_last_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            'hrl',
            {'x': 1280, 'y': 125},
            None,
            id='rtl_exclusive_end_boundary',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            'vrl',
            {'x': 1266, 'y': 125},
            'A',
            id='vertical_rl_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            'vrl',
            {'x': 1266, 'y': 140},
            'B',
            id='vertical_rl_second_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            'vrl',
            {'x': 1146, 'y': 125},
            'r',
            id='vertical_rl_third_column_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            'vrl',
            {'x': 1280, 'y': 125},
            None,
            id='vertical_rl_exclusive_x_end',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            'vlr',
            {'x': 401, 'y': 125},
            'A',
            id='vertical_lr_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            'vlr',
            {'x': 401, 'y': 140},
            'B',
            id='vertical_lr_second_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            'vlr',
            {'x': 521, 'y': 125},
            'r',
            id='vertical_lr_third_column_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            'vlr',
            {'x': 415, 'y': 125},
            None,
            id='vertical_lr_exclusive_x_end',
        ),
    ],
    indirect=['writing_system'],
)
def test_text_stimulus_get_aoi_parameterized(
    filename,
    writing_system,
    row,
    expected_aoi,
    make_example_file,
):
    filepath = make_example_file(filename)
    stimulus = text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=writing_system,
    )

    aoi = stimulus.get_aoi(row=row, x_eye='x', y_eye='y')

    assert aoi['char'].first() == expected_aoi


def test_text_stimulus_rtl_writing_mode_and_line_order(make_example_file):
    filepath = make_example_file('stimuli/toy_text_aoi_rtl.csv')
    text_stimulus_rtl = text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=HORIZONTAL_RL,
    )

    assert text_stimulus_rtl.writing_system == HORIZONTAL_RL

    first_line = (
        text_stimulus_rtl.aois
        .filter(pl.col('line_idx') == 0)
        .select('char', 'top_left_x')
    )

    assert first_line['char'].to_list() == ['T', 'C', 'A', 'R', 'T', 'S', 'B', 'A']
    assert first_line['top_left_x'].to_list() == [
        1160.0, 1175.0, 1190.0, 1205.0, 1220.0, 1235.0, 1250.0, 1265.0,
    ]


def test_text_stimulus_vertical_rl_writing_mode_and_line_order(make_example_file):
    filepath = make_example_file('stimuli/toy_text_aoi_vertical_rtl.csv')
    text_stimulus_vertical_rl = text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=VERTICAL_RL,
    )

    assert text_stimulus_vertical_rl.writing_system == VERTICAL_RL

    first_line = (
        text_stimulus_vertical_rl.aois
        .filter(pl.col('line_idx') == 0)
        .select('char', 'top_left_x', 'top_left_y')
    )
    line_positions = (
        text_stimulus_vertical_rl.aois
        .group_by('line_idx')
        .agg(pl.col('top_left_x').first().alias('x'))
        .sort('line_idx')
    )
    line_indices = sorted(text_stimulus_vertical_rl.aois['line_idx'].unique().to_list())

    assert first_line['char'].to_list() == ['A', 'B', 'S', 'T', 'R', 'A', 'C', 'T']
    assert first_line['top_left_x'].to_list() == [
        1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0,
    ]
    assert first_line['top_left_y'].to_list() == [
        122.0, 140.0, 158.0, 176.0, 194.0, 212.0, 230.0, 248.0,
    ]
    assert line_indices == [0, 1, 2]
    assert line_positions['x'].to_list() == [1265.0, 1205.0, 1145.0]


def test_text_stimulus_vertical_lr_writing_mode_and_line_order(make_example_file):
    filepath = make_example_file('stimuli/toy_text_aoi_vertical_ltr.csv')
    text_stimulus_vertical_lr = text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        writing_system=VERTICAL_LR,
    )

    assert text_stimulus_vertical_lr.writing_system == VERTICAL_LR

    first_line = (
        text_stimulus_vertical_lr.aois
        .filter(pl.col('line_idx') == 0)
        .select('char', 'top_left_x', 'top_left_y')
    )
    line_positions = (
        text_stimulus_vertical_lr.aois
        .group_by('line_idx')
        .agg(pl.col('top_left_x').first().alias('x'))
        .sort('line_idx')
    )
    line_indices = sorted(text_stimulus_vertical_lr.aois['line_idx'].unique().to_list())

    assert first_line['char'].to_list() == ['A', 'B', 'S', 'T', 'R', 'A', 'C', 'T']
    assert first_line['top_left_x'].to_list() == [
        400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
    ]
    assert first_line['top_left_y'].to_list() == [
        122.0, 140.0, 158.0, 176.0, 194.0, 212.0, 230.0, 248.0,
    ]
    assert line_indices == [0, 1, 2]
    assert line_positions['x'].to_list() == [400.0, 460.0, 520.0]


WIDTH_HEIGHT_COLUMNS = {'width_column': 'width', 'height_column': 'height'}
END_XY_COLUMNS = {'end_x_column': 'x_max', 'end_y_column': 'y_max'}


@pytest.fixture(name='sample_page_trial_aoi_dataframe')
def fixture_sample_page_trial_aoi_dataframe():
    """Create a sample AOI dataframe with page and trial columns for testing."""
    return pl.DataFrame({
        'aoi': ['a', 'b', 'c', 'd'],
        'x_min': [0, 100, 0, 100],
        'y_min': [0, 0, 50, 50],
        'width': [100, 100, 100, 100],
        'height': [50, 50, 50, 50],
        'x_max': [100, 200, 100, 200],
        'y_max': [50, 50, 100, 100],
        'page': [1, 1, 2, 2],
        'trial': [1, 2, 1, 2],
    })


@pytest.fixture(name='close_figures', autouse=True)
def fixture_close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


EXPECTED_BOXES_DF = pl.DataFrame(
    {
        'text': ['word1', 'word2', 'word3'],
        'start_x': [0.0, 100.0, 200.0],
        'start_y': [0.0, 0.0, 0.0],
        'width': [100.0, 100.0, 100.0],
        'height': [50.0, 50.0, 50.0],
    },
)


@pytest.mark.parametrize(
    'geometry_columns',
    [
        pytest.param(WIDTH_HEIGHT_COLUMNS, id='width_height'),
        pytest.param(END_XY_COLUMNS, id='end_xy'),
    ],
)
def test_text_stimulus_resolve_boxes_input_forms(sample_aoi_dataframe, geometry_columns):
    aois = sample_aoi_dataframe.with_columns(
        x_max=pl.col('x_min') + pl.col('width'),
        y_max=pl.col('y_min') + pl.col('height'),
    )
    stimulus = TextStimulus(
        aois=aois,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **geometry_columns,
    )

    boxes = stimulus.resolve_boxes()

    assert_frame_equal(boxes, EXPECTED_BOXES_DF)


@pytest.mark.parametrize(
    ('page_column', 'trial_column', 'page', 'trial', 'expected_text'),
    [
        pytest.param('page', None, 1, None, ['a', 'b'], id='page_only'),
        pytest.param(None, 'trial', None, 1, ['a', 'c'], id='trial_only'),
        pytest.param('page', 'trial', 1, 1, ['a'], id='page_and_trial_first'),
        pytest.param('page', 'trial', 2, 2, ['d'], id='page_and_trial_last'),
    ],
)
def test_text_stimulus_resolve_boxes_selection(
    sample_page_trial_aoi_dataframe,
    page_column,
    trial_column,
    page,
    trial,
    expected_text,
):
    stimulus = TextStimulus(
        aois=sample_page_trial_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
        page_column=page_column,
        trial_column=trial_column,
    )

    boxes = stimulus.resolve_boxes(page=page, trial=trial)

    assert boxes['text'].to_list() == expected_text


@pytest.mark.parametrize(
    ('stimulus_kwargs', 'resolve_kwargs', 'message'),
    [
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {'page': 1},
            'page=1 was provided, but no page_column is configured',
            id='page_without_page_column',
        ),
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {'trial': 1},
            'trial=1 was provided, but no trial_column is configured',
            id='trial_without_trial_column',
        ),
        pytest.param(
            {**WIDTH_HEIGHT_COLUMNS, 'page_column': 'page'},
            {},
            "page must be provided because page_column 'page' is configured",
            id='page_column_without_page',
        ),
        pytest.param(
            {**WIDTH_HEIGHT_COLUMNS, 'trial_column': 'trial'},
            {},
            "trial must be provided because trial_column 'trial' is configured",
            id='trial_column_without_trial',
        ),
        pytest.param(
            {**WIDTH_HEIGHT_COLUMNS, 'page_column': 'page'},
            {'page': 99},
            'No AOIs found for page=99',
            id='page_not_found',
        ),
        pytest.param(
            {**WIDTH_HEIGHT_COLUMNS, 'trial_column': 'trial'},
            {'trial': 99},
            'No AOIs found for trial=99',
            id='trial_not_found',
        ),
        pytest.param(
            {'width_column': 'width'},
            {},
            'Both width_column and height_column must be configured together',
            id='width_without_height',
        ),
        pytest.param(
            {'height_column': 'height'},
            {},
            'Both width_column and height_column must be configured together',
            id='height_without_width',
        ),
        pytest.param(
            {'end_x_column': 'x_max'},
            {},
            'Both end_x_column and end_y_column must be configured together',
            id='end_x_without_end_y',
        ),
        pytest.param(
            {'end_y_column': 'y_max'},
            {},
            'Both end_x_column and end_y_column must be configured together',
            id='end_y_without_end_x',
        ),
        pytest.param(
            {},
            {},
            'AOI geometry cannot be resolved',
            id='no_geometry_columns',
        ),
    ],
)
def test_text_stimulus_resolve_boxes_raises(
    sample_page_trial_aoi_dataframe,
    stimulus_kwargs,
    resolve_kwargs,
    message,
):
    stimulus = TextStimulus(
        aois=sample_page_trial_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **stimulus_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        stimulus.resolve_boxes(**resolve_kwargs)


@pytest.mark.parametrize(
    ('stimulus_kwargs', 'columns', 'message'),
    [
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'width': [100.0, None], 'height': [50.0, 50.0],
            },
            'Skipping defective AOI row',
            id='width_height_none',
        ),
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'width': [100.0, 100.0], 'height': [50.0, math.nan],
            },
            'Skipping defective AOI row',
            id='width_height_nan',
        ),
        pytest.param(
            END_XY_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'x_max': [100.0, None], 'y_max': [50.0, 50.0],
            },
            'Skipping defective AOI row',
            id='end_xy_none',
        ),
        pytest.param(
            END_XY_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'x_max': [100.0, 200.0], 'y_max': [50.0, math.nan],
            },
            'Skipping defective AOI row',
            id='end_xy_nan',
        ),
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'width': [100.0, 0.0], 'height': [50.0, 50.0],
            },
            'Skipping AOI with non-positive extent',
            id='width_height_zero_width',
        ),
        pytest.param(
            WIDTH_HEIGHT_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'width': [100.0, 100.0], 'height': [50.0, -50.0],
            },
            'Skipping AOI with non-positive extent',
            id='width_height_negative_height',
        ),
        pytest.param(
            END_XY_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'x_max': [100.0, 100.0], 'y_max': [50.0, 50.0],
            },
            'Skipping AOI with non-positive extent',
            id='end_xy_zero_width',
        ),
        pytest.param(
            END_XY_COLUMNS,
            {
                'x_min': [0.0, 100.0], 'y_min': [0.0, 0.0],
                'x_max': [100.0, 200.0], 'y_max': [50.0, -10.0],
            },
            'Skipping AOI with non-positive extent',
            id='end_xy_negative_height',
        ),
    ],
)
def test_text_stimulus_resolve_boxes_warns_and_skips(stimulus_kwargs, columns, message):
    stimulus = TextStimulus(
        aois=pl.DataFrame({'aoi': ['good', 'bad'], **columns}),
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **stimulus_kwargs,
    )

    with pytest.warns(UserWarning, match=message):
        boxes = stimulus.resolve_boxes()

    assert boxes['text'].to_list() == ['good']


@pytest.mark.parametrize(
    ('page_column', 'page', 'expected_boxes'),
    [
        pytest.param(
            None,
            None,
            [
                (0.0, 0.0, 100.0, 50.0),
                (100.0, 0.0, 100.0, 50.0),
                (200.0, 0.0, 100.0, 50.0),
            ],
            id='all_boxes',
        ),
        pytest.param(
            'page',
            2,
            [(200.0, 0.0, 100.0, 50.0)],
            id='selected_page',
        ),
    ],
)
def test_text_stimulus_plot_box_placement(sample_aoi_dataframe, page_column, page, expected_boxes):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
        page_column=page_column,
    )

    _, ax = stimulus.plot(page=page)

    boxes = [(*patch.get_xy(), patch.get_width(), patch.get_height()) for patch in ax.patches]
    assert boxes == expected_boxes


def test_text_stimulus_plot_labels_centered_by_default(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    _, ax = stimulus.plot()

    assert [text.get_text() for text in ax.texts] == ['word1', 'word2', 'word3']
    assert [text.get_position() for text in ax.texts] == [
        (50.0, 25.0), (150.0, 25.0), (250.0, 25.0),
    ]
    assert all(text.get_ha() == 'center' for text in ax.texts)
    assert all(text.get_va() == 'center' for text in ax.texts)


def test_text_stimulus_plot_text_kwargs_override_defaults(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    _, ax = stimulus.plot(text_kwargs={'ha': 'left', 'va': 'bottom', 'fontsize': 20})

    assert len(ax.texts) == 3
    assert all(text.get_ha() == 'left' for text in ax.texts)
    assert all(text.get_va() == 'bottom' for text in ax.texts)
    assert all(text.get_fontsize() == 20 for text in ax.texts)


def test_text_stimulus_plot_box_kwargs_override_defaults(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    _, ax = stimulus.plot(box_kwargs={'fill': True, 'edgecolor': 'red', 'linewidth': 2})

    assert len(ax.patches) == 3
    assert all(patch.get_fill() for patch in ax.patches)
    assert all(patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0) for patch in ax.patches)
    assert all(patch.get_linewidth() == 2 for patch in ax.patches)


def test_text_stimulus_plot_creates_own_figure_when_ax_not_provided(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    fig, ax = stimulus.plot()

    assert isinstance(fig, plt.Figure)
    assert ax.figure is fig
    assert ax.get_aspect() == 1.0


def test_text_stimulus_plot_uses_provided_ax(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    existing_fig, existing_ax = plt.subplots()

    returned_fig, returned_ax = stimulus.plot(ax=existing_ax)

    assert returned_ax is existing_ax
    assert returned_fig is existing_fig


def test_text_stimulus_plot_show_boxes_false_draws_labels_only(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
    )

    _, ax = stimulus.plot(show_boxes=False)

    assert len(ax.patches) == 0
    assert [text.get_text() for text in ax.texts] == ['word1', 'word2', 'word3']


def test_text_stimulus_plot_empty_selection_raises(sample_aoi_dataframe):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
        page_column='page',
    )

    with pytest.raises(ValueError, match='No AOIs found for page=99'):
        stimulus.plot(page=99)


@pytest.mark.parametrize(
    'writing_system',
    [
        pytest.param(HORIZONTAL_RL, id='horizontal_rtl'),
        pytest.param(VERTICAL_RL, id='vertical_rl'),
        pytest.param(VERTICAL_LR, id='vertical_lr'),
        pytest.param(
            WritingSystem('left-to-right', axis='vertical', lining='right-to-left'),
            id='vertical_axis_ltr_directionality',
        ),
    ],
)
def test_text_stimulus_plot_unsupported_writing_system_raises(
    sample_aoi_dataframe,
    writing_system,
):
    stimulus = TextStimulus(
        aois=sample_aoi_dataframe,
        aoi_column='aoi',
        start_x_column='x_min',
        start_y_column='y_min',
        **WIDTH_HEIGHT_COLUMNS,
        writing_system=writing_system,
    )

    message = 'currently supports only horizontal left-to-right writing systems'
    with pytest.raises(NotImplementedError, match=message):
        stimulus.plot()
