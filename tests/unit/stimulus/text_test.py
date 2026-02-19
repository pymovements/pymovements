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
import polars
import pytest
from polars.testing import assert_frame_equal

from pymovements.stimulus import text
from pymovements.stimulus.text import WritingSystem


HORIZONTAL_LR = WritingSystem('horizontal', 'top-to-bottom', 'left-to-right')
HORIZONTAL_RL = WritingSystem('horizontal', 'top-to-bottom', 'right-to-left')
VERTICAL_RL = WritingSystem('vertical', 'right-to-left', 'top-to-bottom')
VERTICAL_LR = WritingSystem('vertical', 'left-to-right', 'top-to-bottom')


EXPECTED_DF = polars.DataFrame(
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
def test_text_stimulus(filename, custom_read_kwargs, expected, make_example_file):
    aoi_file = make_example_file(filename)
    aois = text.from_file(
        aoi_file,
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


@pytest.mark.parametrize(
    ('filename', 'writing_system', 'expected_n_lines'),
    [
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            HORIZONTAL_LR,
            2,
            id='ltr_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            HORIZONTAL_RL,
            2,
            id='rtl_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            VERTICAL_RL,
            3,
            id='vertical_rl_split',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            VERTICAL_LR,
            3,
            id='vertical_lr_split',
        ),
    ],
)
def test_text_stimulus_split_preserves_writing_mode(
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
            HORIZONTAL_LR,
            {'x': 400, 'y': 125},
            'A',
            id='ltr_inside',
        ),
        pytest.param(
            'stimuli/toy_text_aoi.csv',
            HORIZONTAL_LR,
            {'x': 500, 'y': 300},
            None,
            id='ltr_outside',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            HORIZONTAL_RL,
            {'x': 1161, 'y': 125},
            'T',
            id='rtl_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            HORIZONTAL_RL,
            {'x': 1279, 'y': 125},
            'A',
            id='rtl_last_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_rtl.csv',
            HORIZONTAL_RL,
            {'x': 1280, 'y': 125},
            None,
            id='rtl_exclusive_end_boundary',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            VERTICAL_RL,
            {'x': 1266, 'y': 125},
            'A',
            id='vertical_rl_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            VERTICAL_RL,
            {'x': 1266, 'y': 140},
            'B',
            id='vertical_rl_second_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            VERTICAL_RL,
            {'x': 1146, 'y': 125},
            'r',
            id='vertical_rl_third_column_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_rtl.csv',
            VERTICAL_RL,
            {'x': 1280, 'y': 125},
            None,
            id='vertical_rl_exclusive_x_end',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            VERTICAL_LR,
            {'x': 401, 'y': 125},
            'A',
            id='vertical_lr_first_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            VERTICAL_LR,
            {'x': 401, 'y': 140},
            'B',
            id='vertical_lr_second_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            VERTICAL_LR,
            {'x': 521, 'y': 125},
            'r',
            id='vertical_lr_third_column_char',
        ),
        pytest.param(
            'stimuli/toy_text_aoi_vertical_ltr.csv',
            VERTICAL_LR,
            {'x': 415, 'y': 125},
            None,
            id='vertical_lr_exclusive_x_end',
        ),
    ],
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
        .filter(polars.col('line_idx') == 0)
        .select('char', 'top_left_x')
    )

    assert first_line['char'].to_list() == ['T', 'C', 'A', 'R', 'T', 'S', 'B', 'A']
    assert first_line['top_left_x'].to_list() == [
        1160.0, 1175.0, 1190.0, 1205.0, 1220.0, 1235.0, 1250.0, 1265.0]


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
        .filter(polars.col('line_idx') == 0)
        .select('char', 'top_left_x', 'top_left_y')
    )
    line_positions = (
        text_stimulus_vertical_rl.aois
        .group_by('line_idx')
        .agg(polars.col('top_left_x').first().alias('x'))
        .sort('line_idx')
    )
    line_indices = sorted(text_stimulus_vertical_rl.aois['line_idx'].unique().to_list())

    assert first_line['char'].to_list() == ['A', 'B', 'S', 'T', 'R', 'A', 'C', 'T']
    assert first_line['top_left_x'].to_list() == [
        1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0, 1265.0]
    assert first_line['top_left_y'].to_list() == [
        122.0, 140.0, 158.0, 176.0, 194.0, 212.0, 230.0, 248.0]
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
        .filter(polars.col('line_idx') == 0)
        .select('char', 'top_left_x', 'top_left_y')
    )
    line_positions = (
        text_stimulus_vertical_lr.aois
        .group_by('line_idx')
        .agg(polars.col('top_left_x').first().alias('x'))
        .sort('line_idx')
    )
    line_indices = sorted(text_stimulus_vertical_lr.aois['line_idx'].unique().to_list())

    assert first_line['char'].to_list() == ['A', 'B', 'S', 'T', 'R', 'A', 'C', 'T']
    assert first_line['top_left_x'].to_list() == [
        400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
    assert first_line['top_left_y'].to_list() == [
        122.0, 140.0, 158.0, 176.0, 194.0, 212.0, 230.0, 248.0]
    assert line_indices == [0, 1, 2]
    assert line_positions['x'].to_list() == [400.0, 460.0, 520.0]
