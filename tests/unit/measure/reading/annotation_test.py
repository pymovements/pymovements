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
"""Fixation annotation tests."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.annotation import delta_in
from pymovements.measure.reading.annotation import delta_out
from pymovements.measure.reading.annotation import is_first_fixation
from pymovements.measure.reading.annotation import is_first_pass
from pymovements.measure.reading.annotation import is_reg_in
from pymovements.measure.reading.annotation import is_reg_out
from pymovements.measure.reading.annotation import next_word_idx
from pymovements.measure.reading.annotation import prev_word_idx
from pymovements.measure.reading.annotation import run_id


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': [],
                    'stimulus': [],
                    'page': [],
                    'name': [],
                    'word_idx': [],
                },
                schema={
                    'trial': pl.String,
                    'stimulus': pl.String,
                    'page': pl.String,
                    'name': pl.String,
                    'word_idx': pl.Int64,
                },
            ),
            pl.DataFrame(
                data={
                    'trial': [],
                    'stimulus': [],
                    'page': [],
                    'name': [],
                    'word_idx': [],
                    'run_id': [],
                },
                schema={
                    'trial': pl.String,
                    'stimulus': pl.String,
                    'page': pl.String,
                    'name': pl.String,
                    'word_idx': pl.Int64,
                    'run_id': pl.Int64,
                },
            ),
            id='empty',
        ),
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '1', '1'],
                    'word_idx': [1, 1, 2, 1],
                },
            ),
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '1', '1'],
                    'word_idx': [1, 1, 2, 1],
                    'run_id': [1, 1, 2, 3],
                },
                schema_overrides={'run_id': pl.Int64},
            ),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '2', '2'],
                    'word_idx': [1, 1, 1, 2],
                },
            ),
            pl.DataFrame(
                data={
                    'trial': ['1', '1', '2', '2'],
                    'word_idx': [1, 1, 1, 2],
                    'run_id': [1, 1, 1, 2],
                },
                schema_overrides={'run_id': pl.Int64},
            ),
            id='two_trials',
        ),
    ],
)
def test_run_id(fixations, expected):
    result = fixations.with_columns(run_id().over(['trial']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2]}),
            pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2], 'prev_word_idx': [None, 1]}),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame({'trial': ['1', '2'], 'word_idx': [1, 2]}),
            pl.DataFrame({
                'trial': ['1', '2'],
                'word_idx': [1, 2],
                'prev_word_idx': [None, None],
            }).with_columns(pl.col('prev_word_idx').cast(pl.Int64)),
            id='two_trials',
        ),
        pytest.param(
            pl.DataFrame(
                data={'trial': [], 'word_idx': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64},
            ),
            pl.DataFrame(
                data={'trial': [], 'word_idx': [], 'prev_word_idx': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64, 'prev_word_idx': pl.Int64},
            ),
            id='empty',
        ),
    ],
)
def test_prev_word_idx(fixations, expected):
    result = fixations.with_columns(prev_word_idx().over(['trial']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2]}),
            pl.DataFrame({'trial': ['1', '1'], 'word_idx': [1, 2], 'next_word_idx': [2, None]}),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame({'trial': ['1', '2'], 'word_idx': [1, 2]}),
            pl.DataFrame({
                'trial': ['1', '2'],
                'word_idx': [1, 2],
                'next_word_idx': [None, None],
            }).with_columns(pl.col('next_word_idx').cast(pl.Int64)),
            id='two_trials',
        ),
        pytest.param(
            pl.DataFrame(
                data={'trial': [], 'word_idx': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64},
            ),
            pl.DataFrame(
                data={'trial': [], 'word_idx': [], 'next_word_idx': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64, 'next_word_idx': pl.Int64},
            ),
            id='empty',
        ),
    ],
)
def test_next_word_idx(fixations, expected):
    result = fixations.with_columns(next_word_idx().over(['trial']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'word_idx': [2], 'prev_word_idx': [1]}),
            pl.DataFrame({'word_idx': [2], 'prev_word_idx': [1], 'delta_in': [1]}),
            id='standard',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'prev_word_idx': [3]}),
            pl.DataFrame({'word_idx': [1], 'prev_word_idx': [3], 'delta_in': [-2]}),
            id='regression',
        ),
        pytest.param(
            pl.DataFrame(
                {'word_idx': [1], 'prev_word_idx': [None]},
                schema={'word_idx': pl.Int64, 'prev_word_idx': pl.Int64},
            ),
            pl.DataFrame({
                'word_idx': [1],
                'prev_word_idx': [None],
                'delta_in': [None],
            }).with_columns([
                pl.col('prev_word_idx').cast(pl.Int64),
                pl.col('delta_in').cast(pl.Int64),
            ]),
            id='start_of_sequence',
        ),
    ],
)
def test_delta_in(fixations, expected):
    result = fixations.with_columns(delta_in())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'word_idx': [1], 'next_word_idx': [2]}),
            pl.DataFrame({'word_idx': [1], 'next_word_idx': [2], 'delta_out': [1]}),
            id='standard',
        ),
        pytest.param(
            pl.DataFrame({'word_idx': [3], 'next_word_idx': [1]}),
            pl.DataFrame({'word_idx': [3], 'next_word_idx': [1], 'delta_out': [-2]}),
            id='regression',
        ),
        pytest.param(
            pl.DataFrame(
                {'word_idx': [1], 'next_word_idx': [None]},
                schema={'word_idx': pl.Int64, 'next_word_idx': pl.Int64},
            ),
            pl.DataFrame({
                'word_idx': [1],
                'next_word_idx': [None],
                'delta_out': [None],
            }).with_columns([
                pl.col('next_word_idx').cast(pl.Int64),
                pl.col('delta_out').cast(pl.Int64),
            ]),
            id='end_of_sequence',
        ),
    ],
)
def test_delta_out(fixations, expected):
    result = fixations.with_columns(delta_out())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'delta_in': [1, -1]}),
            pl.DataFrame({'delta_in': [1, -1], 'is_reg_in': [False, True]}),
            id='standard',
        ),
        pytest.param(
            pl.DataFrame({'delta_in': [0, 2]}),
            pl.DataFrame({'delta_in': [0, 2], 'is_reg_in': [False, False]}),
            id='no_regression',
        ),
        pytest.param(
            pl.DataFrame({'delta_in': [None]}, schema={'delta_in': pl.Int64}),
            pl.DataFrame(
                {'delta_in': [None], 'is_reg_in': [None]},
                schema={'delta_in': pl.Int64, 'is_reg_in': pl.Boolean},
            ),
            id='null_delta',
        ),
    ],
)
def test_is_reg_in(fixations, expected):
    result = fixations.with_columns(is_reg_in())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'delta_out': [1, -1]}),
            pl.DataFrame({'delta_out': [1, -1], 'is_reg_out': [False, True]}),
            id='standard',
        ),
        pytest.param(
            pl.DataFrame({'delta_out': [0, 2]}),
            pl.DataFrame({'delta_out': [0, 2], 'is_reg_out': [False, False]}),
            id='no_regression',
        ),
        pytest.param(
            pl.DataFrame({'delta_out': [None]}, schema={'delta_out': pl.Int64}),
            pl.DataFrame(
                {'delta_out': [None], 'is_reg_out': [None]},
                schema={'delta_out': pl.Int64, 'is_reg_out': pl.Boolean},
            ),
            id='null_delta',
        ),
    ],
)
def test_is_reg_out(fixations, expected):
    result = fixations.with_columns(is_reg_out())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({'trial': ['1', '1', '1'], 'word_idx': [1, 1, 2]}),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'word_idx': [1, 1, 2],
                'is_first_fix': [True, False, True],
            }),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame({'trial': ['1', '2'], 'word_idx': [1, 1]}),
            pl.DataFrame({
                'trial': ['1', '2'],
                'word_idx': [1, 1],
                'is_first_fix': [True, True],
            }),
            id='two_trials',
        ),
        pytest.param(
            pl.DataFrame(
                data={'trial': [], 'word_idx': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64},
            ),
            pl.DataFrame(
                data={'trial': [], 'word_idx': [], 'is_first_fix': []},
                schema={'trial': pl.String, 'word_idx': pl.Int64, 'is_first_fix': pl.Boolean},
            ),
            id='empty',
        ),
    ],
)
def test_is_first_fixation(fixations, expected):
    result = fixations.with_columns(is_first_fixation().over(['trial', 'word_idx']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1', '1', '1'],
                'onset': [0, 1, 2, 3, 4],
                'word_idx': [1, 1, 2, 1, 3],
                'run_id': [1, 1, 2, 3, 4],
                'prev_word_idx': [None, 1, 1, 2, 1],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1', '1', '1'],
                'onset': [0, 1, 2, 3, 4],
                'word_idx': [1, 1, 2, 1, 3],
                'run_id': [1, 1, 2, 3, 4],
                'prev_word_idx': [None, 1, 1, 2, 1],
                'is_first_pass': [True, True, True, False, True],
            }),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '2'],
                'onset': [0, 0],
                'word_idx': [1, 1],
                'run_id': [1, 1],
                'prev_word_idx': [None, None],
            }),
            pl.DataFrame({
                'trial': ['1', '2'],
                'onset': [0, 0],
                'word_idx': [1, 1],
                'run_id': [1, 1],
                'prev_word_idx': [None, None],
                'is_first_pass': [True, True],
            }),
            id='two_trials',
        ),
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': [],
                    'onset': [],
                    'word_idx': [],
                    'run_id': [],
                    'prev_word_idx': [],
                },
                schema={
                    'trial': pl.String,
                    'onset': pl.Int64,
                    'word_idx': pl.Int64,
                    'run_id': pl.Int64,
                    'prev_word_idx': pl.Int64,
                },
            ),
            pl.DataFrame(
                data={
                    'trial': [],
                    'onset': [],
                    'word_idx': [],
                    'run_id': [],
                    'prev_word_idx': [],
                    'is_first_pass': [],
                },
                schema={
                    'trial': pl.String,
                    'onset': pl.Int64,
                    'word_idx': pl.Int64,
                    'run_id': pl.Int64,
                    'prev_word_idx': pl.Int64,
                    'is_first_pass': pl.Boolean,
                },
            ),
            id='empty',
        ),
    ],
)
def test_is_first_pass(fixations, expected):
    result = fixations.with_columns(is_first_pass(['trial']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('fixations', 'expected'),
    [
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [1, 3, 2],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 1, 3],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [1, 3, 2],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 1, 3],
                'is_first_pass': [True, True, False],
            }),
            id='skip_regression',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [1, 2, 3],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 1, 2],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [1, 2, 3],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 1, 2],
                'is_first_pass': [True, True, True],
            }),
            id='forward',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [2, 1, 3],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 2, 1],
            }),
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'onset': [0, 1, 2],
                'word_idx': [2, 1, 3],
                'run_id': [1, 2, 3],
                'prev_word_idx': [None, 2, 1],
                'is_first_pass': [True, False, True],
            }),
            id='regression_entry',
        ),
    ],
)
def test_is_first_pass_regression_skip(fixations, expected):
    result = fixations.with_columns(is_first_pass(['trial']))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'events',
    [
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '1', '1'],
                'stimulus': ['s', 's', 's'],
                'page': ['p', 'p', 'p'],
                'name': ['fixation', 'fixation', 'fixation'],
                'word_idx': [1, 2, 1],
                'onset': [0, 100, 200],
            }),
            id='single_trial',
        ),
        pytest.param(
            pl.DataFrame({
                'trial': ['1', '2'],
                'stimulus': ['s', 's'],
                'page': ['p', 'p'],
                'name': ['fixation', 'fixation'],
                'word_idx': [1, 1],
                'onset': [0, 0],
            }),
            id='two_trials',
        ),
        pytest.param(
            pl.DataFrame(
                data={
                    'trial': [],
                    'stimulus': [],
                    'page': [],
                    'name': [],
                    'word_idx': [],
                    'onset': [],
                },
                schema={
                    'trial': pl.String,
                    'stimulus': pl.String,
                    'page': pl.String,
                    'name': pl.String,
                    'word_idx': pl.Int64,
                    'onset': pl.Int64,
                },
            ),
            id='empty',
        ),
    ],
)
def test_annotate_fixations(events):
    result = annotate_fixations(events, group_columns=['trial', 'stimulus', 'page'])

    expected_columns = [
        'trial', 'stimulus', 'page', 'name', 'word_idx', 'onset',
        'fixation_id', 'run_id', 'prev_word_idx', 'next_word_idx',
        'delta_in', 'delta_out', 'is_reg_in', 'is_reg_out',
        'is_first_fix', 'is_first_pass',
    ]
    expected_length = len(
        events.filter(
            (pl.col('name') == 'fixation') & (pl.col('word_idx').is_not_null()),
        ),
    )

    assert all(col in result.columns for col in expected_columns)
    assert len(result) == expected_length


def test_annotate_fixations_default_groups():
    events = pl.DataFrame({
        'trial': ['1'],
        'stimulus': ['s'],
        'page': ['p'],
        'name': ['fixation'],
        'word_idx': [1],
        'onset': [0],
    })
    result = annotate_fixations(events)
    assert 'run_id' in result.columns


def test_annotate_fixations_custom_event_name():
    events = pl.DataFrame({
        'trial': ['1', '1'],
        'stimulus': ['s', 's'],
        'page': ['p', 'p'],
        'name': ['fixation.custom', 'fixation'],
        'word_idx': [1, 2],
        'onset': [0, 100],
    })
    result = annotate_fixations(events, event_name='fixation.custom')
    assert result.height == 1
    assert result['word_idx'].to_list() == [1]


def test_annotate_fixations_warns_when_nothing_to_annotate():
    events = pl.DataFrame({
        'trial': ['1'],
        'stimulus': ['s'],
        'page': ['p'],
        'name': ['saccade'],
        'word_idx': [1],
        'onset': [0],
    })
    with pytest.warns(UserWarning, match='no fixations left to annotate'):
        result = annotate_fixations(events)
    assert result.is_empty()
