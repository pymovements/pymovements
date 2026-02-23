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
"""Test for Experiment class."""

from re import escape

import polars as pl
import pytest

from pymovements import Gaze


@pytest.mark.parametrize(
    'bad_messages',
    [
        pytest.param(123, id='int'),
        pytest.param('foo', id='str'),
        pytest.param({'a': 1}, id='dict'),
        pytest.param([1, 2, 3], id='list'),
        pytest.param(pl.Series('x', [1, 2]), id='polars_series'),
    ],
)
def test_gaze_messages_must_be_polars_dataframe(bad_messages):
    """Ensure that non-DataFrame `messages` raises a TypeError with exact message."""
    expected = (
        f"The `messages` must be a polars DataFrame with columns ['time', 'content'], "
        f'not {type(bad_messages)}.'
    )
    with pytest.raises(TypeError, match=escape(expected)):
        Gaze(messages=bad_messages)


@pytest.mark.parametrize(
    'good_messages',
    [
        pytest.param(None, id='none'),
        pytest.param(pl.DataFrame({'time': [1], 'content': ['hello']}), id='polars_dataframe'),
    ],
)
def test_gaze_messages_accepts_none_or_polars_dataframe(good_messages):
    """Ensure that `messages` accepts None or a polars DataFrame without raising."""
    gaze = Gaze(messages=good_messages)
    if good_messages is None:
        assert gaze.messages is None
    else:
        assert gaze.messages is good_messages


@pytest.mark.parametrize(
    'bad_df',
    [
        pytest.param(pl.DataFrame({'time': [1, 2]}), id='missing_content'),
        pytest.param(pl.DataFrame({'content': ['a', 'b']}), id='missing_time'),
        pytest.param(pl.DataFrame({'timestamp': [1], 'content': ['a']}), id='wrong_time_name'),
        pytest.param(pl.DataFrame({'time': [1], 'message': ['a']}), id='wrong_content_name'),
        pytest.param(pl.DataFrame({'foo': [1], 'bar': ['a']}), id='no_required_cols'),
    ],
)
def test_gaze_messages_dataframe_must_have_time_and_content_columns(bad_df):
    """Ensure that a polars DataFrame missing required columns raises a TypeError."""
    expected = "The `messages` polars DataFrame must contain the columns ['time', 'content']."
    with pytest.raises(TypeError, match=escape(expected)):
        Gaze(messages=bad_df)


@pytest.mark.parametrize(
    ('messages_df', 'expected_fragment'),
    [
        pytest.param(
            pl.DataFrame(schema={'time': pl.Float64, 'content': pl.String}),
            '0 rows',
            id='messages_empty_df',
        ),
        pytest.param(
            pl.DataFrame({'time': [1.0, 2.0], 'content': ['a', 'b']}),
            '2 rows',
            id='messages_two_rows',
        ),
    ],
)
def test_gaze_str_messages_variations(messages_df, expected_fragment):
    """Check __str__ shows clear messages summary."""
    gaze = Gaze(messages=messages_df)
    s = str(gaze)
    assert f'messages={expected_fragment}' in s, s


def test_gaze_str_messages_exluded_if_none():
    """Check __str__ shows clear messages summary."""
    gaze = Gaze(messages=None)
    s = str(gaze)
    assert 'messages=' not in s, s
