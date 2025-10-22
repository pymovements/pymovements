# Copyright (c) 2024-2025 The pymovements Project Authors
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

from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Screen


def test_sampling_rate_setter():
    experiment = Experiment(1280, 1024, 38, 30, sampling_rate=1000.0)
    assert experiment.sampling_rate == 1000.0

    experiment.sampling_rate = 100.0
    assert experiment.sampling_rate == 100.0


@pytest.mark.parametrize(
    'experiment_init_kwargs',
    [
        pytest.param(
            {},
            id='empty',
        ),
        pytest.param(
            {'sampling_rate': 1000},
            id='only_sampling_rate',
        ),
        pytest.param(
            {'screen': Screen(1024, 768)},
            id='only_screen',
        ),
        pytest.param(
            {'screen': Screen(1024, 768), 'eyetracker': EyeTracker(sampling_rate=1000)},
            id='screen_and_eyetracker',
        ),

    ],
)
def test_sampling_rate_trivial_equality(experiment_init_kwargs):
    experiment1 = Experiment(**experiment_init_kwargs)
    experiment2 = Experiment(**experiment_init_kwargs)
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('experiment1', 'experiment2'),
    [
        pytest.param(
            Experiment(sampling_rate=1000),
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='explicit_sampling_rate_and_eyetracker',
        ),
        pytest.param(
            Experiment(1024, 768),
            Experiment(screen=Screen(1024, 768)),
            id='explicit_screen_size_and_screen',
        ),
    ],
)
def test_sampling_rate_equality(experiment1, experiment2):
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('dictionary', 'expected_experiment'),
    [
        pytest.param(
            {'sampling_rate': 1000},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='sampling_rate',
        ),

        pytest.param(
            {'eyetracker': {'sampling_rate': 1000}},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='eyetracker_sampling_rate',
        ),

        pytest.param(
            {'screen_width_px': 1024, 'screen_height_px': 768},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_screen_height_px',
        ),

        pytest.param(
            {'screen': {'width_px': 1024, 'height_px': 768}},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_height_px',
        ),
    ],
)
def test_experiment_from_dict(dictionary, expected_experiment):
    experiment = Experiment.from_dict(dictionary)
    assert experiment == expected_experiment


@pytest.mark.parametrize(
    ('experiment', 'exclude_none', 'expected_dict'),
    [
        pytest.param(
            Experiment(),
            True,
            {},
            id='true_default',
        ),
        pytest.param(
            Experiment(origin=None),
            True,
            {},
            id='true_origin_none',
        ),
        pytest.param(
            Experiment(sampling_rate=18.5),
            True,
            {'eyetracker': {'sampling_rate': 18.5}},
            id='true_sampling_rate_18.5',
        ),
        pytest.param(
            Experiment(sampling_rate=18.5, origin=None),
            True,
            {'eyetracker': {'sampling_rate': 18.5}},
            id='true_sampling_rate_18.5_origin_none',
        ),
        pytest.param(
            Experiment(screen=Screen(height_px=1080), eyetracker=EyeTracker(left=True)),
            True,
            {
                'screen': {
                    'height_px': 1080,
                },
                'eyetracker': {
                    'left': True,
                },
            },
            id='true_screen_eyetracker',
        ),
        pytest.param(
            Experiment(
                screen=Screen(height_px=1080, origin=None),
                eyetracker=EyeTracker(left=True),
            ),
            True,
            {
                'screen': {
                    'height_px': 1080,
                },
                'eyetracker': {
                    'left': True,
                },
            },
            id='true_screen_origin_none_eyetracker',
        ),
        pytest.param(
            Experiment(),
            False,
            {
                'screen': {
                    'width_px': None,
                    'height_px': None,
                    'width_cm': None,
                    'height_cm': None,
                    'distance_cm': None,
                    'origin': None,
                },
                'eyetracker': {
                    'sampling_rate': None,
                    'vendor': None,
                    'model': None,
                    'version': None,
                    'mount': None,
                    'left': None,
                    'right': None,
                },
            },
            id='false_default',
        ),
        pytest.param(
            Experiment(origin=None),
            False,
            {
                'screen': {
                    'width_px': None,
                    'height_px': None,
                    'width_cm': None,
                    'height_cm': None,
                    'distance_cm': None,
                    'origin': None,
                },
                'eyetracker': {
                    'sampling_rate': None,
                    'vendor': None,
                    'model': None,
                    'version': None,
                    'mount': None,
                    'left': None,
                    'right': None,
                },
            },
            id='false_all_none',
        ),
        pytest.param(
            Experiment(
                origin=None, messages=pl.DataFrame(
                    schema={
                        'time': pl.Float64,
                        'content': pl.String,
                    },
                    data=[
                        (12300, 12333, 14666, 14777, 14888, 15555),
                        (
                            'TASK A', 'PRACTICE', 'TRIAL 1',
                            'TASK B', 'PRACTICE', 'TRIAL 1',
                        ),
                    ],
                ),
            ),
            True,
            {
                'messages': {
                    'time': [12300, 12333, 14666, 14777, 14888, 15555],
                    'content': [
                        'TASK A', 'PRACTICE', 'TRIAL 1',
                        'TASK B', 'PRACTICE', 'TRIAL 1',
                    ],
                },
            },
            id='messages_default',
        ),
    ],
)
def test_experiment_to_dict_exclude_none(experiment, exclude_none, expected_dict):
    assert experiment.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    ('experiment', 'expected_bool'),
    [
        pytest.param(
            Experiment(),
            False,
            id='default',
        ),

        pytest.param(
            Experiment(origin=None),
            False,
            id='origin_none',
        ),

        pytest.param(
            Experiment(origin='center'),
            True,
            id='origin_center',
        ),

        pytest.param(
            Experiment(distance_cm=60),
            True,
            id='distance_60',
        ),
    ],
)
def test_experiment_bool(experiment, expected_bool):
    assert bool(experiment) == expected_bool


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
def test_experiment_messages_must_be_polars_dataframe(bad_messages):
    """Ensure that non-DataFrame `messages` raises a TypeError with exact message."""
    expected = (
        f"The `messages` must be a polars DataFrame with columns ['time', 'content'], not {
            type(bad_messages)
        }."
    )
    with pytest.raises(TypeError, match=escape(expected)):
        Experiment(messages=bad_messages)


@pytest.mark.parametrize(
    'good_messages',
    [
        pytest.param(None, id='none'),
        pytest.param(pl.DataFrame({'time': [1], 'content': ['hello']}), id='polars_dataframe'),
    ],
)
def test_experiment_messages_accepts_none_or_polars_dataframe(good_messages):
    """Ensure that `messages` accepts None or a polars DataFrame without raising."""
    experiment = Experiment(messages=good_messages)
    if good_messages is None:
        assert experiment.messages is None
    else:
        assert experiment.messages is good_messages


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
def test_experiment_messages_dataframe_must_have_time_and_content_columns(bad_df):
    """Ensure that a polars DataFrame missing required columns raises a TypeError."""
    expected = (
        "The `messages` polars DataFrame must contain the columns ['time', 'content']."
    )
    with pytest.raises(TypeError, match=escape(expected)):
        Experiment(messages=bad_df)


@pytest.mark.parametrize(
    ('messages_df', 'expected_fragment'),
    [
        pytest.param(
            None,
            'None',
            id='messages_none',
        ),
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
def test_experiment_str_messages_variations(messages_df, expected_fragment):
    """Check __str__ shows clear messages summary."""
    experiment = Experiment(messages=messages_df) if messages_df is not None else Experiment()
    s = str(experiment)
    assert s.startswith('Experiment(')
    assert f"messages={expected_fragment}" in s
