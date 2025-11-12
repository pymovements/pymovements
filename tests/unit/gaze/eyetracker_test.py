# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test for EyeTracker class."""
import pytest

from pymovements.gaze.eyetracker import EyeTracker


def test_eyetracker_with_positive_sampling_rate():
    EyeTracker(
        1000.0, False, True, 'EyeLink 1000 Plus',
        '1.5.3', 'Arm Mount / Monocular / Remote',
    )


def test_eyetracker_with_negative_sampling_rate():
    with pytest.raises(ValueError):
        EyeTracker(
            -500.0, False, True, 'EyeLink 1000 Plus',
            '1.5.3', 'Arm Mount / Monocular / Remote',
        )


def test_eyetracker_without_sampling_rate():
    EyeTracker(
        None, False, True, 'EyeLink 1000 Plus',
        '1.5.3', 'Arm Mount / Monocular / Remote',
    )


@pytest.mark.parametrize(
    ('eyetracker', 'expected_dict', 'exclude_none'),
    [
        pytest.param(
            EyeTracker(),
            {},
            True,
            id='default',
        ),
        pytest.param(
            EyeTracker(right=True, model='test'),
            {'right': True, 'model': 'test'},
            True,
            id='right_model',
        ),
        pytest.param(
            EyeTracker(),
            {
                'sampling_rate': None,
                'left': None,
                'right': None,
                'model': None,
                'version': None,
                'vendor': None,
                'mount': None,
            },
            False,
            id='all_none',
        ),
    ],
)
def test_eyetracker_to_dict_exclude_none(eyetracker, expected_dict, exclude_none):
    assert eyetracker.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    ('eyetracker', 'expected_bool'),
    [
        pytest.param(
            EyeTracker(),
            False,
            id='default',
        ),
        pytest.param(
            EyeTracker(mount='test'),
            True,
            id='mount',
        ),
        pytest.param(
            EyeTracker(sampling_rate=2.0),
            True,
            id='sampling_rate_2',
        ),
    ],
)
def test_eyetracker_bool_all_none(eyetracker, expected_bool):
    assert bool(eyetracker) == expected_bool
