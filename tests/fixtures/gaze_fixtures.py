# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Provide shared fixtures for gaze tests."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements import Events
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Gaze
from pymovements import Screen


@pytest.fixture(name='gaze_all', scope='function')
def fixture_gaze_all():
    """Create a Gaze object with all components and experiment."""
    samples = pl.DataFrame(
        {
            'x': [0, 1, 2, 3],
            'y': [1, 1, 0, 0],
            'pixel': [[260, 150], [270, 120], [271, 122], [240, 22]],
            'trial_id': [0, 1, 1, 2],
        },
        schema={
            'x': pl.Float64,
            'y': pl.Float64,
            'pixel': list,
            'trial_id': pl.Int8,
        },
    )
    return Gaze(
        samples=samples,
        experiment=Experiment(
            screen=Screen(
                width_px=1280,
                height_px=1024,
                width_cm=38.0,
                height_cm=30.0,
                distance_cm=68.0,
                origin='upper left',
            ),
            eyetracker=EyeTracker(
                sampling_rate=1000.0,
                left=None,
                right=None,
                model='MyModel',
                version=None,
                vendor=None,
                mount=None,
            ),
        ),
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )


@pytest.fixture(name='gaze_minimal', scope='function')
def fixture_gaze_minimal():
    """Create a minimal Gaze object without experiment."""
    samples = pl.DataFrame(
        {'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0], 'trial': [1, 1, 2]},
    )
    events = Events(
        pl.DataFrame(
            {
                'name': ['saccade', 'saccade', 'fixation'],
                'onset': [0, 10, 20],
                'offset': [10, 20, 30],
            },
        ),
    )
    return Gaze(
        samples=samples,
        events=events,
        pixel_columns=['x', 'y'],
        trial_columns='trial',
    )
