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
"""Provides gaze related functionality."""
from pymovements.gaze import transforms
from pymovements.gaze import transforms_numpy
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.gaze import Gaze
from pymovements.gaze.gaze_dataframe import GazeDataFrame
from pymovements.gaze.integration import from_numpy
from pymovements.gaze.integration import from_pandas
from pymovements.gaze.io import from_asc
from pymovements.gaze.io import from_csv
from pymovements.gaze.io import from_ipc
from pymovements.gaze.io import from_psychopy_csv
from pymovements.gaze.screen import Screen


__all__ = [
    'Experiment',
    'EyeTracker',
    'from_numpy',
    'from_pandas',
    'Gaze',
    'GazeDataFrame',
    'Screen',
    'transforms_numpy',
    'transforms',
    'from_asc',
    'from_psychopy_csv',
    'from_csv',
    'from_ipc',
]
