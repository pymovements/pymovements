# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Provides top-level access to submodules."""
from pymovements import _version
from pymovements import datasets
from pymovements import events
from pymovements import exceptions
from pymovements import gaze
from pymovements import measure
from pymovements import plotting
from pymovements import reading_measures
from pymovements import stimulus
from pymovements import synthetic
from pymovements import utils
from pymovements.dataset import Dataset
from pymovements.dataset import DatasetDefinition
from pymovements.dataset import DatasetLibrary
from pymovements.dataset import DatasetPaths
from pymovements.dataset import register_dataset
from pymovements.dataset import ResourceDefinition
from pymovements.dataset import ResourceDefinitions
from pymovements.events import EventDataFrame
from pymovements.events import EventGazeProcessor
from pymovements.events import EventProcessor
from pymovements.events import Events
from pymovements.gaze import Experiment
from pymovements.gaze import EyeTracker
from pymovements.gaze import Gaze
from pymovements.gaze import GazeDataFrame
from pymovements.gaze import Screen
from pymovements.measure import register_sample_measure
from pymovements.measure import SampleMeasureLibrary
from pymovements.stimulus import text


__all__ = [
    'Dataset',
    'DatasetDefinition',
    'DatasetLibrary',
    'DatasetPaths',
    'datasets',
    'register_dataset',
    'ResourceDefinition',
    'ResourceDefinitions',

    'events',
    'Events',
    'EventDataFrame',
    'EventGazeProcessor',
    'EventProcessor',

    'gaze',
    'Experiment',
    'EyeTracker',
    'Screen',
    'Gaze',
    'GazeDataFrame',

    'exceptions',

    'measure',
    'SampleMeasureLibrary',
    'register_sample_measure',

    'plotting',
    'reading_measures',
    'stimulus',
    'synthetic',
    'utils',

    'text',
]

__version__ = _version.get_versions()['version']
