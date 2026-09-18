# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Provides functionality for generating synthetic eye gaze data."""
from pymovements.synthetic import generators
from pymovements.synthetic.capabilities import ConditioningMode
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.capabilities import SignalRepresentation
from pymovements.synthetic.config import GeneratorConfig
from pymovements.synthetic.generator import GazeGenerator
from pymovements.synthetic.library import GeneratorLibrary
from pymovements.synthetic.library import register_generator
from pymovements.synthetic.step_function import step_function


__all__ = [
    'ConditioningMode',
    'GazeGenerator',
    'GeneratorCapabilities',
    'GeneratorConfig',
    'GeneratorLibrary',
    'SignalRepresentation',
    'register_generator',

    'generators',
    'step_function',
]
