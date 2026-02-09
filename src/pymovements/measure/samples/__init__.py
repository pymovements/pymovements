# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Module for sample measures."""
from pymovements.measure.samples.library import register_sample_measure
from pymovements.measure.samples.library import SampleMeasureLibrary
from pymovements.measure.samples.measures import amplitude
from pymovements.measure.samples.measures import bcea
from pymovements.measure.samples.measures import data_loss
from pymovements.measure.samples.measures import dispersion
from pymovements.measure.samples.measures import disposition
from pymovements.measure.samples.measures import location
from pymovements.measure.samples.measures import null_ratio
from pymovements.measure.samples.measures import peak_velocity
from pymovements.measure.samples.measures import s2s_rms
from pymovements.measure.samples.measures import std_dev


__all__ = [
    'register_sample_measure',
    'SampleMeasureLibrary',

    'amplitude',
    'bcea',
    'data_loss',
    'dispersion',
    'disposition',
    'location',
    'null_ratio',
    'peak_velocity',
    's2s_rms',
    'std_dev',
]
