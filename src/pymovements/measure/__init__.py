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
"""Provides eye movement measures."""
from pymovements.measure.event_processing import EventProcessor
from pymovements.measure.event_processing import EventSamplesProcessor
from pymovements.measure.sample_measures import amplitude
from pymovements.measure.sample_measures import dispersion
from pymovements.measure.sample_measures import disposition
from pymovements.measure.event_properties import duration
from pymovements.measure.event_properties import EVENT_PROPERTIES
from pymovements.measure.sample_measures import location
from pymovements.measure.sample_measures import peak_velocity
from pymovements.measure.event_properties import register_event_property
from pymovements.measure.library import register_sample_measure
from pymovements.measure.library import SampleMeasureLibrary
from pymovements.measure.sample_measures import null_ratio


__all__ = [
    'SampleMeasureLibrary',
    'register_sample_measure',

    'null_ratio',

    'EventSamplesProcessor',
    'EventProcessor',

    'EVENT_PROPERTIES',
    'register_event_property',
    'amplitude',
    'dispersion',
    'disposition',
    'duration',
    'location',
    'peak_velocity',
]
