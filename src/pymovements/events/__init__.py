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
"""Provides event related functionality."""
from pymovements.events.detection import fill
from pymovements.events.detection import idt
from pymovements.events.detection import ivt
from pymovements.events.detection import microsaccades
from pymovements.events.detection import out_of_screen
from pymovements.events.detection._library import EventDetectionLibrary
from pymovements.events.detection._library import register_event_detection
from pymovements.events.events import Events
from pymovements.events.frame import EventDataFrame
from pymovements.events.precomputed import PrecomputedEventDataFrame
from pymovements.events.segmentation import events2segmentation
from pymovements.events.segmentation import events2timeratio
from pymovements.events.segmentation import segmentation2events

# fmt: off
__all__ = [
    'EventDetectionLibrary',
    'register_event_detection',
    'fill',
    'idt',
    'ivt',
    'microsaccades',
    'out_of_screen',
    'events2segmentation',
    'events2timeratio',
    'segmentation2events',

    'PrecomputedEventDataFrame',
    'Events',
    'EventDataFrame',
]
