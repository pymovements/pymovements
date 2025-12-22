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
"""Contains all event detection methods.

.. rubric:: Detection Methods

.. autosummary::
   :toctree:
   :recursive:

    pymovements.events.fill
    pymovements.events.idt
    pymovements.events.ivt
    pymovements.events.microsaccades

"""
from pymovements.events.detection._fill import fill
from pymovements.events.detection._idt import idt
from pymovements.events.detection._ivt import ivt
from pymovements.events.detection._library import EventDetectionLibrary
from pymovements.events.detection._library import register_event_detection
from pymovements.events.detection._microsaccades import microsaccades

__all__ = [
    'EventDetectionLibrary',
    'register_event_detection',

    'fill',
    'idt',
    'ivt',
    'microsaccades',
]
