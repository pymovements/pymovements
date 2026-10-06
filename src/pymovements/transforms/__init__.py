# Copyright (c) 2026 The pymovements Project Authors
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
"""Provides transformation functions."""
from pymovements.transforms import numpy
from pymovements.transforms.center_origin import center_origin
from pymovements.transforms.clip import clip
from pymovements.transforms.deg2pix import deg2pix
from pymovements.transforms.downsample import downsample
from pymovements.transforms.library import register_transform
from pymovements.transforms.library import TransformLibrary
from pymovements.transforms.norm import norm
from pymovements.transforms.pix2deg import pix2deg
from pymovements.transforms.pos2acc import pos2acc
from pymovements.transforms.pos2vel import pos2vel
from pymovements.transforms.resample import resample
from pymovements.transforms.savitzky_golay import savitzky_golay
from pymovements.transforms.segmentation import events2segmentation
from pymovements.transforms.segmentation import events2timeratio
from pymovements.transforms.segmentation import segmentation2events
from pymovements.transforms.smooth import smooth
from pymovements.transforms.value_impossible import value_impossible


__all__ = [
    'center_origin',
    'clip',
    'deg2pix',
    'downsample',
    'norm',
    'pix2deg',
    'pos2acc',
    'pos2vel',
    'resample',
    'savitzky_golay',
    'smooth',

    'events2segmentation',
    'events2timeratio',
    'segmentation2events',

    'TransformLibrary',
    'register_transform',

    'numpy',

    'value_impossible',
]
