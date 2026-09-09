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
"""Fixation drift correction for gaze data recorded during reading tasks.

Each drift correction algorithm returns a :py:class:`polars.Expr` that computes the
corrected y-coordinates from a column of ``[x, y]`` fixation locations. All coordinates
and distance thresholds are expressed in pixels in the coordinate system of the fixation
locations, with the y-axis pointing downward: lines of text are ordered top to bottom by
increasing y-coordinate.

The implementations follow the reference implementation of Carr et al. :cite:p:`Carr2022`.
Algorithms that build on k-means clustering, numerical optimization or line fitting
('cluster', 'compare', 'merge', 'regress', 'slice', 'split', 'stretch', 'warp') materialize
the fixation sequence inside the expression via ``map_batches``. Their numeric cores use
scikit-learn, scipy and numpy's polyfit.

References & Citations:
- :cite:p:`Abdulin2015`
- :cite:p:`Carr2022`
- :cite:p:`Cohen2013`
- :cite:p:`Glandorf2021`
- :cite:p:`LimaSanches2015`
- :cite:p:`Lohmeier2015`
- :cite:p:`Mercier2024b`
- :cite:p:`Spakov2019`
"""
from pymovements.events.correction.attach import attach
from pymovements.events.correction.chain import chain
from pymovements.events.correction.cluster import cluster
from pymovements.events.correction.compare import compare
from pymovements.events.correction.fixation_correction import correct_fixation_locations
from pymovements.events.correction.fixation_correction import correct_fixations
from pymovements.events.correction.merge import merge
from pymovements.events.correction.regress import regress
from pymovements.events.correction.segment import segment
from pymovements.events.correction.slice import slice  # pylint: disable=redefined-builtin
from pymovements.events.correction.split import split
from pymovements.events.correction.stretch import stretch
from pymovements.events.correction.warp import warp
from pymovements.events.correction.wisdom_of_the_crowd import wisdom_of_the_crowd

__all__ = [
    'correct_fixations',
    'correct_fixation_locations',

    'attach',
    'chain',
    'cluster',
    'compare',
    'merge',
    'regress',
    'segment',
    'slice',
    'split',
    'stretch',
    'warp',
    'wisdom_of_the_crowd',
]
