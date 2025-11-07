# Copyright (c) 2025 The pymovements Project Authors
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
"""Tests for Gaze.map_to_aois preserve_structure flag and narrowed exception."""
from __future__ import annotations

import polars as pl
import pytest

import pymovements as pm
from pymovements.stimulus.text import TextStimulus

# Silence component inference warnings that are irrelevant to AOI mapping behaviour here.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)


@pytest.fixture(name='flat_pixel_samples')
def _flat_pixel_samples() -> pl.DataFrame:
    # Only flat pixel columns, no list columns -> unnest() would raise Warning
    return pl.DataFrame(
        {
            'pixel_xr': [5.0, 15.0],
            'pixel_yr': [5.0, 5.0],
        },
    )


@pytest.fixture(name='list_position_samples')
def _list_position_samples() -> pl.DataFrame:
    # Position as a list column [xl, yl, xr, yr]
    return pl.DataFrame({'position': [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 15.0, 15.0]]})


def test_gaze_map_to_aois_preserve_structure_true_flat_columns(
    simple_stimulus: TextStimulus, flat_pixel_samples: pl.DataFrame,
) -> None:
    # With only flat columns present, unnest() raises Warning internally - we
    # tolerate it and proceed.
    gaze = pm.Gaze(samples=flat_pixel_samples)
    gaze.map_to_aois(simple_stimulus, eye='right', gaze_type='pixel', preserve_structure=True)

    # AOI labels: [5,5] -> inside 'A' - [15,5] -> outside
    labels = gaze.samples.get_column('label').to_list()
    assert labels == ['A', None]


def test_gaze_map_to_aois_preserve_structure_false_list_column(
    simple_stimulus: TextStimulus, list_position_samples: pl.DataFrame,
) -> None:
    # No schema change expected - 'position' list column remains.
    gaze = pm.Gaze(samples=list_position_samples)
    gaze.map_to_aois(simple_stimulus, eye='right', gaze_type='position', preserve_structure=False)

    cols = set(gaze.samples.columns)
    assert 'position' in cols  # schema preserved
    labels = gaze.samples.get_column('label').to_list()
    assert labels == ['A', None]
