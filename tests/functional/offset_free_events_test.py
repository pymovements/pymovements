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
"""Test that detected events without an offset column pass through every offset consumer."""
import matplotlib.pyplot as plt
import polars as pl
import pytest

import pymovements as pm
from pymovements.plotting import tsplot


@pytest.fixture(name='offset_free_gaze')
def fixture_offset_free_gaze():
    # Fixations at index 0 to 4 and 8 to 13 around a saccade at index 5 to 7, 4 ms apart.
    velocities = [[0.0, 0.0]] * 5 + [[100.0, 100.0]] * 3 + [[0.0, 0.0]] * 6
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            'time': [float(i * 4) for i in range(14)],
            'velocity': velocities,
        }),
        experiment=pm.Experiment(sampling_rate=250),
    )
    gaze.detect('ivt', velocity_threshold=20, minimum_duration=8)
    gaze.events = pm.Events(gaze.events.frame.drop('offset'), metadata={'sampling_rate': 250})
    return gaze


def test_offset_free_events_nullify_event_samples(offset_free_gaze):
    offset_free_gaze.nullify_event_samples('fixation', padding=0)

    assert [value is None for value in offset_free_gaze.samples['velocity'].to_list()] == (
        [True] * 5 + [False] * 3 + [True] * 6
    )


def test_offset_free_events_measure_events_ratio(offset_free_gaze):
    ratio = offset_free_gaze.samples.select(
        offset_free_gaze.measure_events_ratio('fixation'),
    ).item()

    # Fixations [0, 20) and [32, 56) ms over the 56 ms the samples span.
    assert ratio == pytest.approx(44 / 56)


def test_offset_free_events_fill(offset_free_gaze):
    offset_free_gaze.detect('fill', minimum_duration=0)

    unclassified = offset_free_gaze.events.frame.filter(pl.col('name') == 'unclassified')
    assert unclassified['onset'].dt.total_microseconds().to_list() == [20000]
    assert unclassified['duration'].dt.total_microseconds().to_list() == [12000]


def test_offset_free_events_merge_subsequent_close_events(offset_free_gaze):
    offset_free_gaze.events.merge_subsequent_close_events('fixation', max_gap=12)

    fixations = offset_free_gaze.events.frame.filter(pl.col('name') == 'fixation')
    assert fixations['onset'].dt.total_microseconds().to_list() == [0]
    assert fixations['duration'].dt.total_microseconds().to_list() == [56000]


def test_offset_free_events_tsplot(offset_free_gaze):
    _, ax = tsplot(gaze=offset_free_gaze, channels='velocity', show_events=True)

    shaded = sorted(
        (vertices.min(), vertices.max())
        for vertices in (
            patch.get_path().transformed(patch.get_patch_transform()).vertices[:, 0]
            for patch in ax.patches
        )
    )
    assert shaded == [(0, 20), (32, 56)]
    plt.close('all')
