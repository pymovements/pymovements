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
"""Benchmark event property computation over events and samples.

``Gaze.compute_event_properties`` delegates to ``EventSamplesProcessor.process``, which assigns
each sample to the event it belongs to and aggregates all requested sample measures. An earlier
implementation scanned the whole samples frame once per event, so the cost grew with
``numEvents * numSamples`` and dominated event detection on long recordings. These benchmarks
track the cost of the vectorized implementation on synthetic 1 kHz recordings.
"""
import numpy as np
import polars as pl

from pymovements.measure.events import EventSamplesProcessor

_RNG = np.random.default_rng(42)

_EVENT_SAMPLES = 240
_GAP_SAMPLES = 10


def _synthetic_events_and_samples(
        n_trials: int,
        events_per_trial: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate non-overlapping fixation events with matching 1 kHz samples."""
    samples = []
    events = []
    for trial in range(n_trials):
        time = 0
        for _ in range(events_per_trial):
            onset = time
            for _ in range(_EVENT_SAMPLES):
                samples.append(
                    (trial, time, [float(_RNG.normal()), float(_RNG.normal())]),
                )
                time += 1
            events.append((trial, 'fixation', onset, time - 1))
            time += _GAP_SAMPLES

    events_frame = pl.DataFrame(
        events, schema=['trial', 'name', 'onset', 'offset'], orient='row',
    ).with_columns(pl.col('trial', 'onset', 'offset').cast(pl.Int64))
    samples_frame = pl.DataFrame(
        samples, schema=['trial', 'time', 'position'], orient='row',
    ).with_columns(
        pl.col('trial', 'time').cast(pl.Int64),
        pl.col('position').cast(pl.List(pl.Float64)),
    )
    return events_frame, samples_frame


def test_event_properties_benchmark(benchmark):
    """Benchmark location and dispersion over 600k samples and 2.4k events."""
    events, samples = _synthetic_events_and_samples(n_trials=12, events_per_trial=200)
    processor = EventSamplesProcessor(['location', 'dispersion'])

    benchmark.pedantic(
        processor.process,
        kwargs={'events': events, 'samples': samples, 'identifiers': 'trial'},
        iterations=1,
        rounds=10,
    )
