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
"""Benchmark ``Gaze.detect`` for trial-grouped samples.

Per-trial event detection used to rebuild and filter the events frame once per trial, even for
detectors that never look at previous events. These benchmarks track the per-trial overhead on
data with many trials.
"""
import numpy as np
import polars as pl

from pymovements import Gaze

_RNG = np.random.default_rng(42)
_N_TRIALS = 5000
_SAMPLES_PER_TRIAL = 200


def _build_gaze() -> Gaze:
    n_samples = _N_TRIALS * _SAMPLES_PER_TRIAL
    samples = pl.DataFrame(
        {
            'trial': np.repeat(np.arange(_N_TRIALS), _SAMPLES_PER_TRIAL),
            'time': list(range(n_samples)),
            'velocity': _RNG.normal(scale=5.0, size=(n_samples, 2)).tolist(),
            'position': _RNG.normal(size=(n_samples, 2)).tolist(),
        },
    ).with_columns(
        pl.col('trial').cast(pl.Int64),
        pl.col('velocity').cast(pl.List(pl.Float64)),
        pl.col('position').cast(pl.List(pl.Float64)),
    )
    return Gaze(samples, trial_columns=['trial'])


def test_gaze_detect_benchmark(benchmark):
    """Benchmark velocity-threshold detection over 5k trials with 200 samples each."""
    gaze = _build_gaze()

    def _setup() -> None:
        # Detection appends to the events frame, so start each round from no events.
        gaze.clear_events()

    def _run() -> None:
        gaze.detect('ivt', velocity_threshold=20.0, minimum_duration=1)

    benchmark.pedantic(_run, setup=_setup, iterations=1, rounds=5)
