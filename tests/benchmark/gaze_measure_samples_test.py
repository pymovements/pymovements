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
"""Benchmark ``Gaze.measure_samples`` for trial-grouped samples.

Computing a measure over many trials used to build a separate one-row frame per trial and
concatenate them. It is now a single grouped aggregation. These benchmarks track that cost on
synthetic data with many trials.
"""
import numpy as np
import polars as pl

from pymovements import Gaze

_RNG = np.random.default_rng(42)
_N_TRIALS = 20000
_SAMPLES_PER_TRIAL = 50


def _build_gaze() -> Gaze:
    n_samples = _N_TRIALS * _SAMPLES_PER_TRIAL
    samples = pl.DataFrame(
        {
            'trial': np.repeat(np.arange(_N_TRIALS), _SAMPLES_PER_TRIAL),
            'time': list(range(n_samples)),
            'position': _RNG.normal(size=(n_samples, 2)).tolist(),
        },
    ).with_columns(
        pl.col('trial').cast(pl.Int64),
        pl.col('position').cast(pl.List(pl.Float64)),
    )
    return Gaze(samples, trial_columns=['trial'])


def test_gaze_measure_samples_benchmark(benchmark):
    """Benchmark location over 20k trials with 50 samples each."""
    gaze = _build_gaze()

    benchmark.pedantic(
        lambda: gaze.measure_samples('location'),
        iterations=1,
        rounds=5,
    )
