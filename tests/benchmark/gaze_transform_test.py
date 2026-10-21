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
"""Benchmark element-wise ``Gaze`` transforms with trial columns.

When ``trial_columns`` is set, ``Gaze.transform`` used to apply *every* transform once per
trial group and concatenate the results. For element-wise transforms such as ``pix2deg`` this
is pure overhead that grows with the number of trials. These benchmarks track that cost.
"""
import numpy as np
import polars as pl

from pymovements import Experiment
from pymovements import Gaze

_RNG = np.random.default_rng(42)
_N_TRIALS = 2660
_SAMPLES_PER_TRIAL = 1000


def _build_gaze() -> Gaze:
    n_samples = _N_TRIALS * _SAMPLES_PER_TRIAL
    samples = pl.DataFrame(
        {
            'trial': np.repeat(np.arange(_N_TRIALS), _SAMPLES_PER_TRIAL),
            'x': _RNG.uniform(0, 1024, size=n_samples),
            'y': _RNG.uniform(0, 768, size=n_samples),
        },
    ).with_columns(pl.col('trial').cast(pl.Int64))
    return Gaze(
        samples=samples,
        experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        pixel_columns=['x', 'y'],
        trial_columns='trial',
    )


def test_gaze_pix2deg_benchmark(benchmark):
    """Benchmark pix2deg over 2.66M samples in 2660 trials."""
    gaze = _build_gaze()
    pristine = gaze.samples.clone()

    def _setup() -> None:
        # pix2deg replaces the pixel column, so restore the pristine samples before each round.
        gaze.samples = pristine.clone()

    def _run() -> None:
        gaze.pix2deg()

    benchmark.pedantic(_run, setup=_setup, iterations=1, rounds=5)
