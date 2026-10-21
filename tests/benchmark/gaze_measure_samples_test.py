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
"""Benchmark ``Gaze.measure_samples`` grouped by trial.

Each measure runs over one million synthetic samples split into many short trials (20000 x 50)
or few long ones (2000 x 500).
"""
import numpy as np
import polars as pl
import pytest

from pymovements import Gaze

_RNG = np.random.default_rng(42)


def _build_gaze(n_trials: int, samples_per_trial: int) -> Gaze:
    n_samples = n_trials * samples_per_trial
    samples = pl.DataFrame(
        {
            'trial': np.repeat(np.arange(n_trials), samples_per_trial),
            'time': list(range(n_samples)),
            'position': _RNG.normal(size=(n_samples, 2)).tolist(),
        },
    ).with_columns(
        pl.col('trial').cast(pl.Int64),
        pl.col('position').cast(pl.List(pl.Float64)),
    )
    return Gaze(samples, trial_columns=['trial'])


@pytest.mark.parametrize(
    ('n_trials', 'samples_per_trial'),
    [
        pytest.param(20000, 50, id='many_short_trials'),
        pytest.param(2000, 500, id='few_long_trials'),
    ],
)
def test_gaze_measure_samples_benchmark(benchmark, n_trials, samples_per_trial):
    """Benchmark location over one million samples split into trials."""
    gaze = _build_gaze(n_trials, samples_per_trial)

    benchmark.pedantic(
        lambda: gaze.measure_samples('location'),
        iterations=1,
        rounds=5,
    )
