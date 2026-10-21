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
is pure overhead that grows with the number of trials. These benchmarks track that cost at a
fixed number of samples, so the runtime should stay flat across trial counts.
"""
import numpy as np
import polars as pl
import pytest

from pymovements import Experiment
from pymovements import Gaze

_RNG = np.random.default_rng(42)
_N_SAMPLES = 2_660_000


def _build_gaze(n_trials: int, columns_kwarg: str) -> Gaze:
    samples_per_trial = _N_SAMPLES // n_trials
    n_samples = n_trials * samples_per_trial
    samples = pl.DataFrame(
        {
            'trial': np.repeat(np.arange(n_trials), samples_per_trial),
            'x': _RNG.uniform(0, 1024, size=n_samples),
            'y': _RNG.uniform(0, 768, size=n_samples),
        },
    ).with_columns(pl.col('trial').cast(pl.Int64))
    return Gaze(
        samples=samples,
        experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        trial_columns='trial',
        **{columns_kwarg: ['x', 'y']},
    )


@pytest.mark.parametrize(
    ('method', 'columns_kwarg'),
    [
        pytest.param('pix2deg', 'pixel_columns', id='pix2deg'),
        pytest.param('deg2pix', 'position_columns', id='deg2pix'),
    ],
)
@pytest.mark.parametrize('n_trials', [1, 200, 2660])
def test_gaze_element_wise_transform_benchmark(benchmark, method, columns_kwarg, n_trials):
    """Benchmark an element-wise transform over 2.66M samples split into ``n_trials`` trials."""
    gaze = _build_gaze(n_trials, columns_kwarg)
    pristine = gaze.samples.clone()

    def _setup() -> None:
        # The transform adds or replaces columns, so restore the pristine samples each round.
        gaze.samples = pristine.clone()

    benchmark.pedantic(getattr(gaze, method), setup=_setup, iterations=1, rounds=5)
