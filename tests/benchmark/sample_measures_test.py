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
"""Benchmark the invalid-value check used by ``null_ratio`` and ``data_loss``.

Both measures count invalid samples (null/NaN/inf elements). The check used to run per row
through ``map_elements``; it is now a vectorized ``list.eval`` expression whenever the column
dtype is known. These benchmarks track that cost on a synthetic list column.
"""
import numpy as np
import polars as pl

from pymovements.measure.samples import data_loss
from pymovements.measure.samples import null_ratio

_RNG = np.random.default_rng(42)
_N_SAMPLES = 500_000


def _synthetic_samples() -> pl.DataFrame:
    """Generate a 1 kHz position column with periodically missing samples."""
    values = [[float(_RNG.normal()), float(_RNG.normal())] for _ in range(_N_SAMPLES)]
    for index in range(0, _N_SAMPLES, 1000):
        values[index] = None
    return pl.DataFrame(
        {'time': list(range(_N_SAMPLES)), 'position': values},
        schema={'time': pl.Int64, 'position': pl.List(pl.Float64)},
    )


def test_null_ratio_benchmark(benchmark):
    """Benchmark null_ratio over 500k list samples."""
    samples = _synthetic_samples()

    benchmark.pedantic(
        lambda: samples.select(null_ratio('position', column_dtype=pl.List(pl.Float64))),
        iterations=1,
        rounds=10,
    )


def test_data_loss_benchmark(benchmark):
    """Benchmark data_loss over 500k list samples."""
    samples = _synthetic_samples()

    benchmark.pedantic(
        lambda: samples.select(
            data_loss('position', sampling_rate=1000.0, column_dtype=pl.List(pl.Float64)),
        ),
        iterations=1,
        rounds=10,
    )
