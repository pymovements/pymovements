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
"""Benchmark the I-HMM event detection algorithm.

Baum-Welch reestimation (the default reestimation_max_iters=1000) is the
costliest path, since it repeats the forward/backward passes every iteration.
These benchmarks track that cost to guard against future regressions.
"""
import numpy as np

from pymovements.events.detection.ihmm import ihmm

_RNG = np.random.default_rng(42)


def _synthetic_velocities(n_samples: int) -> np.ndarray:
    """Generate a synthetic (n_samples, 2) velocity sequence with no NaNs."""
    return _RNG.normal(loc=0.0, scale=2.0, size=(n_samples, 2))


def test_ihmm_benchmark_viterbi_only(benchmark):
    """Benchmark ihmm() decoding without Baum-Welch reestimation."""
    velocities = _synthetic_velocities(5000)

    benchmark.pedantic(
        ihmm,
        kwargs={'velocities': velocities, 'minimum_duration': 1},
        iterations=1,
        rounds=50,
    )


def test_ihmm_benchmark_with_reestimation(benchmark):
    """Benchmark ihmm() with Baum-Welch reestimation enabled (the slow path)."""
    velocities = _synthetic_velocities(2000)

    benchmark.pedantic(
        ihmm,
        kwargs={
            'velocities': velocities,
            'minimum_duration': 1,
            'reestimation': True,
            'reestimation_max_iters': 20,
        },
        iterations=1,
        rounds=20,
    )
