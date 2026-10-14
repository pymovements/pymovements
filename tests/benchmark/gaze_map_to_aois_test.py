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
"""Benchmark ``Gaze.map_to_aois``.

Mapping gaze samples to AOIs used to run a Python loop over all samples, filtering the AOI
frame and checking the spatial bounds once per sample. It is now a single vectorized spatial
join. These benchmarks track that cost on synthetic samples and AOIs.
"""
import numpy as np
import polars as pl

from pymovements import Gaze
from pymovements.stimulus.text import TextStimulus

_RNG = np.random.default_rng(42)
_N_SAMPLES = 1_000_000
_N_AOIS = 200


def _build_stimulus() -> TextStimulus:
    return TextStimulus(
        pl.DataFrame(
            {
                'label': [f'L{index}' for index in range(_N_AOIS)],
                'sx': [float((index % 20) * 100) for index in range(_N_AOIS)],
                'sy': [float((index // 20) * 100) for index in range(_N_AOIS)],
                'ex': [float((index % 20) * 100 + 90) for index in range(_N_AOIS)],
                'ey': [float((index // 20) * 100 + 90) for index in range(_N_AOIS)],
            },
        ),
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
    )


def _build_gaze() -> Gaze:
    xy = _RNG.uniform(0, 2000, size=(_N_SAMPLES, 2))
    positions = [[0.0, 0.0, float(x), float(y)] for x, y in xy]
    samples = pl.DataFrame({'position': positions}).with_columns(
        pl.col('position').cast(pl.List(pl.Float64)),
    )
    return Gaze(samples=samples)


def test_gaze_map_to_aois_benchmark(benchmark):
    """Benchmark mapping 1M samples against 200 AOIs via the list path."""
    stimulus = _build_stimulus()

    def _run() -> None:
        _build_gaze().map_to_aois(
            stimulus, eye='right', gaze_type='position', preserve_structure=False, verbose=False,
        )

    benchmark.pedantic(_run, iterations=1, rounds=5)
