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

Each benchmark maps 1M gaze samples to 200 AOIs on a 20 x 10 grid. It covers the list path
(``preserve_structure=False``), the default flat path with one eye selected directly, and the
flat path that averages the left and right eye.
"""
import numpy as np
import polars as pl
import pytest

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


def _build_samples() -> pl.DataFrame:
    """Build samples with both eyes filled, the left eye a few pixels off the right eye."""
    right = _RNG.uniform(0, 2000, size=(_N_SAMPLES, 2))
    left = right + _RNG.uniform(-5, 5, size=(_N_SAMPLES, 2))
    positions = np.column_stack([left, right]).tolist()
    return pl.DataFrame({'position': positions}).with_columns(
        pl.col('position').cast(pl.List(pl.Float64)),
    )


@pytest.mark.parametrize(
    ('preserve_structure', 'eye'),
    [
        pytest.param(False, 'right', id='list'),
        pytest.param(True, 'right', id='flat_direct'),
        pytest.param(
            True, 'cyclops', id='flat_average_lr',
            marks=pytest.mark.filterwarnings(
                'ignore:Cyclops requested but cyclops components missing.*:UserWarning',
            ),
        ),
    ],
)
def test_gaze_map_to_aois_benchmark(benchmark, preserve_structure, eye):
    """Time one ``map_to_aois`` call per round, without building or resetting the samples."""
    stimulus = _build_stimulus()
    samples = _build_samples()
    gaze = Gaze(samples=samples.clone())

    def _setup() -> None:
        # map_to_aois appends the AOI columns, so reset the pristine samples before each round.
        gaze.samples = samples.clone()

    def _run() -> None:
        gaze.map_to_aois(
            stimulus, eye=eye, gaze_type='position', preserve_structure=preserve_structure,
            verbose=False,
        )

    benchmark.pedantic(_run, setup=_setup, iterations=1, rounds=5)
