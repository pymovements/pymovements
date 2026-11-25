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
"""Benchmark ``correct_fixations`` on a synthetic single-trial reading sequence.

Fixation drift correction runs a Python UDF for most algorithms, and the Wisdom of the Crowd
ensemble combines all of them. These benchmarks track the per-algorithm cost and the ensemble
cost as the number of text lines grows. Two line counts keep any per-line cost of the UDF
algorithms visible while the fixture stays small enough to run in CI.
"""
import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pymovements.events.correction import correct_fixations

_LINE_SEARCH_FILTER = pytest.mark.filterwarnings(
    'ignore:.*line search.*:RuntimeWarning',
)

_RNG_SEED = 42

_N_FIXATIONS = 200
_N_WORDS_PER_LINE = 12
_LINE_HEIGHT = 40.0
_LINE_SPACING = 100.0
_LINE_WIDTH = 1000.0
_DRIFT_SLOPE = 0.03
_DRIFT_OFFSET = 8.0

_ALGORITHMS = [
    'attach',
    'chain',
    'segment',
    'slice',
    'compare',
    'stretch',
    'warp',
    pytest.param('regress', marks=_LINE_SEARCH_FILTER),
    'cluster',
    'split',
    'merge',
    pytest.param('wisdom_of_the_crowd', marks=_LINE_SEARCH_FILTER),
]


def _synthetic_fixations_and_aois(n_lines: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate one trial of fixations and matching word AOIs for the given line count."""
    rng = np.random.default_rng(_RNG_SEED)
    fixations_per_line = _N_FIXATIONS // n_lines

    locations: list[list[float]] = []
    for line_index in range(n_lines):
        line_center = line_index * _LINE_SPACING + _LINE_HEIGHT / 2.0
        x_values = np.linspace(
            40.0, _LINE_WIDTH - 80.0, fixations_per_line,
        ) + rng.normal(scale=4.0, size=fixations_per_line)
        # A few regressions per line keep the greedy merge and DTW paths busy. Both line
        # counts produce the same number of progressive sequences, so the merge column is
        # not line-count sensitive.
        for index in range(3, fixations_per_line - 1, 7):
            x_values[index], x_values[index + 1] = (
                x_values[index + 1], x_values[index],
            )
        y_values = (
            line_center
            + _DRIFT_OFFSET
            + _DRIFT_SLOPE * (x_values - _LINE_WIDTH / 2.0)
            + rng.normal(scale=6.0, size=fixations_per_line)
        )
        locations.extend(
            [float(x), float(y)] for x, y in zip(x_values, y_values)
        )

    n_fixations = len(locations)
    events = pl.DataFrame(
        {
            'trial': ['TRIAL1'] * n_fixations,
            'name': 'fixation',
            'onset': np.arange(n_fixations, dtype=np.int64),
            'offset': np.arange(1, n_fixations + 1, dtype=np.int64),
            'location': locations,
        },
        schema={
            'trial': pl.Utf8,
            'name': pl.Utf8,
            'onset': pl.Int64,
            'offset': pl.Int64,
            'location': pl.List(pl.Float64),
        },
    )

    word_labels: list[str] = []
    start_x: list[float] = []
    start_y: list[float] = []
    end_x: list[float] = []
    line_indices: list[int] = []
    word_width = _LINE_WIDTH / _N_WORDS_PER_LINE
    for line_index in range(n_lines):
        for word_index in range(_N_WORDS_PER_LINE):
            word_labels.append(f'L{line_index}W{word_index}')
            word_start_x = 20.0 + word_index * word_width
            start_x.append(word_start_x)
            start_y.append(line_index * _LINE_SPACING)
            end_x.append(word_start_x + word_width - 20.0)
            line_indices.append(line_index)

    aois = pl.DataFrame(
        {
            'word': word_labels,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'height': [_LINE_HEIGHT] * len(word_labels),
            'line_idx': line_indices,
        },
        schema={
            'word': pl.Utf8,
            'start_x': pl.Float64,
            'start_y': pl.Float64,
            'end_x': pl.Float64,
            'height': pl.Float64,
            'line_idx': pl.Int64,
        },
    )
    return events, aois


@pytest.mark.parametrize('n_lines', [4, 8])
@pytest.mark.parametrize('algorithm', _ALGORITHMS)
def test_fixation_correction_benchmark(
    benchmark: BenchmarkFixture,
    algorithm: str,
    n_lines: int,
) -> None:
    """Benchmark one drift algorithm on 200 fixations over 4 or 8 text lines."""
    events, aois = _synthetic_fixations_and_aois(n_lines)

    def _run() -> None:
        correct_fixations(
            events,
            aois,
            algorithm=algorithm,
            trial_columns='trial',
        )

    benchmark.pedantic(_run, iterations=1, rounds=5, warmup_rounds=1)
