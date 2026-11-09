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
"""Benchmark ``Events.map_to_aois``.

Mapping fixations to AOIs used to run a Python loop over all events, filtering the AOI frame and
checking the spatial bounds once per event. It is now a single vectorized join plus spatial
filter. These benchmarks track that cost on synthetic events and AOIs.
"""
import polars as pl
from pytest_benchmark.fixture import BenchmarkFixture

from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus

_N_EVENTS = 20000
_N_AOIS = 500
_N_TRIALS = 20


def _build_stimulus(*, keyed: bool = False) -> TextStimulus:
    n_groups = _N_TRIALS if keyed else 1
    labels: list[str] = []
    trials: list[int] = []
    start_x: list[float] = []
    start_y: list[float] = []
    end_x: list[float] = []
    end_y: list[float] = []
    for group in range(n_groups):
        for index in range(_N_AOIS):
            labels.append(f'T{group}L{index}' if keyed else f'L{index}')
            trials.append(group)
            start_x.append(float((index % 25) * 40))
            start_y.append(float((index // 25) * 40))
            end_x.append(float((index % 25) * 40 + 38))
            end_y.append(float((index // 25) * 40 + 38))

    data: dict[str, object] = {
        'label': labels,
        'sx': start_x,
        'sy': start_y,
        'ex': end_x,
        'ey': end_y,
    }
    if keyed:
        data['trial'] = trials
    return TextStimulus(
        pl.DataFrame(data),
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
        trial_column='trial' if keyed else None,
    )


def _build_events(*, keyed: bool = False) -> Events:
    locations = [
        [(index % 25) * 40 + 5.0, (index // 25) * 40 + 5.0] for index in range(_N_EVENTS)
    ]
    data: dict[str, object] = {
        'name': ['fixation'] * _N_EVENTS,
        'onset': list(range(_N_EVENTS)),
        'offset': [index + 1 for index in range(_N_EVENTS)],
        'location': locations,
    }
    if keyed:
        data['trial'] = [index % _N_TRIALS for index in range(_N_EVENTS)]
    return Events(pl.DataFrame(data), trial_columns=['trial'] if keyed else None)


def test_events_map_to_aois_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark mapping 20k fixations against 500 AOIs without key columns."""
    stimulus = _build_stimulus()

    def _run(events: Events) -> None:
        events.map_to_aois(stimulus, verbose=False)

    def _setup() -> tuple[tuple[Events, ...], dict[str, object]]:
        return (_build_events(),), {}

    benchmark.pedantic(_run, setup=_setup, rounds=5)


def test_events_map_to_aois_keyed_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark mapping 20k fixations against 20 trials x 500 AOIs."""
    stimulus = _build_stimulus(keyed=True)

    def _run(events: Events) -> None:
        events.map_to_aois(stimulus, verbose=False)

    def _setup() -> tuple[tuple[Events, ...], dict[str, object]]:
        return (_build_events(keyed=True),), {}

    benchmark.pedantic(_run, setup=_setup, rounds=5)
