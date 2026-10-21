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

from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus

_N_EVENTS = 20000
_N_AOIS = 500


def _build_stimulus() -> TextStimulus:
    return TextStimulus(
        pl.DataFrame(
            {
                'label': [f'L{index}' for index in range(_N_AOIS)],
                'sx': [float((index % 25) * 40) for index in range(_N_AOIS)],
                'sy': [float((index // 25) * 40) for index in range(_N_AOIS)],
                'ex': [float((index % 25) * 40 + 38) for index in range(_N_AOIS)],
                'ey': [float((index // 25) * 40 + 38) for index in range(_N_AOIS)],
            },
        ),
        aoi_column='label',
        start_x_column='sx',
        start_y_column='sy',
        end_x_column='ex',
        end_y_column='ey',
    )


def _build_events() -> Events:
    locations = [
        [(index % 25) * 40 + 5.0, (index // 25) * 40 + 5.0] for index in range(_N_EVENTS)
    ]
    return Events(
        pl.DataFrame(
            {
                'name': ['fixation'] * _N_EVENTS,
                'onset': list(range(_N_EVENTS)),
                'offset': [index + 1 for index in range(_N_EVENTS)],
                'location': locations,
            },
        ),
    )


def test_events_map_to_aois_benchmark(benchmark):
    """Benchmark mapping 20k fixations against 500 AOIs."""
    stimulus = _build_stimulus()

    def _run() -> None:
        _build_events().map_to_aois(stimulus, verbose=False)

    benchmark.pedantic(_run, iterations=1, rounds=5)
