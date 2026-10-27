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
"""Benchmark parsing of EyeLink ASC files.

``parse_eyelink`` used to evaluate a large number of non-sample regexes for every sample line.
These benchmarks track the parsing cost on a sample-heavy ASC file.
"""
import io

import pytest

from pymovements.gaze.io import from_asc

_N_SAMPLES = 500_000


def _build_asc_text(binocular: bool) -> str:
    if binocular:
        tracked_eye, start_eyes, samples_eyes = 'LR', 'LEFT\tRIGHT', 'LEFT\tRIGHT'
        sample_values = '964.3\t541.5\t288.0\t960.5\t538.8\t305.0'
    else:
        tracked_eye, start_eyes, samples_eyes = 'L', 'LEFT', 'LEFT'
        sample_values = '964.3\t541.5\t288.0'

    lines = [
        '** DATE: Wed Mar  8 09:25:20 2023',
        '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED',
        '** VERSION: EYELINK II 1',
        '** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)',
        '** RECORDED BY pymovements',
        '**',
        f'MSG\t1408659 RECCFG CR 1000 2 1 {tracked_eye}',
        'MSG\t1408659 ELCLCFG BTABLER',
        'MSG\t1408659 GAZE_COORDS 0.00 0.00 1920.00 1080.00',
        f'START\t1408660 \t{start_eyes}\tSAMPLES\tEVENTS',
        f'SAMPLES\tGAZE\t{samples_eyes}\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2',
    ]
    first_timestamp = 1408660
    for index in range(_N_SAMPLES):
        timestamp = first_timestamp + index
        lines.append(f'{timestamp}\t{sample_values}\t.....')
        if index % 100_000 == 0:
            lines.append(f'SFIX L   {timestamp}')
            lines.append(f'EFIX L   {timestamp}\t{timestamp + 100}\t100\t960.0\t540.0\t280')
    lines.append(f'END\t{first_timestamp + _N_SAMPLES} \tSAMPLES\tEVENTS\tRES\t47.75\t45.92')
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize('binocular', [False, True], ids=['monocular', 'binocular'])
def test_from_asc_benchmark(benchmark, binocular):
    """Benchmark parsing a 500k-sample ASC file."""
    content = _build_asc_text(binocular)

    benchmark.pedantic(
        lambda: from_asc(io.StringIO(content)),
        iterations=1,
        rounds=5,
    )
