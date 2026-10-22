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
"""Provides the provenance record written into generated data."""
from __future__ import annotations

from typing import Any

from pymovements._version import __version__

# The key the provenance record is stored under in the metadata of the generated data.
PROVENANCE_KEY = 'synthetic'


def generation_provenance(
        *,
        generator: str,
        level: str,
        hyperparameters: dict[str, Any],
        run: dict[str, Any],
        seed: int,
) -> dict[str, Any]:
    """Build the provenance record of a generation run.

    Records how the data was produced, so it is not indistinguishable from a recording.

    Parameters
    ----------
    generator: str
        The registered name of the generator.
    level: str
        Whether the generator produces ``'events'`` or ``'samples'``.
    hyperparameters: dict[str, Any]
        The parameters the generator was constructed with.
    run: dict[str, Any]
        The parameters of this run, apart from the seed.
    seed: int
        The seed the run's random generator was created from.

    Returns
    -------
    dict[str, Any]
        The record under a single key, ready to be merged into the metadata.

    Examples
    --------
    >>> provenance = generation_provenance(
    ...     generator='example',
    ...     level='samples',
    ...     hyperparameters={'noise': 0.5},
    ...     run={'sampling_rate': 1000.0},
    ...     seed=42,
    ... )
    >>> provenance['synthetic']['hyperparameters']
    {'noise': 0.5}
    """
    return {
        PROVENANCE_KEY: {
            'generator': generator,
            'level': level,
            'hyperparameters': dict(hyperparameters),
            'run': dict(run),
            'seed': seed,
            'pymovements_version': __version__,
        },
    }
