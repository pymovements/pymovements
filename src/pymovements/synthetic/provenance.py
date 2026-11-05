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
"""Provides the provenance record written into generated gaze data."""
from __future__ import annotations

from typing import Any

from pymovements._version import __version__
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.config import GeneratorConfig

# The key the provenance record is stored under in Gaze.metadata.
PROVENANCE_KEY = 'synthetic'


def generation_provenance(
        *,
        generator: str,
        config: GeneratorConfig | None,
        seed: int,
        capabilities: GeneratorCapabilities | None = None,
        artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance record of a generation run.

    Records how a sequence was produced, so it is not indistinguishable from a recording. The
    recorded configuration carries the seed that was used, not the one that was passed in.

    Parameters
    ----------
    generator: str
        The registered name of the generator.
    config: GeneratorConfig | None
        The configuration of the run, or ``None`` if the generator was called without one.
    seed: int
        The seed the run's random generator was created from.
    capabilities: GeneratorCapabilities | None
        The generator's capability declaration, if it declares one. (default: None)
    artifacts: dict[str, Any] | None
        Identities of external artifacts the run used, such as trained weights. (default: None)

    Returns
    -------
    dict[str, Any]
        The record, under a single key, ready to be merged into
        :py:attr:`~pymovements.Gaze.metadata`.

    Examples
    --------
    >>> provenance = generation_provenance(
    ...     generator='example',
    ...     config=GeneratorConfig(sampling_rate=1000.0, seed=42),
    ...     seed=42,
    ... )
    >>> provenance['synthetic']['config']
    {'sampling_rate': 1000.0, 'duration_s': None, 'seed': 42}
    """
    return {
        PROVENANCE_KEY: {
            'generator': generator,
            'config': None if config is None else config.to_dict(),
            'seed': seed,
            'signal_representations': (
                None if capabilities is None
                else sorted(
                    representation.value
                    for representation in capabilities.signal_representations
                )
            ),
            'artifacts': {} if artifacts is None else dict(artifacts),
            'pymovements_version': __version__,
        },
    }
