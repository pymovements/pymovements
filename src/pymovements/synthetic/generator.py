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
"""Provides the abstract base class for synthetic gaze generators."""
from __future__ import annotations

import abc
import dataclasses
import secrets
from typing import Any
from typing import ClassVar

import numpy as np

from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.config import GeneratorConfig
from pymovements.synthetic.provenance import generation_provenance


class GazeGenerator(abc.ABC):
    """Base class for all synthetic gaze generators.

    Every generation method is reached through this class, whether it is statistical, neural or
    something else. Subclasses put their synthesis logic in ``_generate`` and set ``name``, under
    which they are registered in
    :py:class:`~pymovements.synthetic.GeneratorLibrary`.

    Attributes
    ----------
    name: ClassVar[str]
        The registered name. Must be set by each subclass.
    capabilities: ClassVar[GeneratorCapabilities | None]
        What the generator supports, or ``None`` if it declares nothing. (default: None)
    """

    name: ClassVar[str]

    capabilities: ClassVar[GeneratorCapabilities | None] = None

    def generate(self, config: GeneratorConfig | None = None, **kwargs: Any) -> Gaze:
        """Generate synthetic gaze data.

        The entry point for all generators. Resolves the seed, creates the run's random
        generator and records provenance, and delegates the synthesis itself to ``_generate``.
        A configuration without a seed gets one drawn for it, so a run is reproducible either
        way.

        Parameters
        ----------
        config: GeneratorConfig | None
            The parameters of the run. (default: None)
        **kwargs: Any
            Conditioning inputs, forwarded to ``_generate``.

        Returns
        -------
        Gaze
            The generated synthetic gaze data, with the run recorded under the ``'synthetic'``
            key of its :py:attr:`~pymovements.Gaze.metadata`.
        """
        seed = _draw_seed() if config is None or config.seed is None else config.seed

        if config is not None:
            config = dataclasses.replace(config, seed=seed)

        gaze = self._generate(config=config, rng=np.random.default_rng(seed), **kwargs)

        gaze.metadata = {
            **(gaze.metadata or {}),
            **generation_provenance(
                generator=self.name,
                config=config,
                seed=seed,
                capabilities=self.capabilities,
            ),
        }

        return gaze

    @abc.abstractmethod
    def _generate(
            self,
            config: GeneratorConfig | None,
            rng: np.random.Generator,
            **kwargs: Any,
    ) -> Gaze:
        """Synthesize gaze data. Implemented by each generator; callers use ``generate``.

        All randomness must be drawn from ``rng``, otherwise the run is not reproducible from
        its recorded seed.

        Parameters
        ----------
        config: GeneratorConfig | None
            The parameters of the run, with a resolved seed.
        rng: np.random.Generator
            The run's random generator, seeded by ``generate``.
        **kwargs: Any
            Conditioning inputs.

        Returns
        -------
        Gaze
            The generated synthetic gaze data.
        """


def _draw_seed() -> int:
    """Draw a seed from the operating system's entropy, so it can be recorded."""
    return secrets.randbits(128)
