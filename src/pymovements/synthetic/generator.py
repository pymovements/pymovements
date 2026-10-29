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
from warnings import warn

import numpy as np

from pymovements._version import __version__
from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.capabilities import GeneratorCapabilities
from pymovements.synthetic.config import GeneratorConfig
from pymovements.synthetic.provenance import generation_provenance
from pymovements.synthetic.validation import validate_input
from pymovements.synthetic.validation import validate_output
from pymovements.warnings import ExperimentalWarning


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

    def generate(
            self,
            config: GeneratorConfig | None = None,
            *,
            events: Events | None = None,
            gaze: Gaze | None = None,
            artifacts: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> Gaze:
        """Generate synthetic gaze data.

        The entry point for all generators. Validates the input against what the generator
        declares, resolves the seed, creates the run's random generator, checks the output and
        records provenance, and delegates the synthesis itself to ``_generate``. A configuration
        without a seed gets one drawn for it, so a run is reproducible either way.

        Which conditioning mode a call stands for follows from what is supplied: ``gaze`` means
        gaze-conditioned, ``events`` means event-conditioned, neither means unconditional. Only
        arguments that were supplied are forwarded.

        Parameters
        ----------
        config: GeneratorConfig | None
            The parameters of the run. (default: None)
        events: Events | None
            An event sequence to condition on. (default: None)
        gaze: Gaze | None
            A gaze sequence to condition on. (default: None)
        artifacts: dict[str, Any] | None
            External artifacts, such as trained weights. (default: None)
        **kwargs: Any
            Further generator-specific arguments, forwarded to ``_generate``.

        Returns
        -------
        Gaze
            The generated synthetic gaze data, with the run recorded under the ``'synthetic'``
            key of its :py:attr:`~pymovements.Gaze.metadata`.

        Raises
        ------
        InputContractError
            If the supplied input does not match what the generator declares.
        OutputContractError
            If the generated output does not match what the generator declares.

        Warns
        -----
        ExperimentalWarning
            Always. Synthetic gaze generation is experimental.
        """
        warn(
            'Synthetic gaze generation is experimental. '
            'Names and behavior may change without being considered a breaking change. '
            'Please set the used pymovements version explicitly to prevent unexpected changes. '
            f'The used pymovements version is v{__version__}.',
            ExperimentalWarning,
        )

        validate_input(
            capabilities=self.capabilities,
            config=config,
            events=events,
            gaze=gaze,
            artifacts=artifacts,
            generator=self.name,
        )

        seed = _draw_seed() if config is None or config.seed is None else config.seed

        if config is not None:
            config = dataclasses.replace(config, seed=seed)

        supplied = {
            name: value
            for name, value in (('events', events), ('gaze', gaze), ('artifacts', artifacts))
            if value is not None
        }

        generated = self._generate(
            config=config,
            rng=np.random.default_rng(seed),
            **supplied,
            **kwargs,
        )

        validate_output(
            gaze=generated,
            capabilities=self.capabilities,
            config=config,
            generator=self.name,
        )

        generated.metadata = {
            **(generated.metadata or {}),
            **generation_provenance(
                generator=self.name,
                config=config,
                seed=seed,
                capabilities=self.capabilities,
                artifacts=artifacts,
            ),
        }

        return generated

    @abc.abstractmethod
    def _generate(
            self,
            config: GeneratorConfig | None,
            rng: np.random.Generator,
            **kwargs: Any,
    ) -> Gaze:
        """Synthesize gaze data. Implemented by each generator; callers use ``generate``.

        All randomness must be drawn from ``rng``, otherwise the run is not reproducible from
        its recorded seed. Conditioning inputs arrive as keyword arguments, so a generator names
        the ones it consumes with a default of ``None``: validation has already established that
        the ones its capabilities require are present.

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
