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
"""Provides the public entry point for synthetic gaze generation."""
from __future__ import annotations

from typing import Any

from pymovements.events.events import Events
from pymovements.gaze.gaze import Gaze
from pymovements.synthetic.config import GeneratorConfig
from pymovements.synthetic.library import GeneratorLibrary


def generate(
        name: str,
        config: GeneratorConfig | None = None,
        *,
        events: Events | None = None,
        gaze: Gaze | None = None,
        artifacts: dict[str, Any] | None = None,
        **kwargs: Any,
) -> Gaze:
    """Generate synthetic gaze data with the generator registered under a name.

    The one call every generation method is reached through, whatever paradigm it implements.
    It resolves the name in :py:class:`~pymovements.synthetic.GeneratorLibrary` and runs
    :py:meth:`~pymovements.synthetic.GazeGenerator.generate` on an instance of it, so selecting
    a method needs neither an import of it nor a change here.

    Equivalent to resolving and instantiating the generator yourself::

        GeneratorLibrary.get(name)().generate(config=config, events=events)

    The seed belongs to ``config``, not to this call, so that the parameters reproducing a run
    stay in one object -- the one recorded in the output's metadata.

    Parameters
    ----------
    name: str
        The registered name of the generator.
    config: GeneratorConfig | None
        The parameters of the run. (default: None)
    events: Events | None
        An event sequence to condition on. (default: None)
    gaze: Gaze | None
        A gaze sequence to condition on. (default: None)
    artifacts: dict[str, Any] | None
        External artifacts, such as trained weights. (default: None)
    **kwargs: Any
        Further generator-specific arguments.

    Returns
    -------
    Gaze
        The generated synthetic gaze data, with the run recorded under the ``'synthetic'`` key
        of its :py:attr:`~pymovements.Gaze.metadata`.

    Raises
    ------
    KeyError
        If no generator is registered under that name.
    InputContractError
        If the supplied input does not match what the generator declares.
    OutputContractError
        If the generated output does not match what the generator declares.
    """
    return GeneratorLibrary.get(name)().generate(
        config=config,
        events=events,
        gaze=gaze,
        artifacts=artifacts,
        **kwargs,
    )
