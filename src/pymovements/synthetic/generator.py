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
from typing import Any
from typing import ClassVar

from pymovements.gaze.gaze import Gaze


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
    """

    name: ClassVar[str]

    def generate(self, **kwargs: Any) -> Gaze:
        """Generate synthetic gaze data.

        The entry point for all generators. Holds the shared generation lifecycle and delegates
        the synthesis itself to ``_generate``.

        Parameters
        ----------
        **kwargs: Any
            Generator-specific arguments, forwarded to ``_generate``.

        Returns
        -------
        Gaze
            The generated synthetic gaze data.
        """
        return self._generate(**kwargs)

    @abc.abstractmethod
    def _generate(self, **kwargs: Any) -> Gaze:
        """Synthesize gaze data. Implemented by each generator; callers use ``generate``.

        Parameters
        ----------
        **kwargs: Any
            Generator-specific arguments.

        Returns
        -------
        Gaze
            The generated synthetic gaze data.
        """
