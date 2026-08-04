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
"""Provide shared fixtures for experiment tests."""
from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from pymovements import Experiment


@pytest.fixture(name='make_experiment', scope='function')
def fixture_make_experiment() -> Callable[[str], Experiment]:
    """Return a factory that builds an Experiment from a compact spec string.

    The spec encodes screen resolution and sampling rate as
    ``'<width_px>x<height_px>@<rate>Hz'``, e.g. ``'1280x1024@1000Hz'``. Use for
    inline construction and inside helpers, e.g. ``make_experiment('1024x768@500Hz')``.
    Call without arguments for the default experiment.

    Returns
    -------
    Callable[[str], Experiment]
        Function that takes a spec string and returns an Experiment.

    """
    spec_pattern = re.compile(
        r'^(?P<width>\d+)x(?P<height>\d+)@(?P<rate>\d+(?:\.\d+)?)Hz$', re.IGNORECASE,
    )

    def _make_experiment(spec: str = '1280x1024@1000Hz') -> Experiment:
        """Build an Experiment from a compact spec string.

        The rate may be a decimal (``'@62.5Hz'``) and ``Hz`` is case-insensitive.
        Physical screen dimensions, viewing distance and origin use fixed defaults.

        Parameters
        ----------
        spec : str
            Experiment spec of the form ``'<width_px>x<height_px>@<rate>Hz'``.
            (default: ``'1280x1024@1000Hz'``)

        Returns
        -------
        Experiment
            Experiment with the parsed resolution and sampling rate.

        Raises
        ------
        ValueError
            If *spec* does not match the expected format.

        """
        match = spec_pattern.match(spec.strip())
        if match is None:
            raise ValueError(
                f'invalid experiment spec {spec!r}; expected format '
                "'<width>x<height>@<rate>Hz', e.g. '1280x1024@1000Hz'",
            )
        return Experiment(
            screen_width_px=int(match.group('width')),
            screen_height_px=int(match.group('height')),
            screen_width_cm=38.0,
            screen_height_cm=30.0,
            distance_cm=68.0,
            origin='upper left',
            sampling_rate=float(match.group('rate')),
        )
    return _make_experiment
