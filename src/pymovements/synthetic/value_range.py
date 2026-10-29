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
"""Provides the range type that generator hyperparameters are drawn from."""
from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from pymovements.synthetic._utils import check_name
from pymovements.synthetic._utils import check_not_negative
from pymovements.synthetic._utils import check_number

#: The distributions a value can be drawn from.
DISTRIBUTIONS = ('normal', 'uniform')


@dataclasses.dataclass(frozen=True)
class Range:
    """A range of values with the distribution a value is drawn from.

    Generators describe each random quantity, such as a duration or a peak velocity, by one
    range. A uniform draw spans the range. A normal draw is centred on the range and clipped to
    it, with ``std`` as its spread, or the width of the range if ``std`` is ``None``.

    Attributes
    ----------
    minimum: float
        The smallest value.
    maximum: float
        The largest value. Must not be smaller than ``minimum``.
    distribution: str
        The distribution a value is drawn from, one of ``DISTRIBUTIONS``. (default: 'uniform')
    std: float | None
        The standard deviation of a normal draw, or ``None`` for the width of the range.
        Must not be negative, and is only accepted for a normal range. (default: None)

    Raises
    ------
    TypeError
        If a bound or the standard deviation is not a number.
    ValueError
        If the range is inverted, the distribution is unknown, or the standard deviation is
        negative or given for a uniform range.

    Examples
    --------
    >>> import numpy as np
    >>> Range.from_value((300, 500))
    Range(minimum=300.0, maximum=500.0, distribution='uniform', std=None)
    >>> 300 <= Range(300, 500, 'normal', std=50).draw(np.random.default_rng(42)) <= 500
    True
    """

    minimum: float
    maximum: float
    distribution: str = 'uniform'
    std: float | None = None

    def __post_init__(self) -> None:
        """Validate the range and cast its numbers to float."""
        object.__setattr__(self, 'minimum', check_number(self.minimum, 'minimum'))
        object.__setattr__(self, 'maximum', check_number(self.maximum, 'maximum'))

        if self.minimum > self.maximum:
            raise ValueError(
                f'minimum ({self.minimum}) must not be greater than maximum ({self.maximum})',
            )

        check_name(self.distribution, 'distribution', DISTRIBUTIONS)

        if self.std is not None:
            object.__setattr__(self, 'std', check_not_negative(self.std, 'std'))

            if self.distribution != 'normal':
                raise ValueError(
                    f'std applies to a normal distribution only, but was given as {self.std} '
                    f'for a {self.distribution!r} range',
                )

    @classmethod
    def from_value(
            cls,
            value: Range | tuple[float, float] | list[float] | Mapping[str, Any] | float,
    ) -> Range:
        """Return a range from a range, a ``(minimum, maximum)`` pair, a mapping or a value.

        A pair becomes a uniform range, and a single value a range holding only that value. A
        mapping of the attributes is how the provenance records a range, so a run can be
        replayed from its record.

        Parameters
        ----------
        value: Range | tuple[float, float] | list[float] | Mapping[str, Any] | float
            The value to read a range from.

        Returns
        -------
        Range
            The range.

        Raises
        ------
        TypeError
            If the value is none of the accepted forms, or a mapping has unknown keys.
        """
        if isinstance(value, Range):
            return value

        if isinstance(value, Mapping):
            return cls(**value)

        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(value[0], value[1])

        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return cls(value, value)

        raise TypeError(
            'a range must be a Range, a (minimum, maximum) pair, a mapping of its attributes or '
            'a single number, but is of '
            f'type {type(value).__name__}: {value}',
        )

    def draw(self, rng: np.random.Generator) -> float:
        """Draw one value from the range.

        Parameters
        ----------
        rng: np.random.Generator
            The random generator to draw from.

        Returns
        -------
        float
            The drawn value, within the range.
        """
        if self.distribution == 'uniform':
            return float(rng.uniform(self.minimum, self.maximum))

        std = self.maximum - self.minimum if self.std is None else self.std
        centre = (self.minimum + self.maximum) / 2.0
        return float(np.clip(rng.normal(centre, std), self.minimum, self.maximum))
