# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Holds the EyeTracker class."""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from pymovements._utils import _checks
from pymovements._utils._html import repr_html


@repr_html()
@dataclass
class EyeTracker:
    """EyeTracker class for holding eyetracker properties.

    Attributes
    ----------
    sampling_rate : float | None
        Sample rate of recording (in Hz). (default: None)
    left : bool | None
        Whether the left eye is tracked. (default: None)
    right : bool | None
        Whether the right eye is tracked. (default: None)
    model : str | None
        Eye tracker model (e.g. 'EyeLink II', 'Tobii Pro Spectrum'). (default: None)
    version : str | None
        Eye tracker software version number. (default: None)
    vendor : str | None
        Eye tracker vendor (e.g. 'EyeLink', 'Tobii'). (default: None)
    mount : str | None
        The mounting setup of the eye tracker (e.g. 'Desktop / Monocular / Remote').
        (default: None)

    Examples
    --------
        >>> eyetracker = EyeTracker(
        ...     sampling_rate = 1000.0,
        ...     left = False,
        ...     right = True,
        ...     model = 'EyeLink 1000 Plus',
        ...     version = '1.5.3',
        ...     vendor = 'EyeLink',
        ...     mount = 'Arm Mount / Monocular / Remote',
        ... )
        >>> print(eyetracker)
        EyeTracker(sampling_rate=1000.0, left=False, right=True, model='EyeLink 1000 Plus',
        version='1.5.3', vendor='EyeLink', mount='Arm Mount / Monocular / Remote')
    """

    sampling_rate: float | None = None
    left: bool | None = None
    right: bool | None = None
    model: str | None = None
    version: str | None = None
    vendor: str | None = None
    mount: str | None = None

    def __post_init__(self) -> None:
        """Check that the sampling rate is a positive number."""
        if self.sampling_rate is not None:
            _checks.check_is_greater_than_zero(sampling_rate=self.sampling_rate)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Convert the EyeTracker instance into a dictionary.

        Parameters
        ----------
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
            ``int``, and ``float`` are not excluded.

        Returns
        -------
        dict[str, Any]
            EyeTracker as dictionary.
        """
        _dict = asdict(self)

        # Delete fields that evaluate to False (False, None, [], {})
        if exclude_none:
            for key, value in list(_dict.items()):
                if not isinstance(value, (bool, int, float)) and not value:
                    del _dict[key]

        return _dict

    def __bool__(self) -> bool:
        """Return True if the eyetracker has data defined, else False."""
        return not all(not value for value in self.__dict__.values())
