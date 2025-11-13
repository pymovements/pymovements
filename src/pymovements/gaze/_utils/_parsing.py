# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Module for parsing input data."""
from __future__ import annotations

import re
from typing import Any

import numpy as np


def check_nan(sample_location: str) -> float:
    """Return position as float or np.nan depending on validity of sample.

    Parameters
    ----------
    sample_location: str
        Sample location as extracted from ascii file.

    Returns
    -------
    float
        Returns either the valid sample as a float or np.nan.
    """
    try:
        ret = float(sample_location)
    except ValueError:
        ret = np.nan
    return ret


def compile_patterns(patterns: list[dict[str, Any] | str], msg_prefix: str) -> list[
    dict[str, Any]
]:
    """Compile patterns from strings.

    Parameters
    ----------
    patterns: list[dict[str, Any] | str]
        The list of patterns to compile.
    msg_prefix: str
        The message prefix to prepend to the regex patterns.

    Returns
    -------
    list[dict[str, Any]]
        Returns from string compiled regex patterns.
    """
    compiled_patterns = []

    for pattern in patterns:
        if isinstance(pattern, str):
            compiled_pattern = {'pattern': re.compile(msg_prefix + pattern)}
            compiled_patterns.append(compiled_pattern)
            continue

        if isinstance(pattern, dict):
            if isinstance(pattern['pattern'], str):
                compiled_patterns.append({
                    **pattern,
                    'pattern': re.compile(msg_prefix + pattern['pattern']),
                })
                continue

            if isinstance(pattern['pattern'], (tuple, list)):
                for single_pattern in pattern['pattern']:
                    compiled_patterns.append({
                        **pattern,
                        'pattern': re.compile(msg_prefix + single_pattern),
                    })
                continue

            raise ValueError(f'invalid pattern: {pattern}')

        raise ValueError(f'invalid pattern: {pattern}')

    return compiled_patterns


def get_pattern_keys(compiled_patterns: list[dict[str, Any]], pattern_key: str) -> set[
    str
]:
    """Get names of capture groups or column/metadata keys."""
    keys = set()

    for compiled_pattern_dict in compiled_patterns:
        if pattern_key in compiled_pattern_dict:
            keys.add(compiled_pattern_dict[pattern_key])

        for key in compiled_pattern_dict['pattern'].groupindex.keys():
            keys.add(key)

    return keys


def _calculate_data_loss_ratio(
        num_expected_samples: int,
        num_valid_samples: int,
        num_blink_samples: int,
) -> tuple[float, float]:
    """Calculate the total data loss and data loss due to blinks.

    Parameters
    ----------
    num_expected_samples: int
        Number of total expected samples.
    num_valid_samples: int
        Number of valid samples (excluding blink samples).
    num_blink_samples: int
        Number of blink samples.

    Returns
    -------
    tuple[float, float]
        Data loss ratio and blink loss ratio.
    """
    if num_expected_samples == 0:
        return 0.0, 0.0

    total_data_loss = (num_expected_samples - num_valid_samples) / num_expected_samples
    blink_data_loss = num_blink_samples / num_expected_samples
    return total_data_loss, blink_data_loss
