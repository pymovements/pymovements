# Copyright (c) 2025 The pymovements Project Authors
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
"""Basic self-test for shared TextStimulus fixtures"""
from __future__ import annotations

import pytest

from pymovements.stimulus.text import TextStimulus


@pytest.mark.parametrize(
    "fixture_name",
    [
        pytest.param("stimulus_both_columns", id="both"),
        pytest.param("stimulus_only_trial", id="trial"),
        pytest.param("stimulus_only_page", id="page"),
    ],
)
def test_fixtures_provide_textstimulus(request: pytest.FixtureRequest, fixture_name: str) -> None:
    stim = request.getfixturevalue(fixture_name)
    assert isinstance(stim, TextStimulus)
    # AOIs must contain the configured coordinate columns
    cols = set(stim.aois.columns)
    assert stim.start_x_column in cols
    assert stim.start_y_column in cols
    assert (stim.end_x_column in cols) or (stim.width_column in cols)


@pytest.mark.parametrize(
    ("fixture_name", "row", "expect_none"),
    [
        ("stimulus_both_columns", {"x": 5, "y": 5, "trial": 1, "page": "X"}, False),
        ("stimulus_both_columns", {"x": 5, "y": 5, "trial": 1, "page": "Z"}, True),
        ("stimulus_only_trial", {"x": 5, "y": 5, "trial": 1}, False),
        ("stimulus_only_trial", {"x": 5, "y": 5, "trial": 3}, True),
        ("stimulus_only_page", {"x": 5, "y": 5, "page": "X"}, False),
        ("stimulus_only_page", {"x": 5, "y": 5, "page": "Z"}, True),
    ],
)
def test_fixtures_basic_get_aoi(
    request: pytest.FixtureRequest, fixture_name: str, row: dict, expect_none: bool,
) -> None:
    stim: TextStimulus = request.getfixturevalue(fixture_name)
    aoi = stim.get_aoi(row=row, x_eye="x", y_eye="y")
    assert aoi.shape[0] == 1
    label = aoi.select(stim.aoi_column).item()
    if expect_none:
        assert label is None
    else:
        assert label is not None
