# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test Gaze transform functionality."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements import Gaze


def test_transform_early_return_on_empty_grouped_frames():
    # Create an empty samples frame with only the trial column so grouping yields no groups
    samples = pl.DataFrame(schema={'trial': pl.Int64})
    gaze = Gaze(samples=samples, trial_columns='trial')

    # Calling a transform that would normally require an input column should do nothing
    # because grouped_frames will be empty and the method returns early.
    before = gaze.samples.clone()
    gaze.clip(lower_bound=None, upper_bound=None, input_column='position', output_column='clipped')
    after = gaze.samples

    # Ensure samples are unchanged (no new columns, still empty)
    assert before.schema == after.schema
    assert before.shape == after.shape


def test_transform_returns_early_when_groupby_yields_no_groups(monkeypatch):
    # Create a non-empty samples DataFrame with a trial column so is_empty() is False
    samples = pl.DataFrame({'trial': [1]})
    # Creating a Gaze without identifiable components emits a UserWarning
    with pytest.warns(UserWarning, match='no components could be inferred'):
        gaze = Gaze(samples=samples, trial_columns='trial')

    # Define a dummy transform callable that does not require n_components or specific columns
    def dummy_transform(**_kwargs):  # pragma: no cover - exercised via transform
        return pl.lit(1).alias('dummy')

    # Monkeypatch polars.DataFrame.group_by to return an empty iterator, simulating no groups
    def fake_group_by(self, keys, maintain_order=True):  # pylint: disable=unused-argument
        return iter([])

    monkeypatch.setattr(pl.DataFrame, 'group_by', fake_group_by, raising=True)

    before = gaze.samples.clone()
    # Invoke transform - due to patched group_by producing no groups, it should early-return
    gaze.transform(dummy_transform)
    after = gaze.samples

    # Ensure samples are unchanged (no new columns added)
    assert before.schema == after.schema
    assert before.shape == after.shape


@pytest.mark.parametrize(
    'trials',
    [
        pytest.param([1], id='single_row_single_group'),
        pytest.param([1, 1, 2], id='multiple_rows_multiple_groups'),
    ],
)
def test_transform_grouped_path_non_empty_samples(trials):
    # Non-empty samples so the is_empty() guard is False - ensure grouping yields groups
    samples = pl.DataFrame({'trial': trials})

    # Creating a Gaze without identifiable components emits a UserWarning
    with pytest.warns(UserWarning, match='no components could be inferred'):
        gaze = Gaze(samples=samples, trial_columns='trial')

    # Define a simple transform that doesn't require n_components/columns
    def dummy_transform(**_kwargs):  # pragma: no cover - exercised via transform
        return pl.lit(7).alias('dummy')

    gaze.transform(dummy_transform)

    # Verify that the transform was applied through the grouped path
    assert 'dummy' in gaze.samples.columns
    assert gaze.samples['dummy'].to_list() == [7] * len(trials)


def test_transform_grouped_path_empty_samples_early_return():
    # Empty samples with a trial column: should reach the grouped-path empty check
    # and return early there (not the earlier n_components guard), so we use a dummy
    # transform that does not require n_components.
    samples = pl.DataFrame(schema={'trial': pl.Int64})
    gaze = Gaze(samples=samples, trial_columns='trial')

    def dummy_transform(**_kwargs):  # pragma: no cover - exercised via transform
        return pl.lit(1).alias('dummy')

    before = gaze.samples.clone()
    gaze.transform(dummy_transform)
    after = gaze.samples

    # Ensure samples are unchanged (no new columns, still empty)
    assert before.schema == after.schema
    assert before.shape == after.shape
