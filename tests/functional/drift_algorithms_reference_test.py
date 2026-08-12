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
"""Validate drift correction algorithms against their reference implementations.

These tests download the reference implementation accompanying Carr et al. 2022 (see
``CARR_REFERENCE``) and the GazeGenie implementation of Mercier et al. 2024 (see
``GAZEGENIE_REFERENCE``, providing the Wisdom of the Crowd ensemble), run them side by side
with the pymovements port on the fixation fixtures stored in
``tests/files/drift_correction/``, and assert that both produce identical output.
Running both implementations in the same environment makes the comparison immune to
numerical differences between numpy, scipy and scikit-learn versions: only a genuine
divergence of the port from the reference can fail these tests.

The downloads are pinned to git commits (the raw URLs are content-addressed) and verified
against MD5 checksums, so the executed reference code is immutable. A download failure
fails the tests instead of skipping them, so the reference validation cannot silently
disappear from CI.

Fixture generation: each fixture is a synthetic multi-line reading trial. For each line, one
fixation per word (following the given reading path, which may contain regressions and skips)
is placed at ``x = word_x + N(0, x_sd)`` and ``y = line_y + drift_slope * x +
drift_offset * line_index + N(0, y_sd)``, using ``numpy.random.default_rng(seed)`` and
rounding to 2 decimals. The committed CSV values are canonical. Fixture parameters (seed,
line_ys, word_xs, paths, drift_slope, drift_offset, x_sd, y_sd):

- baseline: 20260812, 100:340:60, 80:800:80, [0,1,2,3,4,5,3,6,7,8,9] per line, 0.02, 2.5, 6, 4
- severe_drift: 20260813, 100:340:60, 80:800:80, [0,1,2,3,1,4,5,6,7,5,8,9] per line,
  0.03, 4.0, 6, 5
- two_lines: 20260814, 150:210:60, 80:710:90, [0,1,2,3,4,2,5,6,7] per line, 0.02, 3.0, 6, 4
- irregular: 20260815, 120:300:60, 80:800:80, per-line paths [0,2,3,5,6,8,9],
  [0,1,3,4,2,6,7,9], [0,4,8], [0,1,2,4,5,3,6,7,8,9], 0.022, 3.0, 6, 4

The fixtures were verified to be stable at creation time with repeated runs (the reference
uses KMeans without a fixed random state) and with input perturbations of 0.01 px, so the
side-by-side comparison of two independent stochastic runs is deterministic on these inputs.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import polars as pl
import pytest

import pymovements.events.correction.drift_algorithms as da
from pymovements import WebSource

pytestmark = pytest.mark.network

CARR_REFERENCE = WebSource(
    url=(
        'https://raw.githubusercontent.com/jwcarr/drift/'
        '5b4b6c475b5118950514dc01960391ef0d95bd19/algorithms/Python/drift_algorithms.py'
    ),
    filename='drift_algorithms_reference.py',
    md5='c70c367328c27fcd3f02c314ee7927f4',
)

GAZEGENIE_REFERENCE = WebSource(
    url=(
        'https://raw.githubusercontent.com/Gittingthehubbing/GazeGenie/'
        '57b60eff78755759e8f76820e48eaddf9a362a0a/classic_correction_algos.py'
    ),
    filename='gazegenie_classic_correction_algos.py',
    md5='524f5e7f338b280c7b8cf41b37672e47',
)


def _import_module_from_file(module_name, filepath):
    """Import a downloaded reference implementation as a module."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALGORITHMS = [
    'attach', 'chain', 'cluster', 'compare', 'merge', 'regress',
    'segment', 'slice', 'split', 'stretch', 'warp',
]

FIXTURE_LINE_YS = {
    'baseline': np.arange(100.0, 341.0, 60.0),
    'severe_drift': np.arange(100.0, 341.0, 60.0),
    'two_lines': np.arange(150.0, 211.0, 60.0),
    'irregular': np.arange(120.0, 301.0, 60.0),
}

FIXTURE_WORD_XS = {
    'baseline': np.arange(80.0, 801.0, 80.0),
    'severe_drift': np.arange(80.0, 801.0, 80.0),
    'two_lines': np.arange(80.0, 711.0, 90.0),
    'irregular': np.arange(80.0, 801.0, 80.0),
}

# The reference implementation raises an IndexError with its default n_nearest_lines=3 on
# texts with fewer than 3 lines, so it is called with an explicit valid value there. The
# pymovements port is still called with its DEFAULT parameters on purpose: this asserts
# that the clamped default reproduces the reference with a valid explicit parameter.
REFERENCE_COMPARE_KWARGS = {'two_lines': {'n_nearest_lines': 2}}


@pytest.fixture(name='reference', scope='session')
def fixture_reference(tmp_path_factory):
    """Download the pinned reference implementation and import it as a module."""
    target_dirpath = tmp_path_factory.mktemp('carr_reference')
    filepath = CARR_REFERENCE.download(target_dirpath, verbose=False)
    return _import_module_from_file('carr_reference_drift_algorithms', filepath)


@pytest.fixture(name='gazegenie_reference', scope='session')
def fixture_gazegenie_reference(tmp_path_factory):
    """Download the pinned GazeGenie reference implementation and import it as a module."""
    target_dirpath = tmp_path_factory.mktemp('gazegenie_reference')
    filepath = GAZEGENIE_REFERENCE.download(target_dirpath, verbose=False)

    # classic_correction_algos.py imports the debug library icecream, which is not a
    # pymovements dependency. Provide a call-compatible stub so the reference module can
    # be imported unmodified.
    if 'icecream' not in sys.modules:
        icecream_stub = types.ModuleType('icecream')
        icecream_stub.ic = types.SimpleNamespace(  # type: ignore[attr-defined]
            configureOutput=lambda **kwargs: None,
        )
        sys.modules['icecream'] = icecream_stub

    return _import_module_from_file('gazegenie_classic_correction_algos', filepath)


def load_fixture(testfiles_dirpath, fixture_name):
    """Load a fixation fixture CSV as a frame with a 'location' column of [x, y] lists."""
    csv_path = testfiles_dirpath / 'drift_correction' / f'{fixture_name}.csv'
    return pl.read_csv(csv_path).select(
        pl.concat_list(['fixation_x', 'fixation_y']).alias('location'),
    )


def load_fixture_array(testfiles_dirpath, fixture_name):
    """Load a fixation fixture CSV as an array of shape (N, 2) for the reference."""
    csv_path = testfiles_dirpath / 'drift_correction' / f'{fixture_name}.csv'
    return pl.read_csv(csv_path).select(['fixation_x', 'fixation_y']).to_numpy()


def word_xy_grid(fixture_name):
    """Build the word center grid of a fixture, y being the line position (Carr et al.)."""
    return np.array([
        [word_x, line_y]
        for line_y in FIXTURE_LINE_YS[fixture_name]
        for word_x in FIXTURE_WORD_XS[fixture_name]
    ])


def word_locations_series(fixture_name):
    """Build the word center grid of a fixture as a series of [x, y] locations."""
    return pl.Series(
        'word_location', word_xy_grid(fixture_name).tolist(), dtype=pl.List(pl.Float64),
    )


def build_algorithm_expression(algorithm, fixture_name):
    """Build the drift correction expression of an algorithm for a fixture."""
    if algorithm in {'compare', 'warp'}:
        return getattr(da, algorithm)(word_locations_series(fixture_name))
    return getattr(da, algorithm)(FIXTURE_LINE_YS[fixture_name].tolist())


@pytest.mark.parametrize('fixture_name', list(FIXTURE_LINE_YS))
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_algorithm_matches_reference_implementation(
        algorithm, fixture_name, reference, testfiles_dirpath,
):
    fixation_xy = load_fixture_array(testfiles_dirpath, fixture_name)
    line_ys = FIXTURE_LINE_YS[fixture_name]

    reference_kwargs = {}
    if algorithm in {'compare', 'warp'}:
        reference_target = word_xy_grid(fixture_name)
        if algorithm == 'compare':
            reference_kwargs = REFERENCE_COMPARE_KWARGS.get(fixture_name, {})
    else:
        reference_target = line_ys

    # The reference implementation mutates its inputs, so it receives its own copies.
    expected = getattr(reference, algorithm)(
        np.array(fixation_xy, copy=True),
        np.array(reference_target, copy=True),
        **reference_kwargs,
    )

    fixations = load_fixture(testfiles_dirpath, fixture_name)
    result = fixations.select(
        build_algorithm_expression(algorithm, fixture_name),
    ).to_series()

    np.testing.assert_array_equal(result.to_numpy(), expected[:, 1])
    # Every output y must be a line position.
    assert set(result.to_list()) <= set(line_ys)


def test_dynamic_time_warping_matches_reference_implementation(reference, testfiles_dirpath):
    fixation_xy = load_fixture_array(testfiles_dirpath, 'baseline')
    word_xy = word_xy_grid('baseline')

    expected_cost, expected_path = reference.dynamic_time_warping(
        np.array(fixation_xy[:11], copy=True), np.array(word_xy[:10], copy=True),
    )
    fixations = load_fixture(testfiles_dirpath, 'baseline')
    cost, path = da.dynamic_time_warping(
        fixations.to_series().head(11), word_locations_series('baseline').head(10),
    )

    assert cost == pytest.approx(expected_cost)
    assert path == expected_path


@pytest.mark.parametrize('fixture_name', list(FIXTURE_LINE_YS))
def test_wisdom_of_the_crowd_matches_reference_implementation(
        fixture_name, gazegenie_reference, testfiles_dirpath,
):
    fixations = load_fixture(testfiles_dirpath, fixture_name)

    # Assignments of the line-based algorithms serve as deterministic ensemble votes.
    line_based_algorithms = [
        'attach', 'chain', 'cluster', 'merge', 'regress',
        'segment', 'slice', 'split', 'stretch',
    ]
    votes = fixations.select([
        build_algorithm_expression(name, fixture_name).alias(name)
        for name in line_based_algorithms
    ])

    expected = gazegenie_reference.wisdom_of_the_crowd(
        [votes[name].to_numpy() for name in line_based_algorithms],
    )
    result = votes.select(da.wisdom_of_the_crowd(line_based_algorithms)).to_series()

    assert result.to_list() == list(expected)


def test_wisdom_of_the_crowd_tie_breaking_matches_reference_implementation(
        gazegenie_reference,
):
    """Ties are broken in favor of the left-most column, as in the reference."""
    vote_scenarios = [
        # Full tie between all algorithms.
        {'a': [100.0, 200.0], 'b': [200.0, 100.0], 'c': [300.0, 300.0]},
        # Two-way tie between second and third candidate.
        {
            'a': [100.0, 100.0], 'b': [200.0, 200.0], 'c': [200.0, 300.0],
            'd': [300.0, 300.0], 'e': [100.0, 200.0],
        },
        # Unanimous vote.
        {'a': [100.0, 100.0], 'b': [100.0, 100.0], 'c': [100.0, 100.0]},
    ]
    for scenario in vote_scenarios:
        votes = pl.DataFrame(scenario)
        expected = gazegenie_reference.wisdom_of_the_crowd(
            [np.array(assignment) for assignment in scenario.values()],
        )
        result = votes.select(da.wisdom_of_the_crowd(list(scenario))).to_series()
        assert result.to_list() == list(expected)
