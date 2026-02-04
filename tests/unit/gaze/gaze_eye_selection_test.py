# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Eye selection behaviour tests for Gaze.map_to_aois()."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import pymovements as pm
from pymovements.stimulus.text import TextStimulus


@pytest.mark.parametrize(
    (
        'columns',
        'eye',
        'gaze_type',
        'expected_labels',
        'expect_warn',
        'warn_match',
    ),
    [
        # mono-only flat pixel components
        pytest.param(
            {'pixel_x': [5.0, 15.0], 'pixel_y': [5.0, 5.0]},
            'mono',
            'pixel',
            ['A', None],
            False,
            None,
            id='flat-mono-exact',
        ),
        pytest.param(
            {'pixel_x': [5.0, 15.0], 'pixel_y': [5.0, 5.0]},
            'left',
            'pixel',
            ['A', None],
            True,
            'Left eye requested .* Using mono',
            id='flat-mono-fallback-left',
        ),
        pytest.param(
            {'pixel_x': [5.0, 15.0], 'pixel_y': [5.0, 5.0]},
            'right',
            'pixel',
            ['A', None],
            True,
            'Right eye requested .* Using mono',
            id='flat-mono-fallback-right',
        ),
        pytest.param(
            {'pixel_x': [5.0, 15.0], 'pixel_y': [5.0, 5.0]},
            'cyclops',
            'pixel',
            ['A', None],
            True,
            'Cyclops requested .* Using mono',
            id='flat-mono-fallback-cyclops',
        ),
        pytest.param(
            {'pixel_xr': [5.0, 15.0], 'pixel_yr': [5.0, 5.0]},
            'auto',
            'pixel',
            ['A', None],
            False,
            None,
            id='flat-right-auto-prefers-right',
        ),
        pytest.param(
            {'position_xl': [5.0, 15.0], 'position_yl': [5.0, 5.0]},
            'left',
            'position',
            ['A', None],
            False,
            None,
            id='flat-left-position',
        ),
        pytest.param(
            {'pixel_xa': [5.0, 15.0], 'pixel_ya': [5.0, 5.0]},
            'cyclops',
            'pixel',
            ['A', None],
            False,
            None,
            id='flat-cyclops-components',
        ),
        pytest.param(
            {
                'pixel_xl': [5.0, 15.0],
                'pixel_yl': [5.0, 5.0],
                'pixel_xr': [6.0, 16.0],
                'pixel_yr': [6.0, 6.0],
            },
            'cyclops',
            'pixel',
            ['A', None],
            True,
            'Cyclops requested .* Averaging left/right',
            id='flat-cyclops-average-lr',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_eye_selection_flat_components(
    simple_stimulus: TextStimulus,
    columns: dict[str, list[float]],
    eye: str,
    gaze_type: str,
    expected_labels: list[str | None],
    expect_warn: bool,
    warn_match: str | None,
) -> None:
    gaze = pm.Gaze(samples=pl.DataFrame(columns))
    if expect_warn:
        with pytest.warns(UserWarning, match=warn_match):
            gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)
    else:
        gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)

    assert gaze.samples.get_column('label').to_list() == expected_labels


@pytest.mark.parametrize(
    ('values', 'eye', 'expected_labels'),
    [
        # length-2 list -> mono
        pytest.param([[5.0, 5.0], [15.0, 5.0]], 'mono', ['A', None], id='list-2-mono'),
        pytest.param(
            [[5.0, 5.0], [15.0, 5.0]],
            'left',
            ['A', None],
            id='list-2-left-fallbacks-to-mono',
        ),
        pytest.param(
            [[5.0, 5.0], [15.0, 5.0]],
            'cyclops',
            ['A', None],
            id='list-2-cyclops-fallbacks-to-mono',
        ),
        # length-4 list -> [xl, yl, xr, yr]
        pytest.param(
            [[5.0, 5.0, 50.0, 50.0], [15.0, 5.0, 50.0, 50.0]],
            'left',
            ['A', None],
            id='list-4-left',
        ),
        pytest.param(
            [[50.0, 50.0, 5.0, 5.0], [50.0, 50.0, 15.0, 5.0]],
            'right',
            ['A', None],
            id='list-4-right',
        ),
        pytest.param(
            [[5.0, 5.0, 5.0, 5.0], [15.0, 5.0, 15.0, 5.0]],
            'cyclops',
            ['A', None],
            id='list-4-cyclops-avg-lr',
        ),
        # length-6 list -> [xl, yl, xr, yr, xa, ya]
        pytest.param(
            [
                [50.0, 50.0, 50.0, 50.0, 5.0, 5.0],
                [
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    15.0,
                    5.0,
                ],
            ],
            'cyclops',
            ['A', None],
            id='list-6-cyclops-direct',
        ),
        pytest.param(
            [[50.0, 50.0, 5.0, 5.0, None, None], [50.0, 50.0, 15.0, 5.0, None, None]],
            'mono',
            [
                'A',
                None,
            ],
            id='list-6-mono-fallback-right',
        ),
    ],
)
@pytest.mark.parametrize('source_col, gaze_type', [('position', 'position'), ('pixel', 'pixel')])
@pytest.mark.parametrize('preserve_structure', [True, False])
@pytest.mark.filterwarnings('ignore:.*requested .* Using .*:UserWarning')
@pytest.mark.filterwarnings('ignore:Cyclops requested .* Averaging left/right.*:UserWarning')
def test_eye_selection_list_columns(
    simple_stimulus: TextStimulus,
    values: list[list[float | None]],
    eye: str,
    expected_labels: list[str | None],
    source_col: str,
    gaze_type: str,
    preserve_structure: bool,
) -> None:
    df = pl.DataFrame({source_col: values})
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(
        simple_stimulus,
        eye=eye,
        gaze_type=gaze_type,
        preserve_structure=preserve_structure,
    )
    assert gaze.samples.get_column('label').to_list() == expected_labels


@pytest.mark.parametrize(
    (
        'flat_columns',
        'list_source',
        'eye',
        'gaze_type',
        'expected_labels',
    ),
    [
        # auto with flat pixel columns present but no valid component pairs ->
        # _select_components_from_flat_columns() returns None via the auto branch
        # and mapping falls back to list-column extraction path.
        pytest.param(
            {'pixel_x': [999.0], 'pixel_xl': [999.0]},  # no matching y components
            {'position': [[5.0, 5.0]]},
            'auto',
            'position',
            ['A'],
            id='auto-no-pair-falls-back-to-list',
        ),
    ],
)
def test_auto_no_pair_fallbacks_to_list(
    simple_stimulus: TextStimulus,
    flat_columns: dict[str, list[float]],
    list_source: dict[str, list[list[float]]],
    eye: str,
    gaze_type: str,
    expected_labels: list[str | None],
) -> None:
    df = pl.DataFrame(flat_columns | list_source)
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == expected_labels


@pytest.mark.parametrize(
    ('columns', 'eye', 'warn_match'),
    [
        pytest.param(
            {'pixel_xa': [5.0], 'pixel_ya': [5.0]},
            'mono',
            'Mono eye requested.*Using cyclops',
            id='mono-fallback-to-cyclops-flat',
        ),
        pytest.param(
            {'pixel_xa': [5.0], 'pixel_ya': [5.0]},
            'left',
            'Left eye requested.*Using cyclops',
            id='left-fallback-to-cyclops-flat',
        ),
        pytest.param(
            {'pixel_xa': [5.0], 'pixel_ya': [5.0]},
            'right',
            'Right eye requested.*Using cyclops',
            id='right-fallback-to-cyclops-flat',
        ),
        pytest.param(
            {'pixel_xl': [5.0], 'pixel_yl': [5.0]},
            'right',
            'Right eye requested .* Using left eye',
            id='right-fallback-to-left-flat',
        ),
        pytest.param(
            {'pixel_xr': [5.0], 'pixel_yr': [5.0]},
            'left',
            'Left eye requested .* Using right eye',
            id='left-fallback-to-right-flat',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_flat_fallbacks_to_cyclops_or_other_eye(
    simple_stimulus: TextStimulus,
    columns: dict[str, list[float]],
    eye: str,
    warn_match: str,
) -> None:
    gaze = pm.Gaze(samples=pl.DataFrame(columns))
    with pytest.warns(UserWarning, match=warn_match):
        gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type='pixel', preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A']


@pytest.mark.parametrize(
    ('values', 'eye', 'expected'),
    [
        pytest.param(
            [[5.0, 5.0, None, None], [15.0, 5.0, None, None]],
            'auto',
            ['A', None],
            id='list-4-auto-right-missing-prefers-left',
        ),
        pytest.param(
            [[None, None, 5.0, 5.0], [None, None, 15.0, 5.0]],
            'auto',
            ['A', None],
            id='list-4-auto-right-only',
        ),
        pytest.param(
            [[5.0, 5.0, 5.0, 5.0], [15.0, 5.0, 15.0, 5.0]],
            'auto',
            ['A', None],
            id='list-4-auto-average-lr',
        ),
        pytest.param(
            [[5.0, 5.0, None, None, None, None]],
            'mono',
            ['A'],
            id='list-6-mono-prefers-mono-slot-if-present',
        ),
        pytest.param(
            [[5.0, 5.0, None, None, None, None]],
            'cyclops',
            ['A'],
            id='list-6-cyclops-fallback-to-left-when-right-missing',
        ),
        pytest.param(
            [[None, None, 5.0, 5.0, None, None]],
            'cyclops',
            [
                'A',
            ],
            id='list-6-cyclops-fallback-to-right-only',
        ),
    ],
)
@pytest.mark.parametrize('src', ['position'])
@pytest.mark.parametrize('preserve_structure', [False])
def test_list_selection_edge_cases(
    simple_stimulus: TextStimulus,
    values: list[list[float | None]],
    eye: str,
    expected: list[str | None],
    src: str,
    preserve_structure: bool,
) -> None:
    df = pl.DataFrame({src: values})
    gaze = pm.Gaze(samples=df)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        gaze.map_to_aois(
            simple_stimulus,
            eye=eye,
            gaze_type=src,
            preserve_structure=preserve_structure,
        )
    assert gaze.samples.get_column('label').to_list() == expected


@pytest.mark.parametrize('exc', [Warning, ValueError, AttributeError])
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_preserve_structure_true_tolerates_unnest_exceptions(
    simple_stimulus: TextStimulus,
    exc: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build flat pixel columns so that, after unnest fails, flat-path selection still works.
    df = pl.DataFrame({'pixel_xr': [5.0], 'pixel_yr': [5.0]})
    gaze = pm.Gaze(samples=df)

    def boom(self):  # noqa: ANN001
        raise exc()

    monkeypatch.setattr(pm.Gaze, 'unnest', boom, raising=True)
    # Should not raise - should still map correctly using flat columns
    gaze.map_to_aois(simple_stimulus, eye='auto', gaze_type='pixel', preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A']


@pytest.mark.parametrize(
    ('columns', 'eye', 'gaze_type'),
    [
        pytest.param(
            {'pixel_xa': [5.0], 'pixel_ya': [5.0]},
            'auto',
            'pixel',
            id='auto-direct-cyclops',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_auto_direct_via_cyclops(
    simple_stimulus: TextStimulus,
    columns: dict[str, list[float]],
    eye: str,
    gaze_type: str,
) -> None:
    gaze = pm.Gaze(samples=pl.DataFrame(columns))
    # No warning expected - direct cyclops chosen under AUTO
    gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A']


@pytest.mark.parametrize(
    ('columns', 'eye', 'gaze_type', 'warn_match'),
    [
        pytest.param(
            {
                'position_xl': [5.0],
                'position_yl': [5.0],
            },
            'mono',
            'position',
            'Mono eye requested .* Using left eye',
            id='mono-fallback-left',
        ),
        pytest.param(
            {
                'position_xa': [5.0],
                'position_ya': [5.0],
            },
            'right',
            'position',
            'Right eye requested .* Using cyclops',
            id='right-fallback-cyclops-position',
        ),
        pytest.param(
            {'pixel_x': [5.0], 'pixel_y': [5.0]},
            'cyclops',
            'pixel',
            'Cyclops requested .* Using mono',
            id='cyclops-fallback-mono',
        ),
        pytest.param(
            {
                'position_xl': [5.0],
                'position_yl': [5.0],
            },
            'cyclops',
            'position',
            'Cyclops requested .* Using left eye',
            id='cyclops-fallback-left',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_additional_flat_fallbacks(
    simple_stimulus: TextStimulus,
    columns: dict[str, list[float]],
    eye: str,
    gaze_type: str,
    warn_match: str,
) -> None:
    gaze = pm.Gaze(samples=pl.DataFrame(columns))
    with pytest.warns(UserWarning, match=warn_match):
        gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A']


@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_average_lr_partial_data(simple_stimulus: TextStimulus) -> None:
    # Left missing X - right present -> average uses only present values, still inside AOI
    df = pl.DataFrame(
        {
            'pixel_xl': [None, None],
            'pixel_yl': [5.0, 5.0],
            'pixel_xr': [5.0, 15.0],
            'pixel_yr': [5.0, 5.0],
        }
    )
    gaze = pm.Gaze(samples=df)
    with pytest.warns(UserWarning, match='Cyclops requested .* Averaging left/right'):
        gaze.map_to_aois(simple_stimulus, eye='cyclops', gaze_type='pixel', preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A', None]


def test_list_fallback_raises_when_no_matching_source(simple_stimulus: TextStimulus) -> None:
    # Only pixel list present, but gaze_type requests position
    # -> should raise ValueError in list-fallback
    df = pl.DataFrame({'pixel': [[5.0, 5.0]]})
    gaze = pm.Gaze(samples=df)
    with pytest.raises(ValueError, match='neither position nor pixel column'):
        gaze.map_to_aois(
            simple_stimulus,
            eye='auto',
            gaze_type='position',
            preserve_structure=False,
        )


@pytest.mark.parametrize(
    (
        'flat_cols',
        'list_source',
        'eye',
        'gaze_type',
        'expected',
    ),
    [
        # req_eye == 'auto' returns None from flat selection (no valid pair) -> fallback to list
        pytest.param(
            {'pixel_x': [999.0], 'pixel_xl': [999.0]},  # no matching y columns
            {'pixel': [[5.0, 5.0]]},
            'auto',
            'pixel',
            ['A'],
            id='flat-auto-none-then-list',
        ),
        # req_eye == 'mono' returns None from flat selection (no mono, no fallbacks) -> list
        # Use pixel gaze_type so flat selector inspects pixel_* columns and walks the mono branch
        pytest.param(
            {'pixel_xl': [999.0]},  # missing y for left, nothing else
            {'pixel': [[5.0, 5.0]]},
            'mono',
            'pixel',
            ['A'],
            id='flat-mono-none-then-list',
        ),
        # req_eye == 'left' returns None from flat selection (no left, no fallbacks) -> list
        pytest.param(
            {'pixel_x': [999.0]},  # mono x only, no usable pairs and no fallbacks
            {'pixel': [[5.0, 5.0]]},
            'left',
            'pixel',
            ['A'],
            id='flat-left-none-then-list',
        ),
        # req_eye == 'right' returns None from flat selection (no right, no fallbacks) -> list
        pytest.param(
            {'pixel_x': [999.0]},  # mono x only, no usable pairs and no fallbacks
            {'pixel': [[5.0, 5.0]]},
            'right',
            'pixel',
            ['A'],
            id='flat-right-none-then-list',
        ),
        # req_eye == 'cyclops' returns None from flat selection (no cx/cy and no fallbacks) -> list
        pytest.param(
            {'pixel_x': [999.0]},
            {'pixel': [[5.0, 5.0]]},
            'cyclops',
            'pixel',
            ['A'],
            id='flat-cyclops-none-then-list',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
@pytest.mark.filterwarnings('ignore:Mono eye requested.*:UserWarning')
@pytest.mark.filterwarnings('ignore:Left eye requested.*:UserWarning')
@pytest.mark.filterwarnings('ignore:Right eye requested.*:UserWarning')
@pytest.mark.filterwarnings('ignore:Cyclops requested.*:UserWarning')
def test_flat_selection_none_then_list_fallback(
    simple_stimulus: TextStimulus,
    flat_cols: dict[str, list[float]],
    list_source: dict[str, list[list[float]]],
    eye: str,
    gaze_type: str,
    expected: list[str | None],
) -> None:
    df = pl.DataFrame(flat_cols | list_source)
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=gaze_type, preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == expected


@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_cyclops_fallback_to_right_warning(simple_stimulus: TextStimulus) -> None:
    # Only the right components present -> cyclops should warn and use the right eye
    df = pl.DataFrame({'pixel_xr': [5.0, 15.0], 'pixel_yr': [5.0, 5.0]})
    gaze = pm.Gaze(samples=df)
    with pytest.warns(UserWarning, match='Cyclops requested .* Using right eye.'):
        gaze.map_to_aois(simple_stimulus, eye='cyclops', gaze_type='pixel', preserve_structure=True)
    assert gaze.samples.get_column('label').to_list() == ['A', None]


@pytest.mark.parametrize('src_col', ['position', 'pixel'])
@pytest.mark.filterwarnings(
    'ignore:Gaze contains samples but no components could be inferred.*:UserWarning',
)
def test_list_values_empty_returns_none_row(simple_stimulus: TextStimulus, src_col: str) -> None:
    # Use uniformly empty lists (n == 0) to hit the (None, None) path without inference errors
    df = pl.DataFrame({src_col: [[], []]})
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye='auto', gaze_type=src_col, preserve_structure=False)
    assert gaze.samples.get_column('label').to_list() == [None, None]


@pytest.mark.parametrize('src_col', ['position', 'pixel'])
@pytest.mark.parametrize('eye', ['mono', 'cyclops', 'auto'])
def test_list_six_prefers_xa_ya(simple_stimulus: TextStimulus, src_col: str, eye: str) -> None:
    # 6-component list with explicit cyclops/mono at positions 4/5 should be chosen directly
    df = pl.DataFrame(
        {
            src_col: [
                [50.0, 50.0, 50.0, 50.0, 5.0, 5.0],
                [50.0, 50.0, 50.0, 50.0, 15.0, 5.0],
            ],
        }
    )
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye=eye, gaze_type=src_col, preserve_structure=False)
    assert gaze.samples.get_column('label').to_list() == ['A', None]


@pytest.mark.parametrize('src_col', ['position', 'pixel'])
def test_list_four_right_pair(simple_stimulus: TextStimulus, src_col: str) -> None:
    # 4-component list: [xl, yl, xr, yr] request right
    df = pl.DataFrame({src_col: [[50.0, 50.0, 5.0, 5.0], [50.0, 50.0, 15.0, 5.0]]})
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye='right', gaze_type=src_col, preserve_structure=False)
    assert gaze.samples.get_column('label').to_list() == ['A', None]


@pytest.mark.parametrize('src_col', ['position', 'pixel'])
def test_list_four_left_pair(simple_stimulus: TextStimulus, src_col: str) -> None:
    # 4-component list: [xl, yl, xr, yr] request left
    df = pl.DataFrame({src_col: [[5.0, 5.0, 50.0, 50.0], [15.0, 5.0, 50.0, 50.0]]})
    gaze = pm.Gaze(samples=df)
    gaze.map_to_aois(simple_stimulus, eye='left', gaze_type=src_col, preserve_structure=False)
    assert gaze.samples.get_column('label').to_list() == ['A', None]
