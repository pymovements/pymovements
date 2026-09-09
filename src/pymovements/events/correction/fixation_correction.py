# Copyright (c) 2022-2026 The pymovements Project Authors
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
"""Module for fixation drift correction routines.

All coordinates and distance thresholds are expressed in pixels in the coordinate system of
the fixation locations, with the y-axis pointing downward: lines of text are ordered top to
bottom by increasing y-coordinate.

Supported Drift Correction Algorithms
-------------------------------------
- **wisdom_of_the_crowd** (or **woc**) : *(Default)* Ensemble correction method combining
  predictions across multiple algorithms via majority voting per fixation
  (:cite:p:`Mercier2024b`).
- **attach** : Snaps each fixation to the vertically closest line of text (:cite:p:`Carr2022`).
- **chain** : Groups fixations into reading chains based on spatio-temporal distance thresholds
  and aligns each chain to line centers (:cite:p:`Carr2022`).
- **cluster** : Uses K-Means clustering to group fixation Y-coordinates into clusters matching
  text lines (:cite:p:`Carr2022`).
- **compare** : Matches fixation sequences to candidate text line paths using Dynamic Time
  Warping (DTW) (:cite:p:`LimaSanches2015,Carr2022`).
- **merge** : Forms progressive sequences and iteratively merges sequences belonging to the same
  text line (:cite:p:`Spakov2019,Carr2022`).
- **regress** : Fits a linear regression model (slope, offset, std) to estimate line assignments
  (:cite:p:`Cohen2013,Carr2022`).
- **segment** : Segments fixations into line subsequences using return sweep identification
  (:cite:p:`Abdulin2015,Carr2022`).
- **slice** : Slices fixation sequence into proto-lines based on vertical drift thresholds
  (:cite:p:`Glandorf2021`).
- **split** : Splits fixations into line subsequences using K-Means return sweep identification
  (:cite:p:`Carr2022`).
- **stretch** : Fits scale and offset parameters to stretch or compress fixations onto line
  centers (:cite:p:`Lohmeier2015,Carr2022`).
- **warp** : Dynamic Time Warping (DTW) alignment between fixations and word centroids
  (:cite:p:`Carr2022`).
"""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import polars as pl

from pymovements.events.correction._aoi import count_text_lines
from pymovements.events.correction._aoi import get_lines_of_text_from_aois
from pymovements.events.correction._aoi import get_word_locations_from_aois
from pymovements.events.correction._aoi import has_word_x_coords
from pymovements.events.correction._aoi import normalize_aois
from pymovements.events.correction._utils import is_right_to_left
from pymovements.events.correction._utils import line_index_to_y
from pymovements.events.correction._utils import location_x
from pymovements.events.correction._utils import nearest_line_index
from pymovements.events.correction.attach import attach
from pymovements.events.correction.chain import chain
from pymovements.events.correction.cluster import cluster
from pymovements.events.correction.compare import compare
from pymovements.events.correction.merge import merge
from pymovements.events.correction.regress import regress
from pymovements.events.correction.segment import segment
from pymovements.events.correction.slice import slice  # pylint: disable=redefined-builtin
from pymovements.events.correction.split import split
from pymovements.events.correction.stretch import stretch
from pymovements.events.correction.warp import warp
from pymovements.events.correction.wisdom_of_the_crowd import wisdom_of_the_crowd

# The insertion order defines the default tie-breaking priority of the ensemble votes.
_DRIFT_ALGORITHMS: dict[str, Callable[..., pl.Expr]] = {
    'attach': attach,
    'chain': chain,
    'cluster': cluster,
    'compare': compare,
    'merge': merge,
    'regress': regress,
    'segment': segment,
    'slice': slice,
    'split': split,
    'stretch': stretch,
    'warp': warp,
}

ALL_DRIFT_ALGORITHMS: list[str] = list(_DRIFT_ALGORITHMS)

# Columns whose presence next to a 'word' column marks a character-level AOI frame.
_CHARACTER_LEVEL_COLUMNS = ('char', 'character', 'char_idx_in_line')


def _character_level_columns(aoi_column: str | None) -> tuple[str, ...]:
    """Resolve the columns whose presence marks a character-level AOI frame.

    With a known AOI content column, a frame is character-level exactly if its content
    column is finer than words; without one, the conventional character column names
    serve as heuristic markers.
    """
    if aoi_column is None:
        return _CHARACTER_LEVEL_COLUMNS
    if aoi_column == 'word':
        return ()
    return (aoi_column,)


def _min_fixation_count(algorithms: set[str], n_lines: int) -> int:
    """Minimum number of fixations the given drift algorithms need to run."""
    minimum = 1
    if 'cluster' in algorithms:
        # cluster fits one KMeans cluster per text line.
        minimum = max(minimum, n_lines)
    if 'split' in algorithms:
        # split fits a 2-cluster KMeans on the saccade x-differences between fixations.
        minimum = max(minimum, 3)
    return minimum


def _select_ensemble_algorithms(
    algorithms: list[str],
    has_word_coords: bool,
    right_to_left: bool,
) -> list[str]:
    """Select candidate algorithms for the ensemble, excluding unsupported ones.

    Parameters
    ----------
    algorithms: list[str]
        Requested algorithm names.
    has_word_coords: bool
        Whether word X coordinates are available for the DTW-based algorithms.
    right_to_left: bool
        Whether the text is read from right to left.

    Returns
    -------
    list[str]
        Candidate algorithm names for the ensemble.

    Raises
    ------
    ValueError
        If an algorithm name is unknown or no candidate algorithms remain.
    """
    unknown_algos = [algo for algo in algorithms if algo not in ALL_DRIFT_ALGORITHMS]
    if unknown_algos:
        raise ValueError(
            f'Unknown drift algorithms {unknown_algos}. '
            f'Valid algorithms are: {ALL_DRIFT_ALGORITHMS}',
        )

    candidate_algos = []
    excluded_algos = []
    for algo in algorithms:
        if algo in {'compare', 'warp'} and not has_word_coords:
            excluded_algos.append(algo)
        else:
            candidate_algos.append(algo)
    if excluded_algos:
        warnings.warn(
            "Word X coordinates ('start_x', 'end_x') are missing from aois DataFrame. "
            'As a consequence, algorithms requiring word X coordinates '
            f"({excluded_algos}) are excluded from Wisdom of the Crowd ensemble.",
            UserWarning,
            stacklevel=4,
        )

    if right_to_left and 'compare' in candidate_algos:
        warnings.warn(
            "Algorithm 'compare' does not support right-to-left reading and is excluded "
            'from Wisdom of the Crowd ensemble.',
            UserWarning,
            stacklevel=4,
        )
        candidate_algos = [algo for algo in candidate_algos if algo != 'compare']

    if not candidate_algos:
        raise ValueError('No candidate algorithms remain for the ensemble.')

    return candidate_algos


def _resolve_algorithms(
    algorithm: str | list[str],
    has_word_coords: bool,
    right_to_left: bool,
) -> tuple[list[str], bool]:
    """Resolve the algorithm argument to candidate names and an ensemble flag.

    Parameters
    ----------
    algorithm: str | list[str]
        Name of a single drift algorithm, 'wisdom_of_the_crowd' (or 'woc'), or a list of
        algorithm names to combine via ensemble correction.
    has_word_coords: bool
        Whether word X coordinates are available for the DTW-based algorithms.
    right_to_left: bool
        Whether the text is read from right to left.

    Returns
    -------
    tuple[list[str], bool]
        Candidate algorithm names and whether they form a Wisdom of the Crowd ensemble.

    Raises
    ------
    ValueError
        If the algorithm list is empty, an algorithm name is unknown, or a single
        'compare' algorithm is requested for right-to-left reading.
    TypeError
        If algorithm is neither a string nor a list of strings.
    """
    if isinstance(algorithm, (list, tuple)):
        if len(algorithm) == 0:
            raise ValueError('At least one algorithm must be provided in the algorithm list.')
        if len(algorithm) == 1:
            # A single-element list is treated exactly like a single algorithm name.
            return _resolve_algorithms(algorithm[0], has_word_coords, right_to_left)
        candidate_algos = _select_ensemble_algorithms(
            list(algorithm), has_word_coords, right_to_left,
        )
        return candidate_algos, True

    if isinstance(algorithm, str):
        if algorithm.lower() in {'wisdom_of_the_crowd', 'woc'}:
            candidate_algos = _select_ensemble_algorithms(
                list(ALL_DRIFT_ALGORITHMS), has_word_coords, right_to_left,
            )
            return candidate_algos, True
        if algorithm not in ALL_DRIFT_ALGORITHMS:
            raise ValueError(
                f"Unknown drift algorithm '{algorithm}'. "
                f'Valid algorithms are: {ALL_DRIFT_ALGORITHMS}',
            )
        if algorithm == 'compare' and right_to_left:
            raise ValueError(
                "Algorithm 'compare' does not support right-to-left reading as its "
                'line break detection assumes left-to-right reading.',
            )
        return [algorithm], False

    raise TypeError('algorithm must be a string or a list of strings.')


def _resolve_line_values(aois: pl.DataFrame, word_locations: pl.Series | None) -> list[float]:
    """Resolve the text line y-coordinates from the AOIs or the word locations."""
    has_line_info = (
        ('start_y' in aois.columns or 'top_left_y' in aois.columns)
        and 'height' in aois.columns
    )
    if has_line_info:
        return get_lines_of_text_from_aois(aois)
    if word_locations is not None:
        return (
            word_locations.cast(pl.List(pl.Float64))
            .list.get(1).unique().sort().to_list()
        )
    # Neither complete line information nor word locations are available: derive from
    # the AOIs anyway so the resulting ValueError names the missing columns.
    return get_lines_of_text_from_aois(aois)


def _route_algorithm_kwargs(
    candidate_algos: list[str],
    algorithm_kwargs: dict[str, Any],
    directionality: str,
) -> dict[str, dict[str, Any]]:
    """Route tuning parameters to those candidate algorithms that accept them.

    Routing algorithm-specific parameters only to the algorithms accepting them keeps
    those parameters from breaking the other algorithms in the ensemble.

    Parameters
    ----------
    candidate_algos: list[str]
        Candidate algorithm names of the ensemble.
    algorithm_kwargs: dict[str, Any]
        Additional tuning parameters for the drift correction algorithms.
    directionality: str
        Reading direction of the text, added for those algorithms that accept it.

    Returns
    -------
    dict[str, dict[str, Any]]
        Call keyword arguments per candidate algorithm.

    Raises
    ------
    ValueError
        If an algorithm_kwargs entry is accepted by none of the candidate algorithms.
    """
    candidate_params = {
        candidate_algo: set(inspect.signature(_DRIFT_ALGORITHMS[candidate_algo]).parameters)
        for candidate_algo in candidate_algos
    }
    unknown_kwargs = [
        key for key in algorithm_kwargs
        if not any(key in params for params in candidate_params.values())
    ]
    if unknown_kwargs:
        raise ValueError(
            f'algorithm_kwargs entries {unknown_kwargs} are not accepted by any of the '
            f'ensemble algorithms {candidate_algos}.',
        )

    routed_kwargs = {}
    for candidate_algo, params in candidate_params.items():
        call_kwargs = {
            key: value for key, value in algorithm_kwargs.items() if key in params
        }
        if 'directionality' in params:
            call_kwargs['directionality'] = directionality
        routed_kwargs[candidate_algo] = call_kwargs
    return routed_kwargs


def _correct_single(
    fixations: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str,
    *,
    directionality: str,
    word_locations: pl.Series | None,
    algorithm_kwargs: dict[str, Any],
    location: str | pl.Expr,
    aoi_column: str | None,
) -> pl.Series:
    """Correct fixation locations with a single drift algorithm."""
    if algorithm in {'compare', 'warp'}:
        if word_locations is None:
            if not has_word_x_coords(aois):
                raise ValueError(
                    f"Algorithm '{algorithm}' requires word X coordinates "
                    "('start_x', 'end_x') in aois DataFrame or the "
                    "'word_locations' parameter.",
                )
            word_locations = get_word_locations_from_aois(
                aois, _character_level_columns(aoi_column),
            )
        target: pl.Series | list[float] = word_locations
    else:
        target = get_lines_of_text_from_aois(aois)

    func = _DRIFT_ALGORITHMS[algorithm]
    call_kwargs = dict(algorithm_kwargs)
    if 'directionality' in inspect.signature(func).parameters:
        call_kwargs['directionality'] = directionality
    corrected_y = func(target, location=location, **call_kwargs)
    return fixations.select(
        pl.concat_list([location_x(location), corrected_y]).alias('location'),
    ).to_series()


def _correct_ensemble(
    fixations: pl.DataFrame,
    aois: pl.DataFrame,
    candidate_algos: list[str],
    *,
    directionality: str,
    word_locations: pl.Series | None,
    algorithm_kwargs: dict[str, Any],
    location: str | pl.Expr,
    aoi_column: str | None,
) -> pl.Series:
    """Correct fixation locations by majority voting across the candidate algorithms."""
    if {'compare', 'warp'} & set(candidate_algos) and word_locations is None:
        word_locations = get_word_locations_from_aois(
            aois, _character_level_columns(aoi_column),
        )

    # Vote on line indices rather than raw y-coordinates so that candidate algorithms cannot
    # split votes through differing float representations of the same text line.
    line_values = _resolve_line_values(aois, word_locations)

    routed_kwargs = _route_algorithm_kwargs(candidate_algos, algorithm_kwargs, directionality)

    vote_exprs = []
    for candidate_algo in candidate_algos:
        func = _DRIFT_ALGORITHMS[candidate_algo]
        if candidate_algo in {'compare', 'warp'}:
            corrected_y = func(word_locations, location=location, **routed_kwargs[candidate_algo])
        else:
            corrected_y = func(line_values, location=location, **routed_kwargs[candidate_algo])
        vote_exprs.append(
            nearest_line_index(corrected_y, line_values).alias(candidate_algo),
        )

    votes = fixations.select(
        [location_x(location).alias('__location_x')] + vote_exprs,  # noqa: SLF001
    )
    return votes.select(
        pl.concat_list([
            pl.col('__location_x'),
            line_index_to_y(wisdom_of_the_crowd(candidate_algos), line_values),
        ]).alias('location'),
    ).to_series()


def _fixation_location(fixations: pl.DataFrame, location_column: str) -> str | pl.Expr:
    """Resolve the location column or expression of a fixations dataframe.

    Parameters
    ----------
    fixations: pl.DataFrame
        Fixations dataframe holding either a location column of [x, y] lists or its
        '_x' and '_y' component columns.
    location_column: str
        Name of the column holding the [x, y] fixation locations.

    Returns
    -------
    str | pl.Expr
        Location column name or expression of [x, y] fixation locations.

    Raises
    ------
    ValueError
        If no location coordinates are found.
    """
    if location_column in fixations.columns and fixations[location_column].dtype != pl.Null:
        return location_column
    x_column, y_column = f'{location_column}_x', f'{location_column}_y'
    if x_column in fixations.columns and y_column in fixations.columns:
        return pl.concat_list([pl.col(x_column), pl.col(y_column)])
    raise ValueError(
        f"No valid location coordinates found in events dataframe: expected a "
        f"'{location_column}' column of [x, y] lists or '{x_column}' and "
        f"'{y_column}' component columns.",
    )


def correct_fixation_locations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    directionality: str = 'left-to-right',
    word_locations: pl.Series | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
    fixation_name: str = 'fixation',
    location_column: str = 'location',
    aoi_column: str | None = None,
) -> pl.Series:
    """Correct fixations based on the specified drift algorithm and AOIs.

    Parameters
    ----------
    events: pl.DataFrame
        Gaze events dataframe.
    aois: pl.DataFrame
        AOIs dataframe for line position extraction. Text line positions are derived
        from the 'start_y' (or 'top_left_y') and 'height' columns, with 'height'
        derived from 'start_y' and 'end_y' if missing. AOIs are grouped into lines by
        a 'line_idx' column if present, otherwise by their y-coordinate. The DTW-based
        algorithms additionally use the word X coordinate columns 'start_x' and
        'end_x', with 'end_x' derived from 'start_x' and 'width' if missing.
    algorithm: str | list[str]
        Name of a single drift algorithm or a list of algorithm names to combine via Wisdom of
        the Crowd (WoC) ensemble correction. Default is 'wisdom_of_the_crowd' (or 'woc'), which
        includes all drift algorithms. If word X coordinates ('start_x', 'end_x') are missing in
        aois, 'compare' and 'warp' are automatically excluded from the ensemble with a UserWarning.
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left',
        mirroring the directionality of a text stimulus writing system; 'top-to-bottom'
        is not supported and raises a ValueError. Passed to those algorithms with
        direction-specific processing ('merge', 'segment', 'split'); direction-agnostic
        algorithms ignore it. The 'compare' algorithm does not support right-to-left
        reading: it is excluded from ensembles with a UserWarning and raises a ValueError
        when selected as a single algorithm. (default: 'left-to-right')
    word_locations: pl.Series | None
        Series of [x, y] word center coordinates for the DTW-based algorithms 'compare'
        and 'warp'. If None, word locations are derived from the aois dataframe. Following
        Carr et al., y-coordinates should be the text line centers. (default: None)
    algorithm_kwargs: dict[str, Any] | None
        Additional tuning parameters passed to underlying drift correction algorithms, e.g.
        ``{'x_thresh': 250.0}``. In ensemble mode, each entry is only passed to those
        candidate algorithms that accept it; a ValueError is raised if an entry is accepted
        by none of the candidate algorithms. Warning: an entry fans out to every candidate
        algorithm whose signature accepts the key, even where defaults and semantics
        differ. For example, ``x_thresh`` is accepted by 'chain' (default 192, chain
        breaking), 'compare' (default 512, line break detection) and 'slice' (default 192,
        run segmentation), so ``{'x_thresh': 250.0}`` reconfigures all three at once.
        (default: None)
    fixation_name: str
        Name of the fixation events to correct. Only events matching this name exactly are
        corrected; unlike :py:meth:`~pymovements.Events.map_to_aois`, no prefix matching
        is applied. (default: 'fixation')
    location_column: str
        Name of the events column holding the [x, y] fixation locations. If missing, the
        component columns '<location_column>_x' and '<location_column>_y' are used
        instead. (default: 'location')
    aoi_column: str | None
        Name of the aois column holding the AOI content. A frame holding a 'word' column
        next to a content column other than 'word' is treated as character-level and its
        AOIs are aggregated to one location per word. If None, character-level frames are
        recognized by a 'word' column next to one of the conventional character columns
        'char', 'character' or 'char_idx_in_line'. (default: None)

    Returns
    -------
    pl.Series
        Series of corrected [x, y] fixation locations.

    Raises
    ------
    ValueError
        If the algorithm name is unknown, the directionality is invalid, an
        algorithm_kwargs entry is accepted by no candidate algorithm, or required
        coordinate data is missing.
    TypeError
        If algorithm is neither a string nor a list of strings.

    Examples
    --------
    Correcting fixations that drift away from three lines of text with their centers at
    y = 100, 200 and 300 snaps each y-coordinate onto its line center:

    >>> import polars as pl
    >>> from pymovements.events.correction import correct_fixation_locations
    >>> events = pl.DataFrame({
    ...     'name': ['fixation', 'fixation', 'fixation'],
    ...     'location': [[100.0, 105.0], [110.0, 195.0], [120.0, 302.0]],
    ... })
    >>> aois = pl.DataFrame({
    ...     'start_y': [80.0, 180.0, 280.0],
    ...     'height': [40.0, 40.0, 40.0],
    ... })
    >>> correct_fixation_locations(events, aois, algorithm='attach')
    shape: (3,)
    Series: 'location' [list[f64]]
    [
        [100.0, 100.0]
        [110.0, 200.0]
        [120.0, 300.0]
    ]

    For updating an events dataframe in place, including per-trial processing and the
    'location_original' and 'correction_algorithm' bookkeeping columns, see
    :py:func:`~pymovements.events.correction.correct_fixations`.
    """
    right_to_left = is_right_to_left(directionality)
    if algorithm_kwargs is None:
        algorithm_kwargs = {}
    for reserved_key in ('directionality', 'word_locations', 'location'):
        if reserved_key in algorithm_kwargs:
            raise ValueError(
                f"'{reserved_key}' must be passed as an explicit parameter, "
                'not via algorithm_kwargs.',
            )

    aois = normalize_aois(aois)

    fixations = events.filter(pl.col('name') == fixation_name)
    location = _fixation_location(fixations, location_column)

    has_word_coords = word_locations is not None or has_word_x_coords(aois)

    candidate_algos, ensemble = _resolve_algorithms(algorithm, has_word_coords, right_to_left)
    if not ensemble:
        corrected = _correct_single(
            fixations, aois, candidate_algos[0],
            directionality=directionality, word_locations=word_locations,
            algorithm_kwargs=algorithm_kwargs, location=location, aoi_column=aoi_column,
        )
    else:
        corrected = _correct_ensemble(
            fixations, aois, candidate_algos,
            directionality=directionality, word_locations=word_locations,
            algorithm_kwargs=algorithm_kwargs, location=location, aoi_column=aoi_column,
        )
    return corrected.rename(location_column)


def _check_not_already_corrected(events: pl.DataFrame, fixation_name: str) -> None:
    """Raise if the fixation events already carry a correction algorithm."""
    if 'correction_algorithm' not in events.columns:
        return
    already_corrected = events.filter(
        (pl.col('name') == fixation_name)
        & pl.col('correction_algorithm').is_not_null(),
    )
    if already_corrected.height > 0:
        raise ValueError(
            f"'{fixation_name}' events have already been corrected with "
            f"'{already_corrected['correction_algorithm'][0]}'.",
        )


def _algorithm_label(algorithm: str | list[str]) -> tuple[str, set[str]]:
    """Resolve the bookkeeping algorithm name and the set of requested algorithms."""
    if isinstance(algorithm, (list, tuple)):
        if len(algorithm) > 1:
            return 'wisdom_of_the_crowd', set(algorithm)
        return algorithm[0], set(algorithm)
    if isinstance(algorithm, str) and algorithm.lower() not in {'wisdom_of_the_crowd', 'woc'}:
        return algorithm, {algorithm}
    return 'wisdom_of_the_crowd', set(ALL_DRIFT_ALGORITHMS)


def _trial_aois(
    aois: pl.DataFrame,
    trial_events: pl.DataFrame,
    aoi_trial_columns: list[str],
) -> pl.DataFrame:
    """Filter the AOIs matching a single trial partition."""
    if not aoi_trial_columns:
        return aois
    # Each partition holds a single combination of trial column values.
    return aois.filter(
        pl.all_horizontal([
            pl.col(column).eq_missing(pl.lit(trial_events[column][0]))
            for column in aoi_trial_columns
        ]),
    )


def _correct_trial(
    fixation_events: pl.DataFrame,
    trial_aois: pl.DataFrame,
    *,
    algorithm: str | list[str],
    requested_algorithms: set[str],
    trial_columns: list[str] | None,
    directionality: str,
    word_locations: pl.Series | None,
    algorithm_kwargs: dict[str, Any] | None,
    fixation_name: str,
    location_column: str,
    aoi_column: str | None,
) -> pl.Series | None:
    """Correct the fixations of a single trial, or return None if the trial is skipped."""
    n_lines = count_text_lines(trial_aois, word_locations)
    if n_lines is not None:
        min_fixations = _min_fixation_count(requested_algorithms, n_lines)
        if fixation_events.height < min_fixations:
            if trial_columns:
                trial_values = {
                    column: fixation_events[column][0] for column in trial_columns
                }
                trial_part = f' for trial {trial_values}'
            else:
                trial_part = ''
            warnings.warn(
                f'Skipping fixation correction{trial_part}: '
                f'{fixation_events.height} fixations are too few for the requested '
                f'algorithms on {n_lines} text lines. The affected fixation '
                'locations stay uncorrected.',
                UserWarning,
                stacklevel=3,
            )
            return None

    return correct_fixation_locations(
        fixation_events, trial_aois, algorithm=algorithm,
        directionality=directionality, word_locations=word_locations,
        algorithm_kwargs=algorithm_kwargs, fixation_name=fixation_name,
        location_column=location_column, aoi_column=aoi_column,
    )


def _preserved_column(
    events: pl.DataFrame,
    column: str,
    corrected_value: pl.Expr,
    dtype: pl.DataType | type[pl.DataType],
) -> pl.Expr:
    """Set corrected_value on corrected rows, preserving any existing column values."""
    if column in events.columns:
        fallback: pl.Expr = pl.col(column)
    else:
        fallback = pl.lit(None, dtype=dtype)
    is_corrected = pl.col('__corrected_location').is_not_null()
    return pl.when(is_corrected).then(corrected_value).otherwise(fallback).alias(column)


def _apply_corrections(
    events: pl.DataFrame,
    indexed_events: pl.DataFrame,
    corrected_indices: list[int],
    corrected_locations: list[pl.Series],
    algo_name: str,
    location_column: str,
) -> pl.DataFrame:
    """Join the corrected locations onto the events and update the location columns."""
    updates = pl.DataFrame({
        '__fixation_correction_index': pl.Series(corrected_indices, dtype=pl.UInt32),
        '__corrected_location': pl.concat(corrected_locations),
    })
    frame = (
        indexed_events
        .join(updates, on='__fixation_correction_index', how='left')
        .sort('__fixation_correction_index')
    )

    x_column, y_column = f'{location_column}_x', f'{location_column}_y'
    is_corrected = pl.col('__corrected_location').is_not_null()
    update_columns = []
    if location_column in events.columns and events[location_column].dtype != pl.Null:
        update_columns.append(
            _preserved_column(
                events, f'{location_column}_original',
                pl.col(location_column), pl.List(pl.Float64),
            ),
        )
        update_columns.append(
            pl.when(is_corrected)
            .then(pl.col('__corrected_location'))
            .otherwise(pl.col(location_column))
            .alias(location_column),
        )
    if x_column in events.columns and y_column in events.columns:
        update_columns.append(
            _preserved_column(events, f'{x_column}_original', pl.col(x_column), pl.Float64),
        )
        update_columns.append(
            _preserved_column(events, f'{y_column}_original', pl.col(y_column), pl.Float64),
        )
        update_columns.append(
            pl.when(is_corrected)
            .then(pl.col('__corrected_location').list.get(0))
            .otherwise(pl.col(x_column))
            .alias(x_column),
        )
        update_columns.append(
            pl.when(is_corrected)
            .then(pl.col('__corrected_location').list.get(1))
            .otherwise(pl.col(y_column))
            .alias(y_column),
        )
    update_columns.append(
        _preserved_column(events, 'correction_algorithm', pl.lit(algo_name), pl.Utf8),
    )

    return (
        frame
        .with_columns(update_columns)
        .drop(['__fixation_correction_index', '__corrected_location'])
    )


def correct_fixations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    trial_columns: list[str] | str | None = None,
    directionality: str = 'left-to-right',
    word_locations: pl.Series | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
    fixation_name: str = 'fixation',
    location_column: str = 'location',
    aoi_column: str | None = None,
) -> pl.DataFrame:
    """Correct fixation locations per trial using the specified drift algorithm.

    The locations of fixation events are replaced with their corrected values. Original
    locations are preserved in a 'location_original' column ('location_x_original' /
    'location_y_original' for split component columns) and the applied algorithm is
    recorded in a 'correction_algorithm' column, which is null for uncorrected rows.

    Trials with too few fixations for the requested algorithms ('cluster' needs at least
    one fixation per text line, 'split' at least three fixations) are skipped with a
    UserWarning: their fixation locations stay uncorrected and their
    'correction_algorithm' entries stay null.

    Parameters
    ----------
    events: pl.DataFrame
        Polars DataFrame containing gaze events.
    aois: pl.DataFrame
        Stimulus AOIs DataFrame. Text line positions are derived from the 'start_y' (or
        'top_left_y') and 'height' columns, with 'height' derived from 'start_y' and
        'end_y' if missing. AOIs are grouped into lines by a 'line_idx' column if
        present, otherwise by their y-coordinate. The DTW-based algorithms additionally
        use the word X coordinate columns 'start_x' and 'end_x', with 'end_x' derived
        from 'start_x' and 'width' if missing.
    algorithm: str | list[str]
        Name of drift algorithm or list of algorithm names. Default is 'wisdom_of_the_crowd'.
        If word X coordinates ('start_x', 'end_x') are not present in aois, 'compare' and 'warp'
        are automatically excluded from the Wisdom of the Crowd ensemble with a UserWarning.
    trial_columns: list[str] | str | None
        Column names identifying trials. Each trial is corrected independently. AOIs are
        filtered on those trial columns that are present in the aois dataframe. If None,
        all events are treated as a single trial. (default: None)
    directionality: str
        Reading direction of the text, either 'left-to-right' or 'right-to-left',
        mirroring the directionality of a text stimulus writing system; 'top-to-bottom'
        is not supported and raises a ValueError. Passed to those algorithms with
        direction-specific processing ('merge', 'segment', 'split'); direction-agnostic
        algorithms ignore it. The 'compare' algorithm does not support right-to-left
        reading and is excluded from ensembles with a UserWarning.
        (default: 'left-to-right')
    word_locations: pl.Series | None
        Series of [x, y] word center coordinates for the DTW-based algorithms 'compare'
        and 'warp'. If None, word locations are derived from the aois dataframe. A
        user-supplied series is reused unchanged for every trial, so with per-trial AOIs
        leave it None to derive the word locations of each trial separately.
        (default: None)
    algorithm_kwargs: dict[str, Any] | None
        Additional tuning parameters passed to underlying drift correction algorithms, e.g.
        ``{'x_thresh': 250.0}``. In ensemble mode, each entry is only passed to those
        candidate algorithms that accept it. Warning: an entry fans out to every candidate
        algorithm whose signature accepts the key, even where defaults and semantics
        differ. For example, ``x_thresh`` is accepted by 'chain' (default 192, chain
        breaking), 'compare' (default 512, line break detection) and 'slice' (default 192,
        run segmentation), so ``{'x_thresh': 250.0}`` reconfigures all three at once.
        (default: None)
    fixation_name: str
        Name of the fixation events to correct. Only events matching this name exactly are
        corrected; unlike :py:meth:`~pymovements.Events.map_to_aois`, no prefix matching
        is applied. If no events match, a UserWarning is emitted and the events dataframe
        is returned unchanged. (default: 'fixation')
    location_column: str
        Name of the events column holding the [x, y] fixation locations, with
        '<location_column>_x' and '<location_column>_y' as the component column
        fallback. The bookkeeping columns preserving the original locations derive
        their '_original' names from this parameter accordingly. (default: 'location')
    aoi_column: str | None
        Name of the aois column holding the AOI content. A frame holding a 'word' column
        next to a content column other than 'word' is treated as character-level and its
        AOIs are aggregated to one location per word. If None, character-level frames are
        recognized by a 'word' column next to one of the conventional character columns
        'char', 'character' or 'char_idx_in_line'. (default: None)

    Returns
    -------
    pl.DataFrame
        Updated events DataFrame with corrected fixation locations.

    Raises
    ------
    ValueError
        If the directionality is invalid, if trial_columns are missing from the events
        dataframe, if no AOIs are found for a trial with fixations to correct, or if the
        fixation events have already been corrected.

    Examples
    --------
    Correcting fixations that drift away from three lines of text with their centers at
    y = 100, 200 and 300 snaps each y-coordinate onto its line center and preserves the
    original locations:

    >>> import polars as pl
    >>> from pymovements.events.correction import correct_fixations
    >>> events = pl.DataFrame({
    ...     'name': ['fixation', 'fixation', 'fixation'],
    ...     'location': [[100.0, 105.0], [110.0, 195.0], [120.0, 302.0]],
    ... })
    >>> aois = pl.DataFrame({
    ...     'start_y': [80.0, 180.0, 280.0],
    ...     'height': [40.0, 40.0, 40.0],
    ... })
    >>> correct_fixations(events, aois, algorithm='attach')
    shape: (3, 4)
    ┌──────────┬────────────────┬───────────────────┬──────────────────────┐
    │ name     ┆ location       ┆ location_original ┆ correction_algorithm │
    │ ---      ┆ ---            ┆ ---               ┆ ---                  │
    │ str      ┆ list[f64]      ┆ list[f64]         ┆ str                  │
    ╞══════════╪════════════════╪═══════════════════╪══════════════════════╡
    │ fixation ┆ [100.0, 100.0] ┆ [100.0, 105.0]    ┆ attach               │
    │ fixation ┆ [110.0, 200.0] ┆ [110.0, 195.0]    ┆ attach               │
    │ fixation ┆ [120.0, 300.0] ┆ [120.0, 302.0]    ┆ attach               │
    └──────────┴────────────────┴───────────────────┴──────────────────────┘
    """
    # Validate eagerly so an invalid directionality raises even without matching fixations.
    is_right_to_left(directionality)

    if isinstance(trial_columns, str):
        trial_columns = [trial_columns]

    if trial_columns is not None:
        missing_columns = [
            column for column in trial_columns if column not in events.columns
        ]
        if missing_columns:
            raise ValueError(
                f'trial columns {missing_columns} are missing from events dataframe.',
            )

    _check_not_already_corrected(events, fixation_name)
    algo_name, requested_algorithms = _algorithm_label(algorithm)

    aois = normalize_aois(aois)
    indexed_events = events.with_row_index('__fixation_correction_index')

    if trial_columns is not None:
        trial_event_frames = indexed_events.partition_by(trial_columns, maintain_order=True)
        aoi_trial_columns = [column for column in trial_columns if column in aois.columns]
    else:
        trial_event_frames = [indexed_events]
        aoi_trial_columns = []

    corrected_indices: list[int] = []
    corrected_locations: list[pl.Series] = []
    matched_fixation_count = 0
    for trial_events in trial_event_frames:
        trial_aois = _trial_aois(aois, trial_events, aoi_trial_columns)

        fixation_events = trial_events.filter(pl.col('name') == fixation_name)
        if fixation_events.height == 0:
            continue
        matched_fixation_count += fixation_events.height

        if aoi_trial_columns and trial_aois.height == 0:
            trial_values = {
                column: trial_events[column][0] for column in aoi_trial_columns
            }
            raise ValueError(f'no AOIs found for trial {trial_values}.')

        corrected_locs = _correct_trial(
            fixation_events, trial_aois, algorithm=algorithm,
            requested_algorithms=requested_algorithms, trial_columns=trial_columns,
            directionality=directionality, word_locations=word_locations,
            algorithm_kwargs=algorithm_kwargs, fixation_name=fixation_name,
            location_column=location_column, aoi_column=aoi_column,
        )
        if corrected_locs is None:
            continue

        corrected_indices.extend(fixation_events['__fixation_correction_index'].to_list())
        corrected_locations.append(corrected_locs)

    if not corrected_indices:
        if events.height > 0 and matched_fixation_count == 0:
            event_names = events['name'].unique().sort().to_list()
            warnings.warn(
                f"No events matched fixation_name '{fixation_name}', so no fixations were "
                f'corrected. Event names present in the events dataframe: {event_names}.',
                UserWarning,
                stacklevel=2,
            )
        return events

    return _apply_corrections(
        events, indexed_events, corrected_indices, corrected_locations, algo_name,
        location_column,
    )
