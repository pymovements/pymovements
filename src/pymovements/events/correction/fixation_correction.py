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
from typing import Any

import numpy as np
import polars as pl

import pymovements.events.correction.drift_algorithms as da


def _with_line_centers(aois: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Annotate each AOI row with the y-center of the text line it belongs to.

    Lines are identified by 'line_idx' if present, otherwise by the top y-coordinate.
    Assumes that the line of text is vertically centered within each AOI bounding box.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to annotate.

    Returns
    -------
    tuple[pl.DataFrame, str]
        AOIs dataframe with an added 'line_center' column in original row order, and the
        name of the column identifying lines.
    """
    y_col = 'start_y' if 'start_y' in aois.columns else 'top_left_y'
    line_key = 'line_idx' if 'line_idx' in aois.columns else y_col

    aois_with_line_centers = (
        aois.filter(pl.col(line_key).is_not_null())
        .with_columns(
            (pl.col(y_col) + pl.col('height') / 2.0)
            .mean()
            .over(line_key)
            .alias('line_center'),
        )
    )
    return aois_with_line_centers, line_key


def _get_lines_of_text_from_aois(aois: pl.DataFrame) -> list[float]:
    """Calculate line positions of text based on AOIs.

    Assumes that the line of text is vertically centered within each AOI.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to calculate line positions from.

    Returns
    -------
    list[float]
        Line center y-coordinates of the text.
    """
    aois_with_line_centers, line_key = _with_line_centers(aois)
    return (
        aois_with_line_centers
        .unique(subset=line_key)
        .sort(line_key)['line_center']
        .to_list()
    )


def _get_word_xy_from_aois(aois: pl.DataFrame) -> np.ndarray:
    """Calculate word center positions from AOIs for DTW-based drift algorithms.

    Following the word_XY convention of Carr et al. :cite:p:`Carr2022`, the y-coordinate of
    each word is the center of the text line the word belongs to, not the center of the
    word's own bounding box. This keeps the y-coordinates identical to the line positions
    returned by _get_lines_of_text_from_aois.

    Parameters
    ----------
    aois: pl.DataFrame
        AOIs dataframe to calculate word positions from.

    Returns
    -------
    np.ndarray
        Array of shape (M, 2) with word center x-coordinates and line center y-coordinates.
    """
    aois_with_line_centers, _ = _with_line_centers(aois)
    word_x = (aois_with_line_centers['start_x'] + aois_with_line_centers['end_x']) / 2.0
    return np.column_stack([
        word_x.to_numpy(), aois_with_line_centers['line_center'].to_numpy(),
    ])


ALL_DRIFT_ALGORITHMS: list[str] = [
    'attach', 'chain', 'cluster', 'compare', 'merge', 'regress',
    'segment', 'slice', 'split', 'stretch', 'warp',
]


def _has_word_x_coords(aois: pl.DataFrame) -> bool:
    """Check if word X coordinates are available in the aois DataFrame."""
    return 'start_x' in aois.columns and 'end_x' in aois.columns


def create_corrected_fixations_locations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    text_right_to_left: bool = False,
    word_XY: np.ndarray | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
    fixation_name: str = 'fixation',
) -> np.ndarray:
    """Correct fixations based on the specified drift algorithm and AOIs.

    Parameters
    ----------
    events: pl.DataFrame
        Gaze events dataframe.
    aois: pl.DataFrame
        AOIs dataframe for line position extraction.
    algorithm: str | list[str]
        Name of a single drift algorithm or a list of algorithm names to combine via Wisdom of
        the Crowd (WoC) ensemble correction. Default is 'wisdom_of_the_crowd' (or 'woc'), which
        includes all drift algorithms. If word X coordinates ('start_x', 'end_x') are missing in
        aois, 'compare' and 'warp' are automatically excluded from the ensemble with a UserWarning.
    text_right_to_left: bool
        Whether the text is read from right to left. Passed to those algorithms with
        direction-specific processing ('merge', 'segment', 'split'); direction-agnostic
        algorithms ignore it. Note that 'compare' currently assumes left-to-right line
        breaks. (default: False)
    word_XY: np.ndarray | None
        Word center coordinates of shape (M, 2) for the DTW-based algorithms 'compare' and
        'warp'. If None, word coordinates are derived from the aois dataframe. Following
        Carr et al., y-coordinates should be the text line centers. (default: None)
    algorithm_kwargs: dict[str, Any] | None
        Additional tuning parameters passed to underlying drift correction algorithms, e.g.
        ``{'x_thresh': 250.0}``. In ensemble mode, each entry is only passed to those
        candidate algorithms that accept it; a ValueError is raised if an entry is accepted
        by none of the candidate algorithms. (default: None)
    fixation_name: str
        Name of the fixation events to correct. Only events matching this name exactly are
        corrected. (default: 'fixation')

    Returns
    -------
    np.ndarray
        Array of corrected fixation locations of shape (N, 2).

    Raises
    ------
    ValueError
        If the algorithm name is unknown, an algorithm_kwargs entry is accepted by no
        candidate algorithm, or required coordinate data is missing.
    TypeError
        If algorithm is neither a string nor a list of strings.
    """
    if algorithm_kwargs is None:
        algorithm_kwargs = {}
    for reserved_key in ('text_right_to_left', 'word_XY'):
        if reserved_key in algorithm_kwargs:
            raise ValueError(
                f"'{reserved_key}' must be passed as an explicit parameter, "
                'not via algorithm_kwargs.',
            )

    # Match the event name exactly so that already corrected fixation events are not
    # corrected again when running on a previously returned events dataframe.
    fixations = events.filter(pl.col('name') == fixation_name)
    if 'location' in fixations.columns and fixations['location'].dtype != pl.Null:
        fixationXY = fixations['location'].to_list()
    elif 'location_x' in fixations.columns and 'location_y' in fixations.columns:
        fixationXY = fixations.select(['location_x', 'location_y']).to_numpy().tolist()
    else:
        raise ValueError('No valid location coordinates found in events dataframe.')

    fixationXY_arr = np.array(fixationXY)

    if isinstance(algorithm, (list, tuple)):
        if len(algorithm) == 0:
            raise ValueError('At least one algorithm must be provided in the algorithm list.')
        if len(algorithm) == 1:
            return create_corrected_fixations_locations(
                events, aois, algorithm=algorithm[0], text_right_to_left=text_right_to_left,
                word_XY=word_XY, algorithm_kwargs=algorithm_kwargs, fixation_name=fixation_name,
            )
        unknown_algos = [algo for algo in algorithm if algo not in ALL_DRIFT_ALGORITHMS]
        if unknown_algos:
            raise ValueError(
                f'Unknown drift algorithms {unknown_algos}. '
                f'Valid algorithms are: {ALL_DRIFT_ALGORITHMS}',
            )
        candidate_algos = []
        has_word_coords = word_XY is not None or _has_word_x_coords(aois)
        excluded_algos = []
        for algo in algorithm:
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
                stacklevel=2,
            )
    elif isinstance(algorithm, str):
        if algorithm.lower() in {'wisdom_of_the_crowd', 'woc'}:
            candidate_algos = list(ALL_DRIFT_ALGORITHMS)
            if not (word_XY is not None or _has_word_x_coords(aois)):
                warnings.warn(
                    "Word X coordinates ('start_x', 'end_x') are missing from aois DataFrame. "
                    "As a consequence, 'compare' and 'warp' algorithms are excluded from "
                    'Wisdom of the Crowd ensemble.',
                    UserWarning,
                    stacklevel=2,
                )
                candidate_algos = [
                    algo for algo in candidate_algos if algo not in {'compare', 'warp'}
                ]
        else:
            if algorithm not in ALL_DRIFT_ALGORITHMS:
                raise ValueError(
                    f"Unknown drift algorithm '{algorithm}'. "
                    f'Valid algorithms are: {ALL_DRIFT_ALGORITHMS}',
                )
            if algorithm in {'compare', 'warp'}:
                if word_XY is not None:
                    target_arg = word_XY
                elif _has_word_x_coords(aois):
                    target_arg = _get_word_xy_from_aois(aois)
                else:
                    raise ValueError(
                        f"Algorithm '{algorithm}' requires word X coordinates ('start_x', 'end_x') "
                        "in aois DataFrame or the 'word_XY' parameter.",
                    )
            else:
                target_arg = np.array(_get_lines_of_text_from_aois(aois))

            func = getattr(da, algorithm)
            call_kwargs = dict(algorithm_kwargs)
            if 'text_right_to_left' in inspect.signature(func).parameters:
                call_kwargs['text_right_to_left'] = text_right_to_left
            # pylint: disable=too-many-function-args
            return func(fixationXY_arr, target_arg, **call_kwargs)
    else:
        raise TypeError('algorithm must be a string or a list of strings.')

    # Vote on line indices rather than raw y-coordinates so that candidate algorithms cannot
    # split votes through differing float representations of the same text line.
    has_line_info = (
        ('start_y' in aois.columns or 'top_left_y' in aois.columns)
        and 'height' in aois.columns
    )
    if has_line_info:
        line_Y = np.array(_get_lines_of_text_from_aois(aois))
    else:
        line_Y = np.unique(np.asarray(word_XY)[:, 1])

    # Route tuning parameters to those candidate algorithms that accept them, so that
    # algorithm-specific parameters do not break the other algorithms in the ensemble.
    candidate_params = {
        candidate_algo: set(inspect.signature(getattr(da, candidate_algo)).parameters)
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

    candidate_line_assignments = []
    for candidate_algo in candidate_algos:
        algo_kwargs = {
            key: value for key, value in algorithm_kwargs.items()
            if key in candidate_params[candidate_algo]
        }
        res = create_corrected_fixations_locations(
            events, aois, algorithm=candidate_algo, text_right_to_left=text_right_to_left,
            word_XY=word_XY, algorithm_kwargs=algo_kwargs, fixation_name=fixation_name,
        )
        y_vals = np.asarray(res[:, 1] if res.ndim == 2 else res)
        candidate_line_assignments.append(
            np.argmin(np.abs(line_Y[:, np.newaxis] - y_vals[np.newaxis, :]), axis=0),
        )
    corrected_line_indices = np.asarray(
        da.wisdom_of_the_crowd(candidate_line_assignments), dtype=int,
    )
    return np.column_stack([fixationXY_arr[:, 0], line_Y[corrected_line_indices]])


def add_corrected_fixations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    trial_columns: list[str] | str | None = None,
    text_right_to_left: bool = False,
    word_XY: np.ndarray | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
    fixation_name: str = 'fixation',
) -> pl.DataFrame:
    """Correct fixations per trial using specified drift algorithm and append corrected events.

    Parameters
    ----------
    events: pl.DataFrame
        Polars DataFrame containing gaze events.
    aois: pl.DataFrame
        Stimulus AOIs DataFrame.
    algorithm: str | list[str]
        Name of drift algorithm or list of algorithm names. Default is 'wisdom_of_the_crowd'.
        If word X coordinates ('start_x', 'end_x') are not present in aois, 'compare' and 'warp'
        are automatically excluded from the Wisdom of the Crowd ensemble with a UserWarning.
    trial_columns: list[str] | str | None
        Column names identifying trials. Each trial is corrected independently. AOIs are
        filtered on those trial columns that are present in the aois dataframe. If None,
        all events are treated as a single trial. (default: None)
    text_right_to_left: bool
        Whether the text is read from right to left. Passed to those algorithms with
        direction-specific processing ('merge', 'segment', 'split'); direction-agnostic
        algorithms ignore it. (default: False)
    word_XY: np.ndarray | None
        Word center coordinates of shape (M, 2) for the DTW-based algorithms 'compare' and
        'warp'. If None, word coordinates are derived from the aois dataframe. (default: None)
    algorithm_kwargs: dict[str, Any] | None
        Additional tuning parameters passed to underlying drift correction algorithms, e.g.
        ``{'x_thresh': 250.0}``. In ensemble mode, each entry is only passed to those
        candidate algorithms that accept it. (default: None)
    fixation_name: str
        Name of the fixation events to correct. Only events matching this name exactly are
        corrected, so previously corrected fixation events are not corrected again when
        running on a returned events dataframe. (default: 'fixation')

    Returns
    -------
    pl.DataFrame
        Updated events DataFrame with corrected fixations appended as new event rows.

    Raises
    ------
    ValueError
        If trial_columns are missing from the events dataframe.
    """
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

    if isinstance(algorithm, (list, tuple)):
        if len(algorithm) > 1:
            algo_name = 'wisdom_of_the_crowd'
        else:
            algo_name = algorithm[0]
    elif isinstance(algorithm, str) and algorithm.lower() not in {'wisdom_of_the_crowd', 'woc'}:
        algo_name = algorithm
    else:
        algo_name = 'wisdom_of_the_crowd'

    if trial_columns is not None:
        trial_event_frames = events.partition_by(trial_columns, maintain_order=True)
        aoi_trial_columns = [column for column in trial_columns if column in aois.columns]
    else:
        trial_event_frames = [events]
        aoi_trial_columns = []

    corrected_rows = []
    for trial_events in trial_event_frames:
        if aoi_trial_columns:
            # Each partition holds a single combination of trial column values.
            trial_aois = aois.filter(
                pl.all_horizontal([
                    pl.col(column).eq_missing(pl.lit(trial_events[column][0]))
                    for column in aoi_trial_columns
                ]),
            )
        else:
            trial_aois = aois

        fixation_events = trial_events.filter(pl.col('name') == fixation_name)
        if fixation_events.height == 0:
            continue

        corrected_locs = create_corrected_fixations_locations(
            fixation_events, trial_aois, algorithm=algorithm,
            text_right_to_left=text_right_to_left, word_XY=word_XY,
            algorithm_kwargs=algorithm_kwargs, fixation_name=fixation_name,
        )

        is_1d = corrected_locs.ndim == 1
        for i, row in enumerate(fixation_events.iter_rows(named=True)):
            row_copy = dict(row)
            row_copy['name'] = f'{fixation_name}_corrected_{algo_name}'
            if is_1d:
                orig_x = row.get('location_x', row.get('location', [0.0, 0.0])[0])
                loc_corr = [float(orig_x), float(corrected_locs[i])]
            else:
                loc_corr = [float(corrected_locs[i][0]), float(corrected_locs[i][1])]
            row_copy['location'] = loc_corr
            if 'location_x' in row_copy:
                row_copy['location_x'] = loc_corr[0]
            if 'location_y' in row_copy:
                row_copy['location_y'] = loc_corr[1]
            corrected_rows.append(row_copy)

    if not corrected_rows:
        return events

    corrected_df = pl.DataFrame(corrected_rows)
    return pl.concat([events, corrected_df], how='diagonal')
