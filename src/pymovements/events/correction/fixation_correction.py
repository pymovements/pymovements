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

import warnings
from typing import Any

import numpy as np
import polars as pl

import pymovements.events.correction.drift_algorithms as da
from pymovements.events.events import Events


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
    y_col = 'start_y' if 'start_y' in aois.columns else 'top_left_y'

    # Assumes text is vertically centered within each AOI bounding box
    if 'line_idx' in aois.columns:
        return (
            aois.filter(pl.col('line_idx').is_not_null())
            .group_by('line_idx')
            .agg(
                (pl.col(y_col) + pl.col('height') / 2.0)
                .mean()
                .alias('line_center'),
            )
            .sort('line_idx')['line_center']
            .to_list()
        )

    return (
        aois.group_by(y_col)
        .agg(
            (pl.col(y_col) + pl.col('height') / 2.0)
            .mean()
            .alias('line_center'),
        )
        .sort(y_col)['line_center']
        .to_list()
    )


ALL_DRIFT_ALGORITHMS: list[str] = [
    'attach', 'chain', 'cluster', 'compare', 'merge', 'regress',
    'segment', 'slice', 'split', 'stretch', 'warp',
]


def _has_word_x_coords(aois: pl.DataFrame, kwargs: dict[str, Any]) -> bool:
    """Check if word X coordinates are available in kwargs or aois DataFrame."""
    if 'word_XY' in kwargs:
        return True
    return 'start_x' in aois.columns and 'end_x' in aois.columns


def create_corrected_fixations_locations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    **kwargs: Any,
) -> np.ndarray:
    """Correct fixations based on the specified drift algorithm and AOIs.

    Parameters
    ----------
    events: pl.DataFrame
        Gaze events dataframe.
    aois: pl.DataFrame
        AOIs dataframe for line position extraction.
    algorithm: str | list[str]
        Name of drift algorithm or a list of algorithm names to combine via Wisdom of the Crowd
        (WoC) ensemble correction. Default is 'wisdom_of_the_crowd' (or 'woc'), which includes all
        drift algorithms. If word X coordinates ('start_x', 'end_x') are not present in aois,
        'compare' and 'warp' are automatically excluded from the ensemble with a UserWarning.
    **kwargs: Any
        Additional keyword arguments passed to underlying drift correction algorithm.

    Returns
    -------
    np.ndarray
        Array of corrected fixation locations of shape (N, 2).
    """
    fixations = events.filter(pl.col('name').str.starts_with('fixation'))
    if 'location' in fixations.columns and fixations['location'].dtype != pl.Null:
        fixationXY = fixations['location'].to_list()
    elif 'location_x' in fixations.columns and 'location_y' in fixations.columns:
        fixationXY = fixations.select(['location_x', 'location_y']).to_numpy().tolist()
    else:
        raise ValueError('No valid location coordinates found in events dataframe.')

    fixationXY_arr = np.array(fixationXY)

    kwargs_copy = dict(kwargs)
    word_xy_arg = kwargs_copy.pop('word_XY', None)

    if isinstance(algorithm, (list, tuple)):
        if len(algorithm) == 0:
            raise ValueError('At least one algorithm must be provided in the algorithm list.')
        if len(algorithm) == 1:
            return create_corrected_fixations_locations(
                events, aois, algorithm=algorithm[0], **kwargs,
            )
        candidate_algos = []
        has_word_coords = word_xy_arg is not None or _has_word_x_coords(aois, kwargs)
        excluded_algos = []
        for algo in algorithm:
            if algo in {'compare', 'warp'} and not has_word_coords:
                excluded_algos.append(algo)
            else:
                candidate_algos.append(algo)
        if excluded_algos:
            warnings.warn(
                "Word X coordinates ('start_x', 'end_x') are not available in aois. "
                f"Automatically excluding {excluded_algos} from Wisdom of the Crowd ensemble.",
                UserWarning,
                stacklevel=2,
            )
    elif isinstance(algorithm, str):
        if algorithm.lower() in {'wisdom_of_the_crowd', 'woc'}:
            candidate_algos = list(ALL_DRIFT_ALGORITHMS)
            if not (word_xy_arg is not None or _has_word_x_coords(aois, kwargs)):
                warnings.warn(
                    "Word X coordinates ('start_x', 'end_x') are not available in aois. "
                    "Automatically excluding 'compare' and 'warp' from Wisdom of the Crowd "
                    'ensemble.',
                    UserWarning,
                    stacklevel=2,
                )
                candidate_algos = [
                    algo for algo in candidate_algos if algo not in {'compare', 'warp'}
                ]
        else:
            if algorithm in {'compare', 'warp'}:
                if word_xy_arg is not None:
                    target_arg = word_xy_arg
                elif _has_word_x_coords(aois, kwargs):
                    word_x = (aois['start_x'] + aois['end_x']) / 2.0
                    word_y = (aois['start_y'] + aois['end_y']) / 2.0
                    target_arg = np.column_stack([word_x.to_numpy(), word_y.to_numpy()])
                else:
                    raise ValueError(
                        f"Algorithm '{algorithm}' requires word X coordinates ('start_x', 'end_x') "
                        "in aois DataFrame or 'word_XY' keyword argument.",
                    )
            else:
                target_arg = np.array(_get_lines_of_text_from_aois(aois))

            func = getattr(da, algorithm)
            # pylint: disable=too-many-function-args
            return func(fixationXY_arr, target_arg, **kwargs_copy)
    else:
        raise TypeError('algorithm must be a string or a list of strings.')

    candidate_y_list: list[np.ndarray | list[float]] = []
    for candidate_algo in candidate_algos:
        res = create_corrected_fixations_locations(
            events, aois, algorithm=candidate_algo, **kwargs,
        )
        y_vals = res[:, 1] if res.ndim == 2 else res
        candidate_y_list.append(y_vals)
    corrected_y = da.wisdom_of_the_crowd(candidate_y_list)
    return np.column_stack([fixationXY_arr[:, 0], corrected_y])


def add_corrected_fixations(
    events: Events | pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str | list[str] = 'wisdom_of_the_crowd',
    trial_id: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Correct fixations for a trial using specified drift algorithm and append corrected events.

    Parameters
    ----------
    events: Events | pl.DataFrame
        Events object or Polars DataFrame containing gaze events.
    aois: pl.DataFrame
        Stimulus AOIs DataFrame.
    algorithm: str | list[str]
        Name of drift algorithm or list of algorithm names. Default is 'wisdom_of_the_crowd'.
        If word X coordinates ('start_x', 'end_x') are not present in aois, 'compare' and 'warp'
        are automatically excluded from the Wisdom of the Crowd ensemble with a UserWarning.
    trial_id: str | None
        Optional trial identifier to filter events and AOIs. (default: None)
    **kwargs: Any
        Additional keyword arguments passed to underlying drift correction algorithm.

    Returns
    -------
    pl.DataFrame
        Updated events DataFrame with corrected fixations appended as new event rows.
    """
    events_frame: pl.DataFrame = events.frame if isinstance(events, Events) else events

    if trial_id is not None and 'trial' in events_frame.columns:
        trial_events = events_frame.filter(pl.col('trial') == trial_id)
        trial_aois = aois.filter(pl.col('trial') == trial_id)
    else:
        trial_events = events_frame
        trial_aois = aois

    fixation_events = trial_events.filter(pl.col('name').str.starts_with('fixation'))
    if fixation_events.height == 0:
        return events_frame

    corrected_locs = create_corrected_fixations_locations(
        fixation_events, trial_aois, algorithm=algorithm, **kwargs,
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

    corrected_rows = []
    is_1d = corrected_locs.ndim == 1
    for i, row in enumerate(fixation_events.iter_rows(named=True)):
        row_copy = dict(row)
        row_copy['name'] = f'fixation_corrected_{algo_name}'
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

    corrected_df = pl.DataFrame(corrected_rows)
    return pl.concat([events_frame, corrected_df], how='diagonal')
