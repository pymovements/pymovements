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

from typing import Any

import numpy as np
import polars as pl

import pymovements.events.correction.drift_algorithms as da
from pymovements.events.events import Events


def _get_lines_of_text_from_aois(aois: pl.DataFrame) -> list[float]:
    """Calculate line positions of text based on AOIs.

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
    heights = list(set(aois['height']))
    height = heights[0]
    return [float(line + height / 2.0) for line in sorted(set(aois[y_col]))]


def create_corrected_fixations_locations(
    events: pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str = 'wisdom_of_the_crowd',
    **kwargs: Any,
) -> np.ndarray:
    """Correct fixations based on the specified drift algorithm and AOIs.

    Parameters
    ----------
    events: pl.DataFrame
        Gaze events dataframe.
    aois: pl.DataFrame
        AOIs dataframe for line position extraction.
    algorithm: str
        Name of drift algorithm:
        - 'wisdom_of_the_crowd' or 'woc' (default)
        - 'attach'
        - 'chain'
        - 'cluster'
        - 'compare'
        - 'merge'
        - 'regress'
        - 'segment'
        - 'slice'
        - 'split'
        - 'stretch'
        - 'warp'
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

    if algorithm.lower() in {'wisdom_of_the_crowd', 'woc'}:
        woc_algos = (
            'attach', 'chain', 'cluster', 'merge', 'regress',
            'segment', 'slice', 'split', 'stretch',
        )
        candidate_y_list: list[np.ndarray | list[float]] = []
        for candidate_algo in woc_algos:
            res = create_corrected_fixations_locations(
                events, aois, algorithm=candidate_algo, **kwargs,
            )
            y_vals = res[:, 1] if res.ndim == 2 else res
            candidate_y_list.append(y_vals)
        corrected_y = da.wisdom_of_the_crowd(candidate_y_list)
        return np.column_stack([fixationXY_arr[:, 0], corrected_y])

    if algorithm in {'compare', 'warp'} and 'word_XY' not in kwargs:
        word_x = (aois['start_x'] + aois['end_x']) / 2.0
        word_y = (aois['start_y'] + aois['end_y']) / 2.0
        target_arg = np.column_stack([word_x.to_numpy(), word_y.to_numpy()])
    else:
        target_arg = np.array(_get_lines_of_text_from_aois(aois))

    func = getattr(da, algorithm)
    # pylint: disable=too-many-function-args
    return func(fixationXY_arr, target_arg, **kwargs)


def add_corrected_fixations(
    events: Events | pl.DataFrame,
    aois: pl.DataFrame,
    algorithm: str = 'wisdom_of_the_crowd',
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
    algorithm: str
        Name of drift algorithm. (default: 'wisdom_of_the_crowd')
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

    corrected_rows = []
    is_1d = corrected_locs.ndim == 1
    for i, row in enumerate(fixation_events.iter_rows(named=True)):
        row_copy = dict(row)
        row_copy['name'] = f'fixation_corrected_{algorithm}'
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
