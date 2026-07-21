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

References & Citations
----------------------
- **Abdulin & Komogortsev (2015)**: Abdulin, E. R., & Komogortsev, O. V. (2015).
  Person verification via eye movement-driven text reading model.
  In *2015 IEEE 7th International Conference on Biometrics Theory, Applications
  and Systems (BTAS)* (pp. 1-8). IEEE.
  https://doi.org/10.1109/BTAS.2015.7358786
- **Al Madi (2025)**: Al Madi, N. (2025).
  Identifying Eye Movement Patterns for An Adaptive Approach to Correcting
  Eye Tracking Data in Reading Tasks.
  *Proceedings of the ACM on Human-Computer Interaction*, 9(PACMHCI), 1-16.
  https://osf.io/khrqp/overview
- **Carr et al. (2022)**: Carr, J. W., Pescuma, V. N., Furlan, M., Ktori, M.,
  & Crepaldi, D. (2022).
  Algorithms for the automated correction of vertical drift in eye-tracking data.
  *Behavior Research Methods*, 54(1), 287-310.
  https://doi.org/10.3758/s13428-021-01554-0
- **Cohen (2013)**: Cohen, A. L. (2013).
  Software for the automatic correction of recorded eye fixation locations
  in reading experiments.
  *Behavior Research Methods*, 45(3), 679-683.
  https://doi.org/10.3758/s13428-012-0280-3
- **Glandorf & Schroeder (2021)**: Glandorf, D., & Schroeder, S. (2021).
  Slice: an algorithm to assign fixations in multi-line texts.
  *Procedia Computer Science*, 192, 2971-2979.
  https://doi.org/10.1016/j.procs.2021.09.069
- **Lima Sanches et al. (2015)**: Lima Sanches, C., Kise, K., & Augereau, O. (2015).
  Eye gaze and text line matching for reading analysis.
  In *Proceedings of the 2015 ACM International Joint Conference on Pervasive
  and Ubiquitous Computing and Proceedings of the 2015 ACM International Symposium
  on Wearable Computers (UbiComp '15)* (pp. 1227-1233).
  https://doi.org/10.1145/2800835.2807936
- **Lohmeier (2015)**: Lohmeier, S. (2015).
  *Experimental evaluation and modelling of the comprehension of indirect anaphors
  in a programming language* (Master's thesis). Technische Universität Berlin.
- **Mercier et al. (2024a)**: Mercier, T. M., Budka, M., Vasilev, M. R.,
  Kirkby, J. A., Angele, B., & Slattery, T. J. (2024).
  Dual input stream transformer for vertical drift correction in eye-tracking reading data.
  *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 46(12),
  8715-8726. https://doi.org/10.1109/TPAMI.2024.3430686
- **Mercier et al. (2024b)**: Mercier, T. M., Budka, M., Angele, B.,
  Vasilev, M. R., Slattery, T. J., & Kirkby, J. A. (2024).
  GazeGenie: Enhancing Multi-Line Reading Research with an Innovative User-Friendly Tool.
  *arXiv preprint arXiv:2410.11873*.
  https://doi.org/10.48550/arXiv.2410.11873
- **Špakov et al. (2019)**: Špakov, O., Istance, H., Hyrskykari, A.,
  Siirtola, H., & Räihä, K.-J. (2019).
  Improving the performance of eye trackers with limited spatial accuracy and low
  sampling rates for reading analysis by heuristic fixation-to-word mapping.
  *Behavior Research Methods*, 51(6), 2661-2687.
  https://doi.org/10.3758/s13428-018-1120-x

Supported Drift Correction Algorithms
-------------------------------------
- **wisdom_of_the_crowd** (or **woc**) : *(Default)* Ensemble correction method combining
  predictions across multiple algorithms via majority voting per fixation (Mercier et al., 2024b).
- **attach** : Snaps each fixation to the vertically closest line of text (Carr et al., 2022).
- **chain** : Groups fixations into reading chains based on spatio-temporal distance thresholds
  and aligns each chain to line centers (Carr et al., 2022).
- **cluster** : Uses K-Means clustering to group fixation Y-coordinates into clusters matching
  text lines (Carr et al., 2022).
- **compare** : Matches fixation sequences to candidate text line paths using Dynamic Time
  Warping (DTW) (Lima Sanches et al., 2015; Carr et al., 2022).
- **merge** : Forms progressive sequences and iteratively merges sequences belonging to the same
  text line (Špakov et al., 2019; Carr et al., 2022).
- **regress** : Fits a linear regression model (slope, offset, std) to estimate line assignments
  (Cohen, 2013; Carr et al., 2022).
- **segment** : Segments fixations into line subsequences using return sweep identification
  (Abdulin & Komogortsev, 2015; Carr et al., 2022).
- **slice** : Slices fixation sequence into proto-lines based on vertical drift thresholds
  (Glandorf & Schroeder, 2021).
- **split** : Splits fixations into line subsequences using K-Means return sweep identification
  (Carr et al., 2022).
- **stretch** : Fits scale and offset parameters to stretch or compress fixations onto line
  centers (Lohmeier, 2015; Carr et al., 2022).
- **warp** : Dynamic Time Warping (DTW) alignment between fixations and word centroids
  (Carr et al., 2022).
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
                events, aois, algorithm=candidate_algo, **kwargs
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
        fixation_events, trial_aois, algorithm=algorithm, **kwargs
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
