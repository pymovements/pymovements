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
"""Algorithms for vertical drift correction in gaze data recorded during reading tasks.

References & Citations:
- Abdulin, E. R., & Komogortsev, O. V. (2015). Person verification via eye movement-driven
  text reading model. In 2015 IEEE 7th International Conference on Biometrics Theory,
  Applications and Systems (BTAS) (pp. 1-8). IEEE. https://doi.org/10.1109/BTAS.2015.7358786
- Al Madi, N. (2025). Identifying Eye Movement Patterns for An Adaptive Approach to Correcting
  Eye Tracking Data in Reading Tasks. Proceedings of the ACM on Human-Computer Interaction,
  9(PACMHCI), 1-16. https://osf.io/khrqp/overview
- Carr, J. W., Pescuma, V. N., Furlan, M., Ktori, M., & Crepaldi, D. (2022). Algorithms for
  the automated correction of vertical drift in eye-tracking data. Behavior Research Methods,
  54(1), 287-310. https://doi.org/10.3758/s13428-021-01554-0
- Cohen, A. L. (2013). Software for the automatic correction of recorded eye fixation
  locations in reading experiments. Behavior Research Methods, 45(3), 679-683.
  https://doi.org/10.3758/s13428-012-0280-3
- Glandorf, D., & Schroeder, S. (2021). Slice: an algorithm to assign fixations in multi-line
  texts. Procedia Computer Science, 192, 2971-2979. https://doi.org/10.1016/j.procs.2021.09.069
- Lima Sanches, C., Kise, K., & Augereau, O. (2015). Eye gaze and text line matching for reading
  analysis. In Proceedings of the 2015 ACM International Joint Conference on Pervasive and
  Ubiquitous Computing and Proceedings of the 2015 ACM International Symposium on Wearable
  Computers (UbiComp '15) (pp. 1227-1233). https://doi.org/10.1145/2800835.2807936
- Lohmeier, S. (2015). Experimental evaluation and modelling of the comprehension of indirect
  anaphors in a programming language (Master's thesis). Technische Universität Berlin.
- Mercier, T. M., Budka, M., Vasilev, M. R., Kirkby, J. A., Angele, B., & Slattery, T. J. (2024).
  Dual input stream transformer for vertical drift correction in eye-tracking reading data.
  IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(12), 8715-8726.
  https://doi.org/10.1109/TPAMI.2024.3430686
- Mercier, T. M., Budka, M., Angele, B., Vasilev, M. R., Slattery, T. J., & Kirkby, J. A. (2024).
  GazeGenie: Enhancing Multi-Line Reading Research with an Innovative User-Friendly Tool.
  arXiv preprint arXiv:2410.11873. https://doi.org/10.48550/arXiv.2410.11873
- Špakov, O., Istance, H., Hyrskykari, A., Siirtola, H., & Räihä, K.-J.
  (2019). Improving the
  performance of eye trackers with limited spatial accuracy and low sampling rates for reading
  analysis by heuristic fixation-to-word mapping. Behavior Research Methods, 51(6), 2661-2687.
  https://doi.org/10.3758/s13428-018-1120-x
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.cluster import KMeans

######################################################################
# ATTACH
######################################################################


def attach(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
) -> np.ndarray:
    """Attach each fixation to the vertically closest text line center.

    Reference: Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2), where column 0 is X and column 1 is Y.
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    for fixation_i in range(n):
        line_i = np.argmin(abs(line_Y - fixation_XY[fixation_i, 1]))
        fixation_XY[fixation_i, 1] = line_Y[line_i]
    return fixation_XY


######################################################################
# CHAIN
######################################################################


def chain(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    x_thresh: float = 192,
    y_thresh: float = 32,
) -> np.ndarray:
    """Group fixations into reading chains based on distance thresholds and align to lines.

    Reference: Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    x_thresh: float
        Horizontal distance threshold to break a chain. (default: 192)
    y_thresh: float
        Vertical distance threshold to break a chain. (default: 32)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    dist_X = abs(np.diff(fixation_XY[:, 0]))
    dist_Y = abs(np.diff(fixation_XY[:, 1]))
    end_chain_indices: list[int] = list(
        (np.where(np.logical_or(dist_X > x_thresh, dist_Y > y_thresh))[0] + 1).tolist(),
    )
    end_chain_indices.append(n)
    start_of_chain = 0
    for end_of_chain in end_chain_indices:
        mean_y = np.mean(fixation_XY[start_of_chain:end_of_chain, 1])
        line_i = np.argmin(abs(line_Y - mean_y))
        fixation_XY[start_of_chain:end_of_chain, 1] = line_Y[line_i]
        start_of_chain = end_of_chain
    return fixation_XY


######################################################################
# CLUSTER
######################################################################


def cluster(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
) -> np.ndarray:
    """Cluster Y-coordinates into clusters matching text lines using K-Means.

    Reference: Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    m = len(line_Y)
    fixation_Y = fixation_XY[:, 1].reshape(-1, 1)
    clusters = KMeans(m, n_init=100, max_iter=300).fit_predict(fixation_Y)
    centers = [fixation_Y[clusters == i].mean() for i in range(m)]
    ordered_cluster_indices = np.argsort(centers)
    for fixation_i, cluster_i in enumerate(clusters):
        line_i = np.where(ordered_cluster_indices == cluster_i)[0][0]
        fixation_XY[fixation_i, 1] = line_Y[line_i]
    return fixation_XY


######################################################################
# COMPARE
######################################################################


def compare(
    fixation_XY: np.ndarray,
    word_XY: np.ndarray,
    x_thresh: float = 512,
    n_nearest_lines: int = 3,
) -> np.ndarray:
    """Match fixation lines to candidate text lines using Dynamic Time Warping (DTW).

    Reference: Lima Sanches et al. (2015); Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    word_XY: np.ndarray
        Word coordinates array of shape (M, 2).
    x_thresh: float
        Threshold for detecting line breaks. (default: 512)
    n_nearest_lines: int
        Number of candidate nearest lines to evaluate with DTW. (default: 3)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    word_XY = np.array(word_XY)
    line_Y = np.unique(word_XY[:, 1])
    n = len(fixation_XY)
    diff_X = np.diff(fixation_XY[:, 0])
    end_line_indices: list[int] = list((np.where(diff_X < -x_thresh)[0] + 1).tolist())
    end_line_indices.append(n)
    start_of_line = 0
    for end_of_line in end_line_indices:
        gaze_line = fixation_XY[start_of_line:end_of_line]
        mean_y = np.mean(gaze_line[:, 1])
        lines_ordered_by_proximity = np.argsort(abs(line_Y - mean_y))
        nearest_line_I = lines_ordered_by_proximity[:n_nearest_lines]
        line_costs = np.zeros(n_nearest_lines)
        for candidate_i in range(n_nearest_lines):
            candidate_line_i = nearest_line_I[candidate_i]
            text_line = word_XY[word_XY[:, 1] == line_Y[candidate_line_i]]
            dtw_cost, _ = dynamic_time_warping(gaze_line[:, 0:1], text_line[:, 0:1])
            line_costs[candidate_i] = dtw_cost
        line_i = nearest_line_I[np.argmin(line_costs)]
        fixation_XY[start_of_line:end_of_line, 1] = line_Y[line_i]
        start_of_line = end_of_line
    return fixation_XY


######################################################################
# MERGE
######################################################################

phases = [
    {'min_i': 3, 'min_j': 3, 'no_constraints': False},  # Phase 1
    {'min_i': 1, 'min_j': 3, 'no_constraints': False},  # Phase 2
    {'min_i': 1, 'min_j': 1, 'no_constraints': False},  # Phase 3
    {'min_i': 1, 'min_j': 1, 'no_constraints': True},   # Phase 4
]


# pylint: disable=too-many-nested-blocks
def merge(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    y_thresh: float = 32,
    g_thresh: float = 0.1,
    e_thresh: float = 20,
    text_right_to_left: bool = False,
) -> np.ndarray:
    """Form progressive sequences and iteratively merge sequences belonging to the same text line.

    Reference: Špakov et al. (2019); Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    y_thresh: float
        Vertical distance threshold for sequence splitting. (default: 32)
    g_thresh: float
        Gradient constraint for sequence merging. (default: 0.1)
    e_thresh: float
        Error constraint for sequence merging. (default: 20)
    text_right_to_left: bool
        If True, adjusts return sweep detection for Right-to-Left reading scripts. (default: False)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    m = len(line_Y)
    diff_X = np.diff(fixation_XY[:, 0])
    dist_Y = abs(np.diff(fixation_XY[:, 1]))

    if text_right_to_left:
        sequence_boundaries = list(
            (np.where(np.logical_or(diff_X > 0, dist_Y > y_thresh))[0] + 1).tolist(),
        )
    else:
        sequence_boundaries = list(
            (np.where(np.logical_or(diff_X < 0, dist_Y > y_thresh))[0] + 1).tolist(),
        )

    sequence_starts = [0] + sequence_boundaries
    sequence_ends = sequence_boundaries + [n]
    sequences = [
        list(range(start, end)) for start, end in zip(sequence_starts, sequence_ends)
    ]

    for phase in phases:
        while len(sequences) > m:
            best_merger = None
            best_error = np.inf
            for i in range(len(sequences) - 1):
                if len(sequences[i]) < phase['min_i']:
                    continue
                for j in range(i + 1, len(sequences)):
                    if len(sequences[j]) < phase['min_j']:
                        continue
                    candidate_XY = fixation_XY[sequences[i] + sequences[j]]
                    gradient, intercept = np.polyfit(
                        candidate_XY[:, 0], candidate_XY[:, 1], 1,
                    )
                    residuals = candidate_XY[:, 1] - (
                        gradient * candidate_XY[:, 0] + intercept
                    )
                    error = np.sqrt(sum(residuals**2) / len(candidate_XY))
                    if phase['no_constraints'] or (
                        abs(gradient) < g_thresh and error < e_thresh
                    ):
                        if error < best_error:
                            best_merger = (i, j)
                            best_error = error
            if best_merger is None:
                break
            merge_i, merge_j = best_merger
            merged_sequence = sequences[merge_i] + sequences[merge_j]
            sequences.append(merged_sequence)
            del sequences[merge_j], sequences[merge_i]
    mean_Y = [fixation_XY[sequence, 1].mean() for sequence in sequences]
    ordered_sequence_indices = np.argsort(mean_Y)
    for line_i, sequence_i in enumerate(ordered_sequence_indices):
        fixation_XY[sequences[sequence_i], 1] = line_Y[line_i]
    return fixation_XY


######################################################################
# REGRESS
######################################################################


def regress(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    k_bounds: tuple[float, float] = (-0.1, 0.1),
    o_bounds: tuple[float, float] = (-50, 50),
    s_bounds: tuple[float, float] = (1, 20),
) -> np.ndarray:
    """Fit linear regression parameters (slope, offset, std) to align fixations to lines of text.

    Reference: Cohen (2013); Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    k_bounds: tuple[float, float]
        Slope bounds. (default: (-0.1, 0.1))
    o_bounds: tuple[float, float]
        Offset bounds. (default: (-50, 50))
    s_bounds: tuple[float, float]
        Standard deviation bounds. (default: (1, 20))

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    m = len(line_Y)

    def fit_lines(params: np.ndarray, return_line_assignments: bool = False) -> np.ndarray | float:
        k = k_bounds[0] + (k_bounds[1] - k_bounds[0]) * norm.cdf(params[0])
        o = o_bounds[0] + (o_bounds[1] - o_bounds[0]) * norm.cdf(params[1])
        s = s_bounds[0] + (s_bounds[1] - s_bounds[0]) * norm.cdf(params[2])
        predicted_Y_from_slope = fixation_XY[:, 0] * k
        line_Y_plus_offset = line_Y + o
        density = np.zeros((n, m))
        for line_i in range(m):
            fit_Y = predicted_Y_from_slope + line_Y_plus_offset[line_i]
            density[:, line_i] = norm.logpdf(fixation_XY[:, 1], fit_Y, s)
        if return_line_assignments:
            return density.argmax(axis=1)
        return float(-sum(density.max(axis=1)))

    best_fit = minimize(fit_lines, [0, 0, 0])
    line_assignments = fit_lines(best_fit.x, True)
    if isinstance(line_assignments, np.ndarray):
        for fixation_i, line_i in enumerate(line_assignments):
            fixation_XY[fixation_i, 1] = line_Y[line_i]
    return fixation_XY


######################################################################
# SEGMENT
######################################################################


def segment(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    text_right_to_left: bool = False,
) -> np.ndarray:
    """Segment fixations into m line subsequences using return sweeps.

    Reference: Abdulin & Komogortsev (2015); Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    text_right_to_left: bool
        If True, identifies return sweeps for Right-to-Left reading scripts. (default: False)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    m = len(line_Y)
    diff_X = np.diff(fixation_XY[:, 0])
    saccades_ordered_by_length = np.argsort(diff_X)

    if text_right_to_left:
        line_change_indices = saccades_ordered_by_length[-(m - 1):]
    else:
        line_change_indices = saccades_ordered_by_length[: m - 1]

    current_line_i = 0
    for fixation_i in range(n):
        fixation_XY[fixation_i, 1] = line_Y[current_line_i]
        if fixation_i in line_change_indices:
            current_line_i += 1
    return fixation_XY


######################################################################
# SPLIT
######################################################################


def split(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    text_right_to_left: bool = False,
) -> np.ndarray:
    """Split fixation sequence into line subsequences using K-Means return sweep identification.

    Reference: Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    text_right_to_left: bool
        If True, identifies return sweeps for Right-to-Left reading scripts. (default: False)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    diff_X = np.diff(fixation_XY[:, 0])
    clusters = KMeans(2, n_init=10, max_iter=300).fit_predict(diff_X.reshape(-1, 1))
    centers = [diff_X[clusters == 0].mean(), diff_X[clusters == 1].mean()]

    sweep_marker = np.argmax(centers) if text_right_to_left else np.argmin(centers)

    end_line_indices: list[int] = list((np.where(clusters == sweep_marker)[0] + 1).tolist())
    end_line_indices.append(n)
    start_of_line = 0
    for end_of_line in end_line_indices:
        mean_y = np.mean(fixation_XY[start_of_line:end_of_line, 1])
        line_i = np.argmin(abs(line_Y - mean_y))
        fixation_XY[start_of_line:end_of_line, 1] = line_Y[line_i]
        start_of_line = end_of_line
    return fixation_XY


######################################################################
# STRETCH
######################################################################


def stretch(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    scale_bounds: tuple[float, float] = (0.9, 1.1),
    offset_bounds: tuple[float, float] = (-50, 50),
) -> np.ndarray:
    """Fit scale and offset bounds to stretch/compress fixations onto line centers.

    Reference: Lohmeier (2015); Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    scale_bounds: tuple[float, float]
        Scaling factor bounds. (default: (0.9, 1.1))
    offset_bounds: tuple[float, float]
        Vertical offset bounds. (default: (-50, 50))

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    fixation_Y = fixation_XY[:, 1]

    def fit_lines(params: np.ndarray, return_correction: bool = False) -> np.ndarray | float:
        candidate_Y = fixation_Y * params[0] + params[1]
        corrected_Y = np.zeros(n)
        for fixation_i in range(n):
            line_i = np.argmin(abs(line_Y - candidate_Y[fixation_i]))
            corrected_Y[fixation_i] = line_Y[line_i]
        if return_correction:
            return corrected_Y
        return float(sum(abs(candidate_Y - corrected_Y)))

    best_fit = minimize(fit_lines, [1, 0], bounds=[scale_bounds, offset_bounds])
    res = fit_lines(best_fit.x, return_correction=True)
    if isinstance(res, np.ndarray):
        fixation_XY[:, 1] = res
    return fixation_XY


######################################################################
# WARP
######################################################################


def warp(
    fixation_XY: np.ndarray,
    word_XY: np.ndarray,
) -> np.ndarray:
    """Dynamic Time Warping alignment between fixation sequence and word positions.

    Reference: Carr et al. (2022).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    word_XY: np.ndarray
        Word coordinates array of shape (M, 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    word_XY = np.array(word_XY)
    _, dtw_path = dynamic_time_warping(fixation_XY, word_XY)
    for fixation_i, words_mapped_to_fixation_i in enumerate(dtw_path):
        candidate_Y = word_XY[words_mapped_to_fixation_i, 1]
        fixation_XY[fixation_i, 1] = mode(candidate_Y)
    return fixation_XY


def mode(values: list[float] | np.ndarray) -> float:
    """Calculate statistical mode of a sequence."""
    values_list = list(values)
    return float(max(set(values_list), key=values_list.count))


######################################################################
# Dynamic Time Warping
######################################################################


def dynamic_time_warping(
    sequence1: np.ndarray,
    sequence2: np.ndarray,
) -> tuple[float, list[list[int]]]:
    """Calculate Dynamic Time Warping (DTW) cost and alignment path between two sequences.

    Parameters
    ----------
    sequence1: np.ndarray
        First sequence array of shape (N, D).
    sequence2: np.ndarray
        Second sequence array of shape (M, D).

    Returns
    -------
    tuple[float, list[list[int]]]
        DTW cost and alignment path list mapping sequence1 elements to sequence2 elements.
    """
    n1 = len(sequence1)
    n2 = len(sequence2)
    dtw_cost = np.zeros((n1 + 1, n2 + 1))
    dtw_cost[0, :] = np.inf
    dtw_cost[:, 0] = np.inf
    dtw_cost[0, 0] = 0
    for i in range(n1):
        for j in range(n2):
            this_cost = np.sqrt(sum((sequence1[i] - sequence2[j]) ** 2))
            dtw_cost[i + 1, j + 1] = this_cost + min(
                dtw_cost[i, j + 1], dtw_cost[i + 1, j], dtw_cost[i, j],
            )
    dtw_cost = dtw_cost[1:, 1:]
    dtw_path: list[list[int]] = [[] for _ in range(n1)]
    i, j = n1 - 1, n2 - 1
    while i > 0 or j > 0:
        dtw_path[i].append(j)
        possible_moves = [np.inf, np.inf, np.inf]
        if i > 0 and j > 0:
            possible_moves[0] = dtw_cost[i - 1, j - 1]
        if i > 0:
            possible_moves[1] = dtw_cost[i - 1, j]
        if j > 0:
            possible_moves[2] = dtw_cost[i, j - 1]
        best_move = np.argmin(possible_moves)
        if best_move == 0:
            i -= 1
            j -= 1
        elif best_move == 1:
            i -= 1
        else:
            j -= 1
    dtw_path[0].append(0)
    return float(dtw_cost[-1, -1]), dtw_path


######################################################################
# SLICE
######################################################################


# pylint: disable=redefined-builtin,consider-using-tuple,consider-using-dict-items
def slice(
    fixation_XY: np.ndarray,
    line_Y: np.ndarray,
    x_thresh: float = 192,
    y_thresh: float = 32,
    w_thresh: float = 32,
    n_thresh: float = 90,
) -> np.ndarray:
    """Slice algorithm to assign fixations in multi-line reading tasks.

    Reference: Glandorf & Schroeder (2021).

    Parameters
    ----------
    fixation_XY: np.ndarray
        Fixation coordinates array of shape (N, 2).
    line_Y: np.ndarray
        Vertical Y-coordinates (midlines) of lines of text.
    x_thresh: float
        Horizontal run segmentation threshold. (default: 192)
    y_thresh: float
        Vertical run segmentation threshold. (default: 32)
    w_thresh: float
        Proto-line merger threshold. (default: 32)
    n_thresh: float
        Adjacent proto-line merger threshold. (default: 90)

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with corrected Y-coordinates.
    """
    fixation_XY = np.array(fixation_XY, copy=True)
    line_Y = np.array(line_Y)
    n = len(fixation_XY)
    line_height = float(np.mean(np.diff(line_Y))) if len(line_Y) > 1 else 32.0
    proto_lines: dict[int, list[int]] = {}
    phantom_proto_lines: dict[int, np.ndarray] = {}

    # 1. Segment runs
    dist_X = abs(np.diff(fixation_XY[:, 0]))
    dist_Y = abs(np.diff(fixation_XY[:, 1]))
    end_run_indices = list(
        (np.where(np.logical_or(dist_X > x_thresh, dist_Y > y_thresh))[0] + 1).tolist(),
    )
    run_starts = [0] + end_run_indices
    run_ends = end_run_indices + [n]
    runs = [list(range(start, end)) for start, end in zip(run_starts, run_ends)]

    # 2. Determine starting run
    longest_run_i = int(
        np.argmax(
            [fixation_XY[run[-1], 0] - fixation_XY[run[0], 0] for run in runs],
        ),
    )
    proto_lines[0] = runs.pop(longest_run_i)

    # 3. Group runs into proto lines
    while runs:
        merger_on_this_iteration = False
        for proto_line_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
            proto_lines[proto_line_i + direction] = []
            if proto_lines[proto_line_i]:
                proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
            else:
                proto_line_XY = phantom_proto_lines[proto_line_i]

            run_differences = np.zeros(len(runs))
            for run_i, run in enumerate(runs):
                y_diffs = [
                    y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                    for x, y in fixation_XY[run]
                ]
                run_differences[run_i] = np.mean(y_diffs)

            merge_into_current = list(np.where(abs(run_differences) < w_thresh)[0])
            merge_into_adjacent = list(
                np.where(
                    np.logical_and(
                        run_differences * direction >= w_thresh,
                        run_differences * direction < n_thresh,
                    ),
                )[0],
            )

            for index in merge_into_current:
                proto_lines[proto_line_i].extend(runs[index])
            for index in merge_into_adjacent:
                proto_lines[proto_line_i + direction].extend(runs[index])

            if not merge_into_adjacent:
                average_x, average_y = np.mean(proto_line_XY, axis=0)
                adjacent_y = average_y + line_height * direction
                phantom_proto_lines[proto_line_i + direction] = np.array(
                    [[average_x, adjacent_y]],
                )

            for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                del runs[index]
                merger_on_this_iteration = True

        if not merger_on_this_iteration:
            break

    # 4. Assign leftover runs
    for run in runs:
        best_pl_distance = np.inf
        best_pl_assignment: int | None = None
        for proto_line_i in proto_lines:
            if proto_lines[proto_line_i]:
                proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
            else:
                proto_line_XY = phantom_proto_lines[proto_line_i]
            y_diffs = [
                y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                for x, y in fixation_XY[run]
            ]
            pl_distance = abs(np.mean(y_diffs))
            if pl_distance < best_pl_distance:
                best_pl_distance = pl_distance
                best_pl_assignment = proto_line_i
        if best_pl_assignment is not None:
            proto_lines[best_pl_assignment].extend(run)

    # 5. Prune proto lines
    while len(proto_lines) > len(line_Y):
        top, bot = min(proto_lines), max(proto_lines)
        if len(proto_lines[top]) < len(proto_lines[bot]):
            proto_lines[top + 1].extend(proto_lines[top])
            del proto_lines[top]
        else:
            proto_lines[bot - 1].extend(proto_lines[bot])
            del proto_lines[bot]

    # 6. Map proto lines to text lines
    for line_i, proto_line_i in enumerate(sorted(proto_lines)):
        fixation_XY[proto_lines[proto_line_i], 1] = line_Y[line_i]
    return fixation_XY


######################################################################
# WISDOM OF THE CROWD (Ensemble Method)
######################################################################


def wisdom_of_the_crowd(
    assignments: Sequence[np.ndarray | list[float]],
) -> list[float]:
    """Ensemble correction choosing line assignment with most votes across algorithms.

    Reference: Mercier et al. (2024b).

    Parameters
    ----------
    assignments: Sequence[np.ndarray | list[float]]
        List of corrected Y-coordinate arrays or line assignments from algorithms of length N.

    Returns
    -------
    list[float]
        Ensemble-corrected line Y-coordinates of length N.
    """
    assignments_matrix = np.column_stack(assignments)
    correction = []
    for row in assignments_matrix:
        candidates = list(row)
        candidate_counts = {y: candidates.count(y) for y in set(candidates)}
        best_count = max(candidate_counts.values())
        best_candidates = [y for y, c in candidate_counts.items() if c == best_count]
        if len(best_candidates) == 1:
            correction.append(best_candidates[0])
        else:
            for y in row:
                if y in best_candidates:
                    correction.append(y)
                    break
    return correction
