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
"""Provides Dynamic Time Warping for the drift correction algorithms."""
from __future__ import annotations

import math


def dynamic_time_warping_points(
    sequence1: list[list[float]],
    sequence2: list[list[float]],
) -> tuple[float, list[list[int]]]:
    """Calculate Dynamic Time Warping (DTW) cost and alignment path between two point lists."""
    n1 = len(sequence1)
    n2 = len(sequence2)
    cost = [[math.inf] * (n2 + 1) for _ in range(n1 + 1)]
    cost[0][0] = 0.0
    for i in range(n1):
        for j in range(n2):
            step_cost = math.sqrt(
                sum(
                    (p - q) ** 2 for p, q in zip(sequence1[i], sequence2[j])
                ),
            )
            cost[i + 1][j + 1] = step_cost + min(
                cost[i][j + 1], cost[i + 1][j], cost[i][j],
            )

    dtw_path: list[list[int]] = [[] for _ in range(n1)]
    i, j = n1 - 1, n2 - 1
    while i > 0 or j > 0:
        dtw_path[i].append(j)
        possible_moves = [
            cost[i][j] if i > 0 and j > 0 else math.inf,
            cost[i][j + 1] if i > 0 else math.inf,
            cost[i + 1][j] if j > 0 else math.inf,
        ]
        best_move = possible_moves.index(min(possible_moves))
        if best_move == 0:
            i -= 1
            j -= 1
        elif best_move == 1:
            i -= 1
        else:
            j -= 1
    dtw_path[0].append(0)
    return cost[n1][n2], dtw_path
