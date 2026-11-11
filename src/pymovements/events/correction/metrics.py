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
"""Metrics for evaluating drift correction algorithms."""
from __future__ import annotations

from functools import partial

import polars as pl

from pymovements.events.correction._utils import location_y
from pymovements.events.correction._utils import nearest_index
from pymovements.events.correction._utils import nearest_line_index
from pymovements.events.correction._utils import resolve_line_values
from pymovements.events.correction._utils import woc_core


def _validate_paired_series(first: pl.Series, second: pl.Series) -> None:
    """Validate two equally sized, non-null metric inputs."""
    if len(first) != len(second):
        raise ValueError(
            f'Metric inputs must have the same length, got {len(first)} and {len(second)}.',
        )
    if first.null_count() or second.null_count():
        raise ValueError('Metric inputs must not contain null values.')


def reassignment_rate(
    corrected_events: pl.DataFrame,
    aois: pl.DataFrame,
    trial_columns: list[str] | None = None,
    fixation_name: str = 'fixation',
) -> float:
    """Compute the fraction of fixations assigned to a different line.

    Compares the nearest line assignment of the original fixation y-coordinate
    with that of the corrected fixation y-coordinate.

    Line positions are resolved independently for each trial using the same
    logic as the correction algorithms.

    Parameters
    ----------
    corrected_events : pl.DataFrame
        Output of correct_fixations, containing 'location_original',
        'location', and 'name' columns.
    aois : pl.DataFrame
        AOIs dataframe used for correction.
    trial_columns : list[str] | None
        Columns identifying trials. Must match the trial partitioning used
        by correct_fixations.
    fixation_name : str
        Event name identifying fixation events.

    Returns
    -------
    float
        Fraction of fixations assigned to a different line.

    Raises
    ------
    ValueError
        If a required column is missing
        If a column contains null values

    Examples
    --------
    >>> reassignment_rate(pl.DataFrame({"name": ["saccade"]}), pl.DataFrame())
    0.0
    """
    fixations = corrected_events.filter(pl.col('name') == fixation_name)

    if fixations.is_empty():
        return 0.0

    for column in ('location_original', 'location'):
        if column not in fixations.columns:
            raise ValueError(f"Missing required column: {column}")
        if fixations[column].null_count() > 0:
            raise ValueError(f"{column} must not contain null values.")

    trial_columns = trial_columns or []

    event_groups = (
        fixations.partition_by(trial_columns, as_dict=True)
        if trial_columns
        else {(): fixations}
    )

    aoi_groups = (
        aois.partition_by(trial_columns, as_dict=True)
        if trial_columns
        else {(): aois}
    )

    reassigned = 0
    total = 0

    for trial_key, events in event_groups.items():
        trial_aois = aoi_groups.get(trial_key)

        if trial_aois is None or trial_aois.is_empty():
            raise ValueError(
                f"No AOIs found for trial {trial_key}.",
            )

        line_ys = resolve_line_values(trial_aois, word_locations=None)

        corrected_indices = nearest_line_index(
            location_y('location'), line_ys,
        ).alias('_corrected_line_idx')

        compared = events.with_columns(
            nearest_line_index(
                location_y('location_original'), line_ys,
            ).alias('_original_line_idx'),
            corrected_indices,
        )

        reassigned += compared.select(
            (
                pl.col('_original_line_idx')
                != pl.col('_corrected_line_idx')
            ).sum(),
        ).item()

        total += events.height

    return float(reassigned / total) if total else 0.0


def mean_vertical_distance(original_ys: pl.Series, corrected_ys: pl.Series) -> float:
    """Compute mean absolute distance between original and corrected y-values.

    The corrected y-values are the line centres. Inputs are physical y-coordinates,
    not line indices. Equal-length, non-null inputs are required; empty inputs
    return ``0.0``.

    Parameters
    ----------
    original_ys : pl.Series
        Series of original fixation y-values
    corrected_ys : pl.Series
        Series of corrected fixation y-values

    Returns
    -------
    float
        Mean of the absolute distance between original and corrected y-values

    Examples
    --------
    >>> mean_vertical_distance(pl.Series([1.0, 2.0]), pl.Series([0.0, 3.0]))
    1.0
    """
    _validate_paired_series(original_ys, corrected_ys)
    if len(original_ys) == 0:
        return 0.0
    vertical_distances = (original_ys - corrected_ys).abs()
    return float(vertical_distances.mean())


def implausible_transitions(
    corrected_events: pl.DataFrame,
    line_ys_by_trial: dict[tuple, list[float]],
    *,
    trial_columns: tuple[str, ...],
    max_forward_line_distance: int = 1,
    max_backward_line_distance: int = 1,
) -> int:
    """Count implausible transitions between corrected line indices.

    Events are grouped by ``trial_columns`` in their existing order. For each
    trial, corrected y-coordinates are mapped to the nearest line centre in
    ``line_ys_by_trial``. A transition is counted when the forward or backward
    change in line index exceeds its corresponding maximum. The first event
    in each trial cannot form a transition.

    Parameters
    ----------
    corrected_events : pl.DataFrame
        Corrected events containing a ``location`` column with non-null y
        coordinates and the columns named by ``trial_columns``.
    line_ys_by_trial : dict[tuple, list[float]]
        Mapping from each trial key to the ordered y-coordinates of that
        trial's line centres. Trial keys must match the keys produced by
        grouping ``corrected_events`` by ``trial_columns``.
    trial_columns : tuple[str, ...]
        Column names identifying trials. An empty tuple groups all events as
        one trial.
    max_forward_line_distance : int
        (default: 1) Largest allowed increase in line index between consecutive events.
    max_backward_line_distance : int
        (default: 1) Largest allowed decrease in line index between consecutive events.

    Returns
    -------
    int
        Number of transitions whose line-index change exceeds the applicable
        forward or backward limit.

    Raises
    ------
    ValueError
        If either line-distance limit is negative, or if corrected locations
        contain null y-coordinates.

    Examples
    --------
    >>> events = pl.DataFrame({
    ...     "trial": ["a", "a"],
    ...     "location": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 2.0}],
    ... })
    >>> implausible_transitions(
    ...     events, {("a",): [0.0, 1.0, 2.0]}, trial_columns=("trial",),
    ... )
    0
    """
    if max_forward_line_distance < 0:
        raise ValueError('max_forward_line_distance must be non-negative.')
    if max_backward_line_distance < 0:
        raise ValueError('max_backward_line_distance must be non-negative.')

    total = 0

    for trial_key, trial in corrected_events.group_by(
        list(trial_columns),
        maintain_order=True,
    ):
        if not isinstance(trial_key, tuple):
            trial_key = (trial_key,)

        line_ys = line_ys_by_trial[trial_key]
        corrected_ys = trial.select(location_y('location')).to_series()
        if corrected_ys.null_count():
            raise ValueError('Corrected locations must not contain null values.')
        corrected_indices = pl.Series(
            [
                nearest_index(line_ys, y)
                for y in corrected_ys
            ],
            dtype=pl.Int64,
        )

        total += int(
            (
                (corrected_indices.diff() > max_forward_line_distance)
                | (corrected_indices.diff() < -max_backward_line_distance)
            ).sum(),
        )

    return total


def ensemble_agreement(votes: pl.DataFrame) -> pl.Series:
    """Compute per-fixation agreement from a WoC vote table.

    ``votes`` must have one column per candidate algorithm and one row per fixation.
    Each result is the frequency of the most common vote divided by the number of
    algorithm columns. Ties receive the same score regardless of WoC's
    tie-breaking rule.

    Parameters
    ----------
    votes: pl.DataFrame
        DataFrame of fixation votes given by each fixation algorithm, with one row
        per algorithm and one row per fixation

    Returns
    -------
    pl.Series
        Float64 agreement values between 0.0 and 1.0, preserving row order.

    Raises
    ------
    ValueError
        If there are no algorithm columns in votes
        If votes contains null values

    Examples
    --------
    >>> ensemble_agreement(pl.DataFrame({"a": [1], "b": [1]})).to_list()
    [1.0]
    """
    if not votes.columns:
        raise ValueError('votes must contain at least one algorithm column.')
    if votes.null_count().to_numpy().sum():
        raise ValueError('votes must not contain null values.')

    algorithm_count = len(votes.columns)
    majority = votes.select(
        pl.struct(votes.columns).map_batches(
            partial(
                woc_core,
                column_priority={
                    column: priority
                    for priority, column in enumerate(votes.columns)
                },
            ),
            return_dtype=votes.dtypes[0],
        ).alias('_majority_vote'),
    )['_majority_vote']
    return votes.select(
        (
            pl.sum_horizontal([
                (pl.col(column) == majority).cast(pl.UInt32)
                for column in votes.columns
            ])
            .cast(pl.Float64)
            / algorithm_count
        ).alias('ensemble_agreement'),
    ).to_series()
