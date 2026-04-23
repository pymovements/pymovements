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
"""Provides the data_loss_histogram plotting function."""
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from pymovements.gaze.gaze import Gaze
from pymovements.measure.samples.measures import _is_invalid
from pymovements.plotting._matplotlib import prepare_figure


def data_loss_histogram(
        gaze: Gaze,
        *,
        column: str = 'position',
        sampling_rate: float | None = None,
        unit: Literal['count', 'time'] = 'count',
        time_column: str = 'time',
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
        ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histogram of consecutive data loss chunk lengths.

    Identifies consecutive regions of data loss (invalid values or time gaps)
    and creates a histogram showing the distribution of chunk lengths.

    Parameters
    ----------
    gaze : Gaze
        The gaze data to analyze.
    column : str, optional
        The column to check for invalid values (i.e. 'pixel' for pixel columns, 'position' for position data), by default 'position'.
    sampling_rate : float | None, optional
        Sampling rate in Hz. Required if unit='time', by default None.
    unit : Literal['count', 'time'], optional
        Unit for chunk length: 'count' for sample count or 'time' for milliseconds,
        by default 'count'.
    time_column : str, optional
        The column name containing timestamps, by default 'time'.
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (12, 6).
    title : str | None, optional
        Title for the plot. Auto-generated if None, by default None.
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on. Creates new figure if None, by default None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects.

    Raises
    ------
    ValueError
        If sampling_rate is not provided when unit='time'.
    ValueError
        If unit is not 'count' or 'time'.
    """
    if unit not in {'count', 'time'}:
        raise ValueError(f"unit must be 'count' or 'time', got {unit!r}")

    if unit == 'time' and sampling_rate is None:
        raise ValueError("sampling_rate must be provided when unit='time'")

    # Create figure if needed
    # Don't pass figsize when using external axes
    effective_figsize = None if ax is not None else figsize
    fig, ax = prepare_figure(ax=ax, figsize=effective_figsize, func_name='data_loss_histogram')

    samples = gaze.samples

    # Compute invalid mask using the same logic as data_loss measure
    invalid_mask = samples.select(
        _is_invalid(column).alias('invalid'),
    )['invalid'].to_numpy()

    # Compute time gap mask
    if time_column not in samples.columns:
        # No time gaps if time column doesn't exist
        time_gap_mask = np.zeros(len(samples), dtype=bool)
    else:
        times = samples[time_column].to_numpy()
        time_diffs = np.diff(times)

        # Expected inter-sample interval
        if len(times) > 1 and sampling_rate is not None:
            expected_isi = 1000.0 / sampling_rate
            # Detect gaps: time gap much larger than expected
            gaps = time_diffs > (1.5 * expected_isi)
            # Mark sample after a gap as the start of a loss
            time_gap_mask = np.concatenate([[False], gaps])
        else:
            time_gap_mask = np.zeros(len(samples), dtype=bool)

    # Combined loss mask: invalid values or time gaps
    loss_mask = invalid_mask | time_gap_mask

    # Identify consecutive loss chunks
    chunks = []
    current_chunk_length = 0

    for is_loss in loss_mask:
        if is_loss:
            current_chunk_length += 1
        else:
            if current_chunk_length > 0:
                chunks.append(current_chunk_length)
                current_chunk_length = 0

    # Don't forget the last chunk if data ends with a loss
    if current_chunk_length > 0:
        chunks.append(current_chunk_length)

    # Convert to requested unit
    if unit == 'time' and sampling_rate is not None:
        chunks_converted = [c / sampling_rate * 1000 for c in chunks]
    else:
        chunks_converted = chunks

    # Create histogram
    if chunks_converted:
        ax.hist(chunks_converted, bins='auto', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Chunk Length' + (' (ms)' if unit == 'time' else ' (samples)'))
        ax.set_ylabel('Frequency')

        # Add statistics
        chunks_array = np.array(chunks_converted)
        stats_text = (
            f'n={len(chunks_converted)}\n'
            f'μ={chunks_array.mean():.2f}\n'
            f'σ={chunks_array.std():.2f}\n'
            f'max={chunks_array.max():.0f}'
        )
        ax.text(
            0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    else:
        ax.text(
            0.5, 0.5, 'No data loss detected', ha='center', va='center',
            transform=ax.transAxes, fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Set title
    if title is None:
        unit_str = 'Time (ms)' if unit == 'time' else 'Samples'
        title = f'Data Loss Chunk Length Distribution ({unit_str})'

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax
