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
"""Test data_loss_histogram plotting function."""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest

from pymovements.gaze import Gaze
from pymovements.plotting import data_loss_histogram


class TestDataLossHistogram:
    """Test data_loss_histogram plotting function."""

    @pytest.fixture
    def sample_gaze_no_loss(self) -> Gaze:
        """Create sample gaze data with no data loss."""
        df = pl.DataFrame({
            'time': [0.0, 1.0, 2.0, 3.0, 4.0],
            'x': [1.0, 1.0, 1.0, 1.0, 1.0],
            'y': [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        return Gaze(samples=df, pixel_columns=['x', 'y'], time_column='time')

    @pytest.fixture
    def sample_gaze_with_loss(self) -> Gaze:
        """Create sample gaze data with consecutive data loss."""
        df = pl.DataFrame({
            'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'x': [1.0, None, None, 1.0, 1.0, None, 1.0],
            'y': [1.0, None, None, 1.0, 1.0, None, 1.0],
        })
        return Gaze(samples=df, pixel_columns=['x', 'y'], time_column='time')

    @pytest.fixture
    def sample_gaze_with_time_gaps(self) -> Gaze:
        """Create sample gaze data with time gaps."""
        df = pl.DataFrame({
            'time': [0.0, 1.0, 2.0, 5.0, 6.0],
            'x': [1.0, 1.0, 1.0, 1.0, 1.0],
            'y': [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        return Gaze(samples=df, pixel_columns=['x', 'y'], time_column='time')

    @pytest.fixture
    def sample_gaze_ends_with_loss(self) -> Gaze:
        """Create sample gaze data ending with data loss."""
        df = pl.DataFrame({
            'time': [0.0, 1.0, 2.0],
            'x': [1.0, 1.0, None],
            'y': [1.0, 1.0, None],
        })
        return Gaze(samples=df, pixel_columns=['x', 'y'], time_column='time')

    @pytest.fixture
    def sample_gaze_no_time_column(self) -> Gaze:
        """Create sample gaze data with no time column."""
        df = pl.DataFrame({
            'x': [1.0, 1.0, None, 1.0],
            'y': [1.0, 1.0, None, 1.0],
        })
        # Explicitly don't pass time_column to Gaze initialization
        return Gaze(samples=df, pixel_columns=['x', 'y'])

    def test_no_data_loss(self, sample_gaze_no_loss) -> None:
        """Test histogram with no data loss."""
        fig, ax = data_loss_histogram(sample_gaze_no_loss, column='pixel', sampling_rate=1000.0)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Should have no bars (no loss)
        assert len(ax.patches) == 0

        plt.close(fig)

    def test_with_invalid_values(self, sample_gaze_with_loss) -> None:
        """Test histogram with consecutive invalid values."""
        fig, ax = data_loss_histogram(
            sample_gaze_with_loss,
            column='pixel',
            unit='count',
            sampling_rate=1000.0,
            bins=2,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Should have 2 chunks: 2 invalid samples, then 1 invalid sample
        assert len(ax.patches) == 2

        plt.close(fig)

    def test_with_time_gaps(self, sample_gaze_with_time_gaps) -> None:
        """Test histogram with time gaps."""
        fig, ax = data_loss_histogram(
            sample_gaze_with_time_gaps,
            column='pixel',
            sampling_rate=1000.0,
            unit='count',
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        assert len(ax.patches) == 1

        plt.close(fig)

    def test_ends_with_loss(self, sample_gaze_ends_with_loss) -> None:
        """Test histogram when data ends with a loss chunk."""
        fig, ax = data_loss_histogram(
            sample_gaze_ends_with_loss,
            column='pixel',
        )
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_no_time_column(self, sample_gaze_no_time_column) -> None:
        """Test histogram when there is no time column."""
        fig, ax = data_loss_histogram(
            sample_gaze_no_time_column,
            column='pixel',
        )
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_unit_time_requires_sampling_rate(self, sample_gaze_no_loss) -> None:
        """Test that unit='time' requires sampling_rate."""
        with pytest.raises(ValueError, match='sampling_rate must be provided'):
            data_loss_histogram(sample_gaze_no_loss, unit='time')

    def test_invalid_unit(self, sample_gaze_no_loss) -> None:
        """Test that invalid unit raises error."""
        with pytest.raises(ValueError, match="unit must be 'count' or 'time'"):
            data_loss_histogram(sample_gaze_no_loss, unit='invalid')  # type: ignore

    def test_custom_title(self, sample_gaze_no_loss) -> None:
        """Test custom title parameter."""
        custom_title = 'Custom Histogram Title'
        fig, ax = data_loss_histogram(
            sample_gaze_no_loss, column='pixel', title=custom_title,
        )

        assert ax.get_title() == custom_title

        plt.close(fig)

    def test_custom_figsize(self, sample_gaze_no_loss) -> None:
        """Test custom figure size."""
        figsize = (10, 5)
        fig, _ = data_loss_histogram(
            sample_gaze_no_loss, column='pixel', figsize=figsize,
        )

        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]

        plt.close(fig)

    def test_external_axes(self, sample_gaze_no_loss) -> None:
        """Test plotting on external axes."""
        fig, external_ax = plt.subplots()
        returned_fig, returned_ax = data_loss_histogram(
            sample_gaze_no_loss, column='pixel', ax=external_ax,
        )

        assert returned_ax is external_ax
        assert returned_fig is fig

        plt.close(fig)

    @pytest.mark.parametrize(
        ('unit', 'xlabel'),
        [
            pytest.param('count', 'samples', id='count'),
            pytest.param('time', 'ms', id='time'),
        ],
    )
    def test_unit_time_conversion_axis_labels(
        self, unit, xlabel, sample_gaze_with_loss,
    ) -> None:
        """Test that axis labels reflect the selected unit."""
        fig, ax = data_loss_histogram(
            sample_gaze_with_loss,
            column='pixel',
            unit=unit,
            sampling_rate=500.0,
        )

        # Check axis label
        assert xlabel in ax.get_xlabel().lower()

        plt.close(fig)

    @pytest.mark.parametrize(
        ('unit', 'expected_text'),
        [
            pytest.param('count', 'max=2', id='count'),
            pytest.param('time', 'max=4', id='time'),
        ],
    )
    def test_unit_time_conversion_statistics(
        self, unit, expected_text, sample_gaze_with_loss,
    ) -> None:
        """Test that statistics reflect the selected unit."""
        fig, ax = data_loss_histogram(
            sample_gaze_with_loss,
            column='pixel',
            unit=unit,
            sampling_rate=500.0,
        )

        # Check calculated statistics
        texts = [t.get_text() for t in ax.texts]

        assert any(expected_text in t for t in texts)

        plt.close(fig)
