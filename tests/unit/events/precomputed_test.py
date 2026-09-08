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
"""Test PrecomputedEventDataFrame class."""
import polars as pl

from pymovements.events.precomputed import PrecomputedEventDataFrame


def test_precomputed_event_dataframe_metadata_defaults_to_empty_dict():
    precomputed = PrecomputedEventDataFrame(data=pl.DataFrame())
    assert precomputed.metadata == {}


def test_precomputed_event_dataframe_metadata_is_stored():
    precomputed = PrecomputedEventDataFrame(
        data=pl.DataFrame(),
        metadata={'sources': ['precomputed_events/events.csv'], 'subject_id': 1},
    )
    assert precomputed.metadata == {
        'sources': ['precomputed_events/events.csv'],
        'subject_id': 1,
    }
