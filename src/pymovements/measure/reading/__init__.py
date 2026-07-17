# Copyright (c) 2024-2026 The pymovements Project Authors
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
"""Provides access to reading measure classes and functions."""
from pymovements.measure.reading.annotation import annotate_fixations
from pymovements.measure.reading.frame import ReadingMeasures
from pymovements.measure.reading.measures import build_word_level_table
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_fixation_count
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import first_reading_time
from pymovements.measure.reading.measures import landing_position
from pymovements.measure.reading.measures import regression_count_in
from pymovements.measure.reading.measures import regression_count_out
from pymovements.measure.reading.measures import regression_path_duration
from pymovements.measure.reading.measures import rereading_time
from pymovements.measure.reading.measures import saccade_length_in
from pymovements.measure.reading.measures import saccade_length_out
from pymovements.measure.reading.measures import total_fixation_count
from pymovements.measure.reading.processing import compute_reading_measures
from pymovements.measure.reading.words import all_tokens_from_aois
from pymovements.measure.reading.words import mark_skipped_tokens
from pymovements.measure.reading.words import repair_word_labels


__all__ = [
    # data container
    'ReadingMeasures',
    'compute_reading_measures',
    # main entry points
    'annotate_fixations',
    'build_word_level_table',
    # word/token utilities
    'all_tokens_from_aois',
    'mark_skipped_tokens',
    'repair_word_labels',
    # individual measures (for users who want just one or two)
    'first_duration',
    'first_fixation_duration',
    'first_pass_fixation_count',
    'first_pass_reading_time',
    'first_reading_time',
    'landing_position',
    'rereading_time',
    'regression_count_in',
    'regression_count_out',
    'regression_path_duration',
    'saccade_length_in',
    'saccade_length_out',
    'total_fixation_count',
]
