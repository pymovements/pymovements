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
from pymovements.measure.reading.annotation import delta_in
from pymovements.measure.reading.annotation import delta_out
from pymovements.measure.reading.annotation import is_first_fixation
from pymovements.measure.reading.annotation import is_first_pass
from pymovements.measure.reading.annotation import is_reg_in
from pymovements.measure.reading.annotation import is_reg_out
from pymovements.measure.reading.annotation import next_word_idx
from pymovements.measure.reading.annotation import prev_word_idx
from pymovements.measure.reading.annotation import regression_path_word
from pymovements.measure.reading.annotation import run_id
from pymovements.measure.reading.frame import ReadingMeasures
from pymovements.measure.reading.measures import first_duration
from pymovements.measure.reading.measures import first_fixation_duration
from pymovements.measure.reading.measures import first_pass_fixation_count
from pymovements.measure.reading.measures import first_pass_reading_time
from pymovements.measure.reading.measures import first_reading_time
from pymovements.measure.reading.measures import landing_position
from pymovements.measure.reading.measures import non_aoi_fixation_count_ratio
from pymovements.measure.reading.measures import non_aoi_fixation_duration_ratio
from pymovements.measure.reading.measures import regression_count_in
from pymovements.measure.reading.measures import regression_count_out
from pymovements.measure.reading.measures import regression_path_duration_exclusive
from pymovements.measure.reading.measures import regression_path_duration_inclusive
from pymovements.measure.reading.measures import rereading_time
from pymovements.measure.reading.measures import right_bounded_reading_time
from pymovements.measure.reading.measures import saccade_length_in
from pymovements.measure.reading.measures import saccade_length_out
from pymovements.measure.reading.measures import total_fixation_count
from pymovements.measure.reading.processing import compute_reading_measures


__all__ = [
    # data container
    'ReadingMeasures',
    'compute_reading_measures',
    # main entry points
    'annotate_fixations',
    # annotation expressions
    'run_id',
    'prev_word_idx',
    'next_word_idx',
    'delta_in',
    'delta_out',
    'is_reg_in',
    'is_reg_out',
    'is_first_fixation',
    'is_first_pass',
    'regression_path_word',
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
    'regression_path_duration_exclusive',
    'regression_path_duration_inclusive',
    'right_bounded_reading_time',
    'saccade_length_in',
    'saccade_length_out',
    'total_fixation_count',
    'non_aoi_fixation_count_ratio',
    'non_aoi_fixation_duration_ratio',
]
