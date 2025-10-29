# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test all Gaze functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm

EXPECTED_DF = {
    'char_left_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_right_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_left_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_right_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_left_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_right_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),

    'word_left_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_right_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_auto_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_auto_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),

    'word_auto_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_auto_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_else_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'char_else_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_else_pixel': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'pixel_xl': pl.Float64,
            'pixel_yl': pl.Float64,
            'pixel_xr': pl.Float64,
            'pixel_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
    'word_else_position': pl.DataFrame(
        [
            (
                1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
            (
                1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, None, None,
                None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema={
            'trialId': pl.Int64,
            'pointId': pl.Int64,
            'time': pl.Int64,
            'position_xl': pl.Float64,
            'position_yl': pl.Float64,
            'position_xr': pl.Float64,
            'position_yr': pl.Float64,
            'char': pl.String,
            'top_left_x': pl.Float64,
            'top_left_y': pl.Float64,
            'width': pl.Float64,
            'height': pl.Int64,
            'char_idx_in_line': pl.Int64,
            'line_idx': pl.Int64,
            'page': pl.String,
            'word': pl.String,
            'bottom_left_x': pl.Float64,
            'bottom_left_y': pl.Float64,
        },
        orient='row',
    ),
}


@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('eye'),
    [
        'right',
        'left',
        'auto',
        'else',
    ],
)
@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('aoi_column'),
    [
        'word',
        'char',
    ],
)
@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('gaze_type'),
    [
        'pixel',
        'position',
    ],
)
def test_gaze_to_aoi_mapping_char_width_height(eye, aoi_column, gaze_type, make_example_file):
    aoi_filepath = make_example_file('toy_text_1_1_aoi.csv')
    gaze_filepath = make_example_file('judo1000_example.csv')

    aoi_df = pm.stimulus.text.from_file(
        aoi_filepath,
        aoi_column=aoi_column,
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    if gaze_type == 'pixel':
        gaze = pm.gaze.io.from_csv(
            gaze_filepath,
            read_csv_kwargs={'separator': '\t'},
            pixel_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )
    elif gaze_type == 'position':
        gaze = pm.gaze.io.from_csv(
            gaze_filepath,
            read_csv_kwargs={'separator': '\t'},
            position_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )
    else:
        assert False, 'unknown gaze_type'

    gaze.map_to_aois(aoi_df, eye=eye, gaze_type=gaze_type)
    assert_frame_equal(gaze.samples, EXPECTED_DF[f'{aoi_column}_{eye}_{gaze_type}'])


@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('eye'),
    [
        'right',
        'left',
        'auto',
        'else',
    ],
)
@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('aoi_column'),
    [
        'word',
        'char',
    ],
)
@pytest.mark.filterwarnings('ignore:GazeDataFrame contains data but no.*:UserWarning')
@pytest.mark.parametrize(
    ('gaze_type'),
    [
        'pixel',
        'position',
    ],
)
def test_gaze_to_aoi_mapping_char_end(eye, aoi_column, gaze_type, make_example_file):
    aoi_filepath = make_example_file('toy_text_1_1_aoi.csv')
    gaze_filepath = make_example_file('judo1000_example.csv')

    aoi_df = pm.stimulus.text.from_file(
        aoi_filepath,
        aoi_column=aoi_column,
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        end_x_column='bottom_left_x',
        end_y_column='bottom_left_y',
        page_column='page',
    )
    if gaze_type == 'pixel':
        gaze = pm.gaze.io.from_csv(
            gaze_filepath,
            read_csv_kwargs={'separator': '\t'},
            pixel_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )
    elif gaze_type == 'position':
        gaze = pm.gaze.io.from_csv(
            gaze_filepath,
            read_csv_kwargs={'separator': '\t'},
            position_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )
    else:
        assert False, 'unknown gaze_type'

    gaze.map_to_aois(aoi_df, eye=eye, gaze_type=gaze_type)
    assert_frame_equal(gaze.samples, EXPECTED_DF[f'{aoi_column}_{eye}_{gaze_type}'])


def test_map_to_aois_raises_value_error(make_example_file):
    aoi_filepath = make_example_file('toy_text_1_1_aoi.csv')
    gaze_filepath = make_example_file('judo1000_example.csv')

    aoi_df = pm.stimulus.text.from_file(
        aoi_filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    gaze = pm.gaze.io.from_csv(
        gaze_filepath,
        read_csv_kwargs={'separator': '\t'},
        position_columns=['x_left', 'y_left', 'x_right', 'y_right'],
    )

    with pytest.raises(ValueError) as excinfo:
        gaze.map_to_aois(aoi_df, eye='right', gaze_type='')
    msg, = excinfo.value.args
    assert msg.startswith('neither position nor pixel column in samples dataframe')
