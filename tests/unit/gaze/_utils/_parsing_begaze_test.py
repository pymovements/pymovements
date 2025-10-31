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
"""Tests pymovements asc to csv processing - BeGaze."""
import datetime

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ._parsing_test import METADATA_PATTERNS
from ._parsing_test import PATTERNS
from pymovements.gaze._utils import parsing_begaze

BEGAZE_TEXT = r"""
## [BeGaze]
## Converted from:	C:\test.idf
## Date:	08.03.2023 09:25:20
## Version:	BeGaze 3.7.40
## IDF Version:	9
## Sample Rate:	1000
## Separator Type:	Msg
## Trial Count:	1
## Uses Plane File:	False
## Number of Samples:	11
## Reversed:	none
## [Run]
## Subject:	P01
## Description:	Run1
## [Calibration]
## Calibration Area:	1680	1050
## Calibration Point 0:	Position(841;526)
## Calibration Point 1:	Position(84;52)
## Calibration Point 2:	Position(1599;52)
## Calibration Point 3:	Position(84;1000)
## Calibration Point 4:	Position(1599;1000)
## Calibration Point 5:	Position(84;526)
## Calibration Point 6:	Position(841;52)
## Calibration Point 7:	Position(1599;526)
## Calibration Point 8:	Position(841;1000)
## [Geometry]
## Stimulus Dimension [mm]:	474	297
## Head Distance [mm]:	700
## [Hardware Setup]
## System ID:	IRX0470703-1007
## Operating System :	6.1
## IView X Version:	2.8.26
## [Filter Settings]
## Heuristics:	False
## Heuristics Stage:	0
## Bilateral:	True
## Gaze Cursor Filter:	True
## Saccade Length [px]:	80
## Filter Depth [ms]:	20
## Format:	LEFT, POR, QUALITY, PLANE, MSG
##
Time	Type	Trial	L POR X [px]	L POR Y [px]	L Pupil Diameter [mm]	Timing	Pupil Confidence	R Plane	Info	R Event Info	Stimulus
10000000123	SMP	1	850.71	717.53	714.00	0	1	1	Fixation	test.bmp
10000001123	MSG	1	# Message: START_A
10000002123	SMP	1	850.71	717.53	714.00	0	1	1	Fixation	test.bmp
10000003234	MSG	1	# Message: STOP_A
10000004123	SMP	1	850.71	717.53	714.00	0	1	1	Fixation	test.bmp
10000004234	MSG	1	# Message: METADATA_1 123
10000005234	MSG	1	# Message: START_B
10000006123	SMP	1	850.71	717.53	714.00	0	1	1	Fixation	test.bmp
10000007234	MSG	1	# Message: START_TRIAL_1
10000008123	SMP	1	850.71	717.53	714.00	0	1	1	Fixation	test.bmp
10000009234	MSG	1	# Message: STOP_TRIAL_1
10000010234	MSG	1	# Message: START_TRIAL_2
10000011123	SMP	1	850.71	717.53	714.00	0	1	1	Saccade	test.bmp
10000012234	MSG	1	# Message: STOP_TRIAL_2
10000013234	MSG	1	# Message: START_TRIAL_3
10000014234	MSG	1	# Message: METADATA_2 abc
10000014235	MSG	1	# Message: METADATA_1 456
10000014345	SMP	1	850.71	717.53	714.00	0	1	1	Saccade	test.bmp
10000015234	MSG	1	# Message: STOP_TRIAL_3
10000016234	MSG	1	# Message: STOP_B
10000017234	MSG	1	# Message: METADATA_3
10000017345	SMP	1	850.71	717.53	714.00	0	1	1	Saccade	test.bmp
10000019123	SMP	1	850.71	717.53	714.00	0	0	-1	Saccade	test.bmp
10000020123	SMP	1	850.71	717.53	714.00	0	0	-1	Blink	test.bmp
10000021123	SMP	1	850.71	717.53	714.00	0	0	-1	Blink	test.bmp
"""  # noqa: E501

BEGAZE_EXPECTED_GAZE_DF = pl.from_dict(
    {
        'time': [
            10000000.123, 10000002.123, 10000004.123, 10000006.123, 10000008.123,
            10000011.123,
            10000014.345, 10000017.345, 10000019.123, 10000020.123, 10000021.123,
        ],
        'x_pix': [
            850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, np.nan,
            np.nan,
        ],
        'y_pix': [
            717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, np.nan,
            np.nan,
        ],
        'pupil': [
            714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, np.nan, 0.0,
            0.0,
        ],
        'task': [None, 'A', None, 'B', 'B', 'B', 'B', None, None, None, None],
        'trial_id': [None, None, None, None, '1', '2', '3', None, None, None, None],
    },
)

BEGAZE_EXPECTED_EVENT_DF = pl.from_dict(
    {
        'name': ['fixation_begaze', 'saccade_begaze', 'blink_begaze'],
        'onset': [10000000.123, 10000011.123, 10000020.123],
        'offset': [10000008.123, 10000019.123, 10000021.123],
        'task': [None, 'B', None],
        'trial_id': [None, '2', None],
    },
)

# TODO: Add more metadata
EXPECTED_METADATA_BEGAZE = {
    'sampling_rate': 1000.00,
    'tracked_eye': 'L',
    'data_loss_ratio_blinks': 0.18181818181818182,
    'data_loss_ratio': 0.2727272727272727,
    'total_recording_duration_ms': 11,
    'datetime': datetime.datetime(2023, 3, 8, 9, 25, 20),
    'blinks': [{
        'duration_ms': 2,
        'num_samples': 2,
        'start_timestamp': 10000019.123,
        'stop_timestamp': 10000021.123,
    }],
    'metadata_1': '123',
    'metadata_2': 'abc',
    'metadata_3': True,
    'metadata_4': None,
}


def test_parse_begaze(tmp_path):
    filepath = tmp_path / 'sub.txt'
    filepath.write_text(BEGAZE_TEXT)

    gaze_df, event_df, metadata = parsing_begaze.parse_begaze(
        filepath,
        patterns=PATTERNS,
        metadata_patterns=METADATA_PATTERNS,
    )

    print(gaze_df.with_columns(pl.all().cast(pl.String)))
    print(BEGAZE_EXPECTED_GAZE_DF.with_columns(pl.all().cast(pl.String)))
    assert_frame_equal(
        gaze_df, BEGAZE_EXPECTED_GAZE_DF, check_column_order=False,
        rtol=0,
    )
    print(event_df.with_columns(pl.all().cast(pl.String)))
    print(BEGAZE_EXPECTED_EVENT_DF.with_columns(pl.all().cast(pl.String)))
    assert_frame_equal(
        event_df, BEGAZE_EXPECTED_EVENT_DF, check_column_order=False,
        rtol=0,
    )
    assert metadata == EXPECTED_METADATA_BEGAZE


def test_parse_begaze_binocular_prefer_left(tmp_path):
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tR POR X [px]\tR POR Y [px]\tR Pupil Diameter [mm]\tPupil Confidence\t'
        'L Event Info\tR Event Info\n'
        '10000000100\tSMP\t1\t10\t20\t3.0\t110\t120\t4.0\t1\tFixation\tSaccade\n'
        '10000001100\tSMP\t1\t11\t21\t3.1\t111\t121\t4.1\t1\tSaccade\tFixation\n'
        '10000002100\tSMP\t1\t12\t22\t3.2\t112\t122\t4.2\t0\tBlink\tFixation\n'
    )
    p = tmp_path / 'begaze_binoc_left.txt'
    p.write_text(text)

    gaze_df, event_df, metadata = parsing_begaze.parse_begaze(
        p, prefer_eye='L',
    )

    # Times in ms
    assert gaze_df['time'].to_list() == [10000000.1, 10000001.1, 10000002.1]
    # Left eye selected: blink row should set NaNs and pupil 0.0
    x_list = gaze_df['x_pix'].to_list()
    y_list = gaze_df['y_pix'].to_list()
    p_list = gaze_df['pupil'].to_list()
    assert x_list[:2] == [10.0, 11.0] and np.isnan(x_list[2])
    assert y_list[:2] == [20.0, 21.0] and np.isnan(y_list[2])
    assert p_list == [3.0, 3.1, 0.0]
    # Events follow L Event Info
    assert event_df['name'].to_list() == [
        'fixation_begaze', 'saccade_begaze', 'blink_begaze',
    ]
    assert event_df['onset'].to_list() == [10000000.1, 10000001.1, 10000002.1]
    assert event_df['offset'].to_list() == [10000000.1, 10000001.1, 10000002.1]
    # tracked_eye should be L
    assert metadata['tracked_eye'] == 'L'


def test_parse_begaze_binocular_prefer_right(tmp_path):
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tR POR X [px]\tR POR Y [px]\tR Pupil Diameter [mm]\tPupil Confidence\t'
        'L Event Info\tR Event Info\n'
        '10000000100\tSMP\t1\t10\t20\t3.0\t110\t120\t4.0\t1\tFixation\tSaccade\n'
        '10000001100\tSMP\t1\t11\t21\t3.1\t111\t121\t4.1\t1\tSaccade\tFixation\n'
        '10000002100\tSMP\t1\t12\t22\t3.2\t112\t122\t4.2\t0\tBlink\tFixation\n'
    )
    p = tmp_path / 'begaze_binoc_right.txt'
    p.write_text(text)

    gaze_df, event_df, metadata = parsing_begaze.parse_begaze(
        p, prefer_eye='R',
    )

    # Times in ms
    assert gaze_df['time'].to_list() == [10000000.1, 10000001.1, 10000002.1]
    # Right eye selected: pupil_confidence 0 on third row -> pupil NaN (no blink on R)
    assert gaze_df['x_pix'].to_list() == [110.0, 111.0, 112.0]
    assert gaze_df['y_pix'].to_list() == [120.0, 121.0, 122.0]
    assert np.isnan(gaze_df['pupil'].to_list()[-1])
    # Events follow R Event Info: saccade -> fixation - 3rd row no change
    assert event_df['name'].to_list() == [
        'saccade_begaze', 'fixation_begaze',
    ]
    assert event_df['onset'].to_list() == [10000000.1, 10000001.1]
    assert event_df['offset'].to_list() == [10000000.1, 10000002.1]
    # tracked_eye should be R
    assert metadata['tracked_eye'] == 'R'
