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
from collections.abc import Callable
from math import nan
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze import io
from pymovements.gaze._utils import _parsing_begaze
from pymovements.gaze.experiment import Experiment

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


PATTERNS: list[dict[str, Any] | str] = [
    {
        'pattern': 'START_A',
        'column': 'task',
        'value': 'A',
    },
    {
        'pattern': 'START_B',
        'column': 'task',
        'value': 'B',
    },
    {
        'pattern': ('STOP_A', 'STOP_B'),
        'column': 'task',
        'value': None,
    },

    r'START_TRIAL_(?P<trial_id>\d+)',
    {
        'pattern': r'STOP_TRIAL',
        'column': 'trial_id',
        'value': None,
    },
]

METADATA_PATTERNS: list[dict[str, Any] | str] = [
    r'METADATA_1 (?P<metadata_1>\d+)',
    {'pattern': r'METADATA_2 (?P<metadata_2>\w+)'},
    {'pattern': r'METADATA_3', 'key': 'metadata_3', 'value': True},
    {'pattern': r'METADATA_4', 'key': 'metadata_4', 'value': True},
]


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
        'Stimulus': ['test.bmp'] * 11,
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

# NOTE: Add more metadata
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


@pytest.mark.parametrize(
    'body,expected_gaze,expected_event',
    [[BEGAZE_TEXT, BEGAZE_EXPECTED_GAZE_DF, BEGAZE_EXPECTED_EVENT_DF]],
    ids=['base_fixture'],
)
def test_parse_begaze(make_text_file, body, expected_gaze, expected_event):
    filepath = make_text_file(filename='sub.txt', body=body, encoding='ascii')

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(
        filepath,
        patterns=PATTERNS,
        metadata_patterns=METADATA_PATTERNS,
    )

    assert_frame_equal(
        gaze_df, expected_gaze, check_column_order=False,
        rtol=0,
    )
    assert_frame_equal(
        event_df, expected_event, check_column_order=False,
        rtol=0,
    )
    assert metadata == EXPECTED_METADATA_BEGAZE


@pytest.fixture(name='begaze_binocular_text')
def _begaze_binocular_text():
    return (
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


@pytest.mark.parametrize(
    (
        'prefer_eye, expected_x, expected_y, expected_pupil, '
        'expected_event_names, expected_onsets, expected_offsets, expected_tracked_eye'
    ),
    [
        (
            'L',
            [10.0, 11.0, nan],
            [20.0, 21.0, nan],
            [3.0, 3.1, 0.0],
            ['fixation_begaze', 'saccade_begaze', 'blink_begaze'],
            [10000000.1, 10000001.1, 10000002.1],
            [10000000.1, 10000001.1, 10000002.1],
            'L',
        ),
        (
            'R',
            [110.0, 111.0, 112.0],
            [120.0, 121.0, 122.0],
            [4.0, 4.1, nan],
            ['saccade_begaze', 'fixation_begaze'],
            [10000000.1, 10000001.1],
            [10000000.1, 10000002.1],
            'R',
        ),
    ],
)
def test_parse_begaze_binocular_parametrized(
    make_text_file,
    begaze_binocular_text,
    prefer_eye,
    expected_x,
    expected_y,
    expected_pupil,
    expected_event_names,
    expected_onsets,
    expected_offsets,
    expected_tracked_eye,
):
    p = make_text_file(
        filename=f'begaze_binoc_{prefer_eye}.txt',
        body=begaze_binocular_text,
        encoding='ascii',
    )

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(
        p, prefer_eye=prefer_eye,
    )

    assert gaze_df['time'].to_list() == [10000000.1, 10000001.1, 10000002.1]

    def _eq_list_with_nans(a, b):
        if len(a) != len(b):
            return False
        for va, vb in zip(a, b):
            if isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb):
                continue
            if va != vb:
                return False
        return True

    assert _eq_list_with_nans(gaze_df['x_pix'].to_list(), expected_x)
    assert _eq_list_with_nans(gaze_df['y_pix'].to_list(), expected_y)
    assert _eq_list_with_nans(gaze_df['pupil'].to_list(), expected_pupil)

    assert event_df['name'].to_list() == expected_event_names
    assert event_df['onset'].to_list() == expected_onsets
    assert event_df['offset'].to_list() == expected_offsets

    assert metadata['tracked_eye'] == expected_tracked_eye


def test_from_begaze_loader_uses_parse_begaze(make_text_file):
    # Exercise the public loader that wraps parse_begaze using the same BEGAZE_TEXT fixture.

    filepath = make_text_file(filename='begaze_loader.txt', body=BEGAZE_TEXT, encoding='ascii')

    gaze = io.from_begaze(
        filepath,
        patterns=PATTERNS,
        metadata_patterns=METADATA_PATTERNS,
    )

    # Samples in Gaze use a combined 'pixel' column instead of separate x/y columns.
    expected_samples = BEGAZE_EXPECTED_GAZE_DF.with_columns(
        pl.concat_list([pl.col('x_pix'), pl.col('y_pix')]).alias('pixel'),
    ).drop(['x_pix', 'y_pix'])
    # Align None/NaN semantics for pupil for comparison: Gaze may store nulls instead of NaN.
    expected_samples = expected_samples.with_columns(
        pl.when(pl.col('pupil').is_nan()).then(None).otherwise(pl.col('pupil')).alias('pupil'),
    )
    # Align None/NaN semantics inside the nested list column as well.
    expected_samples = expected_samples.with_columns(
        pl.col('pixel').list.eval(
            pl.when(pl.element().is_nan()).then(None).otherwise(pl.element()),
        ).alias('pixel'),
    )
    assert_frame_equal(
        gaze.samples.select(expected_samples.columns),
        expected_samples,
        check_column_order=False,
        rtol=0,
    )

    # Events should be attached and match expected.
    assert gaze.events is not None
    # Events returned by the loader may include computed 'duration' - compare on common columns.
    ev_actual = gaze.events.frame
    common_cols = [c for c in BEGAZE_EXPECTED_EVENT_DF.columns if c in ev_actual.columns]
    assert_frame_equal(
        ev_actual.select(common_cols),
        BEGAZE_EXPECTED_EVENT_DF.select(common_cols),
        check_column_order=False,
        rtol=0,
    )

    # Experiment should be filled from metadata (sampling_rate)
    assert pytest.approx(gaze.experiment.sampling_rate, rel=0, abs=1e-9) == 1000.0
    assert gaze.experiment.eyetracker.left is True
    assert gaze.experiment.eyetracker.right is False


@pytest.mark.parametrize('prefer_eye', ['R', 'L'])
def test_from_begaze_loader_prefer_eye_via_definition(make_text_file, prefer_eye):
    # prefer_eye should be read from the DatasetDefinition.custom_read_kwargs path
    # and respected by from_begaze.

    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tR POR X [px]\tR POR Y [px]\tR Pupil Diameter [mm]\tPupil Confidence\t'
        'L Event Info\tR Event Info\n'
        '10000000100\tSMP\t1\t10\t20\t3.0\t110\t120\t4.0\t1\tFixation\tSaccade\n'
        '10000001100\tSMP\t1\t11\t21\t3.1\t111\t121\t4.1\t1\tSaccade\tFixation\n'
    )
    p = make_text_file(filename='begaze_loader_pref_eye.txt', body=text, encoding='ascii')

    definition = DatasetDefinition(
        experiment=Experiment(sampling_rate=None),
        custom_read_kwargs={'gaze': {'prefer_eye': prefer_eye}},
    )

    gaze = io.from_begaze(p, definition=definition)

    # Right eye should be selected per definition: reflected in experiment flags.
    assert gaze.experiment.eyetracker.left is (prefer_eye == 'L')
    assert gaze.experiment.eyetracker.right is (prefer_eye == 'R')
    # Gaze samples expose combined pixel column
    if prefer_eye == 'R':
        assert gaze.samples['pixel'].to_list() == [[110.0, 120.0], [111.0, 121.0]]
    else:
        assert gaze.samples['pixel'].to_list() == [[10.0, 20.0], [11.0, 21.0]]


@pytest.mark.parametrize(
    'text,prefer_eye,expected_time,expected_events', [
        (
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tSaccade\n'
            '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\tBlink\n',
            'L',
            [10000000.1, 10000001.1, 10000002.1],
            ['fixation_begaze', 'saccade_begaze', 'blink_begaze'],
        ),
        (
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tR POR X [px]\tR POR Y [px]\tR Pupil Diameter [mm]\t'
            'Pupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t30.0\t40.0\t4.0\t1\tFixation\n'
            '10000001100\tSMP\t1\t31.0\t41.0\t4.1\t1\tSaccade\n'
            '10040002100\tSMP\t1\t32.0\t42.0\t4.2\t1\tBlink\n',
            'R',
            [10000000.1, 10000001.1, 10040002.1],
            ['fixation_begaze', 'saccade_begaze', 'blink_begaze'],
        ),
        (
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tFixation\n'
            '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\tFixation\n',
            'L',
            [10000000.1, 10000001.1, 10000002.1],
            ['fixation_begaze'],
        ),
        (
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\t-\n'
            '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\tSaccade\n',
            'L',
            [10000000.1, 10000001.1, 10000002.1],
            ['fixation_begaze', 'saccade_begaze'],
        ),
    ], ids=['generic_info_only', 'right_eye_preferred', 'continuous_fixation', 'dash_event_gap'],
)
def test_parse_begaze_generic_info_only(
        make_text_file, text, prefer_eye, expected_time, expected_events,
):
    # When only a generic 'Info' column exists, events should be derived from it.
    p = make_text_file(filename='begaze_info_only.txt', body=text, encoding='ascii')

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(p, prefer_eye=prefer_eye)

    # times in ms
    assert gaze_df['time'].to_list() == expected_time
    assert metadata['tracked_eye'] == prefer_eye
    # Events follow the generic Info
    assert event_df['name'].to_list() == expected_events


@pytest.mark.parametrize(
    'text', [
        (
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\t-\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tFixation\n'
            '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\tFixation\n'
        ),
    ], ids=['initial_dash'],
)
def test_parse_begaze_initial_dash_no_event(make_text_file, text):
    # The first labelled event occurs only after an initial '-' value.
    p = make_text_file(filename='begaze_initial_dash.txt', body=text, encoding='ascii')

    _, event_df, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # Only one fixation event starting from the second sample.
    assert event_df['name'].to_list() == ['fixation_begaze']
    assert event_df['onset'].to_list() == [10000001.1]


@pytest.mark.parametrize(
    (
        'text, encoding, prefer_eye, expect_tracked_eye, expected_event_names, '
        'expect_nan_row_idx, expect_pupil_zero_row_idx, expected_rows'
    ),
    [
        (
            # Header without a Stimulus column should still parse samples and events.
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
            '\tPupil Confidence\tL Event Info\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tSaccade\n',
            'ascii', 'L', 'L', ['fixation_begaze', 'saccade_begaze'], None, None, 2,
        ),
        (
            # Non-ASCII in Stimulus should parse if encoding is provided.
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
            '\tPupil Confidence\tL Event Info\tStimulus\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\tGröße_치맥.bmp\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tSaccade\tGröße_치맥.bmp\n',
            'utf-16', 'L', 'L', ['fixation_begaze', 'saccade_begaze'], None, None, 2,
        ),
        (
            # Plane values -1 and >0 should not affect parsing logic - Blink row
            # forces NaNs and 0.0.
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
            '\tPupil Confidence\tR Plane\tL Event Info\n'
            '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\t-1\tFixation\n'
            '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\t2\tBlink\n'
            '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\t1\tFixation\n',
            'ascii', 'L', None, ['fixation_begaze', 'blink_begaze', 'fixation_begaze'], 1, 1, 3,
        ),
    ],
    ids=['no_stimulus', 'non_ascii_stimulus', 'plane_values'],
)
def test_parse_begaze_misc_samples(
    make_text_file,
    text,
    encoding,
    prefer_eye,
    expect_tracked_eye,
    expected_event_names,
    expect_nan_row_idx,
    expect_pupil_zero_row_idx,
    expected_rows,
):
    p = make_text_file(filename='begaze_misc.txt', body=text, encoding=encoding)

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(
        p, prefer_eye=prefer_eye, encoding=encoding,
    )

    if expect_tracked_eye is not None:
        assert metadata['tracked_eye'] == expect_tracked_eye
    assert gaze_df.shape[0] == expected_rows
    assert event_df['name'].to_list() == expected_event_names
    if expect_nan_row_idx is not None:
        assert np.isnan(gaze_df['x_pix'].to_list()[expect_nan_row_idx])
        assert np.isnan(gaze_df['y_pix'].to_list()[expect_nan_row_idx])
    if expect_pupil_zero_row_idx is not None:
        assert gaze_df['pupil'].to_list()[expect_pupil_zero_row_idx] == 0.0


@pytest.mark.parametrize(
    'text, prefer_eye, expected_tracked_eye, expected_rows, expected_events_rows, extra_assert',
    [
        pytest.param(
            # No eye columns in header -> no samples/events, but sampling_rate captured
            '## [BeGaze]\n'
            '## Date:\t08.03.2023 09:25:20\n'
            '## Sample Rate:\t1000\n'
            'Time\tType\tTrial\tPupil Confidence\tInfo\n'
            '10000000100\tSMP\t1\t1\tFixation\n',
            'L', None, 0, 0, lambda meta: meta['sampling_rate'] == 1000.0,
            id='no_eye_columns_no_samples',
        ),
    ],
)
def test_parse_begaze_unparseable_or_missing_eye_headers(
        make_text_file, text, prefer_eye, expected_tracked_eye, expected_rows, expected_events_rows,
        extra_assert,
):
    p = make_text_file(filename='begaze_header_cases.txt', body=text, encoding='ascii')
    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(p, prefer_eye=prefer_eye)
    if expected_tracked_eye is not None:
        assert metadata['tracked_eye'] == expected_tracked_eye
    assert gaze_df.shape[0] == expected_rows
    assert event_df.shape[0] == expected_events_rows
    if extra_assert is not None:
        assert extra_assert(metadata)


def test_parse_begaze_no_eye_columns_disables_header_parsing(make_text_file):
    # Header present but miss eye columns entirely. Should not parse samples.
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tPupil Confidence\tInfo\n'
        '10000000100\tSMP\t1\t1\tFixation\n'
        '10000001100\tSMP\t1\t1\tSaccade\n'
    )
    p = make_text_file(filename='begaze_no_eye_cols.txt', body=text, encoding='ascii')

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    assert gaze_df.shape[0] == 0
    assert event_df.shape[0] == 0
    # header-derived metadata still present
    assert metadata['sampling_rate'] == 1000.0


@pytest.mark.parametrize(
    'row, eye, header_idx, expected',
    [
        (
            ['t', 'SMP', '1', 'Fixation', 'Saccade'], 'L', {
                'L Event Info': 3, 'R Event Info': 4,
            }, 'Fixation',
        ),
        (
            ['t', 'SMP', '1', 'Fixation', 'Saccade'], 'R', {
                'L Event Info': 3, 'R Event Info': 4,
            }, 'Saccade',
        ),
        (['t', 'SMP', '1', 'Blink'], 'L', {'Info': 3}, 'Blink'),
        (['t', 'SMP', '1', 'Blink'], 'R', {'Info': 3}, 'Blink'),
        (['t', 'SMP', '1', 'Fixation', 'Saccade'], 'L', {}, '-'),
    ],
)
def test_parse_event_for_eye_helpers_param(row, eye, header_idx, expected):
    assert _parsing_begaze.parse_event_for_eye(row, eye, header_idx) == expected


def test_metadata_parsing_exceptions_on_header_lines(make_text_file):
    # Ensure ValueError branches in meta parsing are exercised.
    # Non-numeric sampling rate and malformed date should not crash.
    text = (
        '## [BeGaze]\n'
        '## Date:\t2023/03/08 09:25:20\n'  # malformed for strptime
        '## Sample Rate:\tabc\n'  # non-numeric
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
        'Pupil Confidence\tL Event Info\n'
        '10000000100\tSMP\t1\t10\t20\t3.0\t1\tFixation\n'
    )
    p = make_text_file(filename='begaze_bad_meta.txt', body=text, encoding='ascii')

    _, _, meta = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # sampling_rate kept as the original string in metadata due to a failed float cast
    assert meta['sampling_rate'] == 'abc'
    # datetime kept as the original string due to failed strptime
    assert meta['datetime'] == '2023/03/08 09:25:20'


@pytest.mark.parametrize(
    'text, metadata_patterns, expected_keys',
    [
        (
            # compiled_metadata_patterns branch matches (len(parts) < 3), sets keys
            # and removes patterns
            '## [BeGaze]\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tL Event Info\n'
            'META_ONE: foo\n'
            'META_BOOL\n'
            '10000000100\tSMP\t1\t10\t20\t3.0\t1\tFixation\n',
            [
                {'pattern': r'^META_ONE: (?P<meta_one>\w+)$'},
                {'pattern': r'^META_BOOL$', 'key': 'meta_bool', 'value': True},
            ],
            {'meta_one': 'foo', 'meta_bool': True},
        ),
        (
            # raw string metadata_patterns branch matches (len(parts) < 3)
            '## [BeGaze]\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tL Event Info\n'
            'RAW_ONE: bar\n'
            '10000000100\tSMP\t1\t10\t20\t3.0\t1\tFixation\n',
            [r'^RAW_ONE: (?P<raw_one>\w+)$'],
            {'raw_one': 'bar'},
        ),
        (
            # Ensure false branch for compiled_metadata_patterns on MSG lines
            '## [BeGaze]\n'
            'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
            'Pupil Confidence\tL Event Info\n'
            '10000000100\tMSG\t1\t# Message: START_X\n'
            '10000001100\tSMP\t1\t11\t21\t3.1\t1\tFixation\n',
            None,
            {},
        ),
    ],
    ids=['compiled_patterns_len_lt3', 'raw_string_patterns_len_lt3', 'msg_no_meta_patterns'],
)
def test_metadata_parsing_various(make_text_file, text, metadata_patterns, expected_keys):
    p = make_text_file(filename='begaze_meta_various.txt', body=text, encoding='ascii')
    _, _, meta = _parsing_begaze.parse_begaze(
        p, prefer_eye='L', metadata_patterns=metadata_patterns,
    )
    for k, v in expected_keys.items():
        assert meta.get(k) == v


def exp_with_flags(sampling_rate: float, left: bool, right: bool) -> Experiment:
    e = Experiment(sampling_rate=sampling_rate)
    e.eyetracker.left = left
    e.eyetracker.right = right
    return e


@pytest.mark.parametrize(
    ('header', 'kwargs', 'expected_experiment', 'expect_warning'),
    [
        pytest.param(
            '## Sample Rate:\t1000',
            {},
            Experiment(sampling_rate=1000),
            False,
            id='sampling_rate_1000',
        ),
        pytest.param(
            '## Sample Rate:\t500.5',
            {},
            Experiment(sampling_rate=500.5),
            False,
            id='sampling_rate_float_500_5',
        ),
        pytest.param(
            '## Sample Rate:\t1000',
            {},
            exp_with_flags(1000, True, False),
            False,
            id='tracked_eye_left_from_L_columns',
        ),
        pytest.param(
            '## Sample Rate:\t1000',
            {'prefer_eye': 'R'},
            exp_with_flags(1000, True, False),
            True,
            id='prefer_eye_R_but_only_L_columns_present_falls_back_with_warning',
        ),
        pytest.param(
            '## Sample Rate:\t1000',
            {'experiment': Experiment(sampling_rate=250)},
            Experiment(sampling_rate=250),
            False,
            id='keep_existing_sampling_rate_no_warning',
        ),
        pytest.param(
            '## Sample Rate:\t1000',
            {'experiment': Experiment(sampling_rate=1000)},
            Experiment(sampling_rate=1000),
            False,
            id='no_warning_when_sampling_rate_matches',
        ),
        pytest.param(
            '## Sample Rate:\t1000',
            {'experiment': exp_with_flags(1000, False, True)},
            exp_with_flags(1000, False, True),
            True,
            id='warn_on_overwrite_tracked_eye_keep_experiment_flags',
        ),
    ],
)
def test_from_begaze_loads_expected_experiment(
        header, kwargs, expected_experiment, expect_warning, make_text_file,
):
    # Build a minimal BeGaze file containing only the provided header part and
    # a minimal L-eye data row.
    text = (
        '## [BeGaze]\n'
        f'{header}\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\t'
        'Pupil Confidence\tL Event Info\n'
        '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
    )
    p = make_text_file(filename='begaze_experiment.txt', body=text, encoding='ascii')

    if expect_warning:
        with pytest.warns((UserWarning, RuntimeWarning)):
            gaze = io.from_begaze(p, **kwargs)
    else:
        gaze = io.from_begaze(p, **kwargs)

    # Assert Experiment fields were filled as expected.
    assert gaze.experiment is not None
    assert gaze.experiment.sampling_rate == expected_experiment.sampling_rate
    # Optionally assert tracked eye flags when provided in expected_experiment
    if expected_experiment.eyetracker.left is not None:
        assert gaze.experiment.eyetracker.left == expected_experiment.eyetracker.left
    if expected_experiment.eyetracker.right is not None:
        assert gaze.experiment.eyetracker.right == expected_experiment.eyetracker.right


@pytest.mark.parametrize(
    'trial_header, include_stimulus, stimulus_header, include_task, task_header',
    [
        pytest.param('Trial', False, None, False, None, id='trial_header_camel'),
        pytest.param('TRIAL', False, None, False, None, id='trial_header_upper'),
        pytest.param('trial', False, None, False, None, id='trial_header_lower'),
        pytest.param('Trial', True, 'Stimulus', False, None, id='stimulus_header_camel'),
        pytest.param('Trial', True, 'STIMULUS', False, None, id='stimulus_header_upper'),
        pytest.param('Trial', False, None, True, 'Task', id='task_header_camel'),
        pytest.param('Trial', False, None, True, 'TASK', id='task_header_upper'),
    ],
)
def test_parse_begaze_optional_columns_harmonized(
    make_text_file: Callable,
    trial_header: str,
    include_stimulus: bool,
    stimulus_header: str | None,
    include_task: bool,
    task_header: str | None,
) -> None:
    # Build minimal header row with variable trial/stimulus/task capitalisation
    base_cols = [
        'Time', 'Type', trial_header, 'L POR X [px]', 'L POR Y [px]',
        'L Pupil Diameter [mm]', 'Pupil Confidence', 'L Event Info',
    ]
    values = [
        '10000000100', 'SMP', '1', '10.0', '20.0', '3.0', '1', 'Fixation',
    ]
    if include_stimulus:
        assert stimulus_header is not None
        base_cols.append(stimulus_header)
        values.append('foo.bmp')
    if include_task:
        assert task_header is not None
        base_cols.append(task_header)
        values.append('A')

    header_line = '\t'.join(base_cols)
    sample_line = '\t'.join(values)

    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        f'{header_line}\n'
        f'{sample_line}\n'
    )

    p = make_text_file(filename='begaze_optional_cols.txt', body=text, encoding='ascii')

    gaze_df, _, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # trial header should arrive at trial_id column
    assert 'trial_id' in gaze_df.columns
    assert gaze_df['trial_id'].to_list() == ['1']
    # original trial header should not leak as column
    assert trial_header not in gaze_df.columns

    if include_stimulus:
        assert 'Stimulus' in gaze_df.columns
        assert gaze_df['Stimulus'].to_list() == ['foo.bmp']
    else:
        assert 'Stimulus' not in gaze_df.columns

    if include_task:
        assert 'task' in gaze_df.columns
        assert gaze_df['task'].to_list() == ['A']
    else:
        assert 'task' not in gaze_df.columns


@pytest.mark.parametrize(
    'trial_header', ['Trial', 'TRIAL', 'trial'], ids=['camel', 'upper', 'lower'],
)
def test_parse_begaze_trial_header_ignored_when_patterns_provide_trial_id(
    make_text_file: Callable, trial_header: str,
) -> None:
    # When patterns include trial_id, header-derived trial should be ignored.
    header = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
    )
    cols = (
        f'Time\tType\t{trial_header}\tL POR X [px]\tL POR Y [px]'
        '\tL Pupil Diameter [mm]\tPupil Confidence\tL Event Info\n'
    )
    # Emit a message that sets trial_id via patterns before the sample
    body = (
        header + cols +
        '10000000050\tMSG\t1\t# Message: START_TRIAL_5\n'
        '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
    )
    p = make_text_file(filename='begaze_trial_patterns.txt', body=body, encoding='ascii')

    gaze_df, _, _ = _parsing_begaze.parse_begaze(
        p, patterns=PATTERNS, metadata_patterns=METADATA_PATTERNS, prefer_eye='L',
    )

    # trial_id must come from patterns, not header
    assert 'trial_id' in gaze_df.columns
    assert gaze_df['trial_id'].to_list() == ['5']
    assert trial_header not in gaze_df.columns


@pytest.mark.parametrize(
    'bad_date, bad_sampling, expected_datetime_type',
    [
        pytest.param('08-03-2023 09:25:20', 'abc', str, id='bad_date_and_sampling_str_kept'),
        pytest.param('08/03/2023 09:25:20', '1,000', str, id='slash_date_and_sampling_str'),
    ],
)
def test_meta_parsing_bad_date_and_sampling_kept(
        make_text_file, bad_date, bad_sampling, expected_datetime_type,
):
    # Cover _parse_begaze_meta_line except branches for datetime and sampling_rate casts.
    text = (
        '## [BeGaze]\n'
        f'## Date:\t{bad_date}\n'
        f'## Sample Rate:\t{bad_sampling}\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tPupil Confidence\tInfo\n'
        '10000000100\tSMP\t1\t10\t20\t1\tFixation\n'
    )
    p = make_text_file(filename='begaze_bad_meta.txt', body=text, encoding='ascii')
    _, _, meta = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # datetime should be preserved as original string when parsing fails
    assert isinstance(meta.get('datetime'), expected_datetime_type)
    # sampling_rate should not be converted to float - remains as original string or missing
    if bad_sampling:
        assert not isinstance(meta.get('sampling_rate'), (int, float))


@pytest.mark.parametrize('include_pupil_mm', [False, True], ids=['no_pupil_mm', 'with_pupil_mm'])
def test_missing_pupil_mm_sets_nan(make_text_file, include_pupil_mm):
    # Cover pupil_s fallback to 'nan' when per-eye pupil column missing
    cols = [
        'Time', 'Type', 'Trial', 'L POR X [px]', 'L POR Y [px]', 'Pupil Confidence', 'L Event Info',
    ]
    if include_pupil_mm:
        cols.insert(5, 'L Pupil Diameter [mm]')
    header = '## [BeGaze]\n## Sample Rate:\t1000\n' + '\t'.join(cols) + '\n'
    # If we include pupil mm, provide value - else sample has no pupil diameter field
    sample = '10000000100\tSMP\t1\t10\t20' + \
        ('\t3.0' if include_pupil_mm else '') + '\t1\tFixation\n'
    p = make_text_file(filename='begaze_pupil_missing.txt', body=header + sample, encoding='ascii')

    gaze_df, _, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    if include_pupil_mm:
        assert gaze_df['pupil'].to_list() == [3.0]
    else:
        assert np.isnan(gaze_df['pupil'].to_list()[0])


def test_task_fallback_to_last_field_when_missing(make_text_file):
    # Include Task header but omit it in sample row to trigger IndexError and fallback to parts[-1]
    header = (
        '## [BeGaze]\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tPupil Confidence\tL Event Info\tTask\n'
    )
    # No trailing Task field in sample -> fallback takes last token (the event info)
    sample = '10000000100\tSMP\t1\t10\t20\t1\tFixation\n'
    p = make_text_file(filename='begaze_task_fallback.txt', body=header + sample, encoding='ascii')

    gaze_df, _, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    assert 'task' in gaze_df.columns
    assert gaze_df['task'].to_list() == ['Fixation']


@pytest.mark.parametrize('repeat_meta3', [False, True], ids=['single_meta3', 'repeat_meta3'])
def test_metadata_patterns_on_short_lines_and_removal(make_text_file, repeat_meta3):
    # Ensure len(parts) < 3 branch applies metadata_patterns and removes them after first match
    header = (
        '## [BeGaze]\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tPupil Confidence\n'
    )
    meta_lines = 'METADATA_3\n' + ('METADATA_3\n' if repeat_meta3 else '')
    sample = '10000000100\tSMP\t1\t10\t20\t1\n'
    p = make_text_file(
        filename='begaze_meta_short.txt',
        body=header + meta_lines + sample,
        encoding='ascii',
    )

    _, _, meta = _parsing_begaze.parse_begaze(
        p, metadata_patterns=METADATA_PATTERNS, prefer_eye='L',
    )
    assert meta.get('metadata_3') is True


def test_optional_trial_missing_value_falls_back_to_none(make_text_file):
    # Build header containing Trial so optional_col_map includes it, but omit Trial field in sample
    # Place Trial as the LAST header column so omitting it in the sample triggers IndexError
    header = (
        '## [BeGaze]\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tL POR X [px]\tL POR Y [px]\tPupil Confidence\tL Event Info\tTrial\n'
    )
    # Omit the final Trial field entirely so parts[index_of_Trial] raises IndexError
    sample = '10000000100\tSMP\t10\t20\t1\tFixation\n'
    p = make_text_file(
        filename='begaze_trial_missing_value.txt',
        body=header + sample,
        encoding='ascii',
    )

    gaze_df, _, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # 'trial_id' exists but value is None due to fallback else-branch
    assert 'trial_id' in gaze_df.columns
    assert gaze_df['trial_id'].to_list() == [None]


@pytest.mark.parametrize('repeat', [False, True], ids=['single', 'repeat'])
def test_header_msg_metadata_key_value_branch(make_text_file, repeat):
    # Ensure header-parsing path processes MSG lines with metadata_patterns key/value branch
    header = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tPupil Confidence\tL Event Info\n'
    )
    # A message line matching METADATA_4 which uses {'key':..., 'value': True}
    msgs = '10000000050\tMSG\t1\t# Message: METADATA_4\n'
    if repeat:
        # Second occurrence should be ignored due to single-use removal
        msgs += '10000000060\tMSG\t1\t# Message: METADATA_4\n'
    sample = '10000000100\tSMP\t1\t10\t20\t1\tFixation\n'
    p = make_text_file(
        filename='begaze_header_msg_meta.txt',
        body=header + msgs + sample,
        encoding='ascii',
    )

    _, _, meta = _parsing_begaze.parse_begaze(
        p, metadata_patterns=METADATA_PATTERNS, prefer_eye='L',
    )
    assert meta.get('metadata_4') is True


def test_header_short_line_metadata_patterns_copy_and_remove(make_text_file):
    # Cover the len(parts) < 3 branch that applies metadata_patterns on short, non-tab lines
    # by providing compiled regex patterns that do not rely on the MSG prefix.
    header = (
        '## [BeGaze]\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tPupil Confidence\tL Event Info\n'
    )
    # Two short lines without tabs - first has key/value metadata, second uses groupdict
    short_lines = 'META_ONE\nMETA_TWO abc\n'
    sample = '10000000100\tSMP\t1\t10\t20\t1\tFixation\n'

    meta_patterns = [
        {'pattern': r'^META_ONE$', 'key': 'meta_one', 'value': True},
        {'pattern': r'^META_TWO (?P<meta_two>\w+)$'},
    ]

    p = make_text_file(
        filename='begaze_header_short_meta.txt', body=header + short_lines + sample,
        encoding='ascii',
    )

    _, _, meta = _parsing_begaze.parse_begaze(
        p, metadata_patterns=meta_patterns, prefer_eye='L',
    )

    # Both metadata entries should be set and patterns removed after match
    assert meta.get('meta_one') is True
    assert meta.get('meta_two') == 'abc'
