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
from math import nan

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


PATTERNS = [
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

METADATA_PATTERNS = [
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

    assert_frame_equal(gaze_df, expected_gaze, check_column_order=False, rel_tol=0)
    assert_frame_equal(event_df, expected_event, check_column_order=False, rel_tol=0)

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


@pytest.mark.parametrize(
    'with_trial_columns',
    [False, True], ids=['no_trial_columns', 'with_trial_columns'],
)
def test_from_begaze_loader_uses_parse_begaze(make_text_file, with_trial_columns):
    # Exercise the public loader that wraps parse_begaze using the same BEGAZE_TEXT fixture.

    filepath = make_text_file(filename='begaze_loader.txt', body=BEGAZE_TEXT, encoding='ascii')

    kwargs = {
        'patterns': PATTERNS,
        'metadata_patterns': METADATA_PATTERNS,
    }
    # Add a test case where trial_columns is not None
    if with_trial_columns:
        kwargs['trial_columns'] = 'trial_id'

    gaze = io.from_begaze(filepath, **kwargs)

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
        rel_tol=0,
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
        rel_tol=0,
    )

    # Experiment should be filled from metadata (sampling_rate)
    assert pytest.approx(gaze.experiment.sampling_rate, rel=0, abs=1e-9) == 1000.0
    assert gaze.experiment.eyetracker.left is True
    assert gaze.experiment.eyetracker.right is False

    # When trial_columns is provided, it should be set on the Gaze object
    if with_trial_columns:
        assert gaze.trial_columns == ['trial_id']


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
            # Space-separated header: tracked_eye detected but no samples parsed
            '## [BeGaze]\n'
            '## Date: 08.03.2023 09:25:20\n'
            '## Sample Rate: 1000\n'
            'Time Type Trial L POR X [px] L POR Y [px] L Pupil Diameter [mm] R POR X [px] '
            'R POR Y [px] R Pupil Diameter [mm] Pupil Confidence L Event Info R Event Info\n'
            '10000000100\tSMP\t1\t10\t20\t3.0\t110\t120\t4.0\t1\tFixation\tSaccade\n',
            'L', 'L', 0, 0, None,
            id='space_header_no_samples_L',
        ),
        pytest.param(
            '## [BeGaze]\n'
            '## Date: 08.03.2023 09:25:20\n'
            '## Sample Rate: 1000\n'
            'Time Type Trial L POR X [px] L POR Y [px] L Pupil Diameter [mm] R POR X [px] '
            'R POR Y [px] R Pupil Diameter [mm] Pupil Confidence L Event Info R Event Info\n'
            '10000000100\tSMP\t1\t10\t20\t3.0\t110\t120\t4.0\t1\tFixation\tSaccade\n',
            'R', 'R', 0, 0, None,
            id='space_header_no_samples_R',
        ),
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
        (['t', 'SMP', '1', 'Fixation', 'Saccade'], 'R', {}, '-'),
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
    'text, metadata_patterns, expected_keys, expect_warning',
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
            True,
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
            True,
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
            False,
        ),
    ],
    ids=['compiled_patterns_len_lt3', 'raw_string_patterns_len_lt3', 'msg_no_meta_patterns'],
)
def test_metadata_parsing_various(
        make_text_file, text, metadata_patterns, expected_keys, expect_warning,
):
    p = make_text_file(filename='begaze_meta_various.txt', body=text, encoding='ascii')
    if expect_warning:
        with pytest.warns(RuntimeWarning, match='non-sample lines matched by metadata_patterns'):
            _, _, meta = _parsing_begaze.parse_begaze(
                p, prefer_eye='L', metadata_patterns=metadata_patterns,
            )
    else:
        _, _, meta = _parsing_begaze.parse_begaze(
            p, prefer_eye='L', metadata_patterns=metadata_patterns,
        )
    for k, v in expected_keys.items():
        assert meta.get(k) == v


def test_parse_begaze_raises_when_no_tabular_header(make_text_file):
    body = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'This is not a header line and will not be parsed as table\n'
        'I am also not a header, sorry\n'
    )
    p = make_text_file(filename='begaze_no_header.txt', body=body, encoding='ascii')

    with pytest.raises(
            ValueError, match="could not find a tabular header row containing 'Time' and 'Type'",
    ):
        _parsing_begaze.parse_begaze(p)


def test_metadata_patterns_non_sample_full_coverage(make_text_file):
    text = (
        '## [BeGaze]\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\tPupil Confidence\t'
        'L Event Info\n'
        'nachricht: alpha\n'  # non-sample line -> groupdict pattern
        'SET_TRUE\n'          # non-sample line -> key/value pattern (no second warning)
        '10000000100\tSMP\t1\t10\t20\t3.0\t1\tFixation\n'
    )
    patterns = [
        {'pattern': r'^nachricht: (?P<meta_from_group>\w+)$'},
        {'pattern': r'^SET_TRUE$', 'key': 'flag_true', 'value': True},
        {'pattern': r'^I_WONT_MATCH: (?P<never>\d+)$'},  # will not match any non-sample line
    ]
    p = make_text_file(filename='begaze_meta_non_sample.txt', body=text, encoding='ascii')

    with pytest.warns(RuntimeWarning, match='non-sample lines matched by metadata_patterns'):
        _, _, meta = _parsing_begaze.parse_begaze(p, metadata_patterns=patterns)

    # groupdict branch
    assert meta['meta_from_group'] == 'alpha'
    # key/value branch
    assert meta['flag_true'] is True
    # the non-matching pattern remains unused - absence of the key confirms the if match false path
    assert meta.get('never', None) in {None, ''}


def test_metadata_overwrite_warning_on_non_sample_lines(make_text_file):
    # Trigger both the one-time non-sample warning and the overwrite warning in _set_metadata_key
    text = (
        '## [BeGaze]\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\tPupil Confidence\t'
        'L Event Info\n'
        'SET_KV_A\n'   # non-sample line -> sets dup_key = 'A'
        'SET_KV_B\n'   # non-sample line -> sets dup_key = 'B' (overwrites)
        '10000000100\tSMP\t1\t10\t20\t3.0\t1\tFixation\n'
    )
    metadata_patterns = [
        {'pattern': r'^SET_KV_A$', 'key': 'dup_key', 'value': 'A'},
        {'pattern': r'^SET_KV_B$', 'key': 'dup_key', 'value': 'B'},
    ]
    p = make_text_file(filename='begaze_meta_overwrite_non_sample.txt', body=text, encoding='ascii')

    with pytest.warns(RuntimeWarning) as w:
        _, _, meta = _parsing_begaze.parse_begaze(p, metadata_patterns=metadata_patterns)

    messages = [str(warn.message) for warn in w]
    assert any('non-sample lines matched by metadata_patterns' in m for m in messages)
    assert any("metadata key 'dup_key' is being overwritten" in m for m in messages)
    assert meta['dup_key'] == 'B'


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
