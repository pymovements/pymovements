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
            '10000000123\tSMP\t1\t10.50\t20.75\t3.00\t0\t1\t1\tFixation\tstim.bmp\n'
            '10000001123\tMSG\t1\t# Message: START_TRIAL_1\n'
            '10000002123\tSMP\t1\t10.60\t20.85\t3.10\t0\t1\t1\tSaccade\tstim.bmp\n'
        ),
        (
            '10000000123 SMP 1 10.50 20.75 3.00 0 1 1 Fixation stim.bmp\n'
            '10000001123 MSG 1 # Message: START_TRIAL_1\n'
            '10000002123 SMP 1 10.60 20.85 3.10 0 1 1 Saccade stim.bmp\n'
        ),
    ], ids=['tabs', 'spaces'],
)
def test_parse_begaze_regex_fallback_minimal(make_text_file, text):
    # No header row: should use the legacy regex path BEGAZE_SAMPLE.
    p = make_text_file(filename='begaze_regex_only.txt', body=text, encoding='ascii')

    gaze_df, event_df, _ = _parsing_begaze.parse_begaze(
        p, patterns=PATTERNS, metadata_patterns=METADATA_PATTERNS,
    )

    # basic sanity
    assert gaze_df.shape[0] == 2
    assert event_df.shape[0] >= 1
    assert 'trial_id' in gaze_df.columns  # pattern captured from message


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


def test_parse_begaze_missing_stimulus_column(make_text_file):
    # Header without a Stimulus column should still parse samples and events.
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tPupil Confidence\tL Event Info\n'
        '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\n'
        '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tSaccade\n'
    )
    p = make_text_file(filename='begaze_no_stimulus.txt', body=text, encoding='ascii')

    gaze_df, event_df, metadata = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    assert metadata['tracked_eye'] == 'L'
    assert gaze_df.shape == (2, len(gaze_df.columns))
    assert event_df['name'].to_list() == ['fixation_begaze', 'saccade_begaze']


def test_parse_begaze_non_ascii_stimulus_utf16(make_text_file):
    # Non-ASCII in Stimulus should parse if encoding is provided.
    stimulus = 'Größe_치맥.bmp'
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tPupil Confidence\tL Event Info\tStimulus\n'
        f'10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\tFixation\t{stimulus}\n'
        f'10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\tSaccade\t{stimulus}\n'
    )
    p = make_text_file(filename='begaze_utf8_stimulus.txt', body=text, encoding='utf-16')

    gaze_df, event_df, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L', encoding='utf-16')

    # Basic assertions - presence of non-ASCII should not cause errors.
    assert gaze_df.shape[0] == 2
    assert event_df['name'].to_list() == ['fixation_begaze', 'saccade_begaze']


def test_parse_begaze_plane_values_stability(make_text_file):
    # Plane values -1 and >0 should not affect parsing logic.
    text = (
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]'
        '\tPupil Confidence\tR Plane\tL Event Info\n'
        '10000000100\tSMP\t1\t10.0\t20.0\t3.0\t1\t-1\tFixation\n'
        '10000001100\tSMP\t1\t11.0\t21.0\t3.1\t1\t2\tBlink\n'
        '10000002100\tSMP\t1\t12.0\t22.0\t3.2\t1\t1\tFixation\n'
    )
    p = make_text_file(filename='begaze_plane_values.txt', body=text, encoding='ascii')

    gaze_df, event_df, _ = _parsing_begaze.parse_begaze(p, prefer_eye='L')

    # Blink row forces NaN x/y and pupil 0.0
    assert np.isnan(gaze_df['x_pix'].to_list()[1])
    assert np.isnan(gaze_df['y_pix'].to_list()[1])
    assert gaze_df['pupil'].to_list()[1] == 0.0
    # Events should be fixation -> blink -> fixation
    assert event_df['name'].to_list() == [
        'fixation_begaze', 'blink_begaze', 'fixation_begaze',
    ]


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
