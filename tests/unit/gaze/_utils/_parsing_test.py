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
"""Tests pymovements asc to csv processing."""
import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.gaze._utils import parsing

ASC_TEXT = r"""
** DATE: Wed Mar  8 09:25:20 2023
** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED
** VERSION: EYELINK II 1
** SOURCE: EYELINK CL
** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)
** CAMERA: EyeLink USBCAM Version 1.01
** SERIAL NUMBER: CLU-DAB50
** CAMERA_CONFIG: DAB50200.SCD
** RECORDED BY pymovements
** SREB2.2.299 WIN32 LID:20A87A96 Mod:2023.03.08 11:03 MEZ
**

some
lines
MSG	2095865 DISPLAY_COORDS 0 0 1279 1023
MSG	2154555 RECCFG CR 1000 2 1 L
MSG	2154555 ELCLCFG BTABLER
MSG	2154555 GAZE_COORDS 0.00 0.00 1279.00 1023.00
PRESCALER	1
VPRESCALER	1
PUPIL	AREA
EVENTS	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2
SAMPLES	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2	INPUT
the next line has all additional trial columns set to None
START	10000000 	RIGHT	SAMPLES	EVENTS
SFIX	R	10000000
10000000	  850.7	  717.5	  714.0	    0.0	...
END	10000001 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000001 START_A
START	10000002 	RIGHT	SAMPLES	EVENTS
the next line now should have the task column set to A
10000002	  850.7	  717.5	  714.0	    0.0	...
END	10000003 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000003 STOP_A
the task should be set to None again
START	10000004 	RIGHT	SAMPLES	EVENTS
10000004	  850.7	  717.5	  714.0	    0.0	...
END	10000005 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000005 METADATA_1 123
MSG 10000005 START_B
the next line now should have the task column set to B
START	10000006 	RIGHT	SAMPLES	EVENTS
10000006	  850.7	  717.5	  714.0	    0.0	...
END	10000007 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000007 START_TRIAL_1
the next line now should have the trial column set to 1
START	10000008 	RIGHT	SAMPLES	EVENTS
10000008	  850.7	  717.5	  714.0	    0.0	...
EFIX	R	10000000	10000008	9	850.7	717.5	714.0
END	10000009 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000009 STOP_TRIAL_1
MSG 10000010 START_TRIAL_2
the next line now should have the trial column set to 2
START	10000011 	RIGHT	SAMPLES	EVENTS
SSACC	R	10000011
10000011	  850.7	  717.5	  714.0	    0.0	...
END	10000012 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000012 STOP_TRIAL_2
MSG 10000013 START_TRIAL_3
the next line now should have the trial column set to 3
START	10000014 	RIGHT	SAMPLES	EVENTS
MSG 10000014 METADATA_2 abc
MSG 10000014 METADATA_1 456
10000014	  850.7	  717.5	  714.0	    0.0	...
END	10000015 	SAMPLES	EVENTS	RES	  38.54	  31.12
MSG 10000015 STOP_TRIAL_3
MSG 10000016 STOP_B
task and trial should be set to None again
MSG 10000017 METADATA_3
START	10000017 	RIGHT	SAMPLES	EVENTS
10000017	  850.7	  717.5	  714.0	    0.0	...
10000019	  850.7	  717.5	  .	    0.0	...
SBLINK R 10000020
10000020	   .	   .	    0.0	    0.0	...
10000021	   .	   .	    0.0	    0.0	...
EBLINK R 10000020	10000022	3
ESACC	R	10000011	10000022	12	850.7	717.5	850.7	717.5	19.00	590
END	10000022 	SAMPLES	EVENTS	RES	  38.54	  31.12
"""

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

EXPECTED_GAZE_DF = pl.from_dict(
    {
        'time': [
            10000000.0, 10000002.0, 10000004.0, 10000006.0, 10000008.0, 10000011.0, 10000014.0,
            10000017.0, 10000019.0, 10000020.0, 10000021.0,
        ],
        'x_pix': [
            850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, np.nan, np.nan,
        ],
        'y_pix': [
            717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, np.nan, np.nan,
        ],
        'pupil': [714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, np.nan, 0.0, 0.0],
        'task': [None, 'A', None, 'B', 'B', 'B', 'B', None, None, None, None],
        'trial_id': [None, None, None, None, '1', '2', '3', None, None, None, None],
    },
)

EXPECTED_EVENT_DF = pl.from_dict(
    {
        'name': ['fixation_eyelink', 'blink_eyelink', 'saccade_eyelink'],
        'eye': ['right', 'right', 'right'],
        'onset': [10000000.0, 10000020.0, 10000011.0],
        'offset': [10000008.0, 10000022.0, 10000022.0],
        'task': [None, None, 'B'],
        'trial_id': [None, None, '2'],
    },
)

EXPECTED_METADATA = {
    'weekday': 'Wed',
    'month': 'Mar',
    'day': 8,
    'time': '09:25:20',
    'year': 2023,
    'version_1': 'EYELINK II 1',
    'version_2': 'EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)',
    'model': 'EyeLink Portable Duo',
    'version_number': '6.12',
    'sampling_rate': 1000.0,
    'tracked_eye': 'L',
    'recorded_eye': 'L',
    'pupil_data_type': 'AREA',
    'calibrations': [],
    'validations': [],
    'resolution': (1280, 1024),
    'DISPLAY_COORDS': (0.0, 0.0, 1279.0, 1023.0),
    'data_loss_ratio_blinks': 3 / 12,
    'data_loss_ratio': 4 / 12,
    'total_recording_duration_ms': 12.0,
    'datetime': datetime.datetime(2023, 3, 8, 9, 25, 20),
    'mount_configuration': {
        'mount_type': 'Desktop',
        'head_stabilization': 'stabilized',
        'eyes_recorded': 'binocular / monocular',
        'short_name': 'BTABLER',
    },
    'metadata_1': '123',
    'metadata_2': 'abc',
    'metadata_3': True,
    'metadata_4': None,
    'recording_config': [
        {
            'sampling_rate': '1000',
            'file_sample_filter': '2',
            'link_sample_filter': '1',
            'timestamp': '2154555',
            'tracked_eye': 'L',
            'tracking_mode': 'CR',
            'resolution': (1280.0, 1024.0),
        },
    ],
}


def test_parse_eyelink(tmp_path):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    gaze_df, event_df, metadata = parsing.parse_eyelink(
        filepath,
        patterns=PATTERNS,
        metadata_patterns=METADATA_PATTERNS,
    )

    assert_frame_equal(gaze_df, EXPECTED_GAZE_DF, check_column_order=False, rtol=0)
    assert_frame_equal(event_df, EXPECTED_EVENT_DF, check_column_order=False, rtol=0)
    assert metadata == EXPECTED_METADATA


@pytest.mark.parametrize(
    ('filename', 'kwargs', 'expected_metadata'),
    [
        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'metadata_patterns': [
                    {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<subject_id>-?\d+)'},
                    r'!V TRIAL_VAR STIMULUS_COMBINATION_ID (?P<stimulus_combination_id>.+)',
                ],
            },
            {
                'subject_id': '-1',
                'stimulus_combination_id': 'start',
            },
            id='eyelink_asc_metadata_patterns',
        ),
        pytest.param(
            'eyelink_monocular_example.asc',
            {
                'metadata_patterns': [r'inexistent pattern (?P<value>-?\d+)'],
            },
            {
                'value': None,
            },
            id='eyelink_asc_metadata_pattern_not_found',
        ),
    ],
)
def test_from_asc_metadata_patterns(filename, kwargs, expected_metadata, make_example_file):
    filepath = make_example_file(filename)
    _, _, metadata = parsing.parse_eyelink(filepath=filepath, **kwargs)

    for key, value in expected_metadata.items():
        assert metadata[key] == value


@pytest.mark.parametrize(
    'patterns',
    [
        [1],
        [{'pattern': 1}],
    ],
)
def test_parse_eyelink_raises_value_error(tmp_path, patterns):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    with pytest.raises(ValueError) as excinfo:
        parsing.parse_eyelink(
            filepath,
            patterns=patterns,
        )

    msg, = excinfo.value.args

    expected_substrings = ['invalid pattern', '1']
    for substring in expected_substrings:
        assert substring in msg


@pytest.mark.parametrize(
    ('metadata', 'expected_version', 'expected_model'),
    [
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)',
            '6.12',
            'EyeLink Portable Duo',
            id='eye_link_portable_duo',
        ),
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v5.12 Feb  1 2018\n',
            '5.12',
            'EyeLink 1000 Plus',
            id='eye_link_1000_plus',
        ),
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v4.12 Feb  1 2018',
            '4.12',
            'EyeLink 1000',
            id='eye_link_1000_1',
        ),
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v3.12 Feb  1 2018',
            '3.12',
            'EyeLink 1000',
            id='eye_link_1000_2',
        ),
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v2.12 Feb  1 2018',
            '2.12',
            'EyeLink II',
            id='eye_link_II',
        ),
        pytest.param(
            '** VERSION: EYELINK REVISION 2.00 (Aug 12 1997)',
            '2.00',
            'EyeLink I',
            id='eye_link_I',
        ),
        pytest.param(
            '** VERSION: nothing\n',
            'unknown',
            'unknown',
            id='unknown_version_1',
        ),
        pytest.param(
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL Feb  1 2018 (EyeLink Portable Duo)',
            'unknown',
            'unknown',
            id='unknown_version_2',
        ),
        pytest.param(
            '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED',
            'unknown',
            'unknown',
            id='unknown_version_3',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_parse_eyelink_version(tmp_path, metadata, expected_version, expected_model):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, _, metadata = parsing.parse_eyelink(
        filepath,
    )

    assert metadata['version_number'] == expected_version
    assert metadata['model'] == expected_model


@pytest.mark.parametrize(
    ('metadata', 'expected_msg'),
    [
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n',
            'No metadata found. Please check the file for errors.',
            id='no_metadata',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n',
            'No recording configuration found.',
            id='no_reccfg',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154556 RECCFG CR 2000 2 1 L\n',
            r"Found inconsistent values for 'sampling_rate': \['1000', '2000'\]",
            id='inconsistent_sampling_rate_reccfg',
        ),
        pytest.param(
            'SAMPLES	GAZE	LEFT	RIGHT	RATE	1000.00	TRACKING	CR	FILTER	2\n'
            'SAMPLES	GAZE	LEFT	RIGHT	RATE	2000.00	TRACKING	CR	FILTER	2\n',
            r"Found inconsistent values for 'sampling_rate': \['1000.00', '2000.00'\]",
            id='inconsistent_sampling_rate_samples_config',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 GAZE_COORDS 0 0 1919 1079\n'
            'MSG	2154556 RECCFG CR 1000 2 1 L\n'
            'MSG	2154556 GAZE_COORDS 0 0 1023 767\n',
            "Found inconsistent values for 'resolution': "
            r'\[\(1024.0, 768.0\), \(1920.0, 1080.0\)\]',
            id='inconsistent_resolution',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_metadata_warnings(tmp_path, metadata, expected_msg):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    with pytest.warns(Warning, match=expected_msg):
        _, _, metadata = parsing.parse_eyelink(
            filepath,
        )


@pytest.mark.parametrize(
    ('metadata', 'expected_validation', 'expected_calibration'),
    [
        pytest.param(
            'MSG	7045618 !CAL \n'
            '>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT: <<<<<<<<<\n'
            'MSG	7045618 !CAL Calibration points:  \n'
            'MSG	1076158 !CAL VALIDATION HV9 R RIGHT POOR ERROR 2.40 avg. 6.03 max  '
            'OFFSET 0.19 deg. 4.2,6.3 pix.\n',
            [{
                'error': 'POOR ERROR',
                'tracked_eye': 'RIGHT',
                'num_points': '9',
                'timestamp': '1076158',
                'validation_score_avg': '2.40',
                'validation_score_max': '6.03',
            }],
            [{
                'num_points': '9',
                'timestamp': '7045618',
                'tracked_eye': 'LEFT',
                'type': 'P-CR',
            }],
            id='cal_timestamp_with_space',
        ),
        pytest.param(
            'MSG	7045618 !CAL\n'
            '>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT: <<<<<<<<<\n',
            [],
            [{
                'num_points': '9',
                'timestamp': '7045618',
                'tracked_eye': 'LEFT',
                'type': 'P-CR',
            }],
            id='cal_timestamp_no_space_no_val',
        ),
        pytest.param(
            'MSG	7045618 !CAL\n'
            'MSG	7045618 !CAL\n',
            [],
            [{'timestamp': '7045618'}],
            id='cal_timestamp_no_cal_no_val',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_val_cal_eyelink(tmp_path, metadata, expected_validation, expected_calibration):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, _, parsed_metadata = parsing.parse_eyelink(filepath)

    assert parsed_metadata['calibrations'] == expected_calibration
    assert parsed_metadata['validations'] == expected_validation


def test_check_reccfg_key_warnings_and_behavior():
    """Ensure _check_reccfg_key emits expected warnings and returns correct values."""
    # No recording config -> should warn and return None
    with pytest.warns(UserWarning, match='No recording configuration found.'):
        assert parsing._check_reccfg_key([], 'sampling_rate') is None

    # Recording config exists but key missing -> silently return None
    rc = [{'timestamp': '1'}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert parsing._check_reccfg_key(rc, 'sampling_rate') is None
        assert len(w) == 0

    # Inconsistent values -> should warn and return None
    rc = [{'sampling_rate': '1000'}, {'sampling_rate': '2000'}]
    with pytest.warns(UserWarning, match="Found inconsistent values for 'sampling_rate'"):
        assert parsing._check_reccfg_key(rc, 'sampling_rate') is None

    # Casting failure should return None silently (no warning)
    rc = [{'sampling_rate': 'not_a_number'}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert parsing._check_reccfg_key(rc, 'sampling_rate', float) is None
        assert len(w) == 0


def test_check_samples_config_key_warnings_and_casting(make_example_file):
    """Ensure _check_samples_config_key emits expected warnings and casts values."""
    # No samples config -> should warn and return None
    with pytest.warns(UserWarning, match='No samples configuration found.'):
        assert parsing._check_samples_config_key([], 'sampling_rate') is None

    # Inconsistent values -> should warn and return None
    sc = [{'sampling_rate': '1000.00'}, {'sampling_rate': '2000.00'}]
    with pytest.warns(UserWarning, match="Found inconsistent values for 'sampling_rate'"):
        assert parsing._check_samples_config_key(sc, 'sampling_rate') is None

    # Consistent value should be returned and cast when astype provided
    sc = [{'sampling_rate': '1000.00'}]
    assert parsing._check_samples_config_key(sc, 'sampling_rate', float) == 1000.0

    example_asc_monocular_path = make_example_file('eyelink_monocular_example.asc')
    _, _, metadata = parsing.parse_eyelink(example_asc_monocular_path)

    expected_validation = [{
        'error': 'GOOD ERROR',
        'tracked_eye': 'LEFT',
        'num_points': '9',
        'timestamp': '2148587',
        'validation_score_avg': '0.27',
        'validation_score_max': '0.83',
    }]
    expected_calibration = [{
        'num_points': '9', 'type': 'P-CR', 'tracked_eye': 'LEFT', 'timestamp': '2135819',
    }]

    assert metadata['calibrations'] == expected_calibration
    assert metadata['validations'] == expected_validation


@pytest.mark.parametrize(
    ('metadata', 'expected_blink_ratio', 'expected_overall_ratio'),
    [
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            'SBLINK R 10000019\n'
            '10000019	   .	   .	    0.0	    0.0	...\n'
            '10000020	   .	   .	    0.0	    0.0	...\n'
            'EBLINK R 10000019	10000020	2\n'
            'END	10000020 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            1,
            1,
            id='only_blinks',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            'SBLINK R 10000019\n'
            '10000019	   .	   .	    0.0	...\n'
            '10000020	   .	   .	    0.0	...\n'
            'EBLINK R 10000019	10000020	2\n'
            'END	10000020 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            1,
            1,
            id='only_blinks_no_dummy',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            '10000019	   .	   .	    0.0	    0.0	...\n'
            'END	10000019 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0,
            1,
            id='lost_samples_no_blinks',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            'SBLINK R 10000019\n'
            '10000019	   .	   .	    0.0	    0.0	...\n'
            'EBLINK R 10000019	10000019	1\n'
            '10000020	   .	   .	    0.0	    0.0	...\n'
            'END	10000020 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0.5,
            1.0,
            id='blinks_and_lost_samples',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            '10000019	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000020	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000022	   850.7	  717.5	  714.0	    0.0	...\n'
            'END	10000022 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0,
            0.25,
            id='missing_timestamps',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000018 	RIGHT	SAMPLES	EVENTS\n'
            '10000019	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000020	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000022	   .	   .	    0.0	    0.0	...\n'
            'END	10000022 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0,
            0.5,
            id='missing_timestamps_lost_samples1',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000020 	RIGHT	SAMPLES	EVENTS\n'
            '10000020	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000022	   .	   .	    0.0	    0.0	...\n'
            'SBLINK R 10000024\n'
            '10000024	   .	   .	    0.0	    0.0	...\n'
            'EBLINK R 10000024	10000024	1\n'
            'END	10000024 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0.25,
            0.75,
            id='missing_timestamps_lost_samples_blink',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'START	10000020 	RIGHT	SAMPLES	EVENTS\n'
            'END	10000021 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0,
            1,
            id='no_samples',
        ),
        pytest.param(
            'MSG	10000018.0 RECCFG CR 1000 2 1 L\n'
            'START	10000018.0 	RIGHT	SAMPLES	EVENTS\n'
            '10000019.0	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000020.0	   850.7	  717.5	  714.0	    0.0	...\n'
            'END	10000020.0 	SAMPLES	EVENTS	RES	  38.54	  31.12\n'
            'MSG	10000021.0 RECCFG CR 2000 2 1 L\n'
            'START	10000021.0 	RIGHT	SAMPLES	EVENTS\n'
            'END	10000023.0 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            0,
            2 / 3,
            id='varying_sampling_rate',
        ),
        pytest.param(
            'MSG	10000018.0 RECCFG CR 1000 2 1 L\n'
            'START	10000018.0 	RIGHT	SAMPLES	EVENTS\n'
            '10000019.0	   850.7	  717.5	  714.0	    0.0	...\n'
            '10000020.0	   850.7	  717.5	  714.0	    0.0	...\n'
            'END	10000020.0 	SAMPLES	EVENTS	RES	  38.54	  31.12\n'
            'MSG	10000021.0 RECCFG CR 2000 2 1 L\n'
            'START	10000021.0 	RIGHT	SAMPLES	EVENTS\n'
            'SBLINK R 10000020.0\n'
            '10000020.0	   .	   .	    0.0	    0.0	...\n'
            '10000021.0	   .	   .	    0.0	    0.0	...\n'
            'EBLINK R 10000020.0	10000021.5	2.0\n'
            'END	10000023.0 	SAMPLES	EVENTS	RES	  38.54	  31.12\n',
            2 / 3,
            2 / 3,
            id='varying_sampling_rate_blink',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
@pytest.mark.filterwarnings("ignore:Found inconsistent values for 'sampling_rate':")
def test_parse_eyelink_data_loss_ratio(
        tmp_path, metadata, expected_blink_ratio, expected_overall_ratio,
):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, _, parsed_metadata = parsing.parse_eyelink(filepath)

    assert parsed_metadata['data_loss_ratio_blinks'] == expected_blink_ratio
    assert parsed_metadata['data_loss_ratio'] == expected_overall_ratio


@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_parse_eyelink_datetime(tmp_path):
    metadata = '** DATE: Wed Mar  8 09:25:20 2023\n'
    expected_datetime = datetime.datetime(2023, 3, 8, 9, 25, 20)

    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, _, parsed_metadata = parsing.parse_eyelink(filepath)

    assert parsed_metadata['datetime'] == expected_datetime


@pytest.mark.parametrize(
    ('metadata', 'expected_mount_config'),
    [
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG BTABLER\n',
            {
                'mount_type': 'Desktop',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'BTABLER',
            },
            id='desktop_stabilized_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG MTABLER\n',
            {
                'mount_type': 'Desktop',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'monocular',
                'short_name': 'MTABLER',
            },
            id='desktop_stabilized_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG RTABLER\n',
            {
                'mount_type': 'Desktop',
                'head_stabilization': 'remote',
                'eyes_recorded': 'monocular',
                'short_name': 'RTABLER',
            },
            id='desktop_remote_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG RBTABLER\n',
            {
                'mount_type': 'Desktop',
                'head_stabilization': 'remote',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'RBTABLER',
            },
            id='desktop_remote_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG AMTABLER\n',
            {
                'mount_type': 'Arm Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'monocular',
                'short_name': 'AMTABLER',
            },
            id='arm_stabilized_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG ABTABLER\n',
            {
                'mount_type': 'Arm Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'ABTABLER',
            },
            id='arm_stabilized_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG ARTABLER\n',
            {
                'mount_type': 'Arm Mount',
                'head_stabilization': 'remote',
                'eyes_recorded': 'monocular',
                'short_name': 'ARTABLER',
            },
            id='arm_remote_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG ABRTABLE\n',
            {
                'mount_type': 'Arm Mount',
                'head_stabilization': 'remote',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'ABRTABLE',
            },
            id='arm_remote_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG BTOWER\n',
            {
                'mount_type': 'Binocular Tower Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'BTOWER',
            },
            id='binocular_tower_stabilized_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG TOWER\n',
            {
                'mount_type': 'Tower Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'monocular',
                'short_name': 'TOWER',
            },
            id='tower_stabilized_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG MPRIM\n',
            {
                'mount_type': 'Primate Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'monocular',
                'short_name': 'MPRIM',
            },
            id='primate_stabilized_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG BPRIM\n',
            {
                'mount_type': 'Primate Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'binocular / monocular',
                'short_name': 'BPRIM',
            },
            id='primate_stabilized_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG MLRR\n',
            {
                'mount_type': 'Long-Range Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'monocular',
                'camera_position': 'level',
                'short_name': 'MLRR',
            },
            id='long_range_level_monocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG BLRR\n',
            {
                'mount_type': 'Long-Range Mount',
                'head_stabilization': 'stabilized',
                'eyes_recorded': 'binocular / monocular',
                'camera_position': 'angled',
                'short_name': 'BLRR',
            },
            id='long_range_angled_binocular',
        ),
        pytest.param(
            'MSG	2154555 RECCFG CR 1000 2 1 L\n'
            'MSG	2154555 ELCLCFG XXXXX\n',
            {
                'mount_type': 'unknown',
                'head_stabilization': 'unknown',
                'eyes_recorded': 'unknown',
                'camera_position': 'unknown',
                'short_name': 'XXXXX',
            },
            id='unknown_mount_config',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_parse_eyelink_mount_config(tmp_path, metadata, expected_mount_config):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, _, parsed_metadata = parsing.parse_eyelink(filepath)

    assert parsed_metadata['mount_configuration'] == expected_mount_config


@pytest.mark.parametrize(
    ('bytestring', 'encoding', 'expected_text'),
    [
        pytest.param(
            b'MSG	2154555 H\xe4user\n',
            'latin1',
            'Häuser',
            id='latin1',
        ),
        pytest.param(
            b'MSG	2154555 H\xc3\xa4user\n',
            'utf-8',
            'Häuser',
            id='utf-8',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_parse_eyelink_encoding(tmp_path, bytestring, encoding, expected_text):
    filepath = tmp_path / 'sub.asc'
    filepath.write_bytes(bytestring)

    _, _, parsed_metadata = parsing.parse_eyelink(
        filepath,
        metadata_patterns=[r'(?P<text>.+)'],
        encoding=encoding,
    )

    assert parsed_metadata['text'] == expected_text


def test_data_loss_zero_expected_samples_via_parsing_module():
    """When num_expected_samples == 0 the function must return (0.0, 0.0)."""
    total, blink = parsing._calculate_data_loss_ratio(0, 10, 5)
    assert total == 0.0
    assert blink == 0.0


def test_data_loss_zero_expected_all_zero_via_parsing_module():
    """Sanity: zero inputs produce zero outputs."""
    total, blink = parsing._calculate_data_loss_ratio(0, 0, 0)
    assert total == 0.0
    assert blink == 0.0


def test_parse_eyelink_binocular_simple(tmp_path):
    """Basic test for parsing a binocular ASC snippet."""
    asc_text = r"""
** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED
MSG	1408659 RECCFG CR 1000 2 1 LR
MSG	1408659 ELCLCFG BTABLER
MSG	1408659 GAZE_COORDS 0.00 0.00 1919.00 1079.00
PRESCALER	1
VPRESCALER	1
PUPIL	AREA
EVENTS	GAZE	LEFT	RIGHT	RATE	1000.00	TRACKING	CR	FILTER	2
SAMPLES	GAZE	LEFT	RIGHT	RATE	1000.00	TRACKING	CR	FILTER	2
START	1408660 	LEFT	RIGHT	SAMPLES	EVENTS
1408660	 964.3	 541.5	 288.0	 960.5	 538.8	 305.0	.....
1408661	 964.5	 542.2	 288.0	 960.4	 539.5	 306.0	.....
1408662	 964.9	 543.0	 288.0	 960.3	 540.4	 307.0	.....
SFIX L   1408667
SFIX R   1408667
1408667	 963.7	 543.1	 288.0	 959.3	 538.6	 306.0	.....
1408782	 966.9	 565.5	 276.0	 949.4	 545.5	 308.0	.....
1408783	 970.7	 580.5	 271.0	 945.7	 540.8	 308.0	.....
1408784	 974.4	 594.8	 266.0	 942.4	 538.1	 307.0	.....
1408785	 976.7	 604.8	 262.0	 938.9	 540.1	 305.0	.....
1408786	 976.7	 604.8	 262.0	 935.1	 549.1	 303.0	.....
SBLINK L 1408787
1408787	  .	  .	   0.0	 933.4	 568.2	 298.0	.C...
1408788	  .	  .	   0.0	 934.1	 597.2	 289.0	.C...
1408789	  .	  .	   0.0	 937.7	 634.5	 276.0	.C...
1408790	  .	  .	   0.0	 941.1	 661.7	 266.0	.C...
1408791	  .	  .	   0.0	 942.9	 675.4	 259.0	.C...
1408792	  .	  .	   0.0	 942.9	 675.4	 259.0	.C...
SBLINK R 1408793
1408793	  .	  .	   0.0	  .	  .	   0.0	.C.C.
1408794	  .	  .	   0.0	  .	  .	   0.0	.C.C.
1408795	  .	  .	   0.0	  .	  .	   0.0	.C.C.
END	1408795 	SAMPLES	EVENTS	RES	 38.54	 31.12
"""

    filepath = tmp_path / 'sub_binoc.asc'
    filepath.write_text(asc_text)

    gaze_df, event_df, metadata = parsing.parse_eyelink(filepath)

    assert isinstance(gaze_df, pl.DataFrame)

    # Assert exact binocular sample values (times and left/right coords/pupils)
    expected_gaze = pl.from_dict(
        {
            'time': [
                1408660.0, 1408661.0, 1408662.0, 1408667.0,
                1408782.0, 1408783.0, 1408784.0, 1408785.0, 1408786.0,
                1408787.0, 1408788.0, 1408789.0, 1408790.0, 1408791.0, 1408792.0,
                1408793.0, 1408794.0, 1408795.0,
            ],
            'x_left_pix': [
                964.3, 964.5, 964.9, 963.7,
                966.9, 970.7, 974.4, 976.7, 976.7,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
            ],
            'y_left_pix': [
                541.5, 542.2, 543.0, 543.1,
                565.5, 580.5, 594.8, 604.8, 604.8,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
            ],
            'pupil_left': [
                288.0, 288.0, 288.0, 288.0,
                276.0, 271.0, 266.0, 262.0, 262.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            'x_right_pix': [
                960.5, 960.4, 960.3, 959.3,
                949.4, 945.7, 942.4, 938.9, 935.1,
                933.4, 934.1, 937.7, 941.1, 942.9, 942.9,
                np.nan, np.nan, np.nan,
            ],
            'y_right_pix': [
                538.8, 539.5, 540.4, 538.6,
                545.5, 540.8, 538.1, 540.1, 549.1,
                568.2, 597.2, 634.5, 661.7, 675.4, 675.4,
                np.nan, np.nan, np.nan,
            ],
            'pupil_right': [
                305.0, 306.0, 307.0, 306.0,
                308.0, 308.0, 307.0, 305.0, 303.0,
                298.0, 289.0, 276.0, 266.0, 259.0, 259.0,
                0.0, 0.0, 0.0,
            ],
        },
    )

    assert_frame_equal(gaze_df, expected_gaze, check_column_order=False, rtol=0)

    assert isinstance(event_df, pl.DataFrame)
    assert 'name' in event_df.columns

    # basic metadata expectations
    assert 'sampling_rate' in metadata
    assert metadata['sampling_rate'] == 1000.0
    # resolution should reflect the GAZE_COORDS line (+1 pixel for each
    # dimension because of 0-based indexing, see
    # https://www.sr-research.com/support/thread-9129.html)
    assert metadata.get('resolution') == (
        1920, 1080,
    ) or metadata.get('resolution') == (
        1920.0, 1080.0,
    )


@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_parse_eyelink_binocular_missing_samples_data_loss(tmp_path):
    """Ensure data loss ratios are reported for binocular ASC with missing samples.

    The snippet contains timestamps where the left eye becomes missing (dots) while the
    right eye still has values, and later both eyes are missing. The test checks that
    parse_eyelink returns numeric data loss ratios in the expected range.
    """
    # Build ASC text with continuous timestamps (1 ms steps) to cover blink intervals
    start = 1408782
    end = 1408887

    asc_lines = [
        '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED',
        'MSG\t1408659 RECCFG CR 1000 2 1 LR',
        'MSG\t1408659 ELCLCFG BTABLER',
        'MSG\t1408659 GAZE_COORDS 0.00 0.00 1919.00 1079.00',
        'PRESCALER\t1',
        'VPRESCALER\t1',
        'PUPIL\tAREA',
        'EVENTS\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2',
        'SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2',
        f'START\t{start}\tLEFT\tRIGHT\tSAMPLES\tEVENTS',
    ]

    for t in range(start, end + 1):
        # Insert SBLINK markers at their onsets
        if t == 1408787:
            asc_lines.append(f'SBLINK L {t}')
        if t == 1408793:
            asc_lines.append(f'SBLINK R {t}')
        # Insert EBlink markers at their ends
        if t == 1408873:
            asc_lines.append('EBLINK R 1408793\t1408872\t80')
        if t == 1408884:
            asc_lines.append('EBLINK L 1408787\t1408883\t97')

        # Construct sample lines depending on the timestamp range
        if t <= 1408786:
            # both eyes valid
            asc_lines.append(f'{t}\t 966.9\t 565.5\t 276.0\t 949.4\t 545.5\t 308.0\t.....')
        elif 1408787 <= t <= 1408792:
            # left missing, right valid
            asc_lines.append(f'{t}\t  .\t  .\t   0.0\t 933.4\t 568.2\t 298.0\t.C...')
        elif 1408793 <= t <= 1408872:
            # both eyes missing (overlap of left and right blink)
            asc_lines.append(f'{t}\t  .\t  .\t   0.0\t  .\t  .\t   0.0\t.C.C.')
        elif 1408873 <= t <= 1408883:
            # left missing, right valid (after right blink ended)
            asc_lines.append(f'{t}\t  .\t  .\t   0.0\t 939.7\t 590.6\t 288.0\t.C...')
        else:
            # both eyes valid again
            asc_lines.append(f'{t}\t 1009.6\t 483.1\t 252.0\t 939.4\t 582.3\t 292.0\t..R..')

    asc_lines.append(f'END\t{end}\tSAMPLES\tEVENTS\tRES\t 47.75\t 45.92')

    asc_text = '\n'.join(asc_lines) + '\n'

    filepath = tmp_path / 'sub_binoc_missing.asc'
    filepath.write_text(asc_text)

    _, _, parsed_metadata = parsing.parse_eyelink(filepath)

    # The parser should return both ratios and they should be numeric and valid
    assert 'data_loss_ratio_blinks' in parsed_metadata
    assert 'data_loss_ratio' in parsed_metadata

    blink_ratio = parsed_metadata['data_loss_ratio_blinks']
    overall_ratio = parsed_metadata['data_loss_ratio']

    print(f'Blink ratio: {blink_ratio}, Overall ratio: {overall_ratio}')

    assert isinstance(blink_ratio, (int, float))
    assert isinstance(overall_ratio, (int, float))

    assert blink_ratio == pytest.approx(97 / 105)
    assert overall_ratio == pytest.approx(96 / 105)


@pytest.mark.filterwarnings('ignore:No metadata found.')
# @pytest.mark.filterwarnings('ignore:No recording configuration found.')
# @pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_tracked_vs_recorded_eye_warning(tmp_path):
    """When RECCFG and SAMPLES report different eyes, a warning should be raised."""
    asc_text = (
        'MSG\t2154555 RECCFG CR 1000 2 1 LR\n'
        'SAMPLES\tGAZE\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n'
    )

    filepath = tmp_path / 'sub_mismatch.asc'
    filepath.write_text(asc_text)

    # The parser maps RECCFG 'LR' -> 'LR' and SAMPLES 'RIGHT' -> 'R', so expect [LR, R]
    with pytest.warns(Warning, match=r'inconsistent: \[LR, R\]'):
        parsing.parse_eyelink(filepath)


@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_sampling_rate_inconsistent_warning(tmp_path):
    """When RECCFG and SAMPLES report different sampling rates, a warning should be raised."""
    asc_text = (
        'MSG\t2154555 RECCFG CR 2000 2 1 L\n'
        'SAMPLES\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n'
    )

    filepath = tmp_path / 'sub_rate_mismatch.asc'
    filepath.write_text(asc_text)

    # Depending on internal casting, the warning should contain both values; match the float form
    with pytest.warns(
        Warning,
        match=r"inconsistent values for 'sampling_rate': \[1000\.0, 2000\.0\]",
    ):
        parsing.parse_eyelink(filepath)


@pytest.mark.parametrize(
    ('samples_line', 'expected_tracked'),
    [
        ('SAMPLES\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n', 'L'),
        ('SAMPLES\tGAZE\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n', 'R'),
        ('SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n', 'LR'),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
def test_tracked_eye_mapping_from_samples(tmp_path, samples_line, expected_tracked):
    """SAMPLES tracked_eye strings should map to metadata['tracked_eye'] correctly."""
    asc_text = samples_line

    filepath = tmp_path / 'sub_tracked.asc'
    filepath.write_text(asc_text)

    _, _, metadata = parsing.parse_eyelink(filepath)

    assert metadata['tracked_eye'] == expected_tracked


@pytest.mark.filterwarnings('ignore:No metadata found.')
def test_recording_config_missing_sampling_rate_key(monkeypatch, tmp_path):
    """Simulate a recording_config entry missing 'sampling_rate' to trigger KeyError path."""

    class DummyMatch:
        def groupdict(self):
            # deliberately omit 'sampling_rate' to trigger KeyError in parser
            return {
                'timestamp': '2154555',
                'file_sample_filter': '2',
                'link_sample_filter': '1',
                'tracked_eye': 'L',
                'tracking_mode': 'CR',
            }

    class DummyRegex:
        def match(self, line):
            return DummyMatch() if 'RECCFG' in line else None

    # replace the RECORDING_CONFIG_REGEX with our dummy that returns dict without sampling_rate
    monkeypatch.setattr(parsing, 'RECORDING_CONFIG_REGEX', DummyRegex())

    asc_text = (
        'MSG\t2154555 RECCFG CR ??? 2 1 L\n'
        'START\t10000018 \tRIGHT\tSAMPLES\tEVENTS\n'
        'SBLINK R 10000019\n'
        '10000019\t  .\t  .\t   0.0\t 0.0\t...\n'
        'EBLINK R 10000019\t10000020\t2\n'
        'END\t10000020 \tSAMPLES\tEVENTS\tRES\t 38.54\t 31.12\n'
    )

    filepath = tmp_path / 'sub_missing_rate_key.asc'
    filepath.write_text(asc_text)

    # Now the parser should warn (e.g. missing samples or sampling rate) and
    # set data-loss metrics to None.
    with pytest.warns(Warning):
        _, _, metadata = parsing.parse_eyelink(filepath)

    assert metadata['data_loss_ratio'] is None
    assert metadata['data_loss_ratio_blinks'] is None


def test_parse_eyelink_stop_recording_calculates_expected_samples(tmp_path: Path) -> None:
    """Write a minimal asc file with RECCFG, START, and END to exercise recording_config block.

    The RECCFG line provides a sampling rate of 1000 Hz and the START/END span 1000 ms,
    so expected number of samples = 1000 and total_recording_duration_ms should be 1000.0.
    """
    content = (
        'MSG 0 RECCFG CR 1000 0 0 LR\n'
        'START 0 RIGHT types\n'
        'END 1000 types RES 0 0\n'
    )

    p = tmp_path / 'test_stop_recording.asc'
    p.write_text(content)

    # parse_eyelink will emit a metadata warning for this minimal file; capture it
    with pytest.warns(UserWarning):
        _, _, metadata = parsing.parse_eyelink(str(p))

    # Duration should be 1000 ms
    assert metadata['total_recording_duration_ms'] == 1000.0

    # Because no valid samples were parsed, expected samples should be 1000 -> full data loss
    assert metadata['data_loss_ratio'] == 1.0
    assert metadata['data_loss_ratio_blinks'] == 0.0


def test_check_reccfg_key_warns_on_empty_config() -> None:
    # Empty recording_config should produce a warning and return None
    with pytest.warns(UserWarning):
        assert parsing._check_reccfg_key([], 'sampling_rate') is None


def test_check_reccfg_key_handles_unorderable_values_with_warning() -> None:
    """Ensure the TypeError in sorting unique values is handled and a warning is emitted.

    We create a recording_config with mixed types (int and str) for the same key so
    that sorted(unique_values) raises TypeError and the code falls back to list(unique_values).
    """
    recording_config: list[dict[str, Any]] = [
        {'sampling_rate': 1000},
        {'sampling_rate': '1000'},
    ]

    with pytest.warns(UserWarning) as w:
        result = parsing._check_reccfg_key(recording_config, 'sampling_rate')

    # Function should return None when inconsistent values are found
    assert result is None

    # A warning should have been emitted about inconsistent values
    assert len(w) >= 1
    assert any("Found inconsistent values for 'sampling_rate'" in str(rec.message) for rec in w)
