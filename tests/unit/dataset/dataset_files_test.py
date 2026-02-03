# Copyright (c) 2023-2026 The pymovements Project Authors
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
# flake8: noqa: E101, W191, E501
# pylint: disable=duplicate-code
from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pyreadr
import pytest
from polars.testing import assert_frame_equal

from pymovements import DatasetDefinition
from pymovements import Experiment
from pymovements import Gaze
from pymovements import ResourceDefinition
from pymovements.dataset.dataset_files import DatasetFile
from pymovements.dataset.dataset_files import load_gaze_file
from pymovements.dataset.dataset_files import load_precomputed_event_file
from pymovements.dataset.dataset_files import load_precomputed_event_files
from pymovements.dataset.dataset_files import load_precomputed_reading_measure_file
from pymovements.dataset.dataset_files import load_precomputed_reading_measures
from pymovements.dataset.dataset_files import load_stimulus_file
from pymovements.stimulus import ImageStimulus
from pymovements.stimulus import TextStimulus


ASC_TEXT = r"""\
** CONVERTED FROM D:\SamplePymovements\results\sub_1\sub_1.edf using edfapi 4.2.1 Win32  EyeLink Dataviewer Sub ComponentApr 01 1990 on Wed Sep 20 13:47:57 1989
** DATE: Wed Sep  20 13:47:20 1989
** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED
** VERSION: EYELINK II 1
** SOURCE: EYELINK CL
** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)
** CAMERA: EyeLink USBCAM Version 1.01
** SERIAL NUMBER: CLU-DAB50
** CAMERA_CONFIG: DAB50200.SCD
** RECORDED BY SleepAlc
** SREB2.2.299 WIN32 LID:20A87A96 Mod:2023.03.08 11:03 MEZ
**

MSG	2091650 !CMD 1 select_parser_configuration 0
MSG	2091659 !CMD 0 fixation_update_interval = 50
MSG	2091659 !CMD 0 fixation_update_accumulate = 50
MSG	2091681 !CMD 1 auto_calibration_messages = YES
MSG	2095865 DISPLAY_COORDS 0 0 1279 1023
MSG	2095865 RETRACE_INTERVAL  16.646125144
MSG	2095865 ENVIRONMENT   OpenGL on Windows (6, 2, 9200, 2, '')
MSG	2095980 TRACKER_TIME 0 2095980.470
MSG	2096367 -4 SYNCTIME 766 0
MSG	2100624 SYNCTIME_READING
INPUT	2117909	0
INPUT	2126823	0
MSG	2135819 !CAL
>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT: <<<<<<<<<
MSG	2135819 !CAL Calibration points:
MSG	2135819 !CAL -32.6, -47.7        -0,    227
MSG	2135819 !CAL -32.2, -63.6        -0,  -2267
MSG	2135819 !CAL -32.6, -31.7        -0,   2624
MSG	2135819 !CAL -58.0, -47.3     -3291,    227
MSG	2135819 !CAL -7.7, -46.8      3291,    227
MSG	2135820 !CAL -58.8, -65.2     -3358,  -2267
MSG	2135820 !CAL -7.9, -61.4      3358,  -2267
MSG	2135820 !CAL -55.5, -31.2     -3227,   2624
MSG	2135820 !CAL -9.0, -31.3      3227,   2624
MSG	2135820 !CAL  0.0,  0.0         0,      0
MSG	2135820 !CAL eye check box: (L,R,T,B)
	  -65     6   -72     7
MSG	2135820 !CAL href cal range: (L,R,T,B)
	-5037  5037 -3489  3847
MSG	2135820 !CAL Cal coeff:(X=a+bx+cy+dxx+eyy,Y=f+gx+goaly+ixx+jyy)
  -0.00043008  131.07  1.437  0.051949 -0.1007
   227.35 -1.5024  153.47 -0.1679 -0.22845
MSG	2135820 !CAL Prenormalize: offx, offy = -32.583 -47.715
MSG	2135820 !CAL Quadrant center: centx, centy =
  -0.00043025  227.35
MSG	2135820 !CAL Corner correction:
   9.5364e-06,  3.4194e-05
  -1.6932e-05,  2.9132e-05
   3.3933e-05,  3.5e-06
   1.5902e-05,  8.6479e-06
MSG	2135820 !CAL Gains: cx:152.074 lx:170.107 rx:152.936
MSG	2135820 !CAL Gains: cy:128.550 ty:155.848 by:116.611
MSG	2135820 !CAL Resolution (upd) at screen center: X=1.7, Y=2.0
MSG	2135820 !CAL Gain Change Proportion: X: 0.112 Y: 0.336
MSG	2135821 !CAL Gain Ratio (Gy/Gx) = 0.845
MSG	2135821 !CAL Bad Y/X gain ratio: 0.845
MSG	2135821 !CAL PCR gain ratio(x,y) = 2.507, 2.179
MSG	2135821 !CAL CR gain match(x,y) = 1.010, 1.010
MSG	2135821 !CAL Slip rotation correction OFF
MSG	2135821 !CAL CALIBRATION HV9 L LEFT    GOOD
INPUT	2137650	0
MSG	2148587 !CAL VALIDATION HV9 L LEFT  GOOD ERROR 0.27 avg. 0.83 max  OFFSET 0.11 deg. 3.7,2.4 pix.
MSG	2148587 VALIDATE L POINT 0  LEFT  at 640,512  OFFSET 0.19 deg.  7.2,1.0 pix.
MSG	2148587 VALIDATE L POINT 1  LEFT  at 640,159  OFFSET 0.12 deg.  3.9,-2.2 pix.
MSG	2148587 VALIDATE L POINT 2  LEFT  at 640,864  OFFSET 0.42 deg.  -15.8,0.9 pix.
MSG	2148587 VALIDATE L POINT 3  LEFT  at 172,512  OFFSET 0.83 deg.  26.3,17.5 pix.
MSG	2148587 VALIDATE L POINT 4  LEFT  at 1107,512  OFFSET 0.19 deg.  -4.3,5.7 pix.
MSG	2148587 VALIDATE L POINT 5  LEFT  at 228,201  OFFSET 0.06 deg.  1.3,1.9 pix.
MSG	2148587 VALIDATE L POINT 6  LEFT  at 1051,201  OFFSET 0.33 deg.  -3.0,-12.5 pix.
MSG	2148587 VALIDATE L POINT 7  LEFT  at 228,822  OFFSET 0.18 deg.  -6.8,0.2 pix.
MSG	2148587 VALIDATE L POINT 8  LEFT  at 1051,822  OFFSET 0.18 deg.  3.8,5.5 pix.
INPUT	2153108	0
MSG	2154447 DRIFTCORRECT L LEFT  at 133,133  OFFSET 0.38 deg.  12.5,7.9 pix.
MSG	2154540 TRIALID 0
MSG	2154555 RECCFG CR 1000 2 1 L
MSG	2154555 ELCLCFG BTABLER
MSG	2154555 GAZE_COORDS 0.00 0.00 1279.00 1023.00
MSG	2154555 THRESHOLDS L 102 242
MSG	2154555 ELCL_WINDOW_SIZES 176 188 0 0
MSG	2154555 CAMERA_LENS_FOCAL_LENGTH 27.00
MSG	2154555 PUPIL_DATA_TYPE RAW_AUTOSLIP
MSG	2154555 ELCL_PROC CENTROID (3)
MSG	2154555 ELCL_PCR_PARAM 5 3.0
START	2154556 	LEFT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
PUPIL	AREA
EVENTS	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2
SAMPLES	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2	INPUT
INPUT	2154556	0
MSG	2154557 !MODE RECORD CR 1000 2 1 L
2154557	  139.6	  132.1	  784.0	    0.0	...
MSG	2154558 -11 SYNCTIME_READING_SCREEN_0
2154558	  139.5	  131.9	  784.0	    0.0	...
MSG	2154559 -11 !V DRAW_LIST ../../runtime/dataviewer/sub_1/graphics/VC_1.vcl
2154560	   .	   .	    0.0	    0.0	...
2154561	  850.7	  717.5	  714.0	    0.0	...
MSG	2154562 TRACKER_TIME 1 2222195.987
MSG	2154563 0 READING_SCREEN_0.STOP
MSG	2154447 DRIFTCORRECT L LEFT  at 133,133  OFFSET 0.38 deg.  12.5,7.9 pix.
MSG	2154540 TRIALID 1
MSG	2154555 RECCFG CR 1000 2 1 L
MSG	2154555 ELCLCFG BTABLER
MSG	2154555 GAZE_COORDS 0.00 0.00 1279.00 1023.00
MSG	2154555 THRESHOLDS L 102 242
MSG	2154555 ELCL_WINDOW_SIZES 176 188 0 0
MSG	2154555 CAMERA_LENS_FOCAL_LENGTH 27.00
MSG	2154555 PUPIL_DATA_TYPE RAW_AUTOSLIP
MSG	2154555 ELCL_PROC CENTROID (3)
MSG	2154555 ELCL_PCR_PARAM 5 3.0
MSG	2154564 -11 SYNCTIME_READING_SCREEN_1
2154565	  139.5	  131.9	  784.0	    0.0	...
MSG	2154566 -11 !V DRAW_LIST ../../runtime/dataviewer/sub_1/graphics/VC_2.vcl
2154567	   .	   .	    0.0	    0.0	...
2154568	  850.7	  717.5	  714.0	    0.0	...
MSG	2154569 TRACKER_TIME 1 2222195.987
MSG	2154570 0 READING_SCREEN_1.STOP
"""

EXPECTED_EYELINK_SAMPLES_NO_PATTERNS = pl.from_dict(
    {
        'time': [2154557, 2154558, 2154560, 2154561, 2154565, 2154567, 2154568],
        'pixel': [
            (139.6, 132.1), (139.5, 131.9), (None, None), (850.7, 717.5),
            (139.5, 131.9), (None, None), (850.7, 717.5),
        ],
        'pupil': [784.0, 784.0, 0.0, 714.0, 784.0, 0.0, 714.0],
    },
)

EXPECTED_EYELINK_SAMPLES_PATTERNS = pl.from_dict(
    {
        'time': [2154557, 2154558, 2154560, 2154561, 2154565, 2154567, 2154568],
        'pixel': [
            (139.6, 132.1), (139.5, 131.9), (None, None), (850.7, 717.5),
            (139.5, 131.9), (None, None), (850.7, 717.5),
        ],
        'pupil': [784.0, 784.0, 0.0, 714.0, 784.0, 0.0, 714.0],
        'task': ['reading', 'reading', 'reading', 'reading', 'reading', 'reading', 'reading'],
        'trial_id': [0, 0, 0, 0, 1, 1, 1],
    },
)

EYELINK_PATTERNS = [
    {
        'pattern': 'SYNCTIME_READING',
        'column': 'task',
        'value': 'reading',
    },
    r'TRIALID (?P<trial_id>\d+)',
]


@pytest.mark.parametrize(
    ('load_kwargs', 'definition_kwargs', 'expected_samples'),
    [
        pytest.param(
            None,
            {},
            EXPECTED_EYELINK_SAMPLES_NO_PATTERNS,
            id='no_load_kwargs_empty_definition',
        ),

        pytest.param(
            {'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64}},
            {},
            EXPECTED_EYELINK_SAMPLES_PATTERNS,
            id='patterns_via_load_kwargs',
        ),

        pytest.param(
            None,
            {
                'custom_read_kwargs': {
                    'gaze': {'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64}},
                },
            },
            EXPECTED_EYELINK_SAMPLES_PATTERNS,
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning',
            ),
            id='patterns_via_definition',
        ),

        pytest.param(
            {'patterns': 'eyelink'},
            {
                'custom_read_kwargs': {
                    'gaze': {'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64}},
                },
            },
            EXPECTED_EYELINK_SAMPLES_PATTERNS,
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning',
            ),
            id='patterns_definition_overrides_load_kwargs',
        ),
    ],
)
@pytest.mark.parametrize(
    'load_function',
    [None, 'from_asc'],
)
def test_load_eyelink_file_has_expected_samples(
        load_kwargs, load_function, definition_kwargs, expected_samples, make_text_file,
):
    filepath = make_text_file(filename='sub.asc', body=ASC_TEXT)
    resource_definition = ResourceDefinition(
        content='gaze', load_function=load_function, load_kwargs=load_kwargs,
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_gaze_file(
        file=file,
        dataset_definition=DatasetDefinition(**definition_kwargs),
    )

    assert_frame_equal(gaze.samples, expected_samples, check_column_order=False)
    assert gaze.experiment is not None


@pytest.mark.parametrize(
    ('load_kwargs', 'definition_dict', 'expected_trial_columns'),
    [
        pytest.param(
            {'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64}},
            {},
            None,
            id='no_trial_columns',
        ),

        pytest.param(
            {
                'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64},
                'trial_columns': 'trial_id',
            },
            {},
            ['trial_id'],
            id='trial_columns_via_load_kwargs',
        ),

        pytest.param(
            {'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64}},
            {'trial_columns': ['trial_id']},
            ['trial_id'],
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
            ),
            id='trial_columns_via_definition',
        ),

        pytest.param(
            {
                'patterns': EYELINK_PATTERNS, 'schema': {'trial_id': pl.Int64},
                'trial_columns': 'wrong_trial_id',
            },
            {'trial_columns': ['trial_id']},
            ['trial_id'],
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
            ),
            id='trial_columns_definition_overrides_load_kwargs',
        ),
    ],
)
def test_load_eyelink_file_has_expected_trial_columns(
        load_kwargs, definition_dict, expected_trial_columns, make_text_file,
):
    filepath = make_text_file(filename='sub.asc', body=ASC_TEXT)
    resource_definition = ResourceDefinition(
        content='gaze', load_function='from_asc', load_kwargs=load_kwargs,
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_gaze_file(
        file=file,
        dataset_definition=DatasetDefinition(**definition_dict),
    )

    assert gaze.trial_columns == expected_trial_columns


@pytest.mark.parametrize(
    ('filename', 'rename_extension', 'load_function', 'load_kwargs'),
    [
        pytest.param(
            'monocular_example.csv',
            '.csv',
            None,
            {'pixel_columns': ['x_left_pix', 'y_left_pix']},
            id='load_csv_default',
        ),
        pytest.param(
            'monocular_example.csv',
            '.CSV',
            None,
            {'pixel_columns': ['x_left_pix', 'y_left_pix']},
            id='load_csv_uppercase',
        ),
        pytest.param(
            'monocular_example.csv',
            '.csv',
            'from_csv',
            {'pixel_columns': ['x_left_pix', 'y_left_pix']},
            id='load_csv_from_csv',
        ),
        pytest.param(
            'monocular_example.csv',
            '.renamed',
            'from_csv',
            {'pixel_columns': ['x_left_pix', 'y_left_pix']},
            id='load_csv_rename_from_csv',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.tsv',
            None,
            {'pixel_columns': ['x_left_pix', 'y_left_pix'], 'read_csv_kwargs': {'separator': '\t'}},
            id='load_tsv_default',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.TSV',
            None,
            {'pixel_columns': ['x_left_pix', 'y_left_pix'], 'read_csv_kwargs': {'separator': '\t'}},
            id='load_tsv_uppercase',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.tsv',
            'from_csv',
            {'pixel_columns': ['x_left_pix', 'y_left_pix'], 'read_csv_kwargs': {'separator': '\t'}},
            id='load_tsv_from_csv',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.foo',
            'from_csv',
            {'pixel_columns': ['x_left_pix', 'y_left_pix'], 'read_csv_kwargs': {'separator': '\t'}},
            id='load_tsv_rename_from_csv',
        ),
        pytest.param(
            'monocular_example.feather',
            '.feather',
            None,
            None,
            id='load_feather_default',
        ),
        pytest.param(
            'monocular_example.feather',
            '.FEATHER',
            None,
            None,
            id='load_feather_uppercase',
        ),
        pytest.param(
            'monocular_example.feather',
            '.feather',
            'from_ipc',
            None,
            id='load_feather_from_ipc',
        ),
        pytest.param(
            'monocular_example.feather',
            '.csv',
            'from_ipc',
            None,
            id='load_feather_rename_from_ipc',
        ),
    ],
)
def test_load_example_gaze_file(
        filename, rename_extension, load_function, load_kwargs, tmp_path, make_example_file,
):
    # Copy the file to the temporary path with the new extension
    filepath = make_example_file(filename)
    renamed_filename = filepath.stem + rename_extension
    renamed_filepath = tmp_path / renamed_filename
    renamed_filepath.write_bytes(filepath.read_bytes())

    resource_definition = ResourceDefinition(
        content='gaze', load_function=load_function, load_kwargs=load_kwargs,
    )

    file = DatasetFile(path=renamed_filepath, definition=resource_definition)

    gaze = load_gaze_file(
        file,
        dataset_definition=DatasetDefinition(
            experiment=Experiment(1280, 1024, 38, 30, None, 'center', 1000),
        ),
    )
    expected_df = pl.from_dict(
        {
            'time': list(range(10)),
            'pixel': [[0, 0]] * 10,
        },
    )

    assert_frame_equal(gaze.samples, expected_df, check_column_order=False)


@pytest.mark.parametrize(
    ('example_filename', 'load_function', 'load_kwargs', 'metadata'),
    [
        pytest.param(
            'monocular_example.csv',
            None,
            {'pixel_columns': ['x_left_pix', 'y_left_pix']},
            {'key': 'value'},
            id='csv',
        ),

        pytest.param(
            'monocular_example.feather',
            None,
            None,
            {'foo': 'bar'},
            id='feather',
        ),

        pytest.param(
            'eyelink_monocular_example.asc',
            None,
            None,
            {'meta': 'data'},
            id='eyelink',
        ),

        pytest.param(
            'didec_example.txt',
            'from_begaze',
            None,
            {'hello': 'there', 'how': 'are you?'},
            id='begaze',
        ),
    ],
)
def test_load_gaze_file_has_correct_metadata(
        example_filename, load_function, load_kwargs, metadata, make_example_file,
):
    filepath = make_example_file(example_filename)
    resource_definition = ResourceDefinition(
        content='gaze', load_function=load_function, load_kwargs=load_kwargs,
    )
    file = DatasetFile(path=filepath, definition=resource_definition, metadata=metadata)

    gaze = load_gaze_file(
        file,
        dataset_definition=DatasetDefinition(),
    )

    assert gaze.metadata == metadata


@pytest.mark.parametrize(
    (
        'make_csv_file_kwargs', 'load_function', 'load_kwargs', 'definition_dict', 'expected_gaze',
    ),
    [
        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'t': [1], 'x': [1.23], 'y': [3.45]}),
            },
            None, {'time_column': 't', 'pixel_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [1], 'pixel': [[1.23, 3.45]]})),
            id='time_column_and_pixel_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'trial': [1], 't': [0], 'x': [1.23], 'y': [3.45]}),
            },
            None, {'trial_columns': 'trial', 'time_column': 't', 'pixel_columns': ['x', 'y']},
            {},
            Gaze(
                samples=pl.DataFrame({'trial': [1], 'time': [0], 'pixel': [[1.23, 3.45]]}),
                trial_columns=['trial'],
            ),
            id='trial_columns_time_column_pixel_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'t': [1], 'x': [1.23], 'y': [3.45]}),
            },
            None, {'pixel_columns': ['x', 'y']},
            {'time_column': 't'},
            Gaze(samples=pl.DataFrame({'time': [1], 'pixel': [[1.23, 3.45]]})),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.time_column.*:DeprecationWarning',
            ),
            id='time_column_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'trial': [1], 't': [0], 'x': [1.23], 'y': [3.45]}),
            },
            None, {'time_column': 't', 'pixel_columns': ['x', 'y']},
            {'trial_columns': 'trial'},
            Gaze(
                samples=pl.DataFrame({'trial': [1], 'time': [0], 'pixel': [[1.23, 3.45]]}),
                trial_columns=['trial'],
            ),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
            ),
            id='trial_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [2], 'x': [0.23], 'y': [0.45]}),
            },
            None, {'time_unit': 's', 'pixel_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [2000], 'pixel': [[0.23, 0.45]]})),
            id='time_unit_and_pixel_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [2], 'x': [0.23], 'y': [0.45]}),
            },
            None, {},
            {'time_unit': 's', 'pixel_columns': ['x', 'y']},
            Gaze(samples=pl.DataFrame({'time': [2000], 'pixel': [[0.23, 0.45]]})),
            marks=[
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.time_unit.*:DeprecationWarning',
                ),
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.pixel_columns.*:DeprecationWarning',
                ),
            ],
            id='time_unit_and_pixel_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
            },
            None, {'pixel_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'pixel': [[1.2, 3.4]]})),
            id='pixel_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
            },
            None, None,
            {'pixel_columns': ['x', 'y']},
            Gaze(samples=pl.DataFrame({'time': [0], 'pixel': [[1.2, 3.4]]})),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.pixel_columns.*:DeprecationWarning',
            ),
            id='pixel_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
            },
            None, {'position_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'position': [[1.2, 3.4]]})),
            id='position_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [21.2], 'y': [23.4]}),
            },
            None, None,
            {'position_columns': ['x', 'y']},
            Gaze(samples=pl.DataFrame({'time': [0], 'position': [[21.2, 23.4]]})),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.position_columns.*:DeprecationWarning',
            ),
            id='position_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
            },
            None, {'velocity_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'velocity': [[1.2, 3.4]]})),
            id='velocity_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [21.2], 'y': [23.4]}),
            },
            None, None,
            {'velocity_columns': ['x', 'y']},
            Gaze(samples=pl.DataFrame({'time': [0], 'velocity': [[21.2, 23.4]]})),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.velocity_columns.*:DeprecationWarning',
            ),
            id='velocity_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
            },
            None, {'acceleration_columns': ['x', 'y']},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'acceleration': [[1.2, 3.4]]})),
            id='acceleration_columns',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [21.2], 'y': [23.4]}),
            },
            None, None,
            {'acceleration_columns': ['x', 'y']},
            Gaze(samples=pl.DataFrame({'time': [0], 'acceleration': [[21.2, 23.4]]})),
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.acceleration_columns.*:DeprecationWarning',
            ),
            id='acceleration_columns_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.23], 'y': [3.45], 'd': [123.45]}),
            },
            None, {'pixel_columns': ['x', 'y'], 'distance_column': 'd'},
            {},
            Gaze(
                samples=pl.DataFrame(
                    {'time': [0], 'distance': [123.45], 'pixel': [[1.23, 3.45]]},
                ),
            ),
            id='distance_column',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [41.23], 'y': [53.45], 'd': [567.89]}),
            },
            None, None,
            {'pixel_columns': ['x', 'y'], 'distance_column': 'd'},
            Gaze(
                samples=pl.DataFrame(
                    {'time': [0], 'distance': [567.89], 'pixel': [[41.23, 53.45]]},
                ),
            ),
            marks=[
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.distance_column.*:DeprecationWarning',
                ),
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.pixel_columns.*:DeprecationWarning',
                ),
            ],
            id='distance_column_definition',
        ),

        pytest.param(
            {
                'filename': 'test.tsv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
                'separator': '\t',
            },
            None, {'pixel_columns': ['x', 'y'], 'read_csv_kwargs': {'separator': '\t'}},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'pixel': [[1.2, 3.4]]})),
            id='pixel_columns_read_kwargs',
        ),

        pytest.param(
            {
                'filename': 'test.tsv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4]}),
                'separator': '\t',
            },
            None, None,
            {'pixel_columns': ['x', 'y'], 'custom_read_kwargs': {'gaze': {'separator': '\t'}}},
            Gaze(samples=pl.DataFrame({'time': [0], 'pixel': [[1.2, 3.4]]})),
            marks=[
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning',
                ),
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.pixel_columns.*:DeprecationWarning',
                ),
            ],
            id='pixel_columns_read_kwargs_definition',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4], 'd': [7]}),
            },
            None, {'pixel_columns': ['x', 'y'], 'column_map': {'d': 'test'}},
            {},
            Gaze(samples=pl.DataFrame({'time': [0], 'test': [7], 'pixel': [[1.2, 3.4]]})),
            id='pixel_columns_column_map',
        ),

        pytest.param(
            {
                'filename': 'test.csv',
                'data': pl.DataFrame({'time': [0], 'x': [1.2], 'y': [3.4], 'd': [8]}),
            },
            None, None,
            {'pixel_columns': ['x', 'y'], 'column_map': {'d': 'fest'}},
            Gaze(samples=pl.DataFrame({'time': [0], 'fest': [8], 'pixel': [[1.2, 3.4]]})),
            marks=[
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.column_map.*:DeprecationWarning',
                ),
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.pixel_columns.*:DeprecationWarning',
                ),
            ],
            id='pixel_columns_column_map_definition',
        ),

    ],
)
def test_load_gaze_samples_csv_file(
        make_csv_file, make_csv_file_kwargs, load_function, load_kwargs, definition_dict,
        expected_gaze,
):
    filepath = make_csv_file(**make_csv_file_kwargs)
    resource_definition = ResourceDefinition(
        content='gaze', load_function=load_function, load_kwargs=load_kwargs,
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_gaze_file(
        file=file,
        dataset_definition=DatasetDefinition(**definition_dict),
    )
    assert gaze == expected_gaze


def test_load_gaze_file_unsupported_load_function(make_example_file):
    filepath = make_example_file('monocular_example.csv')
    resource_definition = ResourceDefinition(
        content='gaze',
        load_function='from_a_land_down_under',
        load_kwargs={'pixel_columns': ['x_left_pix', 'y_left_pix']},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    with pytest.raises(ValueError) as exc:
        load_gaze_file(
            file,
            dataset_definition=DatasetDefinition(
                experiment=Experiment(1280, 1024, 38, 30, None, 'center', 1000),
            ),
        )

    msg, = exc.value.args
    assert msg == (
        'Unsupported load_function "from_a_land_down_under". '
        'Available options are: [\'from_csv\', \'from_ipc\', \'from_asc\', \'from_begaze\']'
    )


@pytest.mark.parametrize('target_filename', ['copco_rm_dummy.csv', 'copco_rm_dummy.CSV'])
def test_load_precomputed_rm_file(target_filename, make_example_file):
    filepath = make_example_file('copco_rm_dummy.csv', target_filename=target_filename)
    resource_definition = ResourceDefinition(
        content='precomputed_reading_measures',
        load_kwargs={'separator': ','},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    reading_measure = load_precomputed_reading_measure_file(
        file, dataset_definition=DatasetDefinition(),
    )
    expected_df = pl.read_csv(filepath)

    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=False)


def test_load_precomputed_rm_file_no_kwargs(make_example_file):
    filepath = make_example_file('copco_rm_dummy.csv')
    resource_definition = ResourceDefinition(content='precomputed_reading_measures')
    file = DatasetFile(path=filepath, definition=resource_definition)

    reading_measure = load_precomputed_reading_measure_file(
        file, dataset_definition=DatasetDefinition(),
    )

    expected_df = pl.read_csv(filepath)
    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=False)


def test_load_precomputed_rm_files_rda(make_example_file):
    filepath1 = make_example_file('rda_test_file.rda', '1.rda')
    filepath2 = make_example_file('rda_test_file.rda', '2.rda')

    resource_definition = ResourceDefinition(
        content='precomputed_reading_measures',
        load_kwargs={'r_dataframe_key': 'joint.fix'},
    )

    definition = DatasetDefinition(
        name='rda_dataset',
        resources=[resource_definition],
    )

    files = [
        DatasetFile(path=filepath1, definition=resource_definition, metadata={'subject_id': '1'}),
        DatasetFile(path=filepath2, definition=resource_definition, metadata={'subject_id': '2'}),
    ]

    precomputed_rm_list = load_precomputed_reading_measures(definition, files)

    for file, measures in zip(files, precomputed_rm_list):
        expected_df = pyreadr.read_r(file.path)

        assert_frame_equal(
            measures.frame,
            pl.DataFrame(expected_df['joint.fix']),
            check_column_order=False,
        )


@pytest.mark.parametrize('target_filename', ['rda_test_file.rda', 'rda_test_file.RDA'])
def test_load_precomputed_rm_file_rda(target_filename, make_example_file):
    filepath = make_example_file('rda_test_file.rda', target_filename=target_filename)
    resource_definition = ResourceDefinition(
        content='precomputed_reading_measures',
        load_kwargs={'r_dataframe_key': 'joint.fix'},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_precomputed_reading_measure_file(
        file,
        dataset_definition=DatasetDefinition(),
    )

    expected_df = pyreadr.read_r(filepath)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


@pytest.mark.filterwarnings('ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning')
def test_load_precomputed_rm_file_rda_dataset_definition_kwargs(make_example_file):
    filepath = make_example_file('rda_test_file.rda')
    resource_definition = ResourceDefinition(content='precomputed_reading_measures')
    file = DatasetFile(path=filepath, definition=resource_definition)

    dataset_definition = DatasetDefinition(
        custom_read_kwargs={'precomputed_reading_measures': {'r_dataframe_key': 'joint.fix'}},
    )
    gaze = load_precomputed_reading_measure_file(file, dataset_definition=dataset_definition)

    expected_df = pyreadr.read_r(file.path)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


@pytest.mark.parametrize('target_filename', ['Sentences.xlsx', 'Sentences.XLSX'])
def test_load_precomputed_rm_file_xlsx(target_filename, make_example_file):
    filepath = make_example_file('Sentences.xlsx', target_filename=target_filename)
    resource_definition = ResourceDefinition(
        content='precomputed_reading_measures',
        load_kwargs={'sheet_name': 'Sheet 1'},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    reading_measure = load_precomputed_reading_measure_file(
        file,
        dataset_definition=DatasetDefinition(),
    )

    expected_df = pl.from_dict({'test': ['foo', 'bar'], 'id': [0, 1]})

    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=True)


def test_load_precomputed_rm_file_unsupported_file_format(make_example_file):
    filepath = make_example_file('binocular_example.feather')
    resource_definition = ResourceDefinition(content='precomputed_events')
    file = DatasetFile(path=filepath, definition=resource_definition)

    with pytest.raises(ValueError) as exc:
        load_precomputed_reading_measure_file(file, dataset_definition=DatasetDefinition())

    msg, = exc.value.args
    assert msg == 'unsupported file format ".feather". Supported formats are: '\
        '.csv, .rda, .tsv, .txt, .xlsx'


@pytest.mark.parametrize('target_filename', ['18sat_fixfinal.csv', '18sat_fixfinal.CSV'])
def test_load_precomputed_file_csv(target_filename, make_example_file):
    filepath = make_example_file('18sat_fixfinal.csv', target_filename=target_filename)
    resource_definition = ResourceDefinition(
        content='precomputed_events',
        load_kwargs={'read_csv_kwargs': {'separator': ','}},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_precomputed_event_file(file, dataset_definition=DatasetDefinition())

    expected_df = pl.read_csv(file.path)
    assert_frame_equal(gaze.frame, expected_df, check_column_order=False)


@pytest.mark.parametrize('target_filename', ['test.jsonl', 'test.JSONL'])
def test_load_precomputed_file_json(target_filename, make_example_file):
    filepath = make_example_file('test.jsonl', target_filename=target_filename)
    file = DatasetFile(path=filepath, definition=ResourceDefinition(content='precomputed_events'))

    gaze = load_precomputed_event_file(file, dataset_definition=DatasetDefinition())
    expected_df = pl.read_ndjson(filepath)

    assert_frame_equal(gaze.frame, expected_df, check_column_order=False)


def test_load_precomputed_file_unsupported_file_format(make_example_file):
    filepath = make_example_file('binocular_example.feather')
    file = DatasetFile(path=filepath, definition=ResourceDefinition(content='precomputed_events'))

    with pytest.raises(ValueError) as exc:
        load_precomputed_event_file(file, dataset_definition=DatasetDefinition())

    msg, = exc.value.args
    assert msg == 'unsupported file format ".feather". '\
        'Supported formats are: .csv, .jsonl, .ndjson, .rda, .tsv, .txt'


def test_load_precomputed_files_rda(make_example_file):
    filepath1 = make_example_file('rda_test_file.rda', '1.rda')
    filepath2 = make_example_file('rda_test_file.rda', '2.rda')

    resource_definition = ResourceDefinition(
        content='precomputed_events',
        load_kwargs={'r_dataframe_key': 'joint.fix'},
    )

    definition = DatasetDefinition(
        name='rda_dataset',
        resources=[resource_definition],
    )

    files = [
        DatasetFile(path=filepath1, definition=resource_definition, metadata={'subject_id': '1'}),
        DatasetFile(path=filepath2, definition=resource_definition, metadata={'subject_id': '2'}),
    ]

    precomputed_events_list = load_precomputed_event_files(definition, files)

    for file, events in zip(files, precomputed_events_list):
        expected_df = pyreadr.read_r(file.path)

        assert_frame_equal(
            events.frame,
            pl.DataFrame(expected_df['joint.fix']),
            check_column_order=False,
        )


@pytest.mark.parametrize('target_filename', ['rda_test_file.rda', 'rda_test_file.RDA'])
def test_load_precomputed_file_rda(target_filename, make_example_file):
    filepath = make_example_file('rda_test_file.rda', target_filename=target_filename)
    resource_definition = ResourceDefinition(
        content='precomputed_events',
        load_kwargs={'r_dataframe_key': 'joint.fix'},
    )
    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_precomputed_event_file(file, dataset_definition=DatasetDefinition())

    expected_df = pyreadr.read_r(file.path)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


@pytest.mark.filterwarnings('ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning')
def test_load_precomputed_file_rda_dataset_definition_kwargs(make_example_file):
    filepath = make_example_file('rda_test_file.rda')
    resource_definition = ResourceDefinition(content='precomputed_events')
    file = DatasetFile(path=filepath, definition=resource_definition)

    dataset_definition = DatasetDefinition(
        custom_read_kwargs={'precomputed_events': {'r_dataframe_key': 'joint.fix'}},
    )
    gaze = load_precomputed_event_file(file, dataset_definition=dataset_definition)

    expected_df = pyreadr.read_r(file.path)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


def test_load_precomputed_file_rda_raise_value_error(make_example_file):
    filepath = make_example_file('rda_test_file.rda')
    file = DatasetFile(path=filepath, definition=ResourceDefinition(content='precomputed_events'))

    with pytest.raises(ValueError) as exc:
        load_precomputed_event_file(file, dataset_definition=DatasetDefinition())

    msg, = exc.value.args
    assert msg == 'please specify r_dataframe_key in ResourceDefinition.load_kwargs'


def test_load_precomputed_rm_file_rda_raise_value_error(make_example_file):
    filepath = make_example_file('rda_test_file.rda')
    file = DatasetFile(
        path=filepath, definition=ResourceDefinition(content='precomputed_reading_measures'),
    )

    with pytest.raises(ValueError) as exc:
        load_precomputed_reading_measure_file(file, dataset_definition=DatasetDefinition())

    msg, = exc.value.args
    assert msg == 'please specify r_dataframe_key in ResourceDefinition.load_kwargs'


@pytest.mark.parametrize(
    ('load_kwargs', 'definition_dict'),
    [
        pytest.param(
            {'trial_columns': 'trial_id'},
            {},
            id='trial_columns_via_load_kwargs',
        ),

        pytest.param(
            {},
            {'trial_columns': ['trial_id']},
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
            ),
            id='trial_columns_via_definition',
        ),

        pytest.param(
            {'trial_columns': 'wrong'},
            {'trial_columns': ['trial_id']},
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
            ),
            id='trial_columns_definition_overrides_load_kwargs',
        ),

        pytest.param(
            {},
            {'custom_read_kwargs': {'gaze': {'trial_columns': ['trial_id']}}},
            marks=pytest.mark.filterwarnings(
                'ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning',
            ),
            id='trial_columns_via_custom_read_kwargs',
        ),

        pytest.param(
            {},
            {
                'trial_columns': ['wrong'],
                'custom_read_kwargs': {'gaze': {'trial_columns': ['trial_id']}},
            },
            marks=[
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.custom_read_kwargs.*:DeprecationWarning',
                ),
                pytest.mark.filterwarnings(
                    'ignore:.*DatasetDefinition.trial_columns.*:DeprecationWarning',
                ),
            ],
            id='trial_columns_via_custom_read_kwargs_overrides_definition',
        ),
    ],
)
def test_load_gaze_file_from_begaze(load_kwargs, definition_dict, make_text_file):
    """Load a BeGaze text export via load_gaze_file using from_begaze.

    Validates that samples are parsed, time is in ms, pixel column exists,
    and BeGaze events are present.
    """
    # Inline BeGaze sample and patterns to avoid cross-test imports
    BEGAZE_TEXT = (
        '## [BeGaze]\n'
        '## Converted from:\tC:\\test.idf\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Version:\tBeGaze 3.7.40\n'
        '## IDF Version:\t9\n'
        '## Sample Rate:\t1000\n'
        '## Separator Type:\tMsg\n'
        '## Trial Count:\t1\n'
        '## Uses Plane File:\tFalse\n'
        '## Number of Samples:\t11\n'
        '## Reversed:\tnone\n'
        '## [Run]\n'
        '## Subject:\tP01\n'
        '## Description:\tRun1\n'
        '## [Calibration]\n'
        '## Calibration Area:\t1680\t1050\n'
        '## Calibration Point 0:\tPosition(841;526)\n'
        '## Calibration Point 1:\tPosition(84;52)\n'
        '## Calibration Point 2:\tPosition(1599;52)\n'
        '## Calibration Point 3:\tPosition(84;1000)\n'
        '## Calibration Point 4:\tPosition(1599;1000)\n'
        '## Calibration Point 5:\tPosition(84;526)\n'
        '## Calibration Point 6:\tPosition(841;52)\n'
        '## Calibration Point 7:\tPosition(1599;526)\n'
        '## Calibration Point 8:\tPosition(841;1000)\n'
        '## [Geometry]\n'
        '## Stimulus Dimension [mm]:\t474\t297\n'
        '## Head Distance [mm]:\t700\n'
        '## [Hardware Setup]\n'
        '## System ID:\tIRX0470703-1007\n'
        '## Operating System :\t6.1\n'
        '## IView X Version:\t2.8.26\n'
        '## [Filter Settings]\n'
        '## Heuristics:\tFalse\n'
        '## Heuristics Stage:\t0\n'
        '## Bilateral:\tTrue\n'
        '## Gaze Cursor Filter:\tTrue\n'
        '## Saccade Length [px]:\t80\n'
        '## Filter Depth [ms]:\t20\n'
        '## Format:\tLEFT, POR, QUALITY, PLANE, MSG\n'
        '##\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\tTiming'
        '\tPupil Confidence\tR Plane\tInfo\tR Event Info\tStimulus\n'
        '10000000123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
        '10000001123\tMSG\t1\t# Message: START_A\n'
        '10000002123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
        '10000003234\tMSG\t1\t# Message: STOP_A\n'
        '10000004123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
        '10000004234\tMSG\t1\t# Message: METADATA_1 123\n'
        '10000005234\tMSG\t1\t# Message: START_B\n'
        '10000006123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
        '10000007234\tMSG\t1\t# Message: START_TRIAL_1\n'
        '10000008123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
        '10000009234\tMSG\t1\t# Message: STOP_TRIAL_1\n'
        '10000010234\tMSG\t1\t# Message: START_TRIAL_2\n'
        '10000011123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tSaccade\ttest.bmp\n'
        '10000012234\tMSG\t1\t# Message: STOP_TRIAL_2\n'
        '10000013234\tMSG\t1\t# Message: START_TRIAL_3\n'
        '10000014234\tMSG\t1\t# Message: METADATA_2 abc\n'
        '10000014235\tMSG\t1\t# Message: METADATA_1 456\n'
        '10000014345\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tSaccade\ttest.bmp\n'
        '10000015234\tMSG\t1\t# Message: STOP_TRIAL_3\n'
        '10000016234\tMSG\t1\t# Message: STOP_B\n'
        '10000017234\tMSG\t1\t# Message: METADATA_3\n'
        '10000017345\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tSaccade\ttest.bmp\n'
        '10000019123\tSMP\t1\t850.71\t717.53\t714.00\t0\t0\t-1\tSaccade\ttest.bmp\n'
        '10000020123\tSMP\t1\t850.71\t717.53\t714.00\t0\t0\t-1\tBlink\ttest.bmp\n'
        '10000021123\tSMP\t1\t850.71\t717.53\t714.00\t0\t0\t-1\tBlink\ttest.bmp\n'
    )

    BEGAZE_PATTERNS = [
        {'pattern': 'START_A', 'column': 'task', 'value': 'A'},
        {'pattern': 'START_B', 'column': 'task', 'value': 'B'},
        {'pattern': ('STOP_A', 'STOP_B'), 'column': 'task', 'value': None},
        r'START_TRIAL_(?P<trial_id>\d+)',
        {'pattern': r'STOP_TRIAL', 'column': 'trial_id', 'value': None},
    ]

    BEGAZE_METADATA_PATTERNS = [
        r'METADATA_1 (?P<metadata_1>\d+)',
        {'pattern': r'METADATA_2 (?P<metadata_2>\w+)'},
        {'pattern': r'METADATA_3', 'key': 'metadata_3', 'value': True},
        {'pattern': r'METADATA_4', 'key': 'metadata_4', 'value': True},
    ]

    # Expected numeric values for comparison (subset)
    EXPECTED_TIMES = [
        10000000.123, 10000002.123, 10000004.123, 10000006.123, 10000008.123,
        10000011.123, 10000014.345, 10000017.345, 10000019.123, 10000020.123,
        10000021.123,
    ]
    EXPECTED_X = [850.7] * 9 + [None, None]
    EXPECTED_Y = [717.5] * 9 + [None, None]

    # Create a temporary BeGaze file
    filepath = make_text_file(filename='sub.txt', body=BEGAZE_TEXT, encoding='ascii')

    # Call loader with explicit from_begaze and corresponding kwargs
    resource_definition = ResourceDefinition(
        content='gaze',
        load_function='from_begaze',
        load_kwargs={
            'patterns': BEGAZE_PATTERNS,
            'metadata_patterns': BEGAZE_METADATA_PATTERNS,
            **load_kwargs,
        },
    )

    file = DatasetFile(path=filepath, definition=resource_definition)

    gaze = load_gaze_file(
        file=file,
        dataset_definition=DatasetDefinition(**definition_dict),
    )

    # from_begaze constructs a Gaze with nested pixel column from x_pix/y_pix
    # Build expected samples accordingly (time + pixel)
    # Build expected samples from inline numeric expectations
    expected_df = pl.DataFrame({
        'time': EXPECTED_TIMES,
        'pixel': [[EXPECTED_X[i], EXPECTED_Y[i]] for i in range(len(EXPECTED_TIMES))],
    })
    # Compare only rows where pixel values are present (blink rows are None in expected)
    mask = [all(v is not None for v in pair) for pair in expected_df['pixel'].to_list()]
    expected_df_non_nan = expected_df.filter(pl.Series(mask))
    gaze_non_nan = gaze.samples.filter(pl.col('pixel').list.get(0).is_not_null())
    assert_frame_equal(
        gaze_non_nan.select(['time', 'pixel']).with_columns(pl.col('time').round(3)),
        expected_df_non_nan.select(['time', 'pixel']).with_columns(pl.col('time').round(3)),
        check_column_order=False,
    )

    # Events should include BeGaze-derived names
    assert set(gaze.events.frame['name'].to_list()) == {
        'fixation_begaze', 'saccade_begaze', 'blink_begaze',
    }
    assert gaze.trial_columns == ['trial_id']


@pytest.mark.parametrize(
    ('example_filename', 'expected_shape'),
    [
        pytest.param('toy_text_aoi.csv', (81, 11), id='toy_text_aoi'),
        pytest.param('toy_text_1_1_aoi.csv', (20, 13), id='toy_text_1_1_aoi'),
        pytest.param('toy_text_2_5_aoi.csv', (19, 13), id='toy_text_2_5_aoi'),
        pytest.param('toy_text_3_8_aoi.csv', (19, 13), id='toy_text_3_8_aoi'),
    ],
)
def test_load_stimulus_file_returns_text_stimulus(
        example_filename, expected_shape, make_example_file,
):
    example_filepath = make_example_file('stimuli/' + example_filename)

    file = DatasetFile(
        path=example_filepath,
        definition=ResourceDefinition(
            content='TextStimulus',
            load_kwargs={
                'aoi_column': 'char',
                'start_x_column': 'top_left_x',
                'start_y_column': 'top_left_y',
                'width_column': 'width',
                'height_column': 'height',
                'page_column': 'page',
            },
        ),
    )
    stimulus = load_stimulus_file(file)

    assert isinstance(stimulus, TextStimulus)
    assert stimulus.aois.shape == expected_shape


def test_load_stimulus_file_returns_image_stimulus():
    filepath = Path('tests/files/stimuli/pexels-zoorg-1000498.jpg')
    file = DatasetFile(
        path=filepath,
        definition=ResourceDefinition(
            content='ImageStimulus',
        ),
    )
    stimulus = load_stimulus_file(file)

    assert isinstance(stimulus, ImageStimulus)
    assert stimulus.images == [filepath]


def test_load_stimulus_file_raises_unknown_load_function():
    file = DatasetFile(
        path='tests/files/stimuli/toy_text_1_1_aoi.csv',
        definition=ResourceDefinition(
            content='TextStimulus',
            load_function='fail',
        ),
    )

    message = 'Unknown load_function "fail". Known functions are:'
    with pytest.raises(ValueError, match=message):
        load_stimulus_file(file)


def test_load_stimulus_file_raises_missing_load_kwargs():
    file = DatasetFile(
        path='tests/files/stimuli/toy_text_1_1_aoi.csv',
        definition=ResourceDefinition(content='TextStimulus'),
    )

    message = re.escape(
        'TextStimulus.from_csv() missing 3 required keyword-only arguments: '
        "'aoi_column', 'start_x_column', and 'start_y_column'",
    )
    with pytest.raises(TypeError, match=message):
        load_stimulus_file(file)


def test_load_stimulus_file_raises_unknown_stimulus_content_type():
    file = DatasetFile(
        path='tests/files/stimuli/toy_text_1_1_aoi.csv',
        definition=ResourceDefinition(content='UnknownStimulus'),
    )

    message = re.escape(
        "Could not infer load function from content type 'UnknownStimulus'. "
        "Supported stimulus content types are: ['ImageStimulus', 'TextStimulus']",
    )
    with pytest.raises(ValueError, match=message):
        load_stimulus_file(file)
