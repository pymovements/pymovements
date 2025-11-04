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
# flake8: noqa: E101, W191, E501
# pylint: disable=duplicate-code
import polars as pl
import pyreadr
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements.dataset.dataset_definition import DatasetDefinition


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

EXPECTED_DF_NO_PATTERNS = pl.from_dict(
    {
        'time': [2154557, 2154558, 2154560, 2154561, 2154565, 2154567, 2154568],
        'pixel': [
            (139.6, 132.1), (139.5, 131.9), (None, None), (850.7, 717.5),
            (139.5, 131.9), (None, None), (850.7, 717.5),
        ],
        'pupil': [784.0, 784.0, 0.0, 714.0, 784.0, 0.0, 714.0],
    },
)

EXPECTED_DF_PATTERNS = pl.from_dict(
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

PATTERNS = [
    {
        'pattern': 'SYNCTIME_READING',
        'column': 'task',
        'value': 'reading',
    },
    r'TRIALID (?P<trial_id>\d+)',
]


@pytest.mark.parametrize(
    ('load_kwargs', 'expected_samples'),
    [
        pytest.param(
            {'patterns': PATTERNS, 'schema': {'trial_id': pl.Int64}},
            EXPECTED_DF_PATTERNS,
            id='patterns',
        ),
        pytest.param(
            None,
            EXPECTED_DF_NO_PATTERNS,
            id='none',
        ),
    ],
)
def test_load_eyelink_file(load_kwargs, expected_samples, make_text_file):
    filepath = make_text_file(filename='sub.asc', body=ASC_TEXT)

    gaze = pm.dataset.dataset_files.load_gaze_file(
        filepath,
        fileinfo_row={'load_function': 'from_asc', 'load_kwargs': load_kwargs},
        definition=DatasetDefinition(
            experiment=pm.Experiment(1280, 1024, 38, 30, None, 'center', 1000),
        ),
    )

    assert_frame_equal(gaze.samples, expected_samples, check_column_order=False)
    assert gaze.experiment is not None


@pytest.mark.parametrize(
    ('filename', 'rename_extension', 'load_function', 'load_kwargs', 'preprocessed'),
    [
        pytest.param(
            'monocular_example.csv',
            '.csv',
            'from_csv',
            None,
            False,
            id='load_csv_from_csv',
        ),
        pytest.param(
            'monocular_example.csv',
            '.renamed',
            'from_csv',
            None,
            False,
            id='load_csv_rename_from_csv',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.tsv',
            'from_csv',
            {'read_csv_kwargs': {'separator': '\t'}},
            False,
            id='load_tsv_from_csv',
        ),
        pytest.param(
            'monocular_example.tsv',
            '.foo',
            'from_csv',
            {'read_csv_kwargs': {'separator': '\t'}},
            False,
            id='load_tsv_rename_from_csv',
        ),
        pytest.param(
            'monocular_example.feather',
            '.feather',
            None,
            None,
            True,
            id='load_feather_default_preprocessed',
        ),
        pytest.param(
            'monocular_example.feather',
            '.feather',
            'from_ipc',
            None,
            False,
            id='load_feather_from_ipc',
        ),
        pytest.param(
            'monocular_example.feather',
            '.csv',
            'from_ipc',
            None,
            False,
            id='load_feather_rename_from_ipc',
        ),
    ],
)
def test_load_gaze_file(
        filename, rename_extension, preprocessed, load_function, load_kwargs,
        tmp_path, make_example_file,
):
    # Copy the file to the temporary path with the new extension
    filepath = make_example_file(filename)
    renamed_filename = filepath.stem + rename_extension
    renamed_filepath = tmp_path / renamed_filename
    renamed_filepath.write_bytes(filepath.read_bytes())

    gaze = pm.dataset.dataset_files.load_gaze_file(
        renamed_filepath,
        fileinfo_row={'load_function': load_function, 'load_kwargs': load_kwargs},
        definition=DatasetDefinition(
            experiment=pm.Experiment(1280, 1024, 38, 30, None, 'center', 1000),
            pixel_columns=['x_left_pix', 'y_left_pix'],
        ),
        preprocessed=preprocessed,
    )
    expected_df = pl.from_dict(
        {
            'time': list(range(10)),
            'pixel': [[0, 0]] * 10,
        },
    )

    assert_frame_equal(gaze.samples, expected_df, check_column_order=False)


def test_load_gaze_file_unsupported_load_function(make_example_file):
    filepath = make_example_file('monocular_example.csv')

    with pytest.raises(ValueError) as exc:
        pm.dataset.dataset_files.load_gaze_file(
            filepath,
            fileinfo_row={'load_function': 'from_a_land_down_under', 'load_kwargs': None},
            definition=DatasetDefinition(
                experiment=pm.Experiment(1280, 1024, 38, 30, None, 'center', 1000),
                pixel_columns=['x_left_pix', 'y_left_pix'],
            ),
        )

    msg, = exc.value.args
    assert msg == (
        'Unsupported load_function "from_a_land_down_under". '
        'Available options are: [\'from_csv\', \'from_ipc\', \'from_asc\']'
    )


def test_load_precomputed_rm_file(make_example_file):
    filepath = make_example_file('copco_rm_dummy.csv')

    reading_measure = pm.dataset.dataset_files.load_precomputed_reading_measure_file(
        filepath,
        custom_read_kwargs={'separator': ','},
    )
    expected_df = pl.read_csv(filepath)

    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=False)


def test_load_precomputed_rm_file_no_kwargs(make_example_file):
    filepath = make_example_file('copco_rm_dummy.csv')

    reading_measure = pm.dataset.dataset_files.load_precomputed_reading_measure_file(
        filepath,
    )
    expected_df = pl.read_csv(filepath)

    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=False)


def test_load_precomputed_rm_file_xlsx(make_example_file):
    filepath = make_example_file('Sentences.xlsx')

    reading_measure = pm.dataset.dataset_files.load_precomputed_reading_measure_file(
        filepath,
        custom_read_kwargs={'sheet_name': 'Sheet 1'},
    )

    expected_df = pl.from_dict({'test': ['foo', 'bar'], 'id': [0, 1]})

    assert_frame_equal(reading_measure.frame, expected_df, check_column_order=True)


def test_load_precomputed_rm_file_unsupported_file_format(make_example_file):
    filepath = make_example_file('binocular_example.feather')

    with pytest.raises(ValueError) as exc:
        pm.dataset.dataset_files.load_precomputed_reading_measure_file(filepath)

    msg, = exc.value.args
    assert msg == 'unsupported file format ".feather". Supported formats are: '\
        '.csv, .rda, .tsv, .txt, .xlsx'


def test_load_precomputed_file_csv(make_example_file):
    filepath = make_example_file('18sat_fixfinal.csv')

    gaze = pm.dataset.dataset_files.load_precomputed_event_file(
        filepath,
        custom_read_kwargs={'separator': ','},
    )
    expected_df = pl.read_csv(filepath)

    assert_frame_equal(gaze.frame, expected_df, check_column_order=False)


def test_load_precomputed_file_json(make_example_file):
    filepath = make_example_file('test.jsonl')

    gaze = pm.dataset.dataset_files.load_precomputed_event_file(filepath)
    expected_df = pl.read_ndjson(filepath)

    assert_frame_equal(gaze.frame, expected_df, check_column_order=False)


def test_load_precomputed_file_unsupported_file_format(make_example_file):
    filepath = make_example_file('binocular_example.feather')

    with pytest.raises(ValueError) as exc:
        pm.dataset.dataset_files.load_precomputed_event_file(filepath)

    msg, = exc.value.args
    assert msg == 'unsupported file format ".feather". '\
        'Supported formats are: .csv, .jsonl, .ndjson, .rda, .tsv, .txt'


def test_load_precomputed_file_rda(make_example_file):
    filepath = make_example_file('rda_test_file.rda')

    gaze = pm.dataset.dataset_files.load_precomputed_event_file(
        filepath,
        custom_read_kwargs={'r_dataframe_key': 'joint.fix'},
    )

    expected_df = pyreadr.read_r(filepath)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


def test_load_precomputed_file_rda_raise_value_error(make_example_file):
    filepath = make_example_file('rda_test_file.rda')

    with pytest.raises(ValueError) as exc:
        pm.dataset.dataset_files.load_precomputed_event_file(filepath)

    msg, = exc.value.args
    assert msg == 'please specify r_dataframe_key in custom_read_kwargs'


def test_load_precomputed_rm_file_rda(make_example_file):
    filepath = make_example_file('rda_test_file.rda')

    gaze = pm.dataset.dataset_files.load_precomputed_reading_measure_file(
        filepath,
        custom_read_kwargs={'r_dataframe_key': 'joint.fix'},
    )

    expected_df = pyreadr.read_r(filepath)

    assert_frame_equal(
        gaze.frame,
        pl.DataFrame(expected_df['joint.fix']),
        check_column_order=False,
    )


def test_load_precomputed_rm_file_rda_raise_value_error(make_example_file):
    filepath = make_example_file('rda_test_file.rda')

    with pytest.raises(ValueError) as exc:
        pm.dataset.dataset_files.load_precomputed_reading_measure_file(filepath)

    msg, = exc.value.args
    assert msg == 'please specify r_dataframe_key in custom_read_kwargs'
