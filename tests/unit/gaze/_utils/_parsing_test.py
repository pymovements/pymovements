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
"""Tests pymovements asc to csv processing - shared functionality."""
from pymovements.gaze._utils import _parsing


def test_data_loss_zero_expected_samples_via_parsing_module():
    """When num_expected_samples == 0 the function must return (0.0, 0.0)."""
    total, blink = _parsing._calculate_data_loss_ratio(0, 10, 5)
    assert total == 0.0
    assert blink == 0.0


def test_data_loss_zero_expected_all_zero_via_parsing_module():
    """Sanity: zero inputs produce zero outputs."""
    total, blink = _parsing._calculate_data_loss_ratio(0, 0, 0)
    assert total == 0.0
    assert blink == 0.0


def test_parse_eyelink_binocular_simple(make_text_file):
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

    filepath = make_text_file(filename='sub_binoc.asc', body=asc_text)

    gaze_df, event_df, metadata, _ = _parsing.parse_eyelink(filepath)

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
def test_parse_eyelink_binocular_missing_samples_data_loss(make_text_file):
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
        # Insert SBLINK markers at their onsets…
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

    filepath = make_text_file(filename='sub_binoc_missing.asc', body=asc_text)

    _, _, parsed_metadata, _ = _parsing.parse_eyelink(filepath)

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
def test_tracked_vs_recorded_eye_warning(make_text_file):
    """When RECCFG and SAMPLES report different eyes, a warning should be raised."""
    asc_text = (
        'MSG\t2154555 RECCFG CR 1000 2 1 LR\n'
        'SAMPLES\tGAZE\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n'
    )

    filepath = make_text_file(filename='sub_mismatch.asc', body=asc_text)

    # The parser maps RECCFG 'LR' -> 'LR' and SAMPLES 'RIGHT' -> 'R', so expect [LR, R]
    with pytest.warns(Warning, match=r'inconsistent: \[LR, R\]'):
        _parsing.parse_eyelink(filepath)


@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No recording configuration found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_sampling_rate_inconsistent_warning(make_text_file):
    """When RECCFG and SAMPLES report different sampling rates, a warning should be raised."""
    asc_text = (
        'MSG\t2154555 RECCFG CR 2000 2 1 L\n'
        'SAMPLES\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2\n'
    )

    filepath = make_text_file(filename='sub_rate_mismatch.asc', body=asc_text)

    # Depending on internal casting, the warning should contain both values; match the float form
    with pytest.warns(
        Warning,
        match=r"inconsistent values for 'sampling_rate': \[1000\.0, 2000\.0\]",
    ):
        _parsing.parse_eyelink(filepath)


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
def test_tracked_eye_mapping_from_samples(make_text_file, samples_line, expected_tracked):
    """SAMPLES tracked_eye strings should map to metadata['tracked_eye'] correctly."""
    asc_text = samples_line

    filepath = make_text_file(filename='sub_tracked.asc', body=asc_text)

    _, _, metadata, _ = _parsing.parse_eyelink(filepath)

    assert metadata['tracked_eye'] == expected_tracked


@pytest.mark.filterwarnings('ignore:No metadata found.')
def test_recording_config_missing_sampling_rate_key(monkeypatch, make_text_file):
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

    filepath = make_text_file(filename='sub_missing_rate_key.asc', body=asc_text)

    # Now the parser should warn (e.g. missing samples or sampling rate) and
    # set data-loss metrics to None.
    with pytest.warns(Warning):
        _, _, metadata, _ = _parsing.parse_eyelink(filepath)

    assert metadata['data_loss_ratio'] is None
    assert metadata['data_loss_ratio_blinks'] is None


def test_parse_eyelink_stop_recording_calculates_expected_samples(make_text_file):
    """Write a minimal asc file with RECCFG, START, and END to exercise recording_config block.

    The RECCFG line provides a sampling rate of 1000 Hz and the START/END span 1000 ms,
    so expected number of samples = 1000 and total_recording_duration_ms should be 1000.0.
    """
    content = (
        'MSG 0 RECCFG CR 1000 0 0 LR\n'
        'START 0 RIGHT types\n'
        'END 1000 types RES 0 0\n'
    )

    p = make_text_file(filename='test_stop_recording.asc', body=content)

    # parse_eyelink will emit a metadata warning for this minimal file; capture it
    with pytest.warns(UserWarning):
        _, _, metadata, _ = _parsing.parse_eyelink(str(p))

    # Duration should be 1000 ms
    assert metadata['total_recording_duration_ms'] == 1000.0

    # Because no valid samples were parsed, expected samples should be 1000 -> full data loss
    assert metadata['data_loss_ratio'] == 1.0
    assert metadata['data_loss_ratio_blinks'] == 0.0


def test_check_reccfg_key_warns_on_empty_config() -> None:
    # Empty recording_config should produce a warning and return None
    with pytest.warns(UserWarning):
        assert _parsing._check_reccfg_key([], 'sampling_rate') is None


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
        result = _parsing._check_reccfg_key(recording_config, 'sampling_rate')

    # Function should return None when inconsistent values are found
    assert result is None

    # A warning should have been emitted about inconsistent values
    assert len(w) >= 1
    assert any("Found inconsistent values for 'sampling_rate'" in str(rec.message) for rec in w)


@pytest.mark.parametrize(
    ('event_end', 'event_name'),
    [
        ('EBLINK R 1000\t2000\t3', 'blink'),
        ('EFIX\tR\t1000\t2000\t9\t850.7\t717.5\t714.0', 'fixation'),
        (
            'ESACC\tR\t1000\t2000\t12\t850.7\t717.5\t850.7\t717.5\t19.00\t590',
            'saccade',
        ),
        # Left eye variants
        ('EBLINK L 1000\t2000\t3', 'blink'),
        ('EFIX\tL\t1000\t2000\t9\t850.7\t717.5\t714.0', 'fixation'),
        (
            'ESACC\tL\t1000\t2000\t12\t850.7\t717.5\t850.7\t717.5\t19.00\t590',
            'saccade',
        ),
    ],
)
def test_unmatched_event_end_uses_current_context_and_warns(
    make_text_file, event_end, event_name,
):
    """Unmatched end events should not crash, should warn, and should preserve pattern columns.

    We set task=B and trial_id=42 via MSG patterns before the unmatched end line.
    The event row should include these values in the additional columns.
    """
    header = (
        '** DATE: Wed Mar  8 09:25:20 2023\n'
        '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED\n'
        '** VERSION: EYELINK II 1\n'
        'MSG\t0 RECCFG CR 1000 0 0 R\n'
        'SAMPLES\tGAZE\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t0\n'
        'START\t0 \tRIGHT\tSAMPLES\tEVENTS\n'
        'MSG 1 START_B\n'
        'MSG 2 START_TRIAL_42\n'
    )
    # No matching start event; directly write an end event line
    body = event_end + '\nEND\t2001 \tSAMPLES\tEVENTS\tRES\t 38.54\t 31.12\n'

    filepath = make_text_file(filename='unmatched.asc', header=header, body=body)

    patterns = [
        {'pattern': 'START_B', 'column': 'task', 'value': 'B'},
        r'START_TRIAL_(?P<trial_id>\d+)',
    ]

    # Expect a warning about missing start marker
    with pytest.warns(UserWarning) as warn:
        _, event_df, _, _ = _parsing.parse_eyelink(filepath, patterns=patterns)

    assert len(event_df) == 1
    assert event_df['name'][0] == f'{event_name}_eyelink'
    # additional columns should be preserved from current context
    assert event_df['task'][0] == 'B'
    assert event_df['trial_id'][0] == '42'

    # Check a warning mentioning missing start and onset/offset
    messages = [str(rec.message) for rec in warn]
    assert any('Missing start marker before end for event' in m for m in messages)
    assert any(event_name in m for m in messages)


@pytest.mark.parametrize(
    ('eye', 'onset', 'offset'),
    [
        ('R', 100, 200),
        ('L', 100, 200),
        ('R', 0, 1),
        ('L', 250, 275),
    ],
)
def test_unmatched_blink_no_patterns_warns_and_records_param(make_text_file, eye, onset, offset):
    """Unmatched blink end without patterns should warn and record event for both eyes.

    Parametrized over left/right eyes and multiple onset/offset pairs to ensure
    robust behavior without relying on pattern-derived additional columns.
    """
    samples_eye = 'RIGHT' if eye == 'R' else 'LEFT'
    header = (
        '** DATE: Wed Mar  8 09:25:20 2023\n'
        '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED\n'
        f'MSG\t0 RECCFG CR 1000 0 0 {"R" if eye == "R" else "L"}\n'
        f'SAMPLES\tGAZE\t{samples_eye}\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t0\n'
        f'START\t0 \t{samples_eye}\tSAMPLES\tEVENTS\n'
    )

    body = (
        f'EBLINK {eye} {onset}\t{offset}\t3\n'
        f'END\t{offset + 1} \tSAMPLES\tEVENTS\tRES\t 0\t 0\n'
    )

    filepath = make_text_file(
        filename=f'unmatched_no_patterns_{eye}_{onset}_{offset}.asc',
        header=header, body=body,
    )

    with pytest.warns(UserWarning) as warn:
        _, event_df, _, _ = _parsing.parse_eyelink(filepath)

    assert len(event_df) == 1
    assert event_df['name'][0] == 'blink_eyelink'
    expected_eye = 'right' if eye == 'R' else 'left'
    assert event_df['eye'][0] == expected_eye
    assert event_df['onset'][0] == float(onset)
    assert event_df['offset'][0] == float(offset)

    messages = [str(rec.message) for rec in warn]
    assert any('Missing start marker before end for event' in m for m in messages)


@pytest.mark.parametrize(
    ('reccfg_line', 'has_event_filters'),
    [
        # normal RECCFG without event filter fields
        ('MSG\t0 RECCFG CR 1000 2 1 R\n', False),
        # extended RECCFG with file_event_filter and link_event_filter present
        ('MSG\t0 RECCFG CR 1000 2 1 2 1 R\n', True),
    ],
)
@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_reccfg_with_optional_event_filters_parses(
    make_text_file, reccfg_line, has_event_filters,
):
    """RECCFG lines with and without optional event filters should parse without error.

    Regression test for RECCFG lines of the form:
    MSG xxxxxx RECCFG CR 1000 2 1 2 1 R
    """
    asc_text = reccfg_line

    filepath = make_text_file(filename='sub_reccfg.asc', body=asc_text)

    # Should parse without any warning about missing recording configuration
    _, _, metadata, _ = _parsing.parse_eyelink(filepath)

    # Ensure we captured a recording_config in metadata
    rec_cfg_list = metadata.get('recording_config', [])
    assert isinstance(rec_cfg_list, list) and len(rec_cfg_list) == 1

    rec_cfg = rec_cfg_list[0]
    # Common fields
    assert rec_cfg['tracking_mode'] == 'CR'
    assert rec_cfg['sampling_rate'] == '1000'
    assert rec_cfg['file_sample_filter'] in {'0', '1', '2'}
    assert rec_cfg['link_sample_filter'] in {'0', '1', '2'}
    assert rec_cfg['tracked_eye'] in {'L', 'R', 'LR'}

    # Optional fields
    if has_event_filters:
        assert rec_cfg.get('file_event_filter') in {'0', '1', '2'}
        assert rec_cfg.get('link_event_filter') in {'0', '1', '2'}
    else:
        assert rec_cfg.get('file_event_filter') is None
        assert rec_cfg.get('link_event_filter') is None


@pytest.mark.filterwarnings('ignore:No metadata found.')
@pytest.mark.filterwarnings('ignore:No samples configuration found.')
def test_gaze_coords_without_reccfg_warns_and_skips(make_text_file):
    """GAZE_COORDS before any RECCFG should warn and not crash."""
    asc_text = 'MSG\t12345 GAZE_COORDS 0 0 1919 1079\n'

    filepath = make_text_file(filename='sub_gaze_only.asc', body=asc_text)

    with pytest.warns(
        UserWarning,
        match='GAZE_COORDS encountered before any RECCFG|No recording configuration found',
    ):
        _, _, metadata, _ = _parsing.parse_eyelink(filepath)

    assert metadata.get('recording_config', []) == []
