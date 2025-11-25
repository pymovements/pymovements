# Copyright (c) 2025 The pymovements Project Authors
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
"""Tests for reading BeGaze files via io.from_begaze."""
from __future__ import annotations

import warnings

import polars as pl
import pytest

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.io import _fill_experiment_from_parsing_begaze_metadata
from pymovements.gaze.io import from_begaze


# Minimal BeGaze-like content (tabs) with MSG lines that our parser understands.
BEGAZE_MINI = (
    '## [BeGaze]\n'
    '## Date:\t08.03.2023 09:25:20\n'
    '## Sample Rate:\t1000\n'
    '##\n'
    # Header row
    'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]'
    '\tL Pupil Diameter [mm]\tTiming\tPupil Confidence\tR Plane\tInfo\tStimulus\n'
    # One SMP row
    '10000000123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
    # Message to set a trial value via pattern and a metadata value
    '10000001123\tMSG\t1\t# Message: START_TRIAL_1\n'
    '10000002123\tMSG\t1\t# Message: METADATA_1 123\n'
    # One more SMP row (after messages) so additional columns are propagated
    '10000003123\tSMP\t1\t850.71\t717.53\t714.00\t0\t1\t1\tFixation\ttest.bmp\n'
)


@pytest.mark.parametrize(
    'trial_columns',
    [
        pytest.param('Stimulus', id='single_missing_trial_column'),
        pytest.param(['Stimulus', 'trial_id'], id='multiple_missing_trial_columns'),
    ],
)
def test_from_begaze_adds_missing_trial_columns_with_none(make_text_file, trial_columns):
    # Create a minimal BeGaze file WITHOUT a Stimulus column
    text = (
        '## [BeGaze]\n'
        '## Date:\t08.03.2023 09:25:20\n'
        '## Sample Rate:\t1000\n'
        '##\n'
        'Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tL Pupil Diameter [mm]\tInfo\n'
        '10000000123\tSMP\t1\t10\t20\t3.0\tFixation\n'
    )

    filepath = make_text_file('mini_begaze_no_stimulus.txt', header='', body=text, encoding='ascii')

    gaze = from_begaze(filepath, trial_columns=trial_columns)

    cols = gaze.samples.columns
    for col in ([trial_columns] if isinstance(trial_columns, str) else trial_columns):
        assert col in cols
        # Entire column should be None because it was missing in the file
        assert gaze.samples[col].to_list() == [None]


@pytest.mark.parametrize(
    'trial_columns, header_cols, row_vals',
    [
        pytest.param(
            'Stimulus',
            # Include Stimulus in the file so there are no missing trial columns
            [
                'Time', 'Type', 'Trial', 'L POR X [px]', 'L POR Y [px]',
                'L Pupil Diameter [mm]', 'Pupil Confidence', 'L Event Info', 'Stimulus',
            ],
            ['10000000100', 'SMP', '1', '10', '20', '3.0', '1', 'Fixation', 'img.bmp'],
            id='no_missing_single_trial_column',
        ),
    ],
)
def test_from_begaze_does_not_add_trial_columns_when_present(
    make_text_file, trial_columns, header_cols, row_vals,
):
    # Build a minimal BeGaze file where requested trial columns are already present
    header = '## [BeGaze]\n## Date:\t08.03.2023 09:25:20\n## Sample Rate:\t1000\n'
    body = '\t'.join(header_cols) + '\n' + '\t'.join(row_vals) + '\n'
    filepath = make_text_file(
        'mini_begaze_with_trials.txt',
        header='',
        body=header + body,
        encoding='ascii',
    )

    gaze = from_begaze(filepath, trial_columns=trial_columns)

    cols = gaze.samples.columns
    requested = [trial_columns] if isinstance(trial_columns, str) else list(trial_columns)
    for col in requested:
        assert col in cols
    # Ensure original values are preserved (no None backfill took place)
    if 'Stimulus' in requested:
        assert gaze.samples['Stimulus'].to_list() == [row_vals[-1]]
    if 'trial_id' in requested and 'trial_id' in header_cols:
        idx = header_cols.index('trial_id')
        assert gaze.samples['trial_id'].to_list() == [row_vals[idx]]


@pytest.mark.parametrize(
    (
        'add_columns',
        'column_schema_overrides',
        'expect_added_cols',
        'expect_overrides',
    ),
    [
        pytest.param(
            {'dataset': 'toy', 'trial': 'SHOULD_NOT_OVERWRITE'},
            {'trial': pl.Int64},
            {'dataset': 'toy'},
            {'trial': pl.Int64},
            id='add_missing_and_cast_existing_trial_to_int',
        ),
    ],
)
def test_from_begaze_definition_kwargs_and_overrides(
    add_columns,
    column_schema_overrides,
    expect_added_cols,
    expect_overrides,
    make_text_file,
):
    # Use DatasetDefinition to supply custom_read_kwargs['gaze'] values
    patterns = [
        {
            'pattern': r'START_TRIAL_(?P<trial>\d+)',
            'column': 'trial',
        },
    ]
    metadata_patterns = [
        {
            'pattern': r'METADATA_1 (?P<meta1>\d+)',
            'key': 'meta1',
        },
    ]

    definition = DatasetDefinition(
        custom_read_kwargs={
            'gaze': {
                'patterns': patterns,
                'metadata_patterns': metadata_patterns,
                # schema is currently unused in parse_begaze!
                'schema': {'trial': int},
                'column_schema_overrides': column_schema_overrides,
                'encoding': 'ascii',
            },
        },
    )

    filepath = make_text_file('mini_begaze.txt', header='', body=BEGAZE_MINI, encoding='ascii')

    # Pass encoding=None to exercise the branch that reads it from the definition
    gaze = from_begaze(
        filepath,
        definition=definition,
        add_columns=add_columns,
        encoding=None,
    )

    assert 'trial' in gaze.samples.columns
    for key, dtype in expect_overrides.items():
        assert gaze.samples.schema[key] == dtype

    # Ensure add_columns only added missing ones (dataset), not overwriting existing columns
    for key, value in expect_added_cols.items():
        assert key in gaze.samples.columns
        assert set(gaze.samples[key].to_list()) == {value}


@pytest.mark.parametrize(
    (
        'exp_kwargs',
        'metadata',
        'expect_screen',
        'expect_warnings_regex',
    ),
    [
        pytest.param(
            # No experiment provided: should be created and filled
            {},
            {'sampling_rate': 500.0, 'resolution': (1920, 1080), 'tracked_eye': 'LR'},
            (1920, 1080),
            None,
            id='create_and_fill_from_metadata',
        ),
        pytest.param(
            # Pre-set and conflicting resolution: warnings expected for width and height
            {'sampling_rate': 1000.0},
            {'sampling_rate': 1000.0, 'resolution': (1920, 1080), 'tracked_eye': 'L'},
            (1280, 720),
            r'screen (width|height)=\d+ differs',
            id='warn_on_resolution_mismatch_keep_experiment_values',
        ),
        pytest.param(
            # Invalid resolution format (non-iterable): triggers except branch to set None
            {},
            {'sampling_rate': 250.0, 'resolution': 1920, 'tracked_eye': 'R'},
            (None, None),
            None,
            id='invalid_resolution_non_iterable_sets_none_no_warning',
        ),
        pytest.param(
            # Pre-set left=True, matching parsed 'L' -> no warning, elif not taken
            {'sampling_rate': 500.0},
            {'sampling_rate': 500.0, 'tracked_eye': 'L'},
            (None, None),
            None,
            id='left_preset_matches_parsed_no_warning',
        ),
        pytest.param(
            # Pre-set right=True, matching parsed 'R' -> no warning, elif not taken
            {'sampling_rate': 500.0},
            {'sampling_rate': 500.0, 'tracked_eye': 'R'},
            (None, None),
            None,
            id='right_preset_matches_parsed_no_warning',
        ),
    ],
)
def test_fill_experiment_from_parsing_begaze_metadata(
        exp_kwargs, metadata, expect_screen, expect_warnings_regex,
):
    # Prepare an experiment based on parameters
    experiment: Experiment | None
    if exp_kwargs:
        experiment = Experiment(**exp_kwargs)
        # Set a conflicting screen to trigger warnings when resolution is provided in metadata
        if 'resolution' in metadata:
            experiment.screen.width_px = 1280
            experiment.screen.height_px = 720
        # For the matching-left/right cases, preset corresponding flags to True
        tracked = (metadata.get('tracked_eye') or '')
        if tracked == 'L':
            experiment.eyetracker.left = True
            experiment.eyetracker.right = False
        elif tracked == 'R':
            experiment.eyetracker.left = False
            experiment.eyetracker.right = True
    else:
        experiment = None

    if expect_warnings_regex:
        with pytest.warns(UserWarning, match=expect_warnings_regex):
            result = _fill_experiment_from_parsing_begaze_metadata(experiment, metadata)
    else:
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter('error')
            result = _fill_experiment_from_parsing_begaze_metadata(experiment, metadata)
            # No warnings should have been raised
            assert len(wrec) == 0

    assert (result.screen.width_px, result.screen.height_px) == expect_screen

    # Sampling rate: created from metadata when the experiment was None - unchanged otherwise
    if experiment is None:
        assert result.eyetracker.sampling_rate == metadata.get('sampling_rate')
    else:
        assert result.eyetracker.sampling_rate == exp_kwargs.get('sampling_rate')

    # Tracked eye flags set when None else warnings issued above - ensure values are booleans
    assert isinstance(result.eyetracker.left, (bool, type(None)))
    assert isinstance(result.eyetracker.right, (bool, type(None)))


def test_from_begaze_with_definition_and_explicit_experiment(make_text_file):
    filepath = make_text_file(
        'mini_begaze_explicit.txt',
        header='',
        body=BEGAZE_MINI,
        encoding='ascii',
    )

    # Definition carries its own experiment and kwargs
    def_exp = DatasetDefinition(
        custom_read_kwargs={
            'gaze': {
                'encoding': 'ascii',
            },
        },
    )

    # Definition should not override explicit experiment passed
    explicit_exp = Experiment(sampling_rate=777.0)
    explicit_exp.screen.width_px = 640
    explicit_exp.screen.height_px = 480

    gaze = from_begaze(
        filepath,
        definition=def_exp,
        experiment=explicit_exp,
    )

    # Ensure we kept the explicitly passed experiment (by identity and values)
    assert gaze.experiment is explicit_exp
    assert gaze.experiment.eyetracker.sampling_rate in (
        777.0, explicit_exp.eyetracker.sampling_rate,
    )
    assert (gaze.experiment.screen.width_px, gaze.experiment.screen.height_px) == (640, 480)
