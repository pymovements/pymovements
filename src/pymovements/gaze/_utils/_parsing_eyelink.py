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
"""EyeLink parsing module.

This module provides a dedicated namespace for EyeLink-specific parsing logic.
"""
from __future__ import annotations

__all__ = [
    'parse_eyelink',
]

import calendar
import datetime
import re

import warnings
from collections import defaultdict
from collections.abc import Sequence

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pymovements.gaze._utils._parsing import compile_patterns, get_pattern_keys, \
    check_nan, _calculate_data_loss_ratio


# Define separate regex patterns for monocular and binocular cases
EYE_TRACKING_SAMPLE_MONOCULAR = re.compile(
    r'(?P<time>(\d+[.]?\d*))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil>\d*[.]\d*|\.)?\s*'
    r'(?P<dummy>\d*[.]\d*|\.)?\s*'
    r'(?P<flags>[A-Za-z.]{3,5})?\s*',
)

EYE_TRACKING_SAMPLE_BINOCULAR = re.compile(
    r'(?P<time>(\d+[.]?\d*))\s+'
    r'(?P<x_pix_left>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix_left>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil_left>\d*[.]\d*|\.)?\s*'
    r'(?P<x_pix_right>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix_right>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil_right>\d*[.]\d*|\.)?\s*'
    r'(?P<dummy>\d*[.]\d*|\.)?\s*'
    r'(?P<flags>[A-Za-z.]{3,5})?\s*',
)

EYELINK_META_REGEXES = [
    {'pattern': re.compile(regex)} for regex in (
        r'\*\*\s+VERSION:\s+(?P<version_1>.*)\s+',
        (
            r'\*\*\s+DATE:\s+(?P<weekday>[A-Z,a-z]+)\s+(?P<month>[A-Z,a-z]+)'
            r'\s+(?P<day>\d\d?)\s+(?P<time>\d\d:\d\d:\d\d)\s+(?P<year>\d{4})\s*'
        ),
        r'\*\*\s+(?P<version_2>EYELINK.*)',
        r'MSG\s+\d+[.]?\d*\s+DISPLAY_COORDS\s*=?\s*(?P<DISPLAY_COORDS>.*)',
        r'PUPIL\s+(?P<pupil_data_type>(AREA|DIAMETER))\s*',
        r'MSG\s+\d+[.]?\d*\s+ELCLCFG\s+(?P<mount_configuration>.*)',
    )
]

VALIDATION_REGEX = re.compile(
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s+VALIDATION\s+HV'
    r'(?P<num_points>\d\d?).*'
    r'(?P<tracked_eye>LEFT|RIGHT)\s+'
    r'(?P<error>\D*)\s+'
    r'(?P<validation_score_avg>\d.\d\d)\s+avg\.\s+'
    r'(?P<validation_score_max>\d.\d\d)\s+max',
)

FIXATION_START_REGEX = re.compile(r'SFIX\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*')
FIXATION_STOP_REGEX = re.compile(
    r'EFIX\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s+'
    r'(?P<avg_x_pix>(\d+[.]?\d*))\s+(?P<avg_y_pix>(\d+[.]?\d*))\s+(?P<avg_pupil>(\d+[.]?\d*))\s*.*',
)
SACCADE_START_REGEX = re.compile(r'SSACC\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*')
SACCADE_STOP_REGEX = re.compile(
    r'ESACC\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s+'
    r'(?P<start_x_pix>(\d+[.]?\d*))\s+(?P<start_y_pix>(\d+[.]?\d*))\s+'
    r'(?P<end_x_pix>(\d+[.]?\d*))\s+(?P<end_y_pix>(\d+[.]?\d*))\s+'
    r'(?P<amplitude>(\d+[.]?\d*))\s+(?P<peak_velocity>(\d+[.]?\d*))\s*.*',
)
BLINK_START_REGEX = re.compile(r'SBLINK\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*')
BLINK_STOP_REGEX = re.compile(
    r'EBLINK\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s*',
)

CALIBRATION_TIMESTAMP_REGEX = re.compile(r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s*\n')

CALIBRATION_REGEX = re.compile(
    r'>+\s+CALIBRATION\s+\(HV(?P<num_points>\d\d?),'
    r'(?P<type>.*)\).*'
    r'(?P<tracked_eye>RIGHT|LEFT):\s+<{9}',
)

RECORDING_CONFIG_REGEX = re.compile(
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+'
    r'RECCFG\s+(?P<tracking_mode>[A-Z,a-z]+)\s+'
    r'(?P<sampling_rate>\d+)\s+'
    r'(?P<file_sample_filter>0|1|2)\s+'
    r'(?P<link_sample_filter>0|1|2)\s+'
    r'((?P<file_event_filter>0|1|2)\s+)?'
    r'((?P<link_event_filter>0|1|2)\s+)?'
    r'(?P<tracked_eye>LR|[LR])\s*',
)

# Resolution (GAZE_COORDS) pattern used to extract screen coordinates
GAZE_COORDS_REGEX = re.compile(
    r'MSG\s+\d+[.]?\d*\s+GAZE_COORDS\s*=?\s*(?P<resolution>.*)',
)

# Regex to match SAMPLES lines and capture which eyes are present (LEFT, RIGHT, LEFT RIGHT, LR)
SAMPLES_CONFIG_REGEX = re.compile(
    r'SAMPLES\s+GAZE\s+'
    r'(?P<tracked_eye>(?:LEFT\s+RIGHT|LEFT|RIGHT|LR|[LR]))'
    r'(?:\s+RATE\s+(?P<sampling_rate>\d+(?:\.\d+)?))?'
    r'(?:\s+TRACKING\s+(?P<tracking_method>\S+))?'
    r'(?:\s+FILTER\s+(?P<filter>\d+))?'
    r'(?:\s+(?P<input_flag>INPUT))?',
    re.IGNORECASE,
)
START_RECORDING_REGEX = re.compile(
    r'START\s+(?P<timestamp>(\d+[.]?\d*))\s+(RIGHT|LEFT)\s+(?P<types>.*)',
)
STOP_RECORDING_REGEX = re.compile(
    r'END\s+(?P<timestamp>(\d+[.]?\d*))\s+(?P<types>.*)\s+RES\s+'
    r'(?P<xres>[\d\.]*)\s+(?P<yres>[\d\.]*)\s*',
)

# General message format
MSG_REGEX = re.compile(
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+(?P<content>.*)',
)


def _check_reccfg_key(
        recording_config: list[dict[str, Any]],
        key: str,
        astype: type | None = None,
) -> Any:
    """Check if the recording configs contain consistent values for the specified key and return it.

    Prints a warning if no recording config is found or if the value is inconsistent across entries.

    Parameters
    ----------
    recording_config: list[dict[str, Any]]
        List of dictionaries containing recording config details.
    key: str
        The key in the recording configs to check for consistency.
    astype: type | None
        The type to cast the value to.

    Returns
    -------
    Any
        The value of the specified key if available, otherwise None.
    """
    if not recording_config:
        warnings.warn('No recording configuration found.')
        return None

    # Extract values for the requested key but ignore entries where the key is missing
    raw_values = [d.get(key) for d in recording_config]
    non_none_values = [v for v in raw_values if v is not None]

    if not non_none_values:
        # The recording config exists but the specific key was never present.
        # Return None silently to avoid emitting unexpected warnings in callers.
        return None

    unique_values = set(non_none_values)
    if len(unique_values) != 1:
        # Try to present a sorted list of values for the warning, fall back if not comparable
        try:
            sorted_values: list = sorted(unique_values)
        except TypeError:
            sorted_values = list(unique_values)
        warnings.warn(f"Found inconsistent values for '{key}': {sorted_values}")
        return None

    value = unique_values.pop()
    if astype is not None:
        try:
            value = astype(value)
        except (TypeError, ValueError):
            # If casting fails, return None silently to avoid unexpected warnings.
            return None
    return value


def _check_samples_config_key(
        samples_config: list[dict[str, Any]],
        key: str,
        astype: type | None = None,
) -> Any:
    """Check if the sample configs contain consistent values for the specified key and return it.

    Prints a warning if no sample config is found or if the value is inconsistent across entries.

    Parameters
    ----------
    samples_config: list[dict[str, Any]]
        List of dictionaries containing sample config details.
    key: str
        The key in the recording configs to check for consistency.
    astype: type | None
        The type to cast the value to.

    Returns
    -------
    Any
        The value of the specified key if available, otherwise None.
    """
    if not samples_config:
        warnings.warn('No samples configuration found.')
        return None

    values = {d.get(key) for d in samples_config}
    if len(values) != 1:
        sorted_values: list = sorted(values)
        warnings.warn(f"Found inconsistent values for '{key}': {sorted_values}")
        return None

    value = values.pop()
    if astype is not None:
        value = astype(value)
    return value


def parse_eyelink_event_start(line: str) -> tuple[str, str] | None:
    """Check if the line contains the start of an event and return the event name and eye.

    Returns a tuple (event_name, eye) where eye is 'left' or 'right'.
    Example: ('fixation', 'left')
    """
    if match := FIXATION_START_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'fixation', eye_str
    if match := SACCADE_START_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'saccade', eye_str
    if match := BLINK_START_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'blink', eye_str
    return None


def parse_eyelink_event_end(line: str) -> tuple[str, str, float, float] | None:
    """Check if the line contains the end of an event and return the event name, eye and times.

    Returns a tuple (event_name, eye, onset, offset). Example: ('fixation', 'left', 123.0, 130.0)
    """
    if match := FIXATION_STOP_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'fixation',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    if match := SACCADE_STOP_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'saccade',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    if match := BLINK_STOP_REGEX.match(line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'blink',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    return None


def parse_eyelink(
        filepath: Path | str,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str | None = None,
        messages: bool | Sequence[str] = False,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], pl.DataFrame | None]:
    """Parse EyeLink asc file.

    Parameters
    ----------
    filepath: Path | str
        file name of ascii file to convert.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        list of patterns to match for additional metadata. (default: None)
    encoding: str | None
        Text encoding of the file. If None, the locale encoding is used. (default: None)
    messages: bool | Sequence[str]
        Flag indicating if any additional messages should be parsed from the asc file
        and returned as a DataFrame with 'time' (f64) and 'content' (str) columns.
        The message format is 'MSG <timestamp> <content>'.
        If True, all available messages will be parsed from the asc,
        alternatively, a list of regular expressions can be passed and only the
        messages that match any of the regular expressions will be kept.
        Regular expressions are only applied to the message content,
        implicitly parsing the `MSG <timestamp>` prefix.
        (default: False)

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], pl.DataFrame | None]
        A tuple containing the parsed gaze sample data, the parsed event data, the metadata,
        and, if asked for, the parsed messages.

    Raises
    ------
    Warning
        If no metadata is found in the file.
    ValueError
        If the `messages` parameter is not bool or a list of strings.

    Notes
    -----
    Event onsets and offsets are parsed as they are in the file. However, EyeLink calculates the
    durations in a different way than pymovements, resulting in a difference of 1 sample duration.
    For 1000 Hz recordings, durations calculated by pymovements are 1 ms shorter than the durations
    reported in the asc file.

    Robustness to unmatched end markers: If an event end line (EBLINK/EFIX/ESACC) appears without a
    corresponding start line (SBLINK/SFIX/SSACC) for the same eye earlier in the file, a warning is
    emitted and the event is still recorded. In this case, the parser seeds additional columns from
    the current context (values derived from the provided ``patterns`` at that line), so trial/task
    information is preserved when available.
    """
    # pylint: disable=too-many-branches, too-many-statements
    msg_prefix = r'MSG\s+\d+[.]?\d*\s+'

    if patterns is None:
        patterns = []
    compiled_patterns = compile_patterns(patterns, msg_prefix)

    if metadata_patterns is None:
        metadata_patterns = []
    compiled_metadata_patterns = compile_patterns(metadata_patterns, msg_prefix)

    additional_columns = get_pattern_keys(compiled_patterns, 'column')
    current_additional = {
        additional_column: None for additional_column in additional_columns
    }
    # Track additional metadata and start state per event type and eye
    # Structure: current_event_additional[event_name][eye] -> dict of additional columns
    current_event_additional: dict[str, dict[str, dict[str, Any]]] = {
        'fixation': {'left': {}, 'right': {}},
        'saccade': {'left': {}, 'right': {}},
        'blink': {'left': {}, 'right': {}},
    }
    current_event_started: dict[str, dict[str, bool]] = {
        'fixation': {'left': False, 'right': False},
        'saccade': {'left': False, 'right': False},
        'blink': {'left': False, 'right': False},
    }

    samples: dict[str, list[Any]] = {
        'time': [],
        'x_pix': [],
        'y_pix': [],
        'pupil': [],
        **{additional_column: [] for additional_column in additional_columns},
    }
    events: dict[str, list[Any]] = {
        'name': [],
        'eye': [],
        'onset': [],
        'offset': [],
        **{additional_column: [] for additional_column in additional_columns},
    }

    with open(filepath, encoding=encoding) as asc_file:
        lines = asc_file.readlines()

    # will return an empty string if the key does not exist
    metadata: defaultdict = defaultdict(str)

    # metadata keys specified by the user should have a default value of None
    metadata_keys = get_pattern_keys(compiled_metadata_patterns, 'key')
    for key in metadata_keys:
        metadata[key] = None

    compiled_metadata_patterns.extend(EYELINK_META_REGEXES)

    if (
        not isinstance(messages, (bool, list)) or
        (isinstance(messages, list) and not all(isinstance(regexp, str) for regexp in messages))
    ):
        raise ValueError(
            'Make sure to pass either a bool or a list of regular expressions '
            f"as strings. Received {messages}.",
        )

    messages_list: list[list[str]] = []

    cal_timestamp = ''

    validations = []
    calibrations = []
    recording_config: list[dict[str, Any]] = []
    samples_config: list[dict[str, Any]] = []

    total_recording_duration = 0.0
    num_expected_samples = 0
    num_valid_samples = 0  # excluding blinks
    num_blink_samples = 0
    blink_intervals: list[tuple[float, float]] = []
    blinking = False

    # Detect if the file is binocular or monocular
    # Collect ALL SAMPLES config lines (don't stop at the first match) so
    # inconsistent values (e.g. different sampling rates across SAMPLES lines)
    # can be detected later.
    is_binocular = False
    for line in lines:
        if match := SAMPLES_CONFIG_REGEX.search(line):
            samples_config.append(match.groupdict())
            tracked = match.group('tracked_eye').upper().strip()
            # consider 'LEFT' in tracked and 'RIGHT' in tracked or tracked == 'LR' or
            # tracked == 'L R': set is_binocular to True if any config indicates binocular
            is_binocular = is_binocular or (
                ('LEFT' in tracked and 'RIGHT' in tracked) or
                tracked == 'LR' or tracked == 'L R'
            )

    # Update the samples dictionary to include binocular data with correct column names if needed
    if is_binocular:
        samples.update({
            'x_left_pix': [],
            'y_left_pix': [],
            'pupil_left': [],
            'x_right_pix': [],
            'y_right_pix': [],
            'pupil_right': [],
        })
        # remove monocular-only keys to avoid mismatched column lengths
        for _k in ('x_pix', 'y_pix', 'pupil'):
            samples.pop(_k, None)
    else:
        # Ensure monocular fields are present in the samples dictionary
        samples.update({
            'x_pix': [],
            'y_pix': [],
            'pupil': [],
        })
        # remove binocular-only keys to avoid mismatched column lengths
        for _k in (
            'x_left_pix', 'y_left_pix', 'pupil_left',
            'x_right_pix', 'y_right_pix', 'pupil_right',
        ):
            samples.pop(_k, None)

    for line in lines:
        for pattern_dict in compiled_patterns:

            if match := pattern_dict['pattern'].match(line):
                if 'value' in pattern_dict:
                    current_column = pattern_dict['column']
                    current_additional[current_column] = pattern_dict['value']

                else:
                    current_additional.update(match.groupdict())

        if cal_timestamp:
            # if a calibration timestamp has been found, the next line will be a
            # calibration pattern, if not, there will only be the timestamp added to the overview

            # very ugly pylint solution
            calibrations.append(
                {
                    'timestamp': cal_timestamp,
                    **match.groupdict(),
                }
                if (match := CALIBRATION_REGEX.match(line))
                else {'timestamp': cal_timestamp},
            )
            cal_timestamp = ''

        elif start_event := parse_eyelink_event_start(line):
            event_name, eye = start_event
            # store additional metadata for this event type and eye
            current_event_additional[event_name][eye] = {**current_additional}
            current_event_started[event_name][eye] = True

            if event_name == 'blink':
                blinking = True

        elif end_event := parse_eyelink_event_end(line):
            event_name, eye, event_onset, event_offset = end_event
            events['name'].append(f'{event_name}_eyelink')
            events['eye'].append(eye)
            events['onset'].append(event_onset)
            events['offset'].append(event_offset)

            # If an event end is found but there is no recorded start for this eye, seed from
            # current context and warn the user that the file may be corrupt or incomplete.
            if not current_event_started[event_name][eye]:
                warnings.warn(
                    "Missing start marker before end for event '" + event_name +
                    f"' (onset={event_onset}, offset={event_offset}). "
                    'Using current context to fill additional columns.',
                )
                current_event_additional[event_name][eye] = {**current_additional}

            for additional_column in additional_columns:
                events[additional_column].append(
                    current_event_additional[event_name][eye][additional_column],
                )
            current_event_additional[event_name][eye] = {}

            if event_name == 'blink':
                # collect blink intervals and compute counts later once sampling rate is known
                blink_intervals.append((event_onset, event_offset))
                blinking = False

        elif match := RECORDING_CONFIG_REGEX.match(line):
            # Drop optional groups that weren't present for legacy behaviour
            rec_cfg = {k: v for k, v in match.groupdict().items() if v is not None}
            recording_config.append(rec_cfg)

        elif match := GAZE_COORDS_REGEX.match(line):
            left, top, right, bottom = (float(coord) for coord in match.group('resolution').split())
            # GAZE_COORDS is typically logged after RECCFG - if not, warn and skip assignment.
            if not recording_config:
                warnings.warn(
                    'GAZE_COORDS encountered before any RECCFG. Skipping resolution assignment.',
                )
            else:
                recording_config[-1]['resolution'] = (right - left + 1, bottom - top + 1)

        elif match := START_RECORDING_REGEX.match(line):
            start_recording_timestamp = match.groupdict()['timestamp']

        elif match := STOP_RECORDING_REGEX.match(line):
            stop_recording_timestamp = match.groupdict()['timestamp']

            try:
                # Safely obtain the sampling rate from the last recording_config entry.
                block_duration = float(stop_recording_timestamp) - float(start_recording_timestamp)
                current_sampling_rate = recording_config[-1].get('sampling_rate')
            except UnboundLocalError:
                warnings.warn(
                    'END recording message without associated START recording message. '
                    f"File '{filepath}' may be corrupted. Data-loss metrics may be incorrect.",
                )
            else:  # this will only be executed if no exception was raised in the try block.
                total_recording_duration += block_duration
                if current_sampling_rate:
                    num_expected_samples += round(
                        block_duration * float(current_sampling_rate) / 1000,
                    )

        if messages and (match := MSG_REGEX.match(line)):
            messages_list.append([match.groupdict()['timestamp'], match.groupdict()['content']])

        # Use the appropriate regex based on the file type
        eye_tracking_sample_match = (
            EYE_TRACKING_SAMPLE_BINOCULAR.match(line)
            if is_binocular else
            EYE_TRACKING_SAMPLE_MONOCULAR.match(line)
        )

        if eye_tracking_sample_match:
            timestamp_s = eye_tracking_sample_match.group('time')

            if is_binocular:
                x_left_pix_s = eye_tracking_sample_match.group('x_pix_left')
                y_left_pix_s = eye_tracking_sample_match.group('y_pix_left')
                pupil_left_s = eye_tracking_sample_match.group('pupil_left')
                x_right_pix_s = eye_tracking_sample_match.group('x_pix_right')
                y_right_pix_s = eye_tracking_sample_match.group('y_pix_right')
                pupil_right_s = eye_tracking_sample_match.group('pupil_right')

                samples['x_left_pix'].append(check_nan(x_left_pix_s))
                samples['y_left_pix'].append(check_nan(y_left_pix_s))
                samples['pupil_left'].append(check_nan(pupil_left_s))
                samples['x_right_pix'].append(check_nan(x_right_pix_s))
                samples['y_right_pix'].append(check_nan(y_right_pix_s))
                samples['pupil_right'].append(check_nan(pupil_right_s))
            else:
                x_pix_s = eye_tracking_sample_match.group('x_pix')
                y_pix_s = eye_tracking_sample_match.group('y_pix')
                pupil_s = eye_tracking_sample_match.group('pupil')

                samples['x_pix'].append(check_nan(x_pix_s))
                samples['y_pix'].append(check_nan(y_pix_s))
                samples['pupil'].append(check_nan(pupil_s))

            timestamp = float(timestamp_s)
            samples['time'].append(timestamp)

            for additional_column in additional_columns:
                samples[additional_column].append(current_additional[additional_column])

            # only check monocular validity when parsing monocular files
            if not is_binocular:
                if not blinking and all(
                    (not np.isnan(val)) for val in (
                        samples['x_pix'][-1], samples['y_pix'][-1], samples['pupil'][-1],
                    )
                ):
                    num_valid_samples += 1

            if is_binocular and not blinking and all(
                (not np.isnan(val)) for val in (
                    samples['x_left_pix'][-1],
                    samples['y_left_pix'][-1],
                    samples['pupil_left'][-1],
                    samples['x_right_pix'][-1],
                    samples['y_right_pix'][-1],
                    samples['pupil_right'][-1],
                )
            ):
                num_valid_samples += 1

        elif match := CALIBRATION_TIMESTAMP_REGEX.match(line):
            cal_timestamp = match.groupdict()['timestamp']

        elif match := VALIDATION_REGEX.match(line):
            validations.append(match.groupdict())

        elif compiled_metadata_patterns:
            for pattern_dict in compiled_metadata_patterns.copy():
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict:
                        metadata[pattern_dict['key']] = pattern_dict['value']

                    else:
                        metadata.update(match.groupdict())

                    # each metadata pattern should only match once
                    compiled_metadata_patterns.remove(pattern_dict)

    if not metadata:
        warnings.warn('No metadata found. Please check the file for errors.')

    # the actual tracked eye is in the samples config, not in the recording config
    # the recording config contains the eyes that were recorded
    sampling_rate_samples_config = _check_samples_config_key(samples_config, 'sampling_rate', float)
    sampling_rate_reccfg = _check_reccfg_key(recording_config, 'sampling_rate', float)
    if sampling_rate_samples_config and sampling_rate_reccfg:
        if sampling_rate_samples_config != sampling_rate_reccfg:
            warnings.warn(
                f'The recording configuration message and the samples message'
                f" give inconsistent values for 'sampling_rate': "
                f'[{sampling_rate_samples_config}, {sampling_rate_reccfg}]'
                f' Using the value from the samples message.',
            )
    metadata['sampling_rate'] = sampling_rate_samples_config
    metadata['recorded_eye'] = _check_reccfg_key(recording_config, 'tracked_eye')
    # the actual tracked eye is in the samples config, not in the recording config
    # the recording config contains the eyes that were recorded
    # RECCFG uses L/R/LR, SAMPLES uses LEFT/RIGHT/LEFT RIGHT
    tracked_eye_samples_config = _check_samples_config_key(samples_config, 'tracked_eye')
    if tracked_eye_samples_config == 'LEFT':
        metadata['tracked_eye'] = 'L'
    elif tracked_eye_samples_config == 'RIGHT':
        metadata['tracked_eye'] = 'R'
    elif tracked_eye_samples_config == 'LEFT\tRIGHT':
        metadata['tracked_eye'] = 'LR'

    if metadata['tracked_eye'] and metadata['recorded_eye']:
        if metadata['tracked_eye'] != metadata['recorded_eye']:
            warnings.warn(
                f'The recorded eye in the recording configuration message and'
                f' the samples message are inconsistent: '
                f"[{metadata['recorded_eye']}, {metadata['tracked_eye']}]"
                f' This could be because the -r or -l flag in edf2asc was used'
                f' to obtain monocular data from a binocular EDF file.'
                f' Using the value from the samples message and storing the value from'
                f" the recording configuration message in 'recorded_eye'.",
            )
    metadata['resolution'] = _check_reccfg_key(recording_config, 'resolution')

    pre_processed_metadata: dict[str, Any] = _pre_process_metadata(metadata)
    # is not yet pre-processed but should be
    pre_processed_metadata['calibrations'] = calibrations
    pre_processed_metadata['validations'] = validations
    pre_processed_metadata['recording_config'] = recording_config
    pre_processed_metadata['total_recording_duration_ms'] = total_recording_duration

    # compute num_blink_samples from collected blink intervals to avoid double-counting overlaps
    num_blink_samples = 0
    if blink_intervals and recording_config:
        try:
            sampling_rate = float(recording_config[-1]['sampling_rate'])
        except (KeyError, TypeError, ValueError):
            sampling_rate = None

        if sampling_rate:
            # merge overlapping intervals
            intervals = sorted(blink_intervals, key=lambda x: x[0])
            merged: list[tuple[float, float]] = []
            current_start, current_end = intervals[0]
            for s, e in intervals[1:]:
                if s <= current_end:
                    current_end = max(current_end, e)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = s, e
            merged.append((current_start, current_end))

            sample_length = 1 / sampling_rate * 1000
            for s, e in merged:
                num_blink_samples += round((e - s) / sample_length) + 1

    # If no sampling rate could be determined from either SAMPLES or RECCFG,
    # only warn the user when there is evidence of samples or recording
    # configuration present (or blink intervals) — otherwise keep silent to
    # avoid noisy warnings for minimal metadata-only files.
    # If we were able to compute an expected number of samples from STOP_RECORDING
    # blocks (num_expected_samples > 0), trust that calculation and compute the
    # data-loss metrics even if the SAMPLES/RECCFG metadata keys are missing or
    # inconsistent. Otherwise, fall back to the previous behavior: warn (when
    # appropriate) and set metrics to None.
    if num_expected_samples > 0:
        (
            pre_processed_metadata['data_loss_ratio'],
            pre_processed_metadata['data_loss_ratio_blinks'],
        ) = _calculate_data_loss_ratio(num_expected_samples, num_valid_samples, num_blink_samples)
    else:
        # Determine if the sampling rate keys were present but inconsistent
        def _config_inconsistent(
            config_list: list[dict[str, Any]],
            key: str = 'sampling_rate',
        ) -> bool:
            vals = [d.get(key) for d in config_list if d.get(key) is not None]
            return len(set(vals)) > 1

        inconsistent_reccfg = _config_inconsistent(recording_config)
        inconsistent_samples = _config_inconsistent(samples_config)

        # Only warn if we truly don't have a sampling-rate value from either
        # the SAMPLES or RECCFG messages. If a sampling rate was present
        # (even when num_expected_samples == 0), there's no need to emit
        # the generic warning — the presence of inconsistent config
        # warnings is already handled above.
        if not (sampling_rate_samples_config or sampling_rate_reccfg):
            if (samples_config or recording_config or blink_intervals) and not (
                inconsistent_reccfg or inconsistent_samples
            ):
                warnings.warn(
                    'Could not determine sampling rate from SAMPLES or RECCFG; '
                    'data-loss metrics will be unavailable.',
                )

        pre_processed_metadata['data_loss_ratio'] = None
        pre_processed_metadata['data_loss_ratio_blinks'] = None

    gaze_schema_overrides = {
        'time': pl.Float64,
    }

    if is_binocular:
        gaze_schema_overrides.update({
            'x_left_pix': pl.Float64,
            'y_left_pix': pl.Float64,
            'pupil_left': pl.Float64,
            'x_right_pix': pl.Float64,
            'y_right_pix': pl.Float64,
            'pupil_right': pl.Float64,
        })
    else:
        gaze_schema_overrides.update({
            'x_pix': pl.Float64,
            'y_pix': pl.Float64,
            'pupil': pl.Float64,
        })

    if schema is not None:
        gaze_schema_overrides.update(schema)

    event_schema_overrides = {
        'name': pl.String,
        'eye': pl.String,
        'onset': pl.Float64,
        'offset': pl.Float64,
    }
    if schema is not None:
        event_schema_overrides.update(schema)

    gaze_df = pl.from_dict(data=samples).cast(gaze_schema_overrides)
    event_df = pl.from_dict(data=events).cast(event_schema_overrides)

    # Only return messages if `messages` not False or []. Otherwise, return None.
    if messages:
        messages_df = pl.DataFrame(
            data=messages_list,
            schema={
                'time': pl.Float64,
                'content': pl.String,
            },
            orient='row',
        )
        # Filter messages with regexp if given
        if isinstance(messages, Sequence):
            # keep rows where content matches any of the regex patterns
            # for each row check if content matches any of the regex patterns
            messages_df = messages_df.filter(
                pl.col('content').str.contains(
                    pattern='|'.join(messages),  # RegExps are joined by OR
                    strict=True,  # Raises error if regexp not valid
                ),
            )
    else:
        messages_df = None

    return gaze_df, event_df, pre_processed_metadata, messages_df


def _parse_eyelink_mount_config(mount_config: str) -> dict[str, str]:
    """Return a dictionary with the mount configuration based on the config short name.

    Parameters
    ----------
    mount_config: str
        Short name of the mount configuration.

    Returns
    -------
    dict[str, str]
        Dictionary with the mount configuration spelled out.
    """
    possible_mounts = {
        'MTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'MTABLER',
        },
        'BTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BTABLER',
        },
        'RTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'remote',
            'eyes_recorded': 'monocular',
            'short_name': 'RTABLER',
        },
        'RBTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'remote',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'RBTABLER',
        },
        'AMTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'AMTABLER',
        },
        'ABTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'ABTABLER',
        },
        'ARTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'remote',
            'eyes_recorded': 'monocular',
            'short_name': 'ARTABLER',
        },
        'ABRTABLE': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'remote',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'ABRTABLE',
        },
        'BTOWER': {
            'mount_type': 'Binocular Tower Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BTOWER',
        },
        'TOWER': {
            'mount_type': 'Tower Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'TOWER',
        },
        'MPRIM': {
            'mount_type': 'Primate Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'MPRIM',
        },
        'BPRIM': {
            'mount_type': 'Primate Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BPRIM',
        },
        'MLRR': {
            'mount_type': 'Long-Range Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'camera_position': 'level',
            'short_name': 'MLRR',
        },
        'BLRR': {
            'mount_type': 'Long-Range Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'camera_position': 'angled',
            'short_name': 'BLRR',
        },
    }

    if mount_config in possible_mounts:
        return possible_mounts[mount_config]

    return {
        'mount_type': 'unknown',
        'head_stabilization': 'unknown',
        'eyes_recorded': 'unknown',
        'camera_position': 'unknown',
        'short_name': mount_config,
    }


def _pre_process_metadata(metadata: defaultdict[str, Any]) -> dict[str, Any]:
    """Pre-process metadata to suitable types and formats.

    Parameters
    ----------
    metadata: defaultdict[str, Any]
        Metadata to pre-process.

    Returns
    -------
    dict[str, Any]
        Pre-processed metadata.
    """
    # in case the version strings have not been found, they will be empty strings (defaultdict)
    metadata['version_number'], metadata['model'] = _parse_full_eyelink_version(
        metadata['version_1'], metadata['version_2'],
    )

    if 'DISPLAY_COORDS' in metadata:
        display_coords = tuple(float(coord) for coord in metadata['DISPLAY_COORDS'].split())
        metadata['DISPLAY_COORDS'] = display_coords

    # if the date has been parsed fully, convert the date to a datetime object
    if 'day' in metadata and 'year' in metadata and 'month' in metadata and 'time' in metadata:
        metadata['day'] = int(metadata['day'])
        metadata['year'] = int(metadata['year'])
        month_num = list(calendar.month_abbr).index(metadata['month'])
        date_time = datetime.datetime(day=metadata['day'], month=month_num, year=metadata['year'])
        time = datetime.datetime.strptime(metadata['time'], '%H:%M:%S')
        metadata['datetime'] = datetime.datetime.combine(date_time, time.time())

    if 'mount_configuration' in metadata:
        metadata['mount_configuration'] = _parse_eyelink_mount_config(
            metadata['mount_configuration'],
        )

    return_metadata: dict[str, Any] = dict(metadata)

    return return_metadata


def _parse_full_eyelink_version(version_str_1: str, version_str_2: str) -> tuple[str, str]:
    """Parse the two version strings into an eyelink version number and model.

    Parameters
    ----------
    version_str_1: str
        First version string.
    version_str_2: str
        Second version string.

    Returns
    -------
    tuple[str, str]
        Version number and model as strings or unknown if it cannot be parsed.
    """
    if version_str_1 == 'EYELINK II 1' and version_str_2:
        version_pattern = re.compile(r'.*v(?P<version_number>[0-9]\.[0-9]+).*')
        if match := version_pattern.match(version_str_2):
            version_number = match.groupdict()['version_number']
            if float(version_number) < 3:
                model = 'EyeLink II'
            elif float(version_number) < 5:
                model = 'EyeLink 1000'
            elif float(version_number) < 6:
                model = 'EyeLink 1000 Plus'
            else:
                model = 'EyeLink Portable Duo'

        else:
            version_number = 'unknown'
            model = 'unknown'

    else:
        # taken from R package eyelinker/eyelink_parser.R
        version_pattern = re.compile(r'.*\s+(?P<version_number>[0-9]\.[0-9]+).*')
        model = 'EyeLink I'
        if match := version_pattern.match(version_str_1):
            version_number = match.groupdict()['version_number']

        else:
            model = 'unknown'
            version_number = 'unknown'

    return version_number, model
