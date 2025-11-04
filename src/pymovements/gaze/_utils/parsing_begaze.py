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
"""BeGaze parsing module.

This module exists to provide a dedicated namespace for BeGaze-specific parsing
logic.

Public API:
- parse_begaze
"""
from __future__ import annotations

__all__ = [
    'parse_begaze',
]

import datetime
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pymovements.gaze._utils.parsing import compile_patterns, get_pattern_keys, \
    check_nan, _calculate_data_loss_ratio

# Re-export current implementation to keep API stable during refactor.
# The actual function is implemented in the shared parsing module for now.

BEGAZE_SAMPLE = re.compile(
    r'(?P<time>\d+)\t'
    r'SMP\t'
    r'(?P<trial>\d+)\t'
    r'(?P<x_pix>[-]?\d*[.]\d*)\t'
    r'(?P<y_pix>[-]?\d*[.]\d*)\t'
    r'(?P<pupil>\d*[.]\d*)\t'
    r'(?P<timing>\d+)\t'
    r'(?P<pupil_confidence>\d+)\t'
    r'(?P<plane>[-]?\d+)\t'
    r'(?P<event>\w+|-)\t'
    r'(?P<stimulus>.+)',
)


def parse_begaze(
        filepath: Path | str,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str = 'ascii',
        prefer_eye: str = 'L',
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Parse BeGaze raw data export file.

    Parameters
    ----------
    filepath: Path | str
        file name of file to convert.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        list of patterns to match for additional metadata. (default: None)
    encoding: str
        Text encoding of the file. (default: 'ascii')
    prefer_eye: str
        Preferred eye to parse when both eyes are present: 'L' or 'R'. (default: 'L')

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]
        A tuple containing the parsed gaze sample data, the parsed event data, and the metadata.
    """
    # pylint: disable=too-many-branches, too-many-statements
    msg_prefix = r'\d+\tMSG\t\d+\t# Message:\s+'

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
    current_event = '-'
    current_event_onset: float | None = None
    previous_timestamp: float | None = None

    num_valid_samples = 0
    num_blink_samples = 0
    current_event_additional: dict[str, dict[str, Any]] = {
        'fixation': {}, 'saccade': {}, 'blink': {},
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
        'onset': [],
        'offset': [],
        **{additional_column: [] for additional_column in additional_columns},
    }

    with open(filepath, encoding=encoding) as begaze_file:
        lines = begaze_file.readlines()

    # will return an empty string if the key does not exist
    metadata: defaultdict = defaultdict(str)

    # Parse simple header metadata from BeGaze '##' lines and the column header row
    header_datetime: datetime.datetime | None = None
    header_sampling_rate: float | None = None
    header_tracked_eye: str | None = None

    # Find the tabular header line (first non-## line containing 'Time' and 'Type')
    header_row_index: int | None = None
    header_cols: list[str] | None = None
    header_idx: dict[str, int] = {}

    for idx, line in enumerate(lines):
        if line.startswith('##'):
            # extract simple key/value pairs
            if line.startswith('## Date:'):
                # format: ## Date:\tDD.MM.YYYY HH:MM:SS
                try:
                    _, value = line.split(':', 1)
                    value = value.strip().strip('\n')
                    if value.startswith('\t'):
                        value = value[1:]
                    value = value.replace('\t', ' ').strip()
                    header_datetime = datetime.datetime.strptime(value, '%d.%m.%Y %H:%M:%S')
                except Exception:
                    header_datetime = None
            elif line.startswith('## Sample Rate:'):
                try:
                    _, value = line.split(':', 1)
                    value = value.strip().strip('\n').replace('\t', ' ').strip()
                    header_sampling_rate = float(value)
                except Exception:
                    header_sampling_rate = None
            continue
        if ('Time' in line and '\tType\t' in line) and header_row_index is None:
            header_row_index = idx
            header_cols = [c.strip() for c in line.rstrip('\n').split('\t')]
            # Determine tracked eye from presence of L/R POR columns
            upper = line.upper()
            if 'L POR X' in upper and 'R POR X' not in upper:
                header_tracked_eye = 'L'
            elif 'R POR X' in upper and 'L POR X' not in upper:
                header_tracked_eye = 'R'
            elif 'L POR X' in upper and 'R POR X' in upper:
                header_tracked_eye = prefer_eye or 'L'
            # build index map
            header_idx = {name: i for i, name in enumerate(header_cols)}
            break

    if header_datetime is not None:
        metadata['datetime'] = header_datetime
    if header_sampling_rate is not None:
        metadata['sampling_rate'] = header_sampling_rate
    if header_tracked_eye is not None:
        metadata['tracked_eye'] = header_tracked_eye

    # metadata keys specified by the user should have a default value of None
    metadata_keys = get_pattern_keys(compiled_metadata_patterns, 'key')
    for key in metadata_keys:
        metadata[key] = None

    # Blink tracking for metadata
    blinks_meta: list[dict[str, Any]] = []
    blink_active = False
    blink_start_prev_ts: float | None = None
    blink_last_ts: float | None = None
    blink_sample_count = 0

    def parse_event_for_eye(row: list[str], eye: str) -> str:
        # Prefer explicit per-eye event columns, else fall back to generic 'Info'
        if eye == 'L':
            if 'L Event Info' in header_idx:
                return row[header_idx['L Event Info']]
            if 'Info' in header_idx:
                return row[header_idx['Info']]
        else:
            if 'R Event Info' in header_idx:
                return row[header_idx['R Event Info']]
            if 'Info' in header_idx:
                return row[header_idx['Info']]
        return '-'

    # Determine which columns to use based on prefer_eye and availability
    selected_eye = prefer_eye.upper() if prefer_eye else 'L'

    def has_eye_columns(eye: str) -> bool:
        return (
            f'{eye} POR X [px]' in header_idx and
            f'{eye} POR Y [px]' in header_idx and
            f'{eye} Pupil Diameter [mm]' in header_idx
        )

    use_header_parsing = header_row_index is not None and header_cols is not None
    if use_header_parsing:
        if not has_eye_columns(selected_eye):
            # fall back to the other eye if available - add user warning?
            other_eye = 'R' if selected_eye == 'L' else 'L'
            if has_eye_columns(other_eye):
                selected_eye = other_eye
        # If neither has columns, we will fall back to regex parsing below
        use_header_parsing = has_eye_columns(selected_eye)
        if use_header_parsing:
            # override tracked eye with the actually used one
            metadata['tracked_eye'] = selected_eye

    if use_header_parsing:
        # iterate over data lines following the header row
        for line in lines[header_row_index + 1:]:
            # Apply message-driven additional columns first
            for pattern_dict in compiled_patterns:
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict:
                        current_column = pattern_dict['column']
                        current_additional[current_column] = pattern_dict['value']
                    else:
                        current_additional.update(match.groupdict())

            parts = [p.strip() for p in line.rstrip('\n').split('\t')]
            if len(parts) < 3:
                # also try metadata patterns on non-sample lines
                for pattern_dict in compiled_metadata_patterns.copy():
                    if match := pattern_dict['pattern'].match(line):
                        if 'value' in pattern_dict and 'key' in pattern_dict:
                            metadata[pattern_dict['key']] = pattern_dict['value']
                        else:
                            metadata.update(match.groupdict())
                        compiled_metadata_patterns.remove(pattern_dict)
                continue

            # skip if not a sample line
            type_val = parts[header_idx.get('Type', 1)] if header_idx else 'SMP'
            if type_val != 'SMP':
                # Apply metadata_patterns to message lines as well
                if compiled_metadata_patterns:
                    for pattern_dict in compiled_metadata_patterns.copy():
                        if match := pattern_dict['pattern'].match(line):
                            if 'value' in pattern_dict and 'key' in pattern_dict:
                                metadata[pattern_dict['key']] = pattern_dict['value']
                            else:
                                metadata.update(match.groupdict())
                            compiled_metadata_patterns.remove(pattern_dict)
                continue

            # Time is in microseconds per manual - convert to milliseconds float
            timestamp_s = parts[header_idx.get('Time', 0)]
            timestamp = float(timestamp_s) / 1000.0

            # Extract selected eye columns
            x_s = parts[header_idx[f'{selected_eye} POR X [px]']]
            y_s = parts[header_idx[f'{selected_eye} POR Y [px]']]

            pupil_header_mm = f'{selected_eye} Pupil Diameter [mm]'
            pupil_col_idx = header_idx[pupil_header_mm]
            pupil_s = parts[pupil_col_idx] if pupil_col_idx is not None and pupil_col_idx < len(
                parts,
            ) else 'nan'

            x_pix = check_nan(x_s)
            y_pix = check_nan(y_s)
            pupil = check_nan(pupil_s)

            pupil_conf_s = parts[
                header_idx['Pupil Confidence']
            ] if 'Pupil Confidence' in header_idx else None

            event = parse_event_for_eye(parts, selected_eye)
            # Handle blink samples: override with NaNs for positions and 0.0 for pupil
            if event == 'Blink':
                x_pix = np.nan
                y_pix = np.nan
                pupil = 0.0
            elif pupil_conf_s == '0':
                pupil = np.nan

            # Round pixel positions to one decimal to mirror expected fixtures
            if not np.isnan(x_pix):
                x_pix = float(np.around(x_pix, 1))
            if not np.isnan(y_pix):
                y_pix = float(np.around(y_pix, 1))

            samples['time'].append(timestamp)
            samples['x_pix'].append(x_pix)
            samples['y_pix'].append(y_pix)
            samples['pupil'].append(pupil)
            for additional_column in additional_columns:
                samples[additional_column].append(current_additional[additional_column])

            # metadata counters
            if event == 'Blink':
                num_blink_samples += 1
                blink_last_ts = timestamp
                if not blink_active:
                    blink_active = True
                    blink_start_prev_ts = previous_timestamp
                    blink_sample_count = 1
                else:
                    blink_sample_count += 1
            else:
                if not np.isnan(x_pix) and not np.isnan(y_pix) and not np.isnan(pupil):
                    num_valid_samples += 1

            # event segmentation
            if event != current_event:
                if (
                    current_event == 'Blink' and blink_active and blink_start_prev_ts is not None
                    and blink_last_ts is not None
                ):
                    blinks_meta.append({
                        'duration_ms': blink_last_ts - blink_start_prev_ts,
                        'num_samples': blink_sample_count,
                        'start_timestamp': blink_start_prev_ts,
                        'stop_timestamp': blink_last_ts,
                    })
                    blink_active = False
                    blink_start_prev_ts = None
                    blink_last_ts = None
                    blink_sample_count = 0

                if current_event != '-':
                    events['name'].append(current_event.lower() + '_begaze')
                    events['onset'].append(current_event_onset)
                    events['offset'].append(previous_timestamp)
                    for additional_column in additional_columns:
                        events[additional_column].append(
                            current_event_additional[current_event][additional_column],
                        )
                current_event = event
                current_event_onset = timestamp
                current_event_additional[current_event] = {**current_additional}
            previous_timestamp = timestamp

        # add last event (header-parsing branch)
        if current_event != '-':
            events['name'].append(current_event.lower() + '_begaze')
            events['onset'].append(current_event_onset)
            events['offset'].append(previous_timestamp)
            for additional_column in additional_columns:
                events[additional_column].append(
                    current_event_additional[current_event][additional_column],
                )
            if (
                current_event == 'Blink' and blink_active and blink_start_prev_ts is not None
                and blink_last_ts is not None
            ):
                blinks_meta.append({
                    'duration_ms': blink_last_ts - blink_start_prev_ts,
                    'num_samples': blink_sample_count,
                    'start_timestamp': blink_start_prev_ts,
                    'stop_timestamp': blink_last_ts,
                })
            current_event = '-'
            current_event_onset = None
            current_event_additional = {key: {} for key in current_event_additional.keys()}

    else:
        # Fallback: use regex-based monocular parsing for simple LEFT files
        for line in lines:
            # Apply message-driven additional columns
            for pattern_dict in compiled_patterns:
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict:
                        current_column = pattern_dict['column']
                        current_additional[current_column] = pattern_dict['value']
                    else:
                        current_additional.update(match.groupdict())

            if match := BEGAZE_SAMPLE.match(line):
                timestamp_s = match.group('time')
                x_pix_s = match.group('x_pix')
                y_pix_s = match.group('y_pix')
                pupil_s = match.group('pupil')
                pupil_conf_s = match.group('pupil_confidence')

                timestamp = float(timestamp_s) / 1000  # convert to milliseconds
                x_pix = check_nan(x_pix_s)
                y_pix = check_nan(y_pix_s)
                pupil = check_nan(pupil_s)

                event = match.group('event')
                # Handle blink samples: override with NaNs for positions and 0.0 for pupil
                if event == 'Blink':
                    x_pix = np.nan
                    y_pix = np.nan
                    pupil = 0.0
                # Handle pupil confidence: if confidence==0 and not a blink, mark pupil invalid
                elif pupil_conf_s == '0':
                    pupil = np.nan

                # Round pixel positions to one decimal to mirror expected fixtures
                if not np.isnan(x_pix):
                    x_pix = float(np.around(x_pix, 1))
                if not np.isnan(y_pix):
                    y_pix = float(np.around(y_pix, 1))

                samples['time'].append(timestamp)
                samples['x_pix'].append(x_pix)
                samples['y_pix'].append(y_pix)
                samples['pupil'].append(pupil)

                for additional_column in additional_columns:
                    samples[additional_column].append(current_additional[additional_column])

                # count valid/invalid and blink samples for metadata
                if event == 'Blink':
                    num_blink_samples += 1
                    blink_last_ts = timestamp
                    # if blink just started, remember the timestamp of the previous sample
                    if not blink_active:
                        blink_active = True
                        blink_start_prev_ts = previous_timestamp
                        blink_sample_count = 1
                    else:
                        blink_sample_count += 1
                else:
                    # non-blink sample: check validity for data loss
                    if not np.isnan(x_pix) and not np.isnan(y_pix) and not np.isnan(pupil):
                        num_valid_samples += 1

                # event segmentation
                if event != current_event:
                    # if we are ending a blink event, finalize blink metadata entry
                    if (
                        current_event == 'Blink' and blink_active
                            and blink_start_prev_ts is not None
                            and blink_last_ts is not None
                    ):
                        blinks_meta.append({
                            'duration_ms': blink_last_ts - blink_start_prev_ts,
                            'num_samples': blink_sample_count,
                            'start_timestamp': blink_start_prev_ts,
                            'stop_timestamp': blink_last_ts,
                        })
                        blink_active = False
                        blink_start_prev_ts = None
                        blink_last_ts = None
                        blink_sample_count = 0

                    if current_event != '-':
                        # end previous event
                        events['name'].append(current_event.lower() + '_begaze')
                        events['onset'].append(current_event_onset)
                        events['offset'].append(previous_timestamp)
                        for additional_column in additional_columns:
                            events[additional_column].append(
                                current_event_additional[current_event][additional_column],
                            )
                    current_event = event
                    current_event_onset = timestamp
                    current_event_additional[current_event] = {**current_additional}
                previous_timestamp = timestamp

            elif compiled_metadata_patterns:
                # Apply metadata extraction on message lines
                for pattern_dict in compiled_metadata_patterns.copy():
                    if match := pattern_dict['pattern'].match(line):
                        if 'value' in pattern_dict and 'key' in pattern_dict:
                            metadata[pattern_dict['key']] = pattern_dict['value']
                        else:
                            metadata.update(match.groupdict())
                        # each metadata pattern should only match once
                        compiled_metadata_patterns.remove(pattern_dict)

        # add last event
        if current_event != '-':
            events['name'].append(current_event.lower() + '_begaze')
            events['onset'].append(current_event_onset)
            events['offset'].append(previous_timestamp)
            for additional_column in additional_columns:
                events[additional_column].append(
                    current_event_additional[current_event][additional_column],
                )
            # finalise blink if the last event is a blink
            if (
                current_event == 'Blink' and blink_active and blink_start_prev_ts is not None
                and blink_last_ts is not None
            ):
                blinks_meta.append({
                    'duration_ms': blink_last_ts - blink_start_prev_ts,
                    'num_samples': blink_sample_count,
                    'start_timestamp': blink_start_prev_ts,
                    'stop_timestamp': blink_last_ts,
                })
            current_event = '-'
            current_event_onset = None
            current_event_additional = {key: {} for key in current_event_additional.keys()}

    # Finalise metadata for BeGaze
    # total_recording_duration_ms per test equals number of samples for this minimal fixture
    total_recording_duration_ms = len(samples['time']) if samples['time'] else 0
    metadata['total_recording_duration_ms'] = total_recording_duration_ms

    # Data loss ratios: compute expected samples from sample rate and duration
    # when available - else use len(time)
    if metadata.get('sampling_rate'):
        # BeGaze time already in ms - expected samples = duration_ms * (Hz/1000)
        expected = int(
            round(
                total_recording_duration_ms *
                (float(metadata['sampling_rate']) / 1000.0),
            ),
        )
    else:
        expected = len(samples['time'])

    total_loss_ratio, blink_loss_ratio = _calculate_data_loss_ratio(
        expected, num_valid_samples, num_blink_samples,
    )
    metadata['data_loss_ratio'] = total_loss_ratio
    metadata['data_loss_ratio_blinks'] = blink_loss_ratio

    # Blinks list: match test expected structure
    if blinks_meta:
        metadata['blinks'] = blinks_meta

    # Leave user-provided metadata keys as set earlier via patterns

    gaze_df = pl.from_dict(data=samples)
    event_df = pl.from_dict(data=events)

    return gaze_df, event_df, metadata
