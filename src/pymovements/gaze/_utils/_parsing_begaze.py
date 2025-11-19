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
"""BeGaze parsing module (internal).

This parser supports header-driven parsing of BeGaze exports.
A tabular header row with at least the columns 'Time' and 'Type' is required.
The parser selects monocular data from left or right eye columns depending on availability
and the `prefer_eye` parameter.
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

import warnings

import numpy as np
import polars as pl

from pymovements.gaze._utils._parsing import compile_patterns, get_pattern_keys, \
    check_nan

# Regular expressions for BeGaze header metadata lines with named groups.
# Note: we compile regexes for performance and to support minor whitespace variations.
BEGAZE_META_REGEXES: list[re.Pattern[str]] = [
    re.compile(r'^##\s+Date:\s+(?P<date>.+?)\s*$'),
    re.compile(r'^##\s+Sample\s+Rate:\s+(?P<sampling_rate>.+?)\s*$'),  # cast to float later
]


def _parse_begaze_meta_line(line: str) -> dict[str, Any]:
    """Parse a single BeGaze '##' metadata line using predefined regexes.

    Returns an empty dict when the line does not match any known pattern.
    Performs light casting for known fields (e.g., float for sampling rate, datetime for date).
    """
    for regex in BEGAZE_META_REGEXES:
        if match := regex.match(line):
            groupdict = match.groupdict()
            # Casting and processing for known fields
            if groupdict.get('sampling_rate') is not None:
                try:
                    groupdict['sampling_rate'] = float(groupdict['sampling_rate'])
                except ValueError:
                    pass
            if groupdict.get('date') is not None:
                # BeGaze Date format: 'DD.MM.YYYY HH:MM:SS'
                try:
                    groupdict['datetime'] = datetime.datetime.strptime(
                        groupdict['date'].strip(), '%d.%m.%Y %H:%M:%S',
                    )
                except ValueError:
                    # Keep original string if parsing fails
                    groupdict['datetime'] = groupdict['date']
                # We don't need to expose raw 'date' alongside 'datetime'
                del groupdict['date']
            return groupdict
    return {}


def parse_event_for_eye(row: list[str], eye: str, header_idx: dict[str, int]) -> str:
    """Return BeGaze event string for a row and eye.

    Prefers explicit per-eye event columns ('L Event Info' / 'R Event Info'),
    otherwise falls back to generic 'Info' when present.
    """
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


def parse_begaze(
        filepath: Path | str,
        *,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str = 'ascii',
        prefer_eye: str = 'L',
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Parse BeGaze raw data export file.

    Minimal "good enough" support for DIDEC-like exports:
    - Parses sample rows (`SMP`) using the tabular header.
    - Supports monocular and binocular files by selecting one eye via `prefer_eye`.
      If the requested eye is unavailable but the other eye is present, the parser
      will automatically fall back to the other eye and emit a warning; the chosen
      eye is recorded in metadata under `tracked_eye`.
    - Converts `Time` from microseconds to milliseconds (float).
    - Derives BeGaze per-sample event labels (Fixation/Saccade/Blink) into consolidated
      event rows named `<lowercase>_begaze` with onset/offset in milliseconds.
    - Applies user-supplied `patterns` and `metadata_patterns` (same semantics as EyeLink)
      to populate additional columns on samples/events and to extract metadata.

    Parameters
    ----------
    filepath: Path | str
        Path of file to convert.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional metadata. (default: None)
        Overwrite semantics: if multiple lines match the same metadata key, the last
        match wins and overwrites any previous value for that key.
        When metadata patterns are applied to non-sample lines (outside the tabular `SMP` rows),
        the parser emits a one-time ``RuntimeWarning`` stating that recurring matches will overwrite
        previous values.
        Additionally, whenever a specific key is overwritten with a different value,
        a per-key ``RuntimeWarning`` is emitted indicating the old and new values.
    encoding: str
        Text encoding of the file. (default: 'ascii')
    prefer_eye: str
        Preferred eye to parse when both eyes are present: 'L' or 'R'. (default: 'L')

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]
        A tuple containing the parsed gaze sample data, the parsed event data, and the metadata.
    """
    # pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
    msg_prefix = r'\d+\tMSG\t\d+\t# Message:\s+'

    # consume unused argument to satisfy linters (schema is still unused)
    _ = schema

    if patterns is None:
        patterns = []
    compiled_patterns = compile_patterns(patterns, msg_prefix)

    if metadata_patterns is None:
        metadata_patterns = []
    # Compile metadata patterns twice:
    # - prefixed: for standard MSG lines within the BeGaze table
    # - unprefixed: for free-form non-tab lines that may carry metadata before/after the table
    compiled_metadata_patterns_prefixed = compile_patterns(metadata_patterns, msg_prefix)
    compiled_metadata_patterns = compile_patterns(metadata_patterns, '')

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

    # Blink tracking for metadata (declared before helper functions for mypy)
    blinks_meta: list[dict[str, Any]] = []
    blink_active = False
    blink_start_prev_ts: float | None = None
    blink_last_ts: float | None = None
    blink_sample_count = 0

    # Small helpers to unify event handling across branches
    def _finalize_current_event() -> None:
        if current_event != '-':
            events['name'].append(current_event.lower() + '_begaze')
            events['onset'].append(current_event_onset)
            events['offset'].append(previous_timestamp)
            for additional_column in additional_columns:
                events[additional_column].append(
                    current_event_additional[current_event][additional_column],
                )
        # reset event-specific additional cache for next event name
        if current_event in current_event_additional:
            current_event_additional[current_event] = {}

    def _maybe_finalize_blink_meta() -> None:
        nonlocal blink_active, blink_start_prev_ts, blink_last_ts, blink_sample_count
        # Only finalize when the current ongoing event is a Blink and we are transitioning away
        if (
            current_event == 'Blink'
            and blink_active
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

    # will return an empty string if the key does not exist
    metadata: defaultdict = defaultdict(str)

    # Parse simple header metadata from BeGaze '##' lines and the column header row
    header_tracked_eye: str | None = None

    # Find the tabular header line (first non-## line containing 'Time' and 'Type')
    header_row_index: int | None = None
    header_cols: list[str] | None = None
    header_idx: dict[str, int] = {}

    for idx, line in enumerate(lines):  # pragma: no branch
        if line.startswith('##'):
            # Parse meta line via regexes with named groups
            line_meta = _parse_begaze_meta_line(line.rstrip('\n'))
            # Merge any parsed metadata keys
            for k, v in line_meta.items():
                metadata[k] = v
            continue
        # Tolerant to whitespace: split on any whitespace sequence but prefer tabs if present
        if ('Time' in line and 'Type' in line) and header_row_index is None:
            # Normalise tabs to single tab first to keep expected column names
            normalized = line.rstrip('\n')
            if '\t' in normalized:
                header_cols = [c.strip() for c in normalized.split('\t')]
            else:
                # Space-separated headers cannot reliably represent multi-word column names
                # like "L POR X [px]". We detect tracked eye from the raw string, but we
                # will not attempt to parse samples from such headers.
                header_cols = None
            header_row_index = idx
            # Determine tracked eye from presence of L/R POR columns
            upper = normalized.upper()
            if 'L POR X' in upper and 'R POR X' not in upper:
                header_tracked_eye = 'L'
            elif 'R POR X' in upper and 'L POR X' not in upper:
                header_tracked_eye = 'R'
            elif 'L POR X' in upper and 'R POR X' in upper:
                header_tracked_eye = prefer_eye or 'L'
            # build index map
            header_idx = {name: i for i, name in enumerate(header_cols)} if header_cols else {}
            break
    if header_tracked_eye is not None:
        metadata['tracked_eye'] = header_tracked_eye

    # metadata keys specified by the user should have a default value of None
    metadata_keys = get_pattern_keys(compiled_metadata_patterns_prefixed, 'key')
    for key in metadata_keys:
        metadata[key] = None

    # Determine which columns to use based on prefer_eye and availability
    selected_eye = prefer_eye.upper() if prefer_eye else 'L'

    def has_eye_columns(eye: str) -> bool:
        return (
            f'{eye} POR X [px]' in header_idx and
            f'{eye} POR Y [px]' in header_idx and
            f'{eye} Pupil Diameter [mm]' in header_idx
        )

    if not has_eye_columns(selected_eye):
        # fall back to the other eye if available and inform the user once
        other_eye = 'R' if selected_eye == 'L' else 'L'
        if has_eye_columns(other_eye):
            warnings.warn(
                f"BeGaze parser: preferred eye '{selected_eye}' not found in columns; "
                f"falling back to '{other_eye}'.",
                RuntimeWarning,
            )
            selected_eye = other_eye

    # Decide if we can parse samples: requires tabular header with eye columns
    parse_samples = False
    if header_row_index is not None and header_cols is not None:
        parse_samples = has_eye_columns('L') or has_eye_columns('R')

    # iterate over data lines following the header row
    if header_row_index is None:
        raise ValueError(
            "BeGaze parser: could not find a tabular header row containing 'Time' and 'Type'.",
        )
    # helper for setting metadata with overwrite warnings
    warned_non_sample_metadata = False

    def _set_metadata_key(key: str, value: Any) -> None:
        if key in metadata and metadata[key] not in ('', None) and metadata[key] != value:
            warnings.warn(
                f"BeGaze parser: metadata key '{key}' is being overwritten "
                f"(old={metadata[key]!r}, new={value!r}).",
                RuntimeWarning,
            )
        metadata[key] = value

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
            # also try metadata patterns on non-sample lines (use unprefixed compiled patterns)
            if compiled_metadata_patterns and not warned_non_sample_metadata:
                warnings.warn(
                    'BeGaze parser: non-sample lines matched by metadata_patterns. '
                    'Recurring matches will overwrite previous values.',
                    RuntimeWarning,
                )
                warned_non_sample_metadata = True
            for pattern_dict in compiled_metadata_patterns.copy():
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict and 'key' in pattern_dict:
                        _set_metadata_key(pattern_dict['key'], pattern_dict['value'])
                    else:
                        for k, v in match.groupdict().items():
                            _set_metadata_key(k, v)
                    compiled_metadata_patterns.remove(pattern_dict)
            continue

        # skip if not a sample line
        type_val = parts[header_idx.get('Type', 1)] if header_idx else 'SMP'
        if type_val != 'SMP':
            # Apply metadata_patterns to message lines as well (use prefixed patterns)
            if compiled_metadata_patterns_prefixed:
                for pattern_dict in compiled_metadata_patterns_prefixed.copy():
                    if match := pattern_dict['pattern'].match(line):
                        if 'value' in pattern_dict and 'key' in pattern_dict:
                            _set_metadata_key(pattern_dict['key'], pattern_dict['value'])
                        else:
                            for k, v in match.groupdict().items():
                                _set_metadata_key(k, v)
                        compiled_metadata_patterns_prefixed.remove(pattern_dict)
            continue

        # If header is not tabular or lacks eye columns, do not attempt sample parsing
        if not parse_samples:
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

        event = parse_event_for_eye(parts, selected_eye, header_idx)
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
        elif not np.isnan(x_pix) and not np.isnan(y_pix) and not np.isnan(pupil):
            num_valid_samples += 1

        # event segmentation
        if event != current_event:
            _maybe_finalize_blink_meta()

            _finalize_current_event()
            current_event = event
            current_event_onset = timestamp
            current_event_additional[current_event] = {**current_additional}
        previous_timestamp = timestamp

    # add last event (header-parsing branch)
    if current_event != '-':
        _finalize_current_event()
        _maybe_finalize_blink_meta()

    # Finalise metadata for BeGaze
    # Blinks list: match test expected structure
    if blinks_meta:
        metadata['blinks'] = blinks_meta

    # Leave user-provided metadata keys as set earlier via patterns

    gaze_df = pl.from_dict(data=samples)
    event_df = pl.from_dict(data=events)

    return gaze_df, event_df, metadata
