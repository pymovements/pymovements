# Copyright (c) 2025-2026 The pymovements Project Authors
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
import math
import re

import warnings
from collections import defaultdict
from collections.abc import Sequence

from pathlib import Path
from typing import Any

import polars as pl

from pymovements.gaze._utils._parsing import compile_patterns, get_pattern_keys, \
    check_nan


# Define separate regex patterns for monocular and binocular cases
EYE_TRACKING_SAMPLE_MONOCULAR = (
    r'(?P<time>(\d+[.]?\d*))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil>\d*[.]\d*|\.)?\s*'
    r'(?P<dummy>\d*[.]\d*|\.)?\s*'
    r'(?P<flags>[A-Za-z.]{3,5})?\s*'
)

EYE_TRACKING_SAMPLE_BINOCULAR = (
    r'(?P<time>(\d+[.]?\d*))\s+'
    r'(?P<x_pix_left>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix_left>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil_left>\d*[.]\d*|\.)?\s*'
    r'(?P<x_pix_right>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<y_pix_right>[-]?\d*[.]\d*|\.)?\s*'
    r'(?P<pupil_right>\d*[.]\d*|\.)?\s*'
    r'(?P<dummy>\d*[.]\d*|\.)?\s*'
    r'(?P<flags>[A-Za-z.]{3,5})?\s*'
)

EYELINK_META_REGEXES = [
    r'\*\*\s+VERSION:\s+(?P<version_1>.*)\s+',
    (
        r'\*\*\s+DATE:\s+(?P<weekday>[A-Z,a-z]+)\s+(?P<month>[A-Z,a-z]+)'
        r'\s+(?P<day>\d\d?)\s+(?P<time>\d\d:\d\d:\d\d)\s+(?P<year>\d{4})\s*'
    ),
    r'\*\*\s+(?P<version_2>EYELINK.*)',
    r'\*\*\s+RECORDED\s+BY\s+(?P<recorded_by>.*)',
    r'MSG\s+\d+[.]?\d*\s+DISPLAY_COORDS\s*=?\s*(?P<DISPLAY_COORDS>.*)',
    r'PUPIL\s+(?P<pupil_data_type>(AREA|DIAMETER))\s*',
    r'MSG\s+\d+[.]?\d*\s+ELCLCFG\s+(?P<mount_configuration>.*)',
]

VALIDATION_REGEX = (
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s+VALIDATION\s+HV'
    r'(?P<num_points>\d\d?).*'
    r'(?P<tracked_eye>LEFT|RIGHT)\s+'
    r'(?P<error>\D*)\s+'
    r'(?P<validation_score_avg>\d.\d\d)\s+avg\.\s+'
    r'(?P<validation_score_max>\d.\d\d)\s+max'
)

FIXATION_START_REGEX = r'SFIX\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*'
FIXATION_STOP_REGEX = (
    r'EFIX\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s+'
    r'(?P<avg_x_pix>(\d+[.]?\d*))\s+(?P<avg_y_pix>(\d+[.]?\d*))\s+(?P<avg_pupil>(\d+[.]?\d*))\s*.*'
)
SACCADE_START_REGEX = r'SSACC\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*'
SACCADE_STOP_REGEX = (
    r'ESACC\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s+'
    r'(?P<start_x_pix>(\d+[.]?\d*))\s+(?P<start_y_pix>(\d+[.]?\d*))\s+'
    r'(?P<end_x_pix>(\d+[.]?\d*))\s+(?P<end_y_pix>(\d+[.]?\d*))\s+'
    r'(?P<amplitude>(\d+[.]?\d*))\s+(?P<peak_velocity>(\d+[.]?\d*))\s*.*'
)
BLINK_START_REGEX = r'SBLINK\s+(?P<eye>R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*'
BLINK_STOP_REGEX = (
    r'EBLINK\s+(?P<eye>R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s*'
)

CALIBRATION_TIMESTAMP_REGEX = r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s*\n'

CALIBRATION_REGEX = (
    r'(?:MSG\s+\d+[.]?\d*\s+)?'
    r'>+\s+CALIBRATION\s+\(HV(?P<num_points>\d\d?),'
    r'(?P<type>.*)\).*'
    r'(?P<tracked_eye>RIGHT|LEFT):\s+<{9}'
)

RECORDING_CONFIG_REGEX = (
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+'
    r'RECCFG\s+(?P<tracking_mode>[A-Z,a-z]+)\s+'
    r'(?P<sampling_rate>\d+)\s+'
    r'(?P<file_sample_filter>0|1|2)\s+'
    r'(?P<link_sample_filter>0|1|2)\s+'
    r'((?P<file_event_filter>0|1|2)\s+)?'
    r'((?P<link_event_filter>0|1|2)\s+)?'
    r'(?P<tracked_eye>LR|[LR])?\s*'
)

# Resolution (GAZE_COORDS) pattern used to extract screen coordinates
GAZE_COORDS_REGEX = (
    r'MSG\s+\d+[.]?\d*\s+GAZE_COORDS\s*=?\s*(?P<resolution>.*)'
)

# Regex to match SAMPLES lines and capture which eyes are present (LEFT, RIGHT, LEFT RIGHT, LR)
SAMPLES_CONFIG_REGEX = (
    r'SAMPLES\s+GAZE\s+'
    r'(?P<tracked_eye>(?:LEFT\s+RIGHT|LEFT|RIGHT|LR|[LR]))'
    r'(?:\s+RATE\s+(?P<sampling_rate>\d+(?:\.\d+)?))?'
    r'(?:\s+TRACKING\s+(?P<tracking_method>\S+))?'
    r'(?:\s+FILTER\s+(?P<filter>\d+))?'
    r'(?:\s+(?P<input_flag>INPUT))?'
)
START_RECORDING_REGEX = (
    r'START\s+(?P<timestamp>(\d+[.]?\d*))\s+(RIGHT|LEFT)\s+(?P<types>.*)'
)
STOP_RECORDING_REGEX = (
    r'END\s+(?P<timestamp>(\d+[.]?\d*))\s+(?P<types>.*)\s+RES\s+'
    r'(?P<xres>[\d\.]*)\s+(?P<yres>[\d\.]*)\s*'
)

# General message format
MSG_REGEX = (
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+(?P<content>.*)'
)


def _match_regex(
        pattern: str | re.Pattern[str],
        line: str,
        flags: int = 0,
) -> re.Match[str] | None:
    """Match a parser regex, compiling string patterns on demand."""
    if isinstance(pattern, str):
        return re.compile(pattern, flags).match(line)
    return pattern.match(line)


def _search_regex(
        pattern: str | re.Pattern[str],
        line: str,
        flags: int = 0,
) -> re.Match[str] | None:
    """Search with a parser regex, compiling string patterns on demand."""
    if isinstance(pattern, str):
        return re.compile(pattern, flags).search(line)
    return pattern.search(line)


def _eyelink_meta_regexes() -> list[dict[str, re.Pattern[str]]]:
    """Return lazily compiled default EyeLink metadata regex dictionaries."""
    return [{'pattern': re.compile(regex)} for regex in EYELINK_META_REGEXES]


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


def parse_eyelink_event_start(line: str) -> tuple[str, str, float] | None:
    """Check if the line contains the start of an event and return the event name, eye and time.

    Returns a tuple (event_name, eye, timestamp) where eye is 'left' or 'right'.
    Example: ('fixation', 'left', 100.0)
    """
    if match := _match_regex(FIXATION_START_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'fixation', eye_str, float(match.group('timestamp'))
    if match := _match_regex(SACCADE_START_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'saccade', eye_str, float(match.group('timestamp'))
    if match := _match_regex(BLINK_START_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return 'blink', eye_str, float(match.group('timestamp'))
    return None


def parse_eyelink_event_end(line: str) -> tuple[str, str, float, float] | None:
    """Check if the line contains the end of an event and return the event name, eye and times.

    Returns a tuple (event_name, eye, onset, offset). Example: ('fixation', 'left', 123.0, 130.0)
    """
    if match := _match_regex(FIXATION_STOP_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'fixation',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    if match := _match_regex(SACCADE_STOP_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'saccade',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    if match := _match_regex(BLINK_STOP_REGEX, line):
        eye = match.group('eye').upper()
        eye_str = 'left' if eye == 'L' else 'right'
        return (
            'blink',
            eye_str,
            float(match.group('timestamp_start')),
            float(match.group('timestamp_end')),
        )
    return None


def _check_patterns(line: str, compiled_patterns: list[dict[str, Any]]) -> dict[str, Any]:
    """Check line against compiled patterns and return matched context."""
    context = {}
    for pattern_dict in compiled_patterns:
        if match := pattern_dict['pattern'].match(line):
            if 'value' in pattern_dict:
                context[pattern_dict['column']] = pattern_dict['value']
            else:
                context.update(match.groupdict())
    return context


def _match_events_with_context(
        event_starts: list[tuple[str, str, float]],
        event_ends: list[tuple[str, str, float, float]],
        context_timeline: dict[float, dict[str, Any]],
        additional_columns: set[str],
) -> list[dict[str, Any]]:
    """Match event starts and ends and build complete events using a stack per eye and type.

    This function pairs event start markers (SFIX/SSACC/SBLINK) with their corresponding
    end markers (EFIX/ESACC/EBLINK) using a stack-based approach. For each event type
    and eye, starts are pushed onto a stack and matched with ends by popping (LIFO).

    Parameters
    ----------
    event_starts : list[tuple[str, str, float]]
        List of event starts as tuples of (event_name, eye, timestamp).
    event_ends : list[tuple[str, str, float, float]]
        List of event ends as tuples of (event_name, eye, onset, offset).
    context_timeline : dict[float, dict[str, Any]]
        Dictionary mapping timestamps to context dictionaries containing additional
        column values at that point in time.
    additional_columns : set[str]
        Set of additional column names to include in the output events.

    Returns
    -------
    list[dict[str, Any]]
        List of matched event dictionaries with keys: name, eye, onset, offset,
        and any additional columns from the context.
    """
    matched_events: list[dict[str, Any]] = []

    # Stacks for each event type and eye: (event_name, eye) -> list of start_timestamps
    stacks: dict[tuple[str, str], list[float]] = defaultdict(list)

    # Process events in chronological order.
    # Combining starts and ends into a single timeline of event-related actions
    timeline: list[tuple[float, str, Any]] = []
    for start in event_starts:
        timeline.append((start[2], 'start', start))
    for end in event_ends:
        timeline.append((end[3], 'end', end))

    # Sort by timestamp, then by action type ('start' before 'end' for same timestamp)
    timeline.sort(key=lambda x: (x[0], 0 if x[1] == 'start' else 1))

    for _, action, data in timeline:
        if action == 'start':
            event_name, eye, start_ts = data
            stacks[(event_name, eye)].append(start_ts)
        else:
            event_name, eye, onset, offset = data
            stack = stacks[(event_name, eye)]

            if stack:
                # Normal case: having a matching start.
                start_ts = stack.pop()  # (last-in-first-out)
                # For EyeLink, onset in EFIX/ESACC/EBLINK line should match start_ts from
                # SFIX/SSACC/SBLINK but, use the onset from the 'end' line
                # as it's more complete (duration etc)
                event_onset = onset
            else:
                # Orphaned end: no matching start
                warnings.warn(
                    f"Missing start marker before end for event '{event_name}' "
                    f'(onset={onset}, offset={offset}). '
                    'Using context from end timestamp.',
                )
                event_onset = onset
                start_ts = onset  # fallback

            # Build event context: prefer end context, but fill with start context
            start_context = context_timeline.get(start_ts, {})
            end_context = context_timeline.get(offset, {})

            # Current behaviour: if it was a matched event, it uses context from the start
            # If it was unmatched, it uses context from END.
            if stack or (
                not stack and start_ts ==
                onset and onset in context_timeline
            ):  # Case for matched (we just popped) - uses context from START
                event_context = {}
                for col in additional_columns:
                    event_context[col] = start_context.get(col)
            else:
                # Unmatched case: uses context from END
                event_context = {}
                for col in additional_columns:
                    event_context[col] = end_context.get(col)

            matched_events.append({
                'name': f'{event_name}_eyelink',
                'eye': eye,
                'onset': event_onset,
                'offset': offset,
                **event_context,
            })

    return matched_events


def _parse_tracked_eye_layout(tracked: str) -> str | None:
    r"""Map a SAMPLES config tracked-eye token to an eye layout.

    Parameters
    ----------
    tracked: str
        Upper-cased tracked-eye token from a SAMPLES config line
        (e.g. ``LEFT``, ``RIGHT``, ``LEFT\\tRIGHT``, ``LR``).

    Returns
    -------
    str | None
        ``'left'``, ``'right'`` or ``'both'``, or None for unrecognized tokens.
    """
    if ('LEFT' in tracked and 'RIGHT' in tracked) or tracked in {'LR', 'L R'}:
        return 'both'
    if tracked in {'LEFT', 'L'}:
        return 'left'
    if tracked in {'RIGHT', 'R'}:
        return 'right'
    return None


def _migrate_samples_to_binocular(samples: dict[str, list[Any]], prev_eye: str | None) -> None:
    """Switch the sample column schema from monocular to binocular.

    Samples collected so far were parsed as monocular and belong to ``prev_eye``; they are
    moved to that eye's channels and the other eye is filled with NaN. In standard EyeLink
    ASC files the binocular SAMPLES config precedes all sample lines, so no samples have
    been collected yet and no migration happens, but the code below stays correct if that
    assumption does not hold.

    Parameters
    ----------
    samples: dict[str, list[Any]]
        Dictionary of sample columns to migrate in place.
    prev_eye: str | None
        Eye the already-collected monocular samples belong to (``'left'`` or ``'right'``).
        None (no prior samples) defaults to the left eye.
    """
    prev_x = samples.pop('x_pix', [])
    prev_y = samples.pop('y_pix', [])
    prev_pupil = samples.pop('pupil', [])
    n_prev = len(prev_x)
    if prev_eye == 'right':
        samples['x_left_pix'] = [math.nan] * n_prev
        samples['y_left_pix'] = [math.nan] * n_prev
        samples['pupil_left'] = [math.nan] * n_prev
        samples['x_right_pix'] = prev_x
        samples['y_right_pix'] = prev_y
        samples['pupil_right'] = prev_pupil
    else:
        samples['x_left_pix'] = prev_x
        samples['y_left_pix'] = prev_y
        samples['pupil_left'] = prev_pupil
        samples['x_right_pix'] = [math.nan] * n_prev
        samples['y_right_pix'] = [math.nan] * n_prev
        samples['pupil_right'] = [math.nan] * n_prev


def parse_eyelink(
        filepath: Path | str,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str | None = None,
        messages: bool | Sequence[str] = False,
        extend_resolution: bool | None = None,
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
    extend_resolution: bool | None
        Flag indicating if the screen resolution should be extended by 1 pixel.
        If None, the resolution is extended unless the file was recorded by libeyelink.py.
        (default: None)

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
    # pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
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

    compiled_metadata_patterns.extend(_eyelink_meta_regexes())

    # Flag: whether file was recorded by libeyelink.py (PyGaze)
    recorded_by_libeyelink = False

    # Event collection for deterministic matching
    context_timeline: dict[float, dict[str, Any]] = {}
    event_starts: list[tuple[str, str, float]] = []
    event_ends: list[tuple[str, str, float, float]] = []

    if not isinstance(messages, (bool, list)) or (
        isinstance(messages, list)
        and not all(isinstance(regexp, str) for regexp in messages)
    ):
        raise ValueError(
            'Make sure to pass either a bool or a list of regular expressions '
            f'as strings. Received {messages}.',
        )

    messages_list: list[list[str]] = []

    cal_timestamp = ''

    validations = []
    calibrations = []
    recording_config: list[dict[str, Any]] = []
    samples_config: list[dict[str, Any]] = []

    total_recording_duration = 0.0
    is_binocular = False
    # Eye layout of the current SAMPLES block ('left', 'right' or 'both') and the single
    # monocular eye seen so far. If the tracked eye changes within a file (LEFT then RIGHT,
    # or a monocular block followed by a binocular one), the whole recording is promoted to
    # a binocular (four-channel) layout, with the non-tracked eye set to NaN for monocular
    # blocks. See https://github.com/pymovements/pymovements/issues/1401.
    current_sample_eye: str | None = None
    mono_eye_seen: str | None = None

    # Single pass: collect events, patterns, samples, samples config, and metadata
    for line in lines:
        # Collect event starts/ends for deterministic matching
        # Store context BEFORE processing this line's patterns (context is from previous lines)
        if start_event := parse_eyelink_event_start(line):
            event_name, eye, timestamp = start_event
            event_starts.append((event_name, eye, timestamp))
            context_timeline[timestamp] = {**current_additional}

        elif end_event := parse_eyelink_event_end(line):
            event_name, eye, onset, offset = end_event
            event_ends.append((event_name, eye, onset, offset))
            context_timeline[offset] = {**current_additional}

        # Then process patterns and update context for subsequent lines
        matched_ctx = _check_patterns(line, compiled_patterns)
        if matched_ctx:
            current_additional.update(matched_ctx)

        # Detect the tracking configuration independently of the elif chain below, so a
        # SAMPLES config line is never missed (e.g. if it directly follows a calibration
        # line while cal_timestamp is still set). The switch to binocular has to happen
        # before this config block's samples are parsed further down the loop.
        if samples_match := _search_regex(SAMPLES_CONFIG_REGEX, line, re.IGNORECASE):
            samples_config.append(samples_match.groupdict())
            tracked = samples_match.group('tracked_eye').upper().strip()
            if (layout := _parse_tracked_eye_layout(tracked)) is not None:
                if not is_binocular:
                    if layout == 'both':
                        _migrate_samples_to_binocular(samples, mono_eye_seen)
                        is_binocular = True
                    elif mono_eye_seen is None:
                        mono_eye_seen = layout
                    elif mono_eye_seen != layout:
                        _migrate_samples_to_binocular(samples, mono_eye_seen)
                        is_binocular = True
                current_sample_eye = layout

        if cal_timestamp:
            # if a calibration timestamp has been found, the next line will be a
            # calibration pattern, if not, there will only be the timestamp added to the overview

            # very ugly pylint solution
            calibrations.append(
                {
                    'timestamp': cal_timestamp,
                    **match.groupdict(),
                }
                if (match := _match_regex(CALIBRATION_REGEX, line))
                else {'timestamp': cal_timestamp},
            )
            cal_timestamp = ''

        elif match := _match_regex(RECORDING_CONFIG_REGEX, line):
            # Drop optional groups that weren't present for legacy behaviour
            rec_cfg = {k: v for k, v in match.groupdict().items() if v is not None}
            recording_config.append(rec_cfg)

        elif match := _match_regex(GAZE_COORDS_REGEX, line):
            left, top, right, bottom = (float(coord) for coord in match.group('resolution').split())
            # GAZE_COORDS is typically logged after RECCFG - if not, warn and skip assignment.
            if not recording_config:
                warnings.warn(
                    'GAZE_COORDS encountered before any RECCFG. Skipping resolution assignment.',
                )
            else:
                width = right - left
                height = bottom - top

                # Resolution handling depends on recorder implementation.
                # - Standard EyeLink GAZE_COORDS list the highest pixel index (0-based),
                #   so we must increment to obtain the resolution.
                # - PyGaze (libeyelink.py) logs exact resolution; do not increment there.
                # - If extend_resolution is provided, it overrides this logic.
                if (extend_resolution is True) or (
                    extend_resolution is None and not recorded_by_libeyelink
                ):
                    width += 1
                    height += 1

                recording_config[-1]['resolution'] = (width, height)

        elif match := _match_regex(START_RECORDING_REGEX, line):
            start_recording_timestamp = match.groupdict()['timestamp']

        elif match := _match_regex(STOP_RECORDING_REGEX, line):
            stop_recording_timestamp = match.groupdict()['timestamp']

            try:
                block_duration = float(stop_recording_timestamp) - float(start_recording_timestamp)
            except UnboundLocalError:
                warnings.warn(
                    'END recording message without associated START recording message. '
                    f"File '{filepath}' may be corrupted. "
                    'Total recording duration may be incorrect.',
                )
            else:  # this will only be executed if no exception was raised in the try block.
                total_recording_duration += block_duration

        if messages and (match := _match_regex(MSG_REGEX, line)):
            messages_list.append([match.groupdict()['timestamp'], match.groupdict()['content']])

        # Parse the current block with the binocular regex only if the current SAMPLES
        # config is itself binocular. A monocular block within a promoted binocular file is
        # still parsed with the monocular regex and routed to its tracked eye.
        current_block_binocular = current_sample_eye == 'both'
        eye_tracking_sample_match = (
            _match_regex(EYE_TRACKING_SAMPLE_BINOCULAR, line)
            if current_block_binocular else
            _match_regex(EYE_TRACKING_SAMPLE_MONOCULAR, line)
        )

        if eye_tracking_sample_match:
            timestamp_s = eye_tracking_sample_match.group('time')

            for additional_column in additional_columns:
                samples[additional_column].append(current_additional[additional_column])

            if not is_binocular:
                # Monocular file: a single tracked eye throughout, three channels.
                x_pix = check_nan(eye_tracking_sample_match.group('x_pix'))
                y_pix = check_nan(eye_tracking_sample_match.group('y_pix'))
                pupil = check_nan(eye_tracking_sample_match.group('pupil'))

                samples['x_pix'].append(x_pix)
                samples['y_pix'].append(y_pix)
                samples['pupil'].append(pupil)
            elif current_block_binocular:
                # Binocular block: both eyes present in the sample line.
                x_left_pix = check_nan(eye_tracking_sample_match.group('x_pix_left'))
                y_left_pix = check_nan(eye_tracking_sample_match.group('y_pix_left'))
                pupil_left = check_nan(eye_tracking_sample_match.group('pupil_left'))
                x_right_pix = check_nan(eye_tracking_sample_match.group('x_pix_right'))
                y_right_pix = check_nan(eye_tracking_sample_match.group('y_pix_right'))
                pupil_right = check_nan(eye_tracking_sample_match.group('pupil_right'))

                samples['x_left_pix'].append(x_left_pix)
                samples['y_left_pix'].append(y_left_pix)
                samples['pupil_left'].append(pupil_left)
                samples['x_right_pix'].append(x_right_pix)
                samples['y_right_pix'].append(y_right_pix)
                samples['pupil_right'].append(pupil_right)
            else:
                # Monocular block within a promoted binocular file: route the sample to its
                # tracked eye and fill the other eye with NaN.
                x_pix = check_nan(eye_tracking_sample_match.group('x_pix'))
                y_pix = check_nan(eye_tracking_sample_match.group('y_pix'))
                pupil = check_nan(eye_tracking_sample_match.group('pupil'))

                if current_sample_eye == 'right':
                    samples['x_left_pix'].append(math.nan)
                    samples['y_left_pix'].append(math.nan)
                    samples['pupil_left'].append(math.nan)
                    samples['x_right_pix'].append(x_pix)
                    samples['y_right_pix'].append(y_pix)
                    samples['pupil_right'].append(pupil)
                else:
                    samples['x_left_pix'].append(x_pix)
                    samples['y_left_pix'].append(y_pix)
                    samples['pupil_left'].append(pupil)
                    samples['x_right_pix'].append(math.nan)
                    samples['y_right_pix'].append(math.nan)
                    samples['pupil_right'].append(math.nan)

            timestamp = float(timestamp_s)
            samples['time'].append(timestamp)

        elif match := _match_regex(CALIBRATION_TIMESTAMP_REGEX, line):
            cal_timestamp = match.groupdict()['timestamp']

        elif match := _match_regex(VALIDATION_REGEX, line):
            validations.append(match.groupdict())

        elif compiled_metadata_patterns:
            for pattern_dict in compiled_metadata_patterns.copy():
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict:
                        metadata[pattern_dict['key']] = pattern_dict['value']

                    else:
                        metadata.update(match.groupdict())

                    # Check for libeyelink recorder to skip resolution increment
                    if 'recorded_by' in match.groupdict():
                        recorded_by = match.groupdict()['recorded_by'].strip()
                        metadata['recorded_by'] = recorded_by
                        if recorded_by.lower().startswith('libeyelink.py'):
                            recorded_by_libeyelink = True

                    # each metadata pattern should only match once
                    compiled_metadata_patterns.remove(pattern_dict)

    if not metadata:
        warnings.warn('No metadata found. Please check the file for errors.')

    # Match events using collected starts/ends and context timeline
    matched_events = _match_events_with_context(
        event_starts,
        event_ends,
        context_timeline,
        additional_columns,
    )
    for event in matched_events:
        for key, value in event.items():
            events[key].append(value)

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
    # the actual tracked eye is in the samples config, not in the recording config
    # the recording config contains the eyes that were recorded
    # RECCFG uses L/R/LR, SAMPLES uses LEFT/RIGHT/LEFT RIGHT
    if is_binocular:
        # Either a genuinely binocular recording or one where the tracked eye changes
        # within the file; both use a binocular (four-channel) layout and are reported as
        # 'LR'. Running the consistency helpers over the tracked eye would emit spurious
        # 'inconsistent values' warnings for the changing case, so derive the eyes directly
        # from the observed configs instead. See issue #1401.
        metadata['tracked_eye'] = 'LR'
        recorded_eyes = {d.get('tracked_eye') for d in recording_config if d.get('tracked_eye')}
        if not recorded_eyes:
            metadata['recorded_eye'] = None
        elif len(recorded_eyes) == 1:
            metadata['recorded_eye'] = recorded_eyes.pop()
        else:
            metadata['recorded_eye'] = 'LR'
    else:
        metadata['recorded_eye'] = _check_reccfg_key(recording_config, 'tracked_eye')
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

    event_schema_overrides = {
        'name': pl.String,
        'eye': pl.String,
        'onset': pl.Float64,
        'offset': pl.Float64,
    }

    if schema is not None:
        gaze_schema_overrides.update(schema)
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
